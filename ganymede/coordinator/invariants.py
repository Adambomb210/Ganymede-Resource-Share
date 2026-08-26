"""What "the protocol is broken" means, checked against the database alone.

M4a's exit criteria are stated as things that must *not* happen -- no
double-leasing, no stuck tasks, coverage that advances rather than repeating --
and every one of them is a statement about coordinator state rather than about
any single request. So they are checked here, over the database, rather than
asserted inside a test that happens to be watching at the right moment.

Three things follow from writing them down in one place:

* The concurrency harness asserts against the same definitions an operator
  would run by hand, so a green test and a healthy deployment mean the same
  thing.
* M5's "alert on broken, not on idle" has something concrete to alert on. An
  unscheduled fleet is idle most of the time and that is not an incident; a
  round wedged in ``closing`` is one, and no amount of waiting fixes it.
* A violation names the rows involved, because the useful question on seeing
  one is always "which task?".

**Transients are not violations.** A lease is routinely a little past its
expiry -- nothing reclaims it until the next claim arrives -- and a round sits
in ``closing`` for as long as aggregation takes. Both checks take a grace
period for that reason, and both default to a value comfortably longer than the
operation should ever take.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

from ganymede.coordinator.db import connect

# How long past its expiry a lease may sit before it counts as stuck. Leases are
# reclaimed lazily, on the next claim (app.claim -> rounds.expire_leases), so on
# a fleet with nothing running they linger by design.
LEASE_GRACE_SEC = 300

# How long a round may stay in 'closing'. Closing downloads every submitted
# adapter, combines them and uploads the result, so it is not instant.
#
# A close that *raises* gives the round back on its way out (closer.close_round),
# so this check is not the first line of defence -- it is the second. What it
# catches is the case with no exception to catch: a coordinator killed between
# claiming 'closing' and releasing it. That leaves the round unclaimable and
# unreopenable, with nothing logged and no request returning an error.
CLOSING_GRACE_SEC = 600


@dataclass(frozen=True)
class Violation:
    check: str
    detail: str
    rows: list[str]

    def __str__(self) -> str:
        listed = ", ".join(self.rows[:6])
        more = f" (+{len(self.rows) - 6} more)" if len(self.rows) > 6 else ""
        return f"{self.check}: {self.detail} [{listed}{more}]"


def _parse(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------


def _one_lease_per_worker_per_round(conn: sqlite3.Connection) -> list[Violation]:
    """A worker holds at most one lease in a round.

    ``claim_task`` returns the lease a worker already holds rather than issuing
    a second one, so a worker retrying after a network blip resumes instead of
    forking its own work into two shards. Two live leases for one worker means
    that guard lost a race -- and the two halves would both submit, both pass
    the gates, and both be weighted as independent contributions.
    """
    rows = conn.execute(
        """SELECT worker_id, run_id, round_idx, COUNT(*) AS n,
                  GROUP_CONCAT(id) AS ids
           FROM tasks
           WHERE status = 'leased' AND worker_id IS NOT NULL
           GROUP BY worker_id, run_id, round_idx
           HAVING n > 1"""
    ).fetchall()
    return [
        Violation(
            "double_lease",
            f"worker {r['worker_id']} holds {r['n']} leases in round "
            f"{r['run_id']}#{r['round_idx']}",
            (r["ids"] or "").split(","),
        )
        for r in rows
    ]


def _coverage_before_repetition(conn: sqlite3.Connection) -> list[Violation]:
    """No bucket is trained twice in a round while another goes untouched.

    Not "no two workers ever share a shard" -- that is not something the design
    promises, and it cannot: a worker whose budget is large enough to want every
    bucket will be assigned every bucket, and a second worker in the same round
    then has nowhere else to go. Pigeonholes, not a bug.

    What *is* promised is the ordering. ``_pick_buckets`` is least-trained-first
    and the counter moves at claim rather than at submit, so a fresh bucket is
    always preferred to a repeat. A round holding one bucket twice while another
    sits at zero means that preference lost: two workers trained identical rows,
    the aggregate weighted that duplicated gradient twice, and a shard nobody
    touched still counts as covered. None of that shows up in any metric.

    Expired and abandoned tasks are excluded -- their shards are *meant* to come
    back round.
    """
    held: dict[tuple[str, int], dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in conn.execute(
        """SELECT id, run_id, round_idx, buckets_json FROM tasks
           WHERE status IN ('leased', 'submitted')"""
    ):
        for bucket in json.loads(row["buckets_json"]):
            held[(row["run_id"], row["round_idx"])][int(bucket)].append(row["id"])

    violations = []
    for (run_id, idx), by_bucket in sorted(held.items()):
        total = conn.execute(
            "SELECT num_buckets FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        if total is None:
            continue
        untouched = int(total["num_buckets"]) - len(by_bucket)
        repeated = {b: ids for b, ids in by_bucket.items() if len(ids) > 1}
        if repeated and untouched > 0:
            violations.append(
                Violation(
                    "repeat_before_coverage",
                    f"round {run_id}#{idx} trains {len(repeated)} bucket(s) more than "
                    f"once while {untouched} bucket(s) go untouched",
                    sorted({i for ids in repeated.values() for i in ids}),
                )
            )
    return violations


def _no_stuck_leases(conn: sqlite3.Connection, now: datetime, grace: int) -> list[Violation]:
    """No lease sits past its expiry long enough to hold a shard hostage."""
    stuck = [
        row["id"]
        for row in conn.execute(
            "SELECT id, lease_expires_at FROM tasks "
            "WHERE status = 'leased' AND lease_expires_at IS NOT NULL"
        )
        if (now - _parse(row["lease_expires_at"])).total_seconds() > grace
    ]
    if not stuck:
        return []
    return [
        Violation(
            "stuck_lease",
            f"{len(stuck)} leases more than {grace}s past expiry and not reclaimed",
            stuck,
        )
    ]


def _no_stuck_closes(conn: sqlite3.Connection, now: datetime, grace: int) -> list[Violation]:
    """No round stays in ``closing``.

    ``closing`` is claimed by exactly one caller with a conditional UPDATE and
    released only when that caller finishes. A close that raises hands the round
    back itself, so what reaches here is the case that cannot: a coordinator
    killed between the two writes. The round is then permanently unclaimable and
    unreopenable, the run stops, and nothing reports an error -- every worker
    simply gets 204 forever.
    """
    stuck = [
        f"{row['run_id']}#{row['idx']}"
        for row in conn.execute("SELECT run_id, idx, opened_at FROM rounds WHERE status = 'closing'")
        if (now - _parse(row["opened_at"])).total_seconds() > grace
    ]
    if not stuck:
        return []
    return [
        Violation(
            "stuck_close",
            f"{len(stuck)} rounds have been closing for more than {grace}s",
            stuck,
        )
    ]


def _closed_rounds_produced_something(conn: sqlite3.Connection) -> list[Violation]:
    """A round that closed with accepted work published a result adapter.

    The chain from one round's result to the next round's base is what makes a
    run a sequence. A closed round with accepted submissions and no result
    broke that chain, and every round after it trained from the wrong base.
    """
    rows = conn.execute(
        """SELECT r.run_id, r.idx, COUNT(s.task_id) AS accepted
           FROM rounds r
           JOIN tasks t       ON t.run_id = r.run_id AND t.round_idx = r.idx
           JOIN submissions s ON s.task_id = t.id AND s.accepted = 1
           WHERE r.status = 'closed' AND r.result_adapter_ref IS NULL
           GROUP BY r.run_id, r.idx"""
    ).fetchall()
    return [
        Violation(
            "no_result_adapter",
            f"round {r['run_id']}#{r['idx']} closed with {r['accepted']} accepted "
            "submissions but published no adapter",
            [f"{r['run_id']}#{r['idx']}"],
        )
        for r in rows
    ]


def _the_run_points_at_a_real_round(conn: sqlite3.Connection) -> list[Violation]:
    """``runs.current_round`` names a round that exists.

    Every claim joins through it. A run pointing at a round that was never
    opened is a run no worker can ever join, and the symptom is a silent 204.
    """
    rows = conn.execute(
        """SELECT runs.id, runs.current_round FROM runs
           LEFT JOIN rounds ON rounds.run_id = runs.id AND rounds.idx = runs.current_round
           WHERE runs.status = 'active' AND rounds.idx IS NULL"""
    ).fetchall()
    return [
        Violation(
            "dangling_current_round",
            f"active run {r['id']} points at round {r['current_round']}, which does not exist",
            [r["id"]],
        )
        for r in rows
    ]


def _submissions_belong_to_submitted_tasks(conn: sqlite3.Connection) -> list[Violation]:
    """Work is recorded only against a lease that was still held.

    ``record_submission`` flips the task to ``submitted`` in the same
    transaction that writes the row, so a submission attached to an expired or
    abandoned task means work was accepted against a lease someone else may
    already have been re-issued.
    """
    rows = conn.execute(
        """SELECT s.task_id, t.status FROM submissions s
           JOIN tasks t ON t.id = s.task_id
           WHERE t.status NOT IN ('submitted', 'expired')"""
    ).fetchall()
    # 'expired' is legitimate: close_round expires every lease still outstanding
    # in the round it just closed, including ones that had already submitted.
    return [
        Violation(
            "orphan_submission",
            f"task {r['task_id']} has a submission but is {r['status']}",
            [r["task_id"]],
        )
        for r in rows
    ]


def check(
    conn: sqlite3.Connection,
    now: datetime | None = None,
    lease_grace_sec: int = LEASE_GRACE_SEC,
    closing_grace_sec: int = CLOSING_GRACE_SEC,
) -> list[Violation]:
    """Every invariant, in one pass. An empty list means healthy."""
    now = now or datetime.now(timezone.utc)
    return [
        *_one_lease_per_worker_per_round(conn),
        *_coverage_before_repetition(conn),
        *_no_stuck_leases(conn, now, lease_grace_sec),
        *_no_stuck_closes(conn, now, closing_grace_sec),
        *_closed_rounds_produced_something(conn),
        *_the_run_points_at_a_real_round(conn),
        *_submissions_belong_to_submitted_tasks(conn),
    ]


# --------------------------------------------------------------------------
# Coverage -- reported, not asserted
# --------------------------------------------------------------------------


def coverage(conn: sqlite3.Connection, run_id: str) -> dict:
    """How evenly a run's buckets have been trained.

    Deliberately not an invariant. Least-trained-first keeps the spread small,
    but a fleet whose workers get very different budgets will legitimately show
    some unevenness, and there is no threshold that is wrong everywhere. What
    *is* diagnostic is the shape: ``distinct_trained`` climbing while ``spread``
    stays small is coverage advancing; ``distinct_trained`` flat while ``max``
    climbs is the same shards being retrained.
    """
    rows = conn.execute(
        "SELECT bucket_idx, times_trained FROM buckets WHERE run_id = ? ORDER BY bucket_idx",
        (run_id,),
    ).fetchall()
    counts = [int(r["times_trained"]) for r in rows]
    if not counts:
        return {"buckets": 0, "distinct_trained": 0, "min": 0, "max": 0, "spread": 0,
                "total": 0, "whole_dataset_tasks": 0}
    # Tasks assigned every bucket the run has. One is unremarkable -- a lone
    # fast worker on a small run legitimately gets the lot. Several in the same
    # round means the run stopped sharding: each worker trained the identical
    # dataset and aggregation weighted the copies as independent contributions.
    # It is invisible in `spread`, which stays at zero precisely because
    # everyone trained everything.
    total_buckets = len(counts)
    whole = conn.execute(
        """SELECT COUNT(*) AS n FROM tasks
           WHERE run_id = ? AND status IN ('leased', 'submitted')
             AND json_array_length(buckets_json) >= ?""",
        (run_id, total_buckets),
    ).fetchone()["n"]
    return {
        "buckets": total_buckets,
        "distinct_trained": sum(1 for c in counts if c > 0),
        "min": min(counts),
        "max": max(counts),
        "spread": max(counts) - min(counts),
        "total": sum(counts),
        "whole_dataset_tasks": int(whole),
    }


def loss_curve(conn: sqlite3.Connection, run_id: str) -> list[tuple[int, float]]:
    """``(round_idx, eval_loss)`` for every closed round that has been evaluated.

    Rounds with no ``eval_loss`` are skipped rather than reported as zero: the
    evaluator (scripts/evalround.py) runs separately from the coordinator, so a
    gap means nobody has evaluated that round yet, not that its loss was zero.
    """
    return [
        (int(r["idx"]), float(r["eval_loss"]))
        for r in conn.execute(
            "SELECT idx, eval_loss FROM rounds "
            "WHERE run_id = ? AND eval_loss IS NOT NULL ORDER BY idx",
            (run_id,),
        )
    ]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python3 -m ganymede.coordinator.invariants",
        description="Check a coordinator database for protocol violations.",
    )
    p.add_argument("--db", default=None, help="path to the database; default is GANYMEDE_DB")
    p.add_argument("--run-id", default=None, help="also report coverage for this run")
    p.add_argument("--lease-grace-sec", type=int, default=LEASE_GRACE_SEC)
    p.add_argument("--closing-grace-sec", type=int, default=CLOSING_GRACE_SEC)
    p.add_argument("--json", action="store_true", help="machine-readable output")
    args = p.parse_args(argv)

    db_path = args.db
    if db_path is None:
        import os

        db_path = os.environ.get("GANYMEDE_DB")
        if not db_path:
            print("no database: pass --db or set GANYMEDE_DB", file=sys.stderr)
            return 2

    conn = connect(db_path)
    try:
        violations = check(
            conn,
            lease_grace_sec=args.lease_grace_sec,
            closing_grace_sec=args.closing_grace_sec,
        )
        report: dict = {"violations": [
            {"check": v.check, "detail": v.detail, "rows": v.rows} for v in violations
        ]}
        if args.run_id:
            report["coverage"] = coverage(conn, args.run_id)
            report["loss_curve"] = loss_curve(conn, args.run_id)
    finally:
        conn.close()

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for v in violations:
            print(v, file=sys.stderr)
        if args.run_id:
            print(f"coverage: {report['coverage']}")
            if report["loss_curve"]:
                print("loss: " + "  ".join(
                    f"r{i}={loss:.4f}" for i, loss in report["loss_curve"]
                ))
        if not violations:
            print("ok")
    # Exit non-zero on violations so this is usable as a cron check.
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
