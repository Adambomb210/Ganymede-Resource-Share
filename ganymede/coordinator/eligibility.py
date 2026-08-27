"""Why a worker never gets work (docs/03-roadmap.md M5, 6.9, 6.10).

M5 names this as one of the two things that must stay honest from the start,
the other being the install path -- both because they are what opening the
fleet up later depends on. A contributor who lends a machine and never sees it
do anything has exactly one question, and "check the logs" is not an answer
when the logs are on somebody else's server.

Recorded, not recomputed
------------------------
Every refusal in this table was produced by ``rounds.claim_task`` on a real
poll and written down verbatim. Nothing here re-derives eligibility.

That distinction is the whole design. A diagnostic that reimplements the
decision it explains will eventually disagree with it, and the disagreement
surfaces as a contributor being told they are eligible for a run that keeps
turning them away -- which is worse than saying nothing, because it moves the
suspicion from their machine to the operator's competence. The claim path
already computes a reason for every run it declines; before this module it
built that list and dropped it on the floor.

Cost
----
One upsert per (worker, run) per poll, into a table with one row per pair. A
ten-machine fleet across two runs is twenty rows, rewritten every poll
interval. The claim path is already opening write transactions per run, so
this adds no new class of contention -- and being able to answer the question
at all is worth more than the write.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

# The three outcomes a poll can have for one run. `leased` and `idle` both mean
# "nothing wrong with this machine", and keeping them apart matters: a fleet
# where everyone is `idle` is a run that has no open round, which is an
# operator problem, while a fleet where everyone is `refused` is a run whose
# requirements nobody can meet, which is a different operator problem.
LEASED = "leased"
IDLE = "idle"
REFUSED = "refused"

SCHEMA = """
CREATE TABLE IF NOT EXISTS worker_eligibility (
    worker_id  TEXT NOT NULL REFERENCES workers(id),
    run_id     TEXT NOT NULL REFERENCES runs(id),
    outcome    TEXT NOT NULL,
    reason     TEXT,
    checked_at TEXT NOT NULL,
    PRIMARY KEY (worker_id, run_id)
);
"""


@dataclass(frozen=True)
class Verdict:
    """One run's answer for one worker, as the claim path decided it."""

    run_id: str
    outcome: str
    reason: str | None = None

    @property
    def eligible(self) -> bool:
        return self.outcome != REFUSED


@dataclass(frozen=True)
class Explanation:
    worker_id: str
    verdicts: list[Verdict]
    checked_at: str | None

    @property
    def any_eligible(self) -> bool:
        return any(v.eligible for v in self.verdicts)

    def __str__(self) -> str:
        if not self.verdicts:
            # Distinct from "refused everywhere", and the distinction is the
            # useful half: a worker with no rows has never completed a poll, so
            # the problem is upstream of eligibility entirely -- the agent is
            # not running, or not reaching the coordinator.
            return (f"worker {self.worker_id}: no polls recorded. "
                    f"The worker has not asked for work since this coordinator started.")
        lines = [f"worker {self.worker_id} (last polled {self.checked_at}):"]
        for v in self.verdicts:
            if v.outcome == LEASED:
                lines.append(f"  {v.run_id}: working -- last poll was handed a task")
            elif v.outcome == IDLE:
                lines.append(f"  {v.run_id}: eligible, but the run had nothing to hand out")
            else:
                lines.append(f"  {v.run_id}: REFUSED -- {v.reason}")
        return "\n".join(lines)


def record(conn: sqlite3.Connection, worker_id: str, verdicts: list[Verdict],
           now: datetime | None = None) -> None:
    """Write down what this poll decided. Never raises.

    Diagnostics must not be able to break the thing they diagnose: a failure
    here would turn a successful claim into a 500 and cost the round, which is
    a strictly worse outcome than not knowing why a worker is idle.
    """
    stamp = (now or datetime.now(timezone.utc)).isoformat()
    try:
        conn.executemany(
            """INSERT INTO worker_eligibility (worker_id, run_id, outcome, reason, checked_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(worker_id, run_id) DO UPDATE SET
                 outcome = excluded.outcome,
                 reason = excluded.reason,
                 checked_at = excluded.checked_at""",
            [(worker_id, v.run_id, v.outcome, v.reason, stamp) for v in verdicts],
        )
        conn.commit()
    except sqlite3.Error:
        pass


def explain(conn: sqlite3.Connection, worker_id: str) -> Explanation:
    rows = conn.execute(
        """SELECT run_id, outcome, reason, checked_at
           FROM worker_eligibility WHERE worker_id = ?
           ORDER BY run_id""",
        (worker_id,),
    ).fetchall()
    verdicts = [Verdict(r["run_id"], r["outcome"], r["reason"]) for r in rows]
    checked = max((r["checked_at"] for r in rows), default=None)
    return Explanation(worker_id=worker_id, verdicts=verdicts, checked_at=checked)


def for_contributor(conn: sqlite3.Connection, contributor_id: str) -> list[Explanation]:
    """Every machine one contributor has registered.

    The unit a person actually asks about. Someone who has installed the agent
    on three machines does not know their worker ids, and should not have to.
    """
    rows = conn.execute(
        "SELECT id FROM workers WHERE contributor_id = ? ORDER BY first_seen",
        (contributor_id,),
    ).fetchall()
    return [explain(conn, r["id"]) for r in rows]


def fleet_summary(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Refusal reason -> the workers hitting it, across the whole fleet.

    The operator's view rather than the contributor's. One machine refused for
    insufficient memory is that machine's business; *every* machine refused for
    insufficient memory is a run whose `requires` block is wrong, and the two
    read identically one worker at a time.
    """
    rows = conn.execute(
        """SELECT worker_id, run_id, reason FROM worker_eligibility
           WHERE outcome = ? ORDER BY reason, worker_id""",
        (REFUSED,),
    ).fetchall()
    grouped: dict[str, list[str]] = {}
    for r in rows:
        # Group on the reason's shape, not its exact text: "vram_mb 4096 <
        # 8000" and "vram_mb 6144 < 8000" are one problem, and keeping the
        # numbers would scatter it across as many buckets as there are cards.
        key = f"{r['run_id']}: {_shape(r['reason'] or '')}"
        grouped.setdefault(key, []).append(r["worker_id"])
    return grouped


def _shape(reason: str) -> str:
    """Strip the numbers out of a reason so like groups with like."""
    out, digits = [], False
    for ch in reason:
        if ch.isdigit():
            if not digits:
                out.append("N")
            digits = True
        else:
            digits = False
            out.append(ch)
    return "".join(out)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ganymede-eligibility",
        description="Why a registered worker is never leased (roadmap M5).",
    )
    p.add_argument("--db", required=True)
    p.add_argument("--worker-id", help="one machine")
    p.add_argument("--contributor-id", help="every machine one person registered")
    p.add_argument("--fleet", action="store_true",
                   help="group refusals across the fleet -- an operator's view")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    if args.fleet:
        summary = fleet_summary(conn)
        if args.json:
            print(json.dumps(summary, indent=2))
        elif not summary:
            print("no worker is being refused for any run")
        else:
            for reason, workers in sorted(summary.items(), key=lambda kv: -len(kv[1])):
                print(f"{len(workers):3d}  {reason}")
                for w in workers[:6]:
                    print(f"       {w}")
                if len(workers) > 6:
                    print(f"       ... and {len(workers) - 6} more")
        return 0

    if args.contributor_id:
        explanations = for_contributor(conn, args.contributor_id)
    elif args.worker_id:
        explanations = [explain(conn, args.worker_id)]
    else:
        p.error("one of --worker-id, --contributor-id or --fleet is required")

    if args.json:
        print(json.dumps([
            {"worker_id": e.worker_id, "checked_at": e.checked_at,
             "verdicts": [{"run_id": v.run_id, "outcome": v.outcome, "reason": v.reason}
                          for v in e.verdicts]}
            for e in explanations
        ], indent=2))
    else:
        for e in explanations:
            print(e)
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
