"""Is it training, and who is contributing? (docs/03-roadmap.md M5)

Two of M5's exit criteria, in one command: answer that question in a glance,
and be told about a stalled run without having to look.

Nothing here maintains state. Every number is derived from the coordinator's
own tables at read time, which is 6.11's rule -- an inventory that is written
down separately is an inventory that goes stale and then lies. ``--alert`` adds
no new facts either; it just decides which of them are worth waking someone up
for, and exits non-zero, so cron is the whole scheduler.

What counts as stalled
----------------------
This is the hard part, and getting it wrong in the obvious direction would make
the alert useless. **An idle fleet is normal operation** -- 3.2 is explicit
that contributors come and go, that a round can sit open overnight, and that
this is the design working rather than failing. An alert that fires on
idleness would fire most nights, and would train the operator to ignore the one
that matters. ``invariants.py`` deliberately says nothing about idleness for
exactly this reason.

So a stall here needs two things at once: a round that should have advanced,
*and* evidence that workers were awake while it did not. The second half comes
from ``worker_eligibility``, which records every poll -- so "nobody asked for
work" and "workers asked and the round did not move" are finally
distinguishable. The second is the M4a wedge signature: every worker polling,
every poll answered 204, the round open forever, and nothing anywhere
returning an error.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from ganymede.coordinator import eligibility, invariants

# How far past its own backstop a round has to be before a stall is suspected.
# A generous multiple, because max_round_sec is a target rather than a
# deadline: the close fires on the next request after the backstop passes, and
# on a slow fleet that request can be minutes away for entirely healthy reasons.
STALL_GRACE_MULT = 3.0

# A worker that polled within this window counts as awake. Comfortably longer
# than the default poll interval, so one missed poll does not make a live fleet
# look asleep and silence a real stall.
AWAKE_WINDOW_SEC = 900

# Below this, a run is not really a swarm. 3.2 makes distinct contributors per
# round the number that says whether the overhead is being earned: if most
# rounds close with one machine, a single-node job would have been simpler and
# faster, and that should be visible rather than inferred months later.
COHORT_FLOOR = 2


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        parsed = datetime.fromisoformat(ts)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@dataclass
class RoundLine:
    idx: int
    status: str
    contributors: int
    eval_loss: float | None
    divergence: float | None
    age_sec: float | None


@dataclass
class RunStatus:
    run_id: str
    base_model: str
    current_round: int
    target_rounds: int
    recent: list[RoundLine] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)

    @property
    def median_cohort(self) -> float:
        closed = sorted(r.contributors for r in self.recent if r.status == "closed")
        if not closed:
            return 0.0
        mid = len(closed) // 2
        return float(closed[mid] if len(closed) % 2 else (closed[mid - 1] + closed[mid]) / 2)

    @property
    def loss_trend(self) -> str:
        losses = [r.eval_loss for r in reversed(self.recent) if r.eval_loss is not None]
        if len(losses) < 2:
            return "not enough evaluated rounds"
        delta = losses[-1] - losses[0]
        arrow = "down" if delta < 0 else "up"
        return f"{losses[0]:.3f} -> {losses[-1]:.3f} ({arrow} {abs(delta):.3f} over {len(losses)})"


@dataclass(frozen=True)
class Stall:
    run_id: str
    round_idx: int
    detail: str

    def __str__(self) -> str:
        return f"{self.run_id}#{self.round_idx}: {self.detail}"


def run_status(conn: sqlite3.Connection, run_id: str, *, rounds_shown: int = 5,
               now: datetime | None = None) -> RunStatus:
    now = now or datetime.now(timezone.utc)
    run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if run is None:
        raise KeyError(run_id)

    rows = conn.execute(
        """SELECT idx, status, distinct_contributors, eval_loss, adapter_divergence,
                  opened_at, closed_at
           FROM rounds WHERE run_id = ? ORDER BY idx DESC LIMIT ?""",
        (run_id, rounds_shown),
    ).fetchall()

    recent = []
    for r in rows:
        opened = _parse(r["opened_at"])
        closed = _parse(r["closed_at"])
        age = None
        if opened is not None:
            age = ((closed or now) - opened).total_seconds()
        recent.append(RoundLine(
            idx=r["idx"], status=r["status"], contributors=r["distinct_contributors"],
            eval_loss=r["eval_loss"], divergence=r["adapter_divergence"], age_sec=age,
        ))

    return RunStatus(
        run_id=run_id,
        base_model=run["base_model"],
        current_round=run["current_round"],
        target_rounds=run["target_rounds"],
        recent=recent,
        coverage=invariants.coverage(conn, run_id),
    )


def awake_workers(conn: sqlite3.Connection, *, window_sec: int = AWAKE_WINDOW_SEC,
                  now: datetime | None = None) -> list[str]:
    """Workers that have polled recently.

    Derived from the eligibility table rather than from a heartbeat or a
    "last seen" column, because a poll is the only event every worker
    generates whether or not it is given work -- a machine that is refused
    every time is exactly the one an operator most wants counted as present.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = (now - timedelta(seconds=window_sec)).isoformat()
    rows = conn.execute(
        """SELECT DISTINCT worker_id FROM worker_eligibility
           WHERE checked_at >= ? ORDER BY worker_id""",
        (cutoff,),
    ).fetchall()
    return [r["worker_id"] for r in rows]


def stalls(conn: sqlite3.Connection, *, now: datetime | None = None,
           grace_mult: float = STALL_GRACE_MULT,
           awake_window_sec: int = AWAKE_WINDOW_SEC) -> list[Stall]:
    """Rounds that should have advanced while workers were awake.

    Both halves are required. Without the "awake" half this fires every night
    on a healthy volunteer fleet; without the "overdue" half it never fires at
    all. Neither half alone is worth an alert.
    """
    now = now or datetime.now(timezone.utc)
    awake = awake_workers(conn, window_sec=awake_window_sec, now=now)
    if not awake:
        # Nobody is asking for work. That is 3.2's normal operation, not a
        # fault, and nothing below applies.
        return []

    found = []
    for r in conn.execute(
        """SELECT rounds.run_id, rounds.idx, rounds.opened_at, rounds.max_round_sec
           FROM rounds JOIN runs ON runs.id = rounds.run_id
           WHERE rounds.status = 'open' AND runs.status = 'active'"""
    ).fetchall():
        opened = _parse(r["opened_at"])
        if opened is None:
            continue
        age = (now - opened).total_seconds()
        limit = r["max_round_sec"] * grace_mult
        if age > limit:
            found.append(Stall(
                r["run_id"], r["idx"],
                f"open {age / 60:.0f} min, {grace_mult:g}x past its "
                f"{r['max_round_sec'] / 60:.0f} min backstop, while "
                f"{len(awake)} worker(s) were polling",
            ))
    return found


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _render(conn: sqlite3.Connection, run_ids: list[str], now: datetime) -> list[str]:
    out: list[str] = []
    awake = awake_workers(conn, now=now)
    out.append(f"{len(awake)} worker(s) polled in the last {AWAKE_WINDOW_SEC // 60} min")

    for run_id in run_ids:
        st = run_status(conn, run_id, now=now)
        out.append("")
        out.append(f"{st.run_id}  {st.base_model}  round {st.current_round}/{st.target_rounds}")
        cov = st.coverage
        if cov:
            out.append(
                f"  coverage   {cov['distinct_trained']}/{cov['buckets']} buckets, "
                f"spread {cov['spread']}"
            )
        out.append(f"  loss       {st.loss_trend}")

        cohort = st.median_cohort
        note = ""
        if 0 < cohort < COHORT_FLOOR:
            # 3.2's question, asked out loud. A swarm that closes every round
            # with one machine is an expensive way to run a single-node job.
            note = "  <- most rounds are closing with one machine"
        out.append(f"  cohort     median {cohort:g} contributors/round{note}")

        out.append("  round  state     cohort   loss      divergence  age")
        for r in st.recent:
            loss = f"{r.eval_loss:.4f}" if r.eval_loss is not None else "   --   "
            div = f"{r.divergence:.4f}" if r.divergence is not None else "   --   "
            age = f"{r.age_sec / 60:.0f}m" if r.age_sec is not None else "--"
            out.append(f"  {r.idx:>5}  {r.status:<9} {r.contributors:>6}   {loss}  {div}    {age}")
    return out


def _active_runs(conn: sqlite3.Connection) -> list[str]:
    return [r["id"] for r in conn.execute(
        "SELECT id FROM runs WHERE status = 'active' ORDER BY created_at"
    ).fetchall()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ganymede-status",
        description="Is it training, and who is contributing? (roadmap M5)",
    )
    p.add_argument("--db", required=True)
    p.add_argument("--run-id", help="one run; default is every active run")
    p.add_argument("--json", action="store_true")
    p.add_argument(
        "--alert", action="store_true",
        help="print only what is wrong and exit non-zero if anything is. For cron.",
    )
    p.add_argument("--stall-grace-mult", type=float, default=STALL_GRACE_MULT)
    args = p.parse_args(argv)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    now = datetime.now(timezone.utc)

    run_ids = [args.run_id] if args.run_id else _active_runs(conn)
    violations = invariants.check(conn, now=now)
    stalled = stalls(conn, now=now, grace_mult=args.stall_grace_mult)

    if args.alert:
        # Silence is the success case. A cron job that mails a report every
        # fifteen minutes is a cron job whose mail gets filtered, and then the
        # one that mattered gets filtered too.
        for v in violations:
            print(v, file=sys.stderr)
        for s in stalled:
            print(f"stalled: {s}", file=sys.stderr)
        return 1 if (violations or stalled) else 0

    if args.json:
        print(json.dumps({
            "checked_at": now.isoformat(),
            "awake_workers": awake_workers(conn, now=now),
            "runs": [
                {
                    "run_id": st.run_id,
                    "base_model": st.base_model,
                    "current_round": st.current_round,
                    "target_rounds": st.target_rounds,
                    "median_cohort": st.median_cohort,
                    "coverage": st.coverage,
                    "rounds": [vars(r) for r in st.recent],
                }
                for st in (run_status(conn, rid, now=now) for rid in run_ids)
            ],
            "violations": [{"check": v.check, "detail": v.detail} for v in violations],
            "stalls": [{"run_id": s.run_id, "round_idx": s.round_idx, "detail": s.detail}
                       for s in stalled],
            "refusals": eligibility.fleet_summary(conn),
        }, indent=2, default=str))
        return 1 if (violations or stalled) else 0

    if not run_ids:
        print("no active runs")
    else:
        print("\n".join(_render(conn, run_ids, now)))

    refusals = eligibility.fleet_summary(conn)
    if refusals:
        print("\nrefusals across the fleet:")
        for reason, workers in sorted(refusals.items(), key=lambda kv: -len(kv[1])):
            print(f"  {len(workers):3d}  {reason}")

    if violations or stalled:
        print()
        for v in violations:
            print(f"BROKEN  {v}")
        for s in stalled:
            print(f"STALLED {s}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
