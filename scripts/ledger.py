"""Run the contribution-ledger sweep (docs/09-ledger.md 1, 5.3).

The provisioned-accrual engine is not a daemon and does no work on the claim
path; it is a periodic sweep driven by this script from cron, next to
``ganymede.coordinator.invariants`` and ``scripts/status.py --alert``. Each run:

1. ``ledger.settle_windows`` -- integrate good-standing ``availability_ticks``
   into settled hour windows and write one ``credit_events`` row each.
2. ``ledger.evaluate_reputation`` -- recompute the per-machine ``reputation``
   scalar from recorded outcomes and drive ``workers.standing`` transitions.
3. ``identity.gc_expired_sessions`` -- drop dead web sessions (docs/08 "runs on
   the audit / availability_ticks cron").
4. ``fairness.recompute_shares`` -- rebuild the share rollup the scheduler reads
   (docs/13 §1.5). Rides here rather than in its own cron for the same reason
   retention rides the image sweep: same cadence, and a separate cron entry for
   one rollup is a thing to forget to install.
5. ``fairness.autopreempt`` -- at most one preemption (docs/13 §4.6). Off unless
   ``GANYMEDE_AUTOPREEMPT``, and it returns before touching the database when it
   is off.

Recommended cadence: once a minute. A 30-minute settle delay plus a minute
sweep means a window settles within a minute of eligibility, and a standing
change lands within the minute a rejection would have justified it. Nothing is
latency-critical -- the sweep is idempotent and recomputes to a fixpoint.

**KNOWN DEFECT, found by review 2026-09-21: step 2 is NOT idempotent, and at
this cadence that is severe.** ``ledger.recompute_reputation`` starts from the
*stored* reputation and then applies *trailing-window totals* --
``_trailing_outcomes``, ``spotcheck.outcomes_for`` and ``_minorities_since``
all return 30-day counts, not deltas, and no watermark column exists anywhere --
so every sweep re-convicts a machine for the same historical events. Solving
the per-rejection recurrence ``s' = (s + inc)*0.5 - REP_REJECT_FLOOR`` gives a
fixed point of ``inc - 0.20``; with spot checks off (the default) ``inc`` is
capped at ``min(acc, 8) * REP_CLEAN_STEP = 0.16``, which is below it. So **any
machine with a single rejection in its 30-day window converges to 0.0 and is
``revoked``** -- terminal for accrual, admin-only to reverse -- after about two
sweeps, i.e. about two minutes at the cadence recommended above, no matter how
much clean work it has done. Reproduced in
``tests/test_ledger.py::test_the_reputation_sweep_currently_revokes_on_repetition_alone``;
the property this paragraph's previous sentence claims is asserted by the
``xfail(strict=True)`` beside it.

Not fixed in place because the fix is a **semantics decision about contributor
credit**, not a typo: either count each event exactly once (a watermark column,
which needs a migration) or make the score a pure function of the window rather
than a mutated scalar -- and the latter changes the "earned slowly, asymptotic
to 1.0" behaviour that the accumulation is currently what produces. Until it is
decided, do not install this on a cron against a fleet you care about.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from ganymede.coordinator import fairness, identity, ledger, spotcheck
from ganymede.coordinator.db import connect


def run(db_path: str, probation_monthly_cap_hours: float, settings=None) -> dict:
    """Execute one sweep, returning a small report dict.

    ``settings`` is optional so the existing two-argument call keeps working;
    without it the Phase D steps that need configuration (autopreempt) sit out,
    while the share rollup -- which needs none -- still runs.
    """
    conn = connect(db_path)
    now = datetime.now(timezone.utc)
    try:
        settled = ledger.settle_windows(
            conn, probation_monthly_cap_hours=probation_monthly_cap_hours, now=now
        )
        # Before reputation, not after: a probe whose task never came back must
        # be retired before the scorer counts it as anything.
        voided = spotcheck.void_stale(conn, now)
        evaluated = ledger.evaluate_reputation(conn, now)
        sessions_gc = identity.gc_expired_sessions(conn)
        shares = fairness.recompute_shares(conn, now)
        preempted = (
            fairness.autopreempt(conn, settings, now) if settings is not None else None
        )
    finally:
        conn.close()
    return {
        "at": now.isoformat(),
        "windows_settled": settled,
        "reputations_evaluated": evaluated,
        "sessions_gc": sessions_gc,
        "shares_recomputed": len(shares),
        "probes_voided": voided,
        "preempted": preempted,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ganymede-ledger",
        description="Run the contribution-ledger sweep (docs/09). For cron.",
    )
    p.add_argument("--db", default=None,
                   help="path to the coordinator database; default GANYMEDE_DB")
    p.add_argument("--probation-monthly-cap-hours", type=float,
                   default=ledger.PROBATION_MONTHLY_CAP_HOURS,
                   help="hard ceiling on provisioned hours while on probation")
    args = p.parse_args(argv)
    db_path = args.db
    if db_path is None:
        import os
        db_path = os.environ.get("GANYMEDE_DB")
        if not db_path:
            print("no database: pass --db or set GANYMEDE_DB", file=sys.stderr)
            return 2
    # ``from_env`` needs the storage block, which a bare ledger cron may not
    # have configured. The sweep's ledger half predates Phase D and must not
    # start depending on it, so a settings failure costs only the autopreempt
    # step -- loudly, on stderr, not silently.
    settings = None
    try:
        from ganymede.coordinator.config import Settings

        settings = Settings.from_env()
    except Exception as exc:  # noqa: BLE001
        print(f"note: no settings ({exc}); autopreempt sits out", file=sys.stderr)

    report = run(db_path, args.probation_monthly_cap_hours, settings)
    print(f"{report['at']} settled {report['windows_settled']} window(s), "
          f"evaluated {report['reputations_evaluated']} machine(s), "
          f"gc'd {report['sessions_gc']} session(s), "
          f"shares for {report['shares_recomputed']} submitter(s)"
          + (f", preempted {report['preempted']}" if report["preempted"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())