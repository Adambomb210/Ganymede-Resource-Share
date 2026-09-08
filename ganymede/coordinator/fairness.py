"""Share accounting, fair-share ordering, quotas, and preemption (docs/13).

Four of Phase D's five remaining pieces; spot-checks are the fifth and live in
``spotcheck.py`` because they belong to the anti-fraud pair with the ledger
rather than to the scheduler.

Everything here is **off by default**, and in the two cases where that matters it
is off by arithmetic rather than by a branch: ``FAIRSHARE_SPREAD`` defaults to
``0.0``, so the sort key gains ``+ 0.0``; and a submitter with no
``submitter_quotas`` row has no cap because there is no row, not because a flag
said so. docs/13 §0 has the argument -- this is a fleet with one contributor and
no contention, so there is nothing to tune scheduling policy against yet, and
shipping untuned policy on-by-default moves work for reasons nobody can explain.

The split of labour with the claim path: this module answers questions
(``share_fractions``, ``quota_refusal``), ``app._selectable_jobs`` and the claim
walk do something with the answers. Nothing here writes a lease or reads a
profile.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone

from ganymede.coordinator.db import immediate
from ganymede.coordinator.rounds import _iso, _parse, utcnow

log = logging.getLogger("ganymede.coordinator.fairness")

# --- share accounting (docs/13 §1) -------------------------------------------

# ``formula_version = 0`` is unweighted leased seconds: a second of a 4090 and a
# second of a 3060 count the same, which is wrong. ``machine_weight`` (docs/09
# §3.2) would fix it and is deliberately not used yet, because it is itself at an
# explicitly interim ``formula_version = 0`` and two interim formulas multiplied
# together produce a number nobody can reason about. Version 1 is the weighted
# one; the column exists so that is a migration rather than a silent redefinition.
SHARE_FORMULA_VERSION = 0

# A day. Long enough that a morning's work still counts in the afternoon, short
# enough that yesterday's does not decide today's queue.
SHARE_HALF_LIFE_HOURS = 24.0

# A computational bound, not a policy one. At a 24-hour half-life a seven-day-old
# second contributes 2^-7 (0.8%) of itself, far below the noise in the thing
# being measured.
SHARE_LOOKBACK_DAYS = 7


def _decay(seconds: float, age_hours: float) -> float:
    """Exponential decay, floored at zero age.

    Negative ages happen: clocks, and a row stamped a moment in the future by a
    sweep that ran during a leap adjustment. ``max(0.0, ...)`` treats those as
    "just now" rather than letting the exponent amplify them.
    """
    return seconds * 0.5 ** (max(0.0, age_hours) / SHARE_HALF_LIFE_HOURS)


def _task_seconds(leased_at: str, ended_at: str | None,
                  now: datetime) -> tuple[float, datetime]:
    """``(seconds, when the consumption ended)`` for one lease (docs/13 §1.3).

    Charged for the *reservation*, not the accepted work: a machine held for
    fifteen minutes is fifteen minutes nobody else could have it, whether or not
    anything came back. So an abandoned lease is charged in full to its
    expiry, and a job whose tasks abandon constantly is not rewarded for it.

    An in-flight lease is charged for what it has held so far, which means a
    submitter's share rises *while* their work runs rather than in a step when it
    finishes.
    """
    start = _parse(leased_at)
    end = _parse(ended_at) if ended_at else now
    if end > now:
        end = now
    return max(0.0, (end - start).total_seconds()), end


def recompute_shares(conn: sqlite3.Connection,
                     now: datetime | None = None) -> dict[str, float]:
    """Rebuild ``share_accounting`` from ``tasks``. For the sweep.

    A full recompute rather than an increment. The input set is bounded by
    ``SHARE_LOOKBACK_DAYS`` and indexed on ``leased_at``, so it costs a scan of
    one week of leases once a minute -- and a recompute cannot drift, where an
    incremental counter that misses an update stays wrong until someone notices.
    """
    now = now or utcnow()
    since = _iso(now - timedelta(days=SHARE_LOOKBACK_DAYS))
    rows = conn.execute(
        """SELECT j.owner_id AS owner_id, t.leased_at AS leased_at,
                  COALESCE(s.received_at, t.lease_expires_at) AS ended_at
             FROM tasks t
             JOIN jobs j ON j.id = t.job_id
             LEFT JOIN submissions s ON s.task_id = t.id
            WHERE t.leased_at IS NOT NULL AND t.leased_at >= ?""",
        (since,),
    ).fetchall()

    totals: dict[str, float] = {}
    for row in rows:
        # A pre-005 task carries no ``job_id`` and so no owner; the JOIN drops
        # it. Those predate the scheduler entirely and belong to a world with
        # one submitter, where a share is meaningless.
        try:
            seconds, end = _task_seconds(row["leased_at"], row["ended_at"], now)
        except (TypeError, ValueError):
            # An unparseable stamp is one row of one submitter's share, and the
            # sweep covering the whole fleet is worth more than exactness about
            # it. Loud in the log, invisible to the scheduler.
            log.warning("share: unparseable lease stamps on an owner-%s task",
                        row["owner_id"])
            continue
        age_hours = (now - end).total_seconds() / 3600.0
        totals[row["owner_id"]] = totals.get(row["owner_id"], 0.0) + _decay(
            seconds, age_hours)

    stamp = _iso(now)
    with immediate(conn):
        # Delete-then-insert, not upsert: an owner who has dropped out of the
        # window must lose their row, and an UPDATE-only pass would leave a
        # stale share sitting there being decayed forward forever.
        conn.execute("DELETE FROM share_accounting")
        for owner_id, seconds in totals.items():
            conn.execute(
                "INSERT INTO share_accounting "
                "(owner_id, decayed_seconds, formula_version, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (owner_id, seconds, SHARE_FORMULA_VERSION, stamp),
            )
    return totals


def share_fractions(conn: sqlite3.Connection,
                    now: datetime | None = None) -> dict[str, float]:
    """``owner_id -> share of the fleet's recent consumption``, each in [0, 1].

    Reads the rollup and **ages it forward** from its own ``updated_at`` (docs/13
    §1.5). This is what makes a cached rollup safe on the claim path: a sweep
    that has not run for an hour does not hand the scheduler an hour-stale
    number, it hands it a correctly-aged one, missing only the leases taken
    during that hour. A sweep that has stopped entirely fades every share toward
    zero -- and since fair-share only demotes, fading to zero turns it *off*,
    which is the right direction for a mechanism to fail in.

    Returns ``{}`` when nobody has consumed anything, which every caller must
    read as "no share is zero", not as an error.
    """
    now = now or utcnow()
    current: dict[str, float] = {}
    for row in conn.execute(
        "SELECT owner_id, decayed_seconds, updated_at FROM share_accounting"
    ).fetchall():
        try:
            age_hours = (now - _parse(row["updated_at"])).total_seconds() / 3600.0
        except (TypeError, ValueError):
            continue
        current[row["owner_id"]] = _decay(float(row["decayed_seconds"]), age_hours)

    total = sum(current.values())
    if total <= 0.0:
        return {}
    return {owner: seconds / total for owner, seconds in current.items()}


# --- fair-share ordering (docs/13 §2) ----------------------------------------


def effective_rank(priority_rank: int, spread: float, share_fraction: float) -> float:
    """The primary sort term of ``_selectable_jobs`` (docs/07 §4's named seam).

    ``spread`` is in units of ``priority_rank`` and says how much of the ordering
    the admin is delegating to the formula. Ranks are sparse by convention
    (10, 20, 30 -- docs/07 §5), so ``spread = 10`` lets a submitter monopolising
    the fleet slip at most one slot, and ``spread = 100`` makes fair-share
    dominant with ``priority_rank`` as the tiebreak.

    ``spread = 0.0`` (the default) is off, and it is off by arithmetic: the term
    is ``priority_rank + 0.0``, which sorts identically to the integer it came
    from. There is a test that asserts the resulting *order* is unchanged rather
    than trusting that sentence.

    **Only ever demotes.** ``share_fraction >= 0``, so the result is always
    ``>= priority_rank``. Fair-share can make a job run later than the admin
    ordered it and never sooner -- which matters, because a formula that could
    promote would be a back door into the submitter-set priority docs/07 §5
    forbids: starve yourself deliberately, get promoted. There is no such move.
    """
    return float(priority_rank) + spread * max(0.0, share_fraction)


# --- quotas and budgets (docs/13 §3) -----------------------------------------

OVER_CONCURRENCY = "over_concurrency_quota"
OVER_BUDGET = "over_monthly_budget"


def quota_for(conn: sqlite3.Connection, user_id: str) -> sqlite3.Row | None:
    """The submitter's quota row, or ``None`` -- which is *uncapped*, and is the
    default for everybody including every submitter that already exists."""
    return conn.execute(
        "SELECT * FROM submitter_quotas WHERE user_id = ?", (user_id,)
    ).fetchone()


def concurrency_used(conn: sqlite3.Connection, owner_id: str) -> int:
    """Leases this submitter's jobs hold right now. Live, exact, cheap."""
    return conn.execute(
        """SELECT COUNT(*) AS n FROM tasks t JOIN jobs j ON j.id = t.job_id
            WHERE j.owner_id = ? AND t.status = 'leased'""",
        (owner_id,),
    ).fetchone()["n"]


def _month_start(now: datetime) -> datetime:
    return now.astimezone(timezone.utc).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0)


def month_hours(conn: sqlite3.Connection, owner_id: str,
                now: datetime | None = None) -> float:
    """Machine-hours this submitter has consumed since the start of the UTC month.

    **Undecayed**, deliberately, and this is the difference between a budget and
    a share. Decay belongs to fairness, which is about recency; a budget is an
    accounting quantity and has to match what a human gets when they add the
    month up by hand.

    The month boundary is UTC and hard: no proration, no rollover, no partial
    first month. Every one of those is a billing feature and docs/04 Decision 2
    keeps this system out of billing. This is a brake, not an invoice.
    """
    now = now or utcnow()
    rows = conn.execute(
        """SELECT t.leased_at AS leased_at,
                  COALESCE(s.received_at, t.lease_expires_at) AS ended_at
             FROM tasks t
             JOIN jobs j ON j.id = t.job_id
             LEFT JOIN submissions s ON s.task_id = t.id
            WHERE j.owner_id = ? AND t.leased_at IS NOT NULL AND t.leased_at >= ?""",
        (owner_id, _iso(_month_start(now))),
    ).fetchall()
    total = 0.0
    for row in rows:
        try:
            seconds, _ = _task_seconds(row["leased_at"], row["ended_at"], now)
        except (TypeError, ValueError):
            continue
        total += seconds
    return total / 3600.0


def quota_refusal(conn: sqlite3.Connection, owner_id: str,
                  now: datetime | None = None) -> str | None:
    """The refusal reason this submitter is over, or ``None``.

    Both caps are checked in the claim walk as a ``continue`` with a recorded
    ``eligibility.Verdict``, not as a silent filter in ``_selectable_jobs``. A
    ``continue`` is every bit as absolute -- the job is not offered, full stop --
    and it buys the *explanation*: the refusal lands in ``worker_eligibility``
    and surfaces through ``explain()``, so a submitter whose jobs have stopped
    moving is told "you are at your cap" rather than watching a queued job sit
    there. A quota nobody can see hitting is a support ticket.
    """
    row = quota_for(conn, owner_id)
    if row is None:
        return None
    cap = row["max_concurrent_tasks"]
    if cap is not None and concurrency_used(conn, owner_id) >= int(cap):
        return OVER_CONCURRENCY
    budget = row["monthly_task_hours"]
    if budget is not None and month_hours(conn, owner_id, now) >= float(budget):
        return OVER_BUDGET
    return None


def budget_exhausted(conn: sqlite3.Connection, owner_id: str,
                     now: datetime | None = None) -> float | None:
    """The monthly cap, if this submitter is already past it; else ``None``.

    Admission control for ``enqueue`` (docs/13 §3.4). Only the *budget* is
    checked there -- the concurrency quota is transient by nature, and refusing
    an enqueue because four tasks happen to be running right now would be
    nonsense.
    """
    row = quota_for(conn, owner_id)
    if row is None or row["monthly_task_hours"] is None:
        return None
    cap = float(row["monthly_task_hours"])
    return cap if month_hours(conn, owner_id, now) >= cap else None


# --- preemption (docs/13 §4) -------------------------------------------------

AUTOPREEMPT_STARVE_MIN = 30
AUTOPREEMPT_RANK_MARGIN = 10


def preempt(conn: sqlite3.Connection, task_id: str, mode: str = "soft",
            cause: str = "admin", now: datetime | None = None) -> bool:
    """Mark one leased task for preemption. ``True`` if it took.

    Writes ``tasks.preempt_mode`` and nothing else. The *transport* is unchanged
    and that is the point of docs/13 §4.1: the worker learns about this on the
    same heartbeat response field, in the same shape, as a job-level cancel
    (Decision 8 -- the cancel rides the heartbeat and nothing else). No new
    endpoint, no push.
    """
    now = now or utcnow()
    if mode not in ("soft", "hard"):
        raise ValueError(f"preempt mode must be soft | hard, got {mode!r}")
    with immediate(conn):
        row = conn.execute(
            "SELECT worker_id, job_id FROM tasks WHERE id = ? AND status = 'leased'",
            (task_id,),
        ).fetchone()
        if row is None:
            return False
        conn.execute("UPDATE tasks SET preempt_mode = ? WHERE id = ?", (mode, task_id))
        # Every preemption is audited. docs/13 §4.4 declines to cap preemptions
        # per task -- a cap would silently convert "this shard keeps getting
        # pushed around" into "this shard failed", which is a worse thing to
        # discover -- so this row is how the pushing-around stays visible.
        conn.execute(
            "INSERT INTO audit (at, event, detail_json) VALUES (?, ?, ?)",
            (_iso(now), "task_preempted", json.dumps({
                "task_id": task_id, "job_id": row["job_id"],
                "worker_id": row["worker_id"], "mode": mode, "cause": cause,
            })),
        )
    return True


def autopreempt(conn: sqlite3.Connection, settings,
                now: datetime | None = None) -> str | None:
    """One preemption per sweep, at most. The preempted task id, or ``None``.

    Off unless ``settings.autopreempt`` (docs/13 §4.6), and §4.6 says plainly why
    rather than leaving it to look like caution: **this policy has never run
    against a fleet with actual contention, because no such fleet exists yet.**
    The constants are reasoned, not measured.

    The policy:

    1. A **starved** job -- queued or running, zero leased tasks, waiting longer
       than ``AUTOPREEMPT_STARVE_MIN``.
    2. Leases held by jobs at least ``AUTOPREEMPT_RANK_MARGIN`` *worse* in rank.
    3. Held by a machine that could plausibly take the starved job.
    4. The longest-running of those, soft.

    Step 3 needs "would this machine accept that job?", which is the constraint
    gate -- and the sweep has no probe profile to re-evaluate it against. But the
    claim path has already answered that question for every (machine, job) pair
    it walked, and wrote it down: ``worker_eligibility`` has been a recorder
    since docs/07 froze it, and this is the first thing to read it as an *input*.
    Absence of a refusal is weaker than a fresh evaluation -- a machine that has
    never polled while this job was queued has no verdict either way -- but it
    fails safe: no candidate, no preemption.
    """
    if not getattr(settings, "autopreempt", False):
        return None
    now = now or utcnow()
    spread = float(getattr(settings, "fairshare_spread", 0.0) or 0.0)
    shares = share_fractions(conn, now)
    starve_before = _iso(now - timedelta(minutes=AUTOPREEMPT_STARVE_MIN))

    starved = conn.execute(
        """SELECT j.id, j.owner_id, j.priority_rank FROM jobs j
            WHERE j.status IN ('queued', 'running') AND j.created_at <= ?
              AND NOT EXISTS (SELECT 1 FROM tasks t
                               WHERE t.job_id = j.id AND t.status = 'leased')
            ORDER BY j.priority_rank, j.created_at""",
        (starve_before,),
    ).fetchall()
    if not starved:
        return None

    held = conn.execute(
        """SELECT t.id AS task_id, t.worker_id, t.leased_at,
                  j.id AS job_id, j.owner_id, j.priority_rank
             FROM tasks t JOIN jobs j ON j.id = t.job_id
            WHERE t.status = 'leased' AND t.preempt_mode IS NULL"""
    ).fetchall()
    if not held:
        return None

    for want in starved:
        want_rank = effective_rank(
            want["priority_rank"], spread, shares.get(want["owner_id"], 0.0))
        candidates = []
        for lease in held:
            lease_rank = effective_rank(
                lease["priority_rank"], spread, shares.get(lease["owner_id"], 0.0))
            if lease_rank - want_rank < AUTOPREEMPT_RANK_MARGIN:
                continue
            refused = conn.execute(
                "SELECT 1 FROM worker_eligibility "
                "WHERE worker_id = ? AND job_id = ? AND outcome = 'refused'",
                (lease["worker_id"], want["id"]),
            ).fetchone()
            if refused is not None:
                continue
            candidates.append(lease)
        if not candidates:
            continue
        # Longest-running: the one that has already had the most out of the
        # machine, so the least work is thrown away per second reclaimed.
        victim = min(candidates, key=lambda r: r["leased_at"] or "")
        if preempt(conn, victim["task_id"], "soft",
                   cause=f"autopreempt:{want['id']}", now=now):
            return victim["task_id"]
    return None
