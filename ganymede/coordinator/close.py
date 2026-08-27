"""Generic round-close dispatch (the type-agnostic remnant of
``coordinator/closer.py``).

docs/02-architecture-v2.md sections 5.1, 5.2 and 3.1; docs/10-jobtype-sdk.md §3.

Phase A split ``closer.py`` in two. The cohort gate, the outer combine and the
next-round open are the job type's ``reduce`` and moved into
``ganymede.jobtypes.collab_lora_finetune.reduce``. What stays here is generic:
``advance_job`` (the per-poll entry point, docs/05 reconciliation #7) and
``close_round`` (the atomic ``status='closing'`` fence with reopen-on-exception).
The reduce is dispatched through the registry -- this module never imports a
type module directly.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime

from ganymede.coordinator import rounds
from ganymede.coordinator.config import DEFAULT_DOMINANCE_CAP, DEFAULT_NORM_REJECT_K
from ganymede.coordinator.db import immediate
from ganymede.jobtypes import resolve

log = logging.getLogger("ganymede.coordinator.close")

# Phase A: every ``runs`` row is a ``collab_lora_finetune`` job (no ``job_type``
# column until Phase B's migration 003), so the type is resolved by name here.
# This becomes a lookup keyed off the job row once ``jobs`` exists.
_JOB_TYPE = "collab_lora_finetune"


def close_round(
    conn: sqlite3.Connection,
    store,
    run_id: str,
    round_idx: int,
    reason: str,
    now: datetime | None = None,
    settings=None,
):
    """Aggregate a round's accepted submissions and advance the run.

    Returns None when another caller is already closing this round. Closing is
    driven opportunistically from the request path, so two workers whose
    submissions land together will both evaluate the close rule and both find
    it satisfied -- that is the normal case, not an error.
    """
    now = now or rounds.utcnow()
    norm_k = getattr(settings, "norm_reject_k", DEFAULT_NORM_REJECT_K)
    cap = getattr(settings, "dominance_cap", DEFAULT_DOMINANCE_CAP)

    # Claim the close atomically. Reading the status and then writing it in a
    # separate statement leaves a window where two callers both see 'open' and
    # both run the whole aggregate-and-advance path; the loser then fails on the
    # rounds-table UNIQUE constraint, which reaches a blameless worker as a 500.
    # The conditional UPDATE makes exactly one caller the winner, and it is the
    # same write that fences off late submissions with a clean 409.
    with immediate(conn):
        claimed = conn.execute(
            """UPDATE rounds SET status = 'closing'
               WHERE run_id = ? AND idx = ? AND status = 'open'""",
            (run_id, round_idx),
        ).rowcount
    if not claimed:
        return None

    try:
        return resolve(_JOB_TYPE).reduce_close(
            conn, store, run_id, round_idx, reason, now, norm_k, cap
        )
    except Exception:
        # Give the round back. Everything between claiming 'closing' and the
        # status update below is storage I/O and tensor arithmetic, any of which
        # can fail -- and 'closing' is claimed by exactly one caller and
        # released only by that caller finishing. Leaving it set would wedge the
        # round permanently: no worker could claim it, no submission could
        # reopen it, and nothing would report an error. Every worker would
        # simply get 204 forever.
        #
        # Reopening is safe to retry. The result adapter is written under a key
        # derived from (run, round), so a second attempt overwrites the same
        # object rather than accumulating; the submissions it aggregates are
        # unchanged; and the close rule that fired once will fire again on the
        # next submit or claim.
        conn.execute(
            """UPDATE rounds SET status = 'open'
               WHERE run_id = ? AND idx = ? AND status = 'closing'""",
            (run_id, round_idx),
        )
        log.exception("closing round %s#%s failed; reopened for retry", run_id, round_idx)
        raise


def advance_job(
    conn: sqlite3.Connection,
    store,
    run_id: str,
    now: datetime | None = None,
    settings=None,
):
    """Evaluate the close rule for a run's current round and act on it.

    Called opportunistically from the request path (after a submit) rather than
    from a background scheduler: with rounds measured in tens of minutes, a
    close that lands a few seconds late costs nothing, and one fewer moving
    part is worth more than the precision.
    """
    now = now or rounds.utcnow()
    run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if run is None or run["status"] != "active":
        return None

    jt = resolve(_JOB_TYPE)
    idx = int(run["current_round"])
    close, reason = jt.should_close(conn, run_id, idx, now)
    if close:
        return close_round(conn, store, run_id, idx, reason, now, settings)

    # Backstop reached with nothing submitted: restart the clock, silently.
    jt.reopen_empty_round(conn, run_id, idx, now)
    return None
