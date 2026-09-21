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

import json
import logging
import sqlite3
from datetime import datetime

from ganymede.coordinator import rounds
from ganymede.coordinator.config import DEFAULT_DOMINANCE_CAP, DEFAULT_NORM_REJECT_K
from ganymede.coordinator.db import immediate
from ganymede.coordinator import events
from ganymede.jobtypes import resolve

log = logging.getLogger("ganymede.coordinator.close")

# How many times a static-type task may be leased before the dispatcher stops
# re-serving it. Bounds the retry of an expired / abandoned / rejected shard and
# the re-dispatch of a deterministically-disagreeing ``attempt_group`` -- a hard
# per-shard failure path (dead-letter, job-fail) is Phase D.
MAX_TASK_ATTEMPTS = 5

# The type behind a ``runs`` row. Every ``runs`` row is a
# ``collab_lora_finetune`` job; a ``batch_inference`` job has no ``runs`` child
# and reaches ``advance_job`` by ``job_id`` instead.
_JOB_TYPE = "collab_lora_finetune"


def _job_type_of(conn: sqlite3.Connection, job_id: str | None) -> str:
    if job_id is None:
        return _JOB_TYPE
    row = conn.execute(
        "SELECT job_type FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()
    return row["job_type"] if row is not None else _JOB_TYPE


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
    rep_weighted = bool(getattr(settings, "reputation_weighted_agg", False))

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
        # Bound, not returned: ``_publish_close`` below has to run on the
        # success path, and a ``return`` here made it -- and the return after
        # it -- unreachable from the day it was added (89fb980). The symptom
        # was silent and looked like a UI problem: ``round.close`` has exactly
        # one publisher in the codebase and it never fired, so an operator's
        # dashboard never refreshed on an ordinary round close. Nothing in the
        # accounting path depended on it, which is why the suite stayed green.
        result = resolve(_JOB_TYPE).reduce_close(
            conn, store, run_id, round_idx, reason, now, norm_k, cap, rep_weighted
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
    _publish_close(conn, run_id)
    return result


def _publish_close(conn: sqlite3.Connection, run_id: str) -> None:
    """After a successful close: ``round.close`` + ``job.status`` to the run's
    owner + admin (docs/12 audience table). Envelopes carry the job id -- the
    fragments key on job_id -- so resolve runs -> jobs here, once."""
    try:
        row = conn.execute(
            "SELECT j.id AS job_id, j.owner_id FROM runs r "
            "JOIN jobs j ON j.id = r.job_id WHERE r.id = ?",
            (run_id,),
        ).fetchone()
    except Exception:
        return
    if row is None:
        return
    events.hub.publish("round.close", job_id=row["job_id"], owner_id=row["owner_id"])
    events.hub.publish("job.status", job_id=row["job_id"], owner_id=row["owner_id"])


def advance_job(
    conn: sqlite3.Connection,
    store,
    run_id: str | None = None,
    now: datetime | None = None,
    settings=None,
    *,
    job_id: str | None = None,
):
    """Evaluate the completion rule for a job and act on it.

    Called opportunistically from the request path (after a submit, and on the
    claim poll) rather than from a background scheduler: with rounds measured in
    tens of minutes, a close that lands a few seconds late costs nothing, and
    one fewer moving part is worth more than the precision.

    A type that returns a ``ReduceState`` (``collab_lora_finetune``) is driven
    by ``run_id`` through the round machinery below. A ``reduce -> None`` type
    (``batch_inference``) has no ``runs`` child and is passed by ``job_id``: the
    dispatcher's own rule -- every ``plan`` task accepted and every
    ``attempt_group`` agreed -- decides completion (docs/10 §5).
    """
    now = now or rounds.utcnow()

    if run_id is None:
        if job_id is None:
            return None
        return _advance_parallel_job(conn, store, job_id, now)

    run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if run is None:
        return None
    if run["status"] != "active":
        # Still mirror a terminal run onto its generic jobs row -- a run flipped
        # 'done' by an earlier close must leave the scheduler's queue walk even
        # if nothing advances it again.
        _mirror_job_status(conn, run)
        return None

    jt = resolve(_JOB_TYPE)
    idx = int(run["current_round"])
    close, reason = jt.should_close(conn, run_id, idx, now)
    result = None
    if close:
        result = close_round(conn, store, run_id, idx, reason, now, settings)
    else:
        # Backstop reached with nothing submitted: restart the clock, silently.
        jt.reopen_empty_round(conn, run_id, idx, now)

    # The type's reduce may have flipped runs.status to 'done'. Keep the generic
    # jobs row in step so the queue walk and worker_eligibility's non-terminal
    # filter both see it (docs/07 §1: advance_job is the per-poll entry point).
    run = conn.execute(
        "SELECT id, status, job_id FROM runs WHERE id = ?", (run_id,)
    ).fetchone()
    _mirror_job_status(conn, run)
    return result


# ``runs.status`` -> ``jobs.status``, terminal states only. The queued -> running
# flip happens on the first lease (docs/07 §1), in ``app.py``, not here.
_RUN_TO_JOB_TERMINAL = {"done": "done", "failed": "failed"}


def _mirror_job_status(conn: sqlite3.Connection, run) -> None:
    if run is None or run["job_id"] is None:
        return
    target = _RUN_TO_JOB_TERMINAL.get(run["status"])
    if target is None:
        return
    with immediate(conn):
        changed = conn.execute(
            # ``terminal_at`` is stamped in the same statement that makes the
            # job terminal, so the two can never disagree (docs/11 §1.2's
            # retention clock reads it).
            "UPDATE jobs SET status = ?, terminal_at = ? "
            "WHERE id = ? AND status NOT IN ('done', 'failed', 'cancelled')",
            (target, rounds._iso(rounds.utcnow()), run["job_id"]),
        ).rowcount
    if changed:
        events.hub.publish("job.status", job_id=run["job_id"], owner_id=_owner_of(conn, run["job_id"]))


def _owner_of(conn: sqlite3.Connection, job_id: str) -> str | None:
    try:
        row = conn.execute("SELECT owner_id FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return row["owner_id"] if row is not None else None
    except Exception:
        return None


# --------------------------------------------------------------------------
# Embarrassingly-parallel completion (docs/10 §5) -- the dispatcher's rule for
# a ``reduce -> None`` type. Owned here, not by the type: ``is_complete`` has
# neither ``conn`` nor a ``ReduceState`` to count from.
# --------------------------------------------------------------------------


def _advance_parallel_job(
    conn: sqlite3.Connection, store, job_id: str, now: datetime
):
    """A job whose ``plan`` fixed a finite task set is ``done`` once every task
    has an accepted verdict and every ``attempt_group`` has agreed.

    Disagreement in a group re-dispatches its members (back to ``planned``,
    worker cleared) and marks the offending submissions ``accepted = 0`` --
    uncredited, but kept, with each member's ``worker_id`` recorded in the audit
    event so a Phase-D scorer can still find the offending machine. Probation /
    per-machine sampling scoring is Phase D (docs/10 §4); a group that never
    agrees within ``MAX_TASK_ATTEMPTS`` leaves the job ``running`` rather than
    spinning (a job-fail path is also Phase D).
    """
    job = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if job is None or job["status"] in ("done", "failed", "cancelled"):
        return None

    jt = resolve(job["job_type"])
    # This path is for a ``reduce -> None`` type only -- one with a real reduce
    # is driven by ``run_id`` through the round machinery instead.
    if hasattr(jt, "shape_claim"):  # pragma: no cover - defensive
        return None

    # Known-answer probes are excluded from both the completion gate and the
    # ``attempt_group`` comparison below (docs/13 §5.4). A *failed* probe would
    # otherwise wedge the job forever -- it never passes, so the job is never
    # done -- and a passed one would join a group it was never part of. The
    # probe's work is still real and still credited to the machine that did it;
    # it just does not get a vote on whether the job is finished, because the
    # shard it duplicates already cast that vote.
    tasks = conn.execute(
        "SELECT * FROM tasks WHERE job_id = ? "
        "AND id NOT IN (SELECT task_id FROM spot_check_issues)",
        (job_id,),
    ).fetchall()
    if not tasks:
        return None

    def _accepted(task_id: str) -> bool:
        row = conn.execute(
            "SELECT accepted FROM submissions WHERE task_id = ?", (task_id,)
        ).fetchone()
        return row is not None and row["accepted"] == 1

    if not all(_accepted(t["id"]) for t in tasks):
        return None

    spec = json.loads(job["spec_json"] or "{}")
    redundancy = spec.get("redundancy") or {}
    agree_on = redundancy.get("agree_on", "output")
    sample_rows = int(redundancy.get("sample_rows", 8))

    groups: dict[str, list[sqlite3.Row]] = {}
    for t in tasks:
        if t["attempt_group"]:
            groups.setdefault(t["attempt_group"], []).append(t)

    from ganymede.jobtypes.batch_inference import run as bi_run
    from ganymede.jobtypes.batch_inference import validate as bi_validate

    unresolved = False
    for group, members in groups.items():
        if all(m["attempts"] >= MAX_TASK_ATTEMPTS for m in members):
            _record_unresolved_once(conn, group, members, now)
            unresolved = True
            continue
        outputs = []
        ok = True
        for m in members:
            sub = conn.execute(
                "SELECT artifact_ref FROM submissions WHERE task_id = ?", (m["id"],)
            ).fetchone()
            try:
                outputs.append(bi_run.parse_jsonl(store.get_bytes(sub["artifact_ref"])))
            except Exception:
                ok = False
                break
        if not ok:
            return None
        if not bi_validate.sample_agreement(outputs, sample_rows, agree_on):
            _redispatch_group(conn, members, group, now)
            return None
    if unresolved:
        return None

    with immediate(conn):
        conn.execute(
            "UPDATE jobs SET status = 'done' "
            "WHERE id = ? AND status NOT IN ('done', 'failed', 'cancelled')",
            (job_id,),
        )
    return None


def _group_detail(group: str, members: list[sqlite3.Row]) -> str:
    return json.dumps({
        "attempt_group": group,
        # worker_id captured BEFORE re-dispatch nulls it -- the only record of
        # which machine produced each side of the disagreement (Phase D input).
        "members": [{"task": m["id"], "worker_id": m["worker_id"]} for m in members],
    })


def _redispatch_group(
    conn: sqlite3.Connection, members: list[sqlite3.Row], group: str, now: datetime
) -> None:
    with immediate(conn):
        for m in members:
            conn.execute(
                "UPDATE tasks SET status = 'planned', worker_id = NULL, "
                "lease_expires_at = NULL WHERE id = ?",
                (m["id"],),
            )
            # Kept, not deleted (record_submission's durability-before-validation
            # rule) -- just marked uncredited so completion cannot count it.
            conn.execute(
                "UPDATE submissions SET accepted = 0, "
                "reject_reason = 'attempt_group_disagreement' WHERE task_id = ?",
                (m["id"],),
            )
        conn.execute(
            "INSERT INTO audit (at, event, detail_json) VALUES (?, ?, ?)",
            (rounds._iso(now), "attempt_group_disagreement",
             _group_detail(group, members)),
        )


def _record_unresolved_once(
    conn: sqlite3.Connection, group: str, members: list[sqlite3.Row], now: datetime
) -> None:
    seen = conn.execute(
        "SELECT 1 FROM audit WHERE event = 'attempt_group_unresolved' "
        "AND detail_json LIKE ?",
        (f'%"attempt_group": "{group}"%',),
    ).fetchone()
    if seen is None:
        with immediate(conn):
            conn.execute(
                "INSERT INTO audit (at, event, detail_json) VALUES (?, ?, ?)",
                (rounds._iso(now), "attempt_group_unresolved",
                 _group_detail(group, members)),
            )
