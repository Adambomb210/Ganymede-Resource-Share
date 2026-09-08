"""Lease lifecycle and the shared time helpers (docs/02-architecture-v2.md 3.1, 3.2, 6.4).

Phase A moved the round lifecycle, per-machine claim sizing and the outer
combine into ``ganymede.jobtypes.collab_lora_finetune`` (docs/10-jobtype-sdk.md
§3). What stays here is generic: reclaiming expired leases, heartbeats,
voluntary abandon, recording a submission, and the ``utcnow`` / ``_iso`` /
``_parse`` helpers the rest of the coordinator (and the type) still share.

The one seam this leaves behind is the round-closed 409: ``heartbeat`` and
``record_submission`` used to read ``rounds.status`` directly -- state that is
now type-private -- so they call the job type's ``still_accepting`` hook
instead, through the registry.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

from ganymede.coordinator.db import immediate


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class RoundClosed(Exception):
    """The round moved on while a worker was still holding a lease. -> HTTP 409."""


class LeaseLost(Exception):
    """The lease expired or was reassigned. The worker must stop. -> HTTP 410."""


class NotEligible(Exception):
    """Worker or contributor may not work this run. Carries a reason for the log."""


def _round_still_accepting(conn: sqlite3.Connection, task: sqlite3.Row):
    """The 409 seam (docs/10 §3).

    Whether the task's job type still accepts work for the unit a held lease
    belongs to. ``collab_lora_finetune`` checks its ``rounds`` row via
    ``still_accepting``; a type with no round (``batch_inference``) has no
    ``run_id`` / ``round_idx`` and always accepts. Returns ``None`` to continue,
    or a ``RoundClosed`` for the caller to raise. Imported lazily so this
    generic module has no import-time dependency on the job-type package.
    """
    if task["run_id"] is None:
        return None
    from ganymede.jobtypes import resolve

    job_type = "collab_lora_finetune"
    if task["job_id"] is not None:
        row = conn.execute(
            "SELECT job_type FROM jobs WHERE id = ?", (task["job_id"],)
        ).fetchone()
        if row is not None:
            job_type = row["job_type"]
    jt = resolve(job_type)
    still = getattr(jt, "still_accepting", None)
    if still is None:
        return None
    return still(conn, task["run_id"], task["round_idx"])


# --------------------------------------------------------------------------
# Leases
# --------------------------------------------------------------------------


def cancel_outstanding(conn: sqlite3.Connection, task_id: str) -> str | None:
    """The cancel mode owed to this task, or ``None`` (docs/11 §3, docs/13 §4.2).

    Two causes, one answer. A **job cancel** is a fact about the job; a task
    learns about it on its next heartbeat, which means one write cancels a job
    however many tasks it has out, and there is no fan-out to get wrong. A
    **preemption** (docs/13 §4) is a fact about one task, on a job that is still
    perfectly alive.

    Task first, because the job-level clause below reads ``jobs.status =
    'cancelled'`` and a preempted task's job is ``queued`` or ``running`` -- it
    would find nothing. The transport is identical either way: the same
    ``{"cancel": mode}`` field on the same heartbeat response (Decision 8), which
    is the whole point of aiming the existing mechanism at a task rather than
    inventing a second one.
    """
    row = conn.execute(
        "SELECT preempt_mode FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    if row is not None and row["preempt_mode"]:
        return row["preempt_mode"]
    row = conn.execute(
        """SELECT j.cancel_mode FROM tasks t
             JOIN jobs j ON j.id = t.job_id
            WHERE t.id = ? AND j.status = 'cancelled'""",
        (task_id,),
    ).fetchone()
    if row is None:
        return None
    # A job cancelled without a mode is a soft cancel: the gentler reading is
    # the safe one when nobody said.
    return row["cancel_mode"] or "soft"


def expire_leases(conn: sqlite3.Connection, now: datetime | None = None) -> int:
    """Reclaim leases whose holder stopped heartbeating. Returns the count.

    A machine that was shut down mid-task is the common case, not an anomaly,
    so its shard has to become available again without operator involvement.

    Three destinations, not one. A lease belonging to a *cancelled* job lands on
    ``cancelled`` (docs/11 §3) and a *preempted* one on ``preempted`` (docs/13
    §4.3) -- both here as well as in ``abandon``, because this is the wedged-
    worker path where nobody acknowledged anything and the soft/hard distinction
    collapsed to hard.

    Keeping all three apart is not cosmetic. ``expired`` and ``abandoned`` are
    what the ledger counts as a machine's infractions (docs/09 5.2), and neither
    an operator cancelling a job nor the scheduler reclaiming a machine is the
    contributor's fault. ``preempted`` is further distinct from ``cancelled`` in
    the other direction: a cancelled task's work is not wanted, a preempted
    task's is, so ``preempted`` is non-terminal and goes back in the pool.

    Order matters: preempted first, then cancelled, then the rest. A task that is
    both preempted and on a cancelled job is *cancelled* -- the job going away
    outranks a scheduling decision about it -- so the preempt arm excludes those.
    """
    now = now or utcnow()
    with immediate(conn):
        preempted = conn.execute(
            """UPDATE tasks SET status = 'preempted', lease_expires_at = NULL,
                                worker_id = NULL, preempt_mode = NULL
               WHERE status = 'leased' AND lease_expires_at IS NOT NULL
                 AND lease_expires_at < ? AND preempt_mode IS NOT NULL
                 AND job_id NOT IN (SELECT id FROM jobs WHERE status = 'cancelled')""",
            (_iso(now),),
        ).rowcount
        cancelled = conn.execute(
            """UPDATE tasks SET status = 'cancelled', lease_expires_at = NULL
               WHERE status = 'leased' AND lease_expires_at IS NOT NULL
                 AND lease_expires_at < ?
                 AND job_id IN (SELECT id FROM jobs WHERE status = 'cancelled')""",
            (_iso(now),),
        ).rowcount
        expired = conn.execute(
            """UPDATE tasks SET status = 'expired'
               WHERE status = 'leased' AND lease_expires_at IS NOT NULL
                 AND lease_expires_at < ?""",
            (_iso(now),),
        ).rowcount
        return preempted + cancelled + expired


def heartbeat(
    conn: sqlite3.Connection,
    task_id: str,
    worker_id: str,
    steps_completed: int,
    settings,
    now: datetime | None = None,
) -> datetime:
    """Extend a lease. Raises RoundClosed (409) or LeaseLost (410)."""
    now = now or utcnow()
    with immediate(conn):
        task = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if task is None or task["worker_id"] != worker_id:
            raise LeaseLost("no such task for this worker")
        if task["status"] != "leased":
            raise LeaseLost(f"task is {task['status']}")

        closed = _round_still_accepting(conn, task)
        if closed is not None:
            raise closed

        expires = now + timedelta(seconds=settings.lease_duration_sec)
        # Progress only ever moves forward. A worker that restarts mid-task and
        # resumes from a checkpoint may report a lower figure than its own last
        # heartbeat; taking the max stops that from being read as the worker
        # having un-trained steps, which gate 5 would reject.
        conn.execute(
            """UPDATE tasks
               SET lease_expires_at = ?,
                   last_heartbeat_steps = MAX(COALESCE(last_heartbeat_steps, 0), ?)
               WHERE id = ?""",
            (_iso(expires), steps_completed, task_id),
        )
        conn.execute(
            """INSERT INTO audit (at, worker_id, event, detail_json)
               VALUES (?, ?, 'heartbeat', ?)""",
            (_iso(now), worker_id, json.dumps({"task": task_id, "steps": steps_completed})),
        )
        conn.execute("UPDATE workers SET last_seen = ? WHERE id = ?", (_iso(now), worker_id))
        return expires


def last_heartbeat_steps(conn: sqlite3.Connection, task_id: str) -> int | None:
    """Highest progress figure the worker reported, for acceptance gate 5."""
    row = conn.execute(
        "SELECT last_heartbeat_steps FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    return None if row is None else row["last_heartbeat_steps"]


def abandon(conn: sqlite3.Connection, task_id: str, worker_id: str) -> str:
    """Voluntary release. The host is going away; give the shard back cleanly.

    Returns the status the task landed on -- one of three (docs/11 §3 step 4,
    docs/13 §4.3):

    - ``cancelled`` when the *job* was cancelled. The shard is not re-dispatched
      while it drains -- no second container on one unit of work -- and the
      release is not counted against the machine, because a job the operator
      cancelled is not an infraction of the contributor's (docs/09 5.2).
    - ``preempted`` when this *task* was preempted. Not an infraction either, for
      the same reason: the machine was told to stop by the scheduler and did
      nothing wrong. But unlike ``cancelled`` it is **not terminal** -- the work
      is still wanted, it was interrupted rather than abandoned as an idea, so
      the shard goes back in the pool and ``_reserve_static_task`` re-serves it
      without burning an attempt (docs/13 §4.4).
    - ``abandoned`` otherwise: the host is going away and this counts.

    A preempted task clears ``worker_id`` and ``preempt_mode`` on the way out.
    Leaving the marker set would preempt whichever machine picked the shard up
    next, on the strength of a decision made about a different machine.

    Both reads happen **inside** the transaction. Deciding which of the three
    this is takes two SELECTs, and ``POST /v1/jobs/{id}/cancel`` writes under
    ``immediate()`` -- so reading outside leaves a window where a job cancel
    lands between them and the task ends up ``preempted`` (non-terminal) on a
    job that is now terminal. Nothing would ever clear it.
    """
    with immediate(conn):
        job_cancelled = conn.execute(
            """SELECT 1 FROM tasks t JOIN jobs j ON j.id = t.job_id
                WHERE t.id = ? AND j.status = 'cancelled'""",
            (task_id,),
        ).fetchone() is not None
        if job_cancelled:
            status = "cancelled"
        elif cancel_outstanding(conn, task_id):
            status = "preempted"
        else:
            status = "abandoned"
        if status == "preempted":
            conn.execute(
                """UPDATE tasks SET status = 'preempted', lease_expires_at = NULL,
                                    worker_id = NULL, preempt_mode = NULL
                   WHERE id = ? AND worker_id = ? AND status = 'leased'""",
                (task_id, worker_id),
            )
        else:
            conn.execute(
                """UPDATE tasks SET status = ?, lease_expires_at = NULL
                   WHERE id = ? AND worker_id = ? AND status = 'leased'""",
                (status, task_id, worker_id),
            )
    return status


def record_submission(
    conn: sqlite3.Connection,
    task_id: str,
    worker_id: str,
    artifact_ref: str,
    steps_completed: int,
    tokens_seen: int,
    metrics: dict,
    now: datetime | None = None,
) -> None:
    """Persist a submission and close out its lease. Gating happens separately.

    Submission and gating are deliberately separate: the bytes are durably
    recorded before any validation runs, so a coordinator crash inside the
    gates cannot lose work a worker already did.
    """
    now = now or utcnow()
    with immediate(conn):
        task = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if task is None or task["worker_id"] != worker_id:
            raise LeaseLost("no such task for this worker")
        if task["status"] != "leased":
            raise LeaseLost(f"task is {task['status']}")

        closed = _round_still_accepting(conn, task)
        if closed is not None:
            raise closed

        conn.execute(
            """INSERT OR REPLACE INTO submissions
                 (task_id, artifact_ref, steps_completed, tokens_seen,
                  metrics_json, accepted, reject_reason, received_at)
               VALUES (?, ?, ?, ?, ?, NULL, NULL, ?)""",
            (task_id, artifact_ref, steps_completed, tokens_seen,
             json.dumps(metrics), _iso(now)),
        )
        conn.execute(
            "UPDATE tasks SET status = 'submitted', lease_expires_at = NULL WHERE id = ?",
            (task_id,),
        )
        conn.execute(
            "UPDATE workers SET steps_total = steps_total + ?, last_seen = ? WHERE id = ?",
            (steps_completed, _iso(now), worker_id),
        )
