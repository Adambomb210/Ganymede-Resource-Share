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


def _round_still_accepting(conn: sqlite3.Connection, run_id: str, round_idx: int):
    """The 409 seam (docs/10 §3).

    Whether the run's job type still accepts work for the round a held lease
    belongs to. ``collab_lora_finetune`` checks its ``rounds`` row; a type
    without a reduce step always accepts. Returns ``None`` to continue, or a
    ``RoundClosed`` for the caller to raise. Imported lazily so this generic
    module has no import-time dependency on the job-type package.
    """
    from ganymede.jobtypes import resolve

    return resolve("collab_lora_finetune").still_accepting(conn, run_id, round_idx)


# --------------------------------------------------------------------------
# Leases
# --------------------------------------------------------------------------


def expire_leases(conn: sqlite3.Connection, now: datetime | None = None) -> int:
    """Reclaim leases whose holder stopped heartbeating. Returns the count.

    A machine that was shut down mid-task is the common case, not an anomaly,
    so its shard has to become available again without operator involvement.
    """
    now = now or utcnow()
    with immediate(conn):
        cur = conn.execute(
            """UPDATE tasks SET status = 'expired'
               WHERE status = 'leased' AND lease_expires_at IS NOT NULL
                 AND lease_expires_at < ?""",
            (_iso(now),),
        )
        return cur.rowcount


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

        closed = _round_still_accepting(conn, task["run_id"], task["round_idx"])
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


def abandon(conn: sqlite3.Connection, task_id: str, worker_id: str) -> None:
    """Voluntary release. The host is going away; give the shard back cleanly."""
    with immediate(conn):
        conn.execute(
            """UPDATE tasks SET status = 'abandoned', lease_expires_at = NULL
               WHERE id = ? AND worker_id = ? AND status = 'leased'""",
            (task_id, worker_id),
        )


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

        closed = _round_still_accepting(conn, task["run_id"], task["round_idx"])
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
