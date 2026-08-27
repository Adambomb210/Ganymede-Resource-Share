"""Submit-time acceptance gates for ``collab_lora_finetune``
(ex-``coordinator/closer.py``).

Relocated in Phase A (docs/10-jobtype-sdk.md §3): ``gate_submission``,
``expected_manifest``, ``_record_verdict`` and ``_MANIFEST_CACHE`` move here as
the type's ``validate``. The structural checks themselves are
``aggregate.check_structural`` (this package). Bodies are byte-identical to the
originals -- only the import path of ``aggregate`` changed.

The gates are split across two moments on purpose. The structural gates (1-3, 5
in 5.1) run at **submit** time, because they are per-submission and a worker can
be told immediately that its adapter was malformed -- feedback at the moment the
bug happened rather than forty minutes later. Gate 4 is cohort-relative: an
adapter is an outlier only with respect to the other adapters in the round, so
it cannot run until the round is done (see ``reduce.py``).
"""

from __future__ import annotations

import json
import sqlite3
import weakref
from datetime import datetime

from ganymede.coordinator import rounds
from ganymede.coordinator.db import immediate
from ganymede.coordinator.store import Store
from ganymede.jobtypes.collab_lora_finetune import aggregate

# A round's base adapter never changes once written, so its manifest is safe to
# cache by key. Without this every submit re-downloads and re-parses ~25 MB just
# to learn a key set the coordinator already knew -- eight workers a round would
# pay for that eight times over an artifact none of them changed.
# Keyed by the Store the bytes actually live in, not by the ref string alone. A
# key path is only unique *within* a bucket: the same run id recreated against
# fresh storage, or a process holding two stores, would otherwise be served a
# manifest describing an adapter it never read. Weak keys so a discarded Store
# takes its cache with it.
_MANIFEST_CACHE: "weakref.WeakKeyDictionary[object, dict[str, dict]]" = (
    weakref.WeakKeyDictionary()
)


def expected_manifest(store: Store, base_adapter_ref: str) -> dict:
    """Key set and shapes a submission must match, from the round's base adapter."""
    per_store = _MANIFEST_CACHE.setdefault(store, {})
    cached = per_store.get(base_adapter_ref)
    if cached is None:
        cached = aggregate.manifest_of(
            aggregate.load_adapter(store.get_bytes(base_adapter_ref))
        )
        # One entry per round; a long run would otherwise accumulate one dict
        # per round for the lifetime of the process.
        if len(per_store) > 4:
            per_store.clear()
        per_store[base_adapter_ref] = cached
    return cached


def gate_submission(
    conn: sqlite3.Connection,
    store: Store,
    task_id: str,
    expected: dict,
    now: datetime | None = None,
) -> tuple[bool, str | None]:
    """Run the structural gates against a just-received artifact.

    Records the verdict on the submission row and, on rejection, in the audit
    log keyed by contributor -- that log is the raw material for Phase 2
    reputation scoring, so it accumulates from day one rather than being
    retrofitted.
    """
    now = now or rounds.utcnow()
    sub = conn.execute("SELECT * FROM submissions WHERE task_id = ?", (task_id,)).fetchone()
    if sub is None:
        return False, "no_submission"

    task = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    hb = rounds.last_heartbeat_steps(conn, task_id)

    try:
        raw = store.get_bytes(sub["artifact_ref"])
    except Exception:
        verdict, _ = aggregate.GateResult(False, "missing_artifact"), None
        _record_verdict(conn, task_id, task, verdict, now)
        return False, "missing_artifact"

    verdict, _adapter = aggregate.check_structural(
        raw, expected, sub["steps_completed"], heartbeat_steps=hb
    )
    _record_verdict(conn, task_id, task, verdict, now)
    return verdict.accepted, verdict.reason


def _record_verdict(conn, task_id, task, verdict, now) -> None:
    with immediate(conn):
        conn.execute(
            "UPDATE submissions SET accepted = ?, reject_reason = ? WHERE task_id = ?",
            (1 if verdict.accepted else 0, verdict.reason, task_id),
        )
        if not verdict.accepted and task is not None:
            row = conn.execute(
                "SELECT contributor_id FROM workers WHERE id = ?", (task["worker_id"],)
            ).fetchone()
            conn.execute(
                """INSERT INTO audit (at, contributor_id, worker_id, event, detail_json)
                   VALUES (?, ?, ?, 'submission_rejected', ?)""",
                (
                    rounds._iso(now),
                    row["contributor_id"] if row else None,
                    task["worker_id"],
                    json.dumps({"task": task_id, "reason": verdict.reason,
                                "detail": verdict.detail}),
                ),
            )
