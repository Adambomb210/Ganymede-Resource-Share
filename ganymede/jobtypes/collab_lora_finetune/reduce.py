"""The round close for ``collab_lora_finetune`` (ex-``closer._close_claimed_round``).

Relocated in Phase A (docs/10-jobtype-sdk.md §3). This is the type's ``reduce``:
gate 4 (cohort-relative outlier rejection), the outer combine, momentum
load/store, ``adapter_divergence``, the ``base_adapter_key`` write, the
next-round open and the throughput fold-back.

The generic close fence -- the atomic ``status='closing'`` claim and the
reopen-on-exception -- stays in ``coordinator/close.py`` and calls this through
the registry. The body below is byte-identical to the original except that
``update_throughput`` / ``open_round`` now live in ``plan`` (they moved out of
``rounds`` in the same pass).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from ganymede.coordinator import rounds
from ganymede.coordinator.db import immediate
from ganymede.coordinator.store import base_adapter_key, momentum_key
from ganymede.jobtypes.collab_lora_finetune import aggregate, plan
from ganymede.jobtypes.collab_lora_finetune.validate import _record_verdict


@dataclass(frozen=True)
class CloseResult:
    run_id: str
    round_idx: int
    reason: str
    accepted: int
    rejected: int
    total_steps: int
    distinct_contributors: int
    result_adapter_ref: str | None
    divergence: float | None
    next_round_opened: bool


def _close_claimed_round(
    conn: sqlite3.Connection,
    store,
    run_id: str,
    round_idx: int,
    reason: str,
    now: datetime,
    norm_k: float,
    cap: float,
) -> CloseResult | None:
    """The body of ``close_round``, after the close has been claimed."""
    run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    rnd = conn.execute(
        "SELECT * FROM rounds WHERE run_id = ? AND idx = ?", (run_id, round_idx)
    ).fetchone()

    subs = conn.execute(
        """SELECT s.*, t.worker_id FROM submissions s
           JOIN tasks t ON t.id = s.task_id
           WHERE t.run_id = ? AND t.round_idx = ? AND s.accepted = 1
           ORDER BY s.received_at""",
        (run_id, round_idx),
    ).fetchall()

    base = aggregate.load_adapter(store.get_bytes(rnd["base_adapter_ref"]))
    keys = sorted(base.keys())

    adapters: list[dict] = []
    kept: list[sqlite3.Row] = []
    for s in subs:
        try:
            adapters.append(aggregate.load_adapter(store.get_bytes(s["artifact_ref"])))
            kept.append(s)
        except Exception:
            # Already passed the structural gates at submit; a failure here means
            # the object went missing between then and now. Drop it and carry on
            # rather than failing the whole round for one absent artifact.
            _record_verdict(conn, s["task_id"], None,
                            aggregate.GateResult(False, "missing_artifact"), now)

    rejected = len(subs) - len(kept)

    # Gate 4: cohort-relative outlier rejection, only meaningful now.
    if adapters:
        norm_results = aggregate.check_norms(adapters, k=norm_k)
        surviving, surviving_subs = [], []
        for adapter, sub, res in zip(adapters, kept, norm_results, strict=True):
            if res.accepted:
                surviving.append(adapter)
                surviving_subs.append(sub)
            else:
                rejected += 1
                task = conn.execute(
                    "SELECT * FROM tasks WHERE id = ?", (sub["task_id"],)
                ).fetchone()
                _record_verdict(conn, sub["task_id"], task, res, now)
        adapters, kept = surviving, surviving_subs

    total_steps = sum(int(s["steps_completed"]) for s in kept)
    contributors = conn.execute(
        """SELECT COUNT(DISTINCT w.contributor_id) AS n FROM submissions s
           JOIN tasks t ON t.id = s.task_id
           JOIN workers w ON w.id = t.worker_id
           WHERE t.run_id = ? AND t.round_idx = ? AND s.accepted = 1""",
        (run_id, round_idx),
    ).fetchone()["n"]

    result_ref: str | None = None
    divergence: float | None = None

    if adapters:
        weights = aggregate.dense_weights(
            [int(s["steps_completed"]) for s in kept], keys, cap=cap
        )
        momentum = None
        if run["outer_momentum_ref"]:
            try:
                momentum = aggregate.load_adapter(store.get_bytes(run["outer_momentum_ref"]))
            except Exception:
                momentum = None  # rebuilt from zeros; costs one round of momentum

        combined, new_momentum = aggregate.combine(
            base, adapters, weights,
            mode=run["combine_mode"],
            lr_outer=float(run["lr_outer"]),
            beta=float(run["outer_beta"]),
            momentum=momentum,
        )
        divergence = aggregate.adapter_divergence(adapters)

        result_ref = base_adapter_key(run_id, round_idx + 1)
        store.put_bytes(result_ref, aggregate.save_adapter(combined))
        if new_momentum is not None:
            mref = momentum_key(run_id)
            store.put_bytes(mref, aggregate.save_adapter(new_momentum))
            with immediate(conn):
                conn.execute(
                    "UPDATE runs SET outer_momentum_ref = ? WHERE id = ?", (mref, run_id)
                )

    with immediate(conn):
        conn.execute(
            """UPDATE rounds
               SET status = 'closed', closed_at = ?, result_adapter_ref = ?,
                   distinct_contributors = ?, adapter_divergence = ?
               WHERE run_id = ? AND idx = ?""",
            (rounds._iso(now), result_ref, contributors, divergence, run_id, round_idx),
        )
        # Any lease still outstanding belongs to a round that no longer exists.
        conn.execute(
            """UPDATE tasks SET status = 'expired', lease_expires_at = NULL
               WHERE run_id = ? AND round_idx = ? AND status = 'leased'""",
            (run_id, round_idx),
        )

    # Fold observed throughput back in, so the next round's budgets are measured
    # rather than guessed. This is what makes the cold-start default cheap.
    for s in kept:
        metrics = json.loads(s["metrics_json"] or "{}")
        spm, gpu = metrics.get("steps_per_min"), metrics.get("gpu_model")
        if spm and gpu:
            plan.update_throughput(conn, run_id, gpu, float(spm), now)

    next_opened = False
    if result_ref is not None and round_idx + 1 < run["target_rounds"]:
        plan.open_round(
            conn, run_id, round_idx + 1, result_ref,
            target_steps=rnd["target_steps"],
            min_round_sec=rnd["min_round_sec"],
            max_round_sec=rnd["max_round_sec"],
        )
        next_opened = True
    elif result_ref is not None:
        with immediate(conn):
            conn.execute("UPDATE runs SET status = 'done' WHERE id = ?", (run_id,))

    return CloseResult(
        run_id=run_id, round_idx=round_idx, reason=reason,
        accepted=len(kept), rejected=rejected, total_steps=total_steps,
        distinct_contributors=int(contributors), result_adapter_ref=result_ref,
        divergence=divergence, next_round_opened=next_opened,
    )
