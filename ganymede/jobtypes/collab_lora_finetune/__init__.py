# -> ganymede/jobtypes/collab_lora_finetune/__init__.py
"""``collab_lora_finetune`` -- the collaborative-LoRA training control flow,
relocated behind the ``JobType`` seam in Phase A (docs/10-jobtype-sdk.md §3).

Phase A is a **numerically inert relocation**: the round lifecycle
(``plan``/``claim``), the acceptance gates (``validate``) and the outer combine
(``reduce`` + ``aggregate``) are the coordinator's old bodies, moved verbatim.
The generic coordinator drives them through the run-centric methods below; the
``JobRow``-based seven-method protocol (docs/05) is Phase B's, when the ``jobs``
table exists. ``shape_claim`` and ``still_accepting`` are the two optional seams
docs/10 §3 adds so ``claim_task`` and the round-closed 409 path could move
without being rewritten.
"""

from __future__ import annotations

from ganymede.coordinator import rounds
from ganymede.jobtypes.collab_lora_finetune import (
    aggregate,
    claim,
    inputs,
    plan,
    reduce,
    validate,
)

__all__ = ["CollabLoraFinetune", "aggregate", "claim", "inputs", "plan", "reduce", "validate"]


class CollabLoraFinetune:
    """The one Phase A job type. Methods delegate to the relocated bodies."""

    name = "collab_lora_finetune"
    version = 1

    # -- round lifecycle (ex-coordinator/rounds.py) ------------------------
    def open_round(self, conn, run_id, idx, base_adapter_ref, target_steps,
                   min_round_sec, max_round_sec):
        return plan.open_round(conn, run_id, idx, base_adapter_ref, target_steps,
                               min_round_sec, max_round_sec)

    def current_round(self, conn, run_id):
        return plan.current_round(conn, run_id)

    def round_progress(self, conn, run_id, round_idx):
        return plan.round_progress(conn, run_id, round_idx)

    def should_close(self, conn, run_id, round_idx, now=None):
        return plan.should_close(conn, run_id, round_idx, now)

    def reopen_empty_round(self, conn, run_id, round_idx, now=None):
        return plan.reopen_empty_round(conn, run_id, round_idx, now)

    def task_seed(self, run_id, round_idx, task_id):
        return plan.task_seed(run_id, round_idx, task_id)

    def update_throughput(self, conn, run_id, gpu_model, steps_per_min, now=None):
        return plan.update_throughput(conn, run_id, gpu_model, steps_per_min, now)

    # -- claim seam (ex-rounds.claim_task) --------------------------------
    # docs/10 §3 freezes shape_claim(job, profile, settings, conn, now); Phase A
    # has no `jobs` table, so this keeps claim_task's own run-centric signature
    # and body verbatim. Refusal is still `raise rounds.NotEligible`, recorded
    # in worker_eligibility exactly as today.
    def shape_claim(self, conn, run_id, worker_id, contributor_clearance, profile,
                    settings, now=None, worker_image_tag=None):
        return claim.claim_task(conn, run_id, worker_id, contributor_clearance,
                                profile, settings, now, worker_image_tag)

    # -- 409 seam (ex-rounds heartbeat/record_submission round-status read) --
    def still_accepting(self, conn, run_id, round_idx):
        """None if the round still accepts work, else a RoundClosed to raise."""
        rnd = conn.execute(
            "SELECT status FROM rounds WHERE run_id = ? AND idx = ?",
            (run_id, round_idx),
        ).fetchone()
        if rnd is None or rnd["status"] != "open":
            return rounds.RoundClosed("round is no longer open")
        return None

    # -- validate (ex-coordinator/closer.py gate_submission / expected_manifest) --
    def expected_manifest(self, store, base_adapter_ref):
        return validate.expected_manifest(store, base_adapter_ref)

    def gate_submission(self, conn, store, task_id, expected, now=None):
        return validate.gate_submission(conn, store, task_id, expected, now)

    # -- reduce (ex-closer._close_claimed_round) --------------------------
    def reduce_close(self, conn, store, run_id, round_idx, reason, now, norm_k, cap):
        return reduce._close_claimed_round(conn, store, run_id, round_idx, reason,
                                           now, norm_k, cap)

    # -- inputs_for (docs/10 §1) ----------------------------------------
    def inputs_for(self, task, store):
        return inputs.inputs_for(task, store)
