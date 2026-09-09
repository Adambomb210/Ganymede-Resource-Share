# -> ganymede/jobtypes/batch_inference/__init__.py
"""``batch_inference`` -- the second first-party job type (docs/10-jobtype-sdk.md §4).

The proof the ``JobType`` seam is not welded to LoRA: no base adapter, no
``reduce``, no round. ``plan`` fixes a finite task set at enqueue; the generic
dispatcher decides completion (docs/10 §5, "``None`` = embarrassingly
parallel"). Redundant execution is a ``fraction`` of shards dispatched ``n``
ways under a shared ``attempt_group``, agreed on by the coordinator.

It omits ``shape_claim`` / ``still_accepting`` -- its ``plan`` output is claimed
as-is and no round can close under a worker, so there is no 409.
"""

from __future__ import annotations

from ganymede.jobtypes.base import ReduceState, Verdict, WorkUnits
from ganymede.jobtypes.batch_inference import inputs, plan, run, validate
from ganymede.jobtypes.batch_inference.run import InferResult, InferTask

__all__ = [
    "BatchInference",
    "InferTask",
    "InferResult",
    "inputs",
    "plan",
    "run",
    "validate",
]


class BatchInference:
    """Embarrassingly-parallel batch generation. All seven protocol methods; no
    optional claim seam."""

    name = "batch_inference"
    version = 1

    # docs/13 §5.2: the type opts in to known-answer probes by being
    # deterministic, and this one is -- a redundancy job must decode greedily
    # (``plan.validate_spec``). It is the only type that opts in, which is what
    # makes ``spotcheck.judge`` reaching for *this* type's comparator correct
    # rather than an accident of it having been the only static type.
    spot_checkable = True
    # First-party and in-tree: ``jobs.image_id`` is NULL and there is no
    # contained body to run one in (docs/11 §4).
    requires_image = False

    # -- spec validation (docs/06 "POST /v1/jobs") ------------------------
    def validate_spec(self, spec) -> None:
        plan.validate_spec(spec)

    # -- plan: one task per shard, n copies for a fraction (docs/10 §4) ---
    def plan(self, job, conn):
        return plan.plan(job, conn)

    # -- inputs_for: model GET + output PUT, no base adapter -------------
    def inputs_for(self, task, store):
        return inputs.inputs_for(task, store)

    # -- run: the worker body ------------------------------------------
    def run(self, task, task_inputs, on_step=None, should_stop=None, **kw):
        return run.run(task, task_inputs, on_step, should_stop, **kw)

    # -- validate: per-submission structural gate (docs/10 §4) ----------
    def validate(self, task, result, conn, store) -> Verdict:
        return validate.validate(task, result, conn, store)

    # -- reduce: always None -- embarrassingly parallel ----------------
    def reduce(self, job, results, conn, store) -> ReduceState | None:
        return None

    # -- is_complete: not consulted; completion is the dispatcher's ----
    def is_complete(self, job, state) -> bool:
        return True

    # -- credit: trusted work-done units for the 'work' signal --------
    def credit(self, task, result) -> WorkUnits:
        return WorkUnits("rows", int(getattr(result, "rows", 0)))
