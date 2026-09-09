# -> ganymede/jobtypes/contained_batch/__init__.py
"""``contained_batch`` -- the third first-party job type (docs/10 §7, docs/11 §2).

docs/04's Phase E asks for "one more genuinely different type ... to confirm the
SDK isn't just training with two spellings." ``batch_inference`` was weak
evidence for that: it still loads a model, still runs on a GPU, still submits an
artifact to a presigned key. This one runs **a body Ganymede did not write**, in
a container, with no network -- and it is the first consumer
``sandbox.JobContainer`` has ever had.

What it changed about the SDK, which is the part worth reading:

* **Nothing structural.** ``plan`` / ``inputs_for`` / ``validate`` / ``credit``
  are the same shapes ``batch_inference`` uses, and ``ContainedResult`` reuses
  ``InferResult``'s field set so the worker's existing ``_submit_shard`` carries
  it unchanged. The seven-method protocol did not grow a member.
* **One thing it could not inherit: the assumption of determinism.** Two of the
  coordinator's mechanisms -- ``attempt_group`` redundancy and docs/13 §5's
  spot-checks -- compare two machines' outputs for equality. Both are claims
  about a body's reproducibility, and neither can be made about a submitter's
  image. ``validate_spec`` refuses ``redundancy``; ``spot_checkable`` is the
  opt-in that stops probes being issued. See docs/10 §7's deviation note.

``jobs.image_id`` is **required**, the inverse of every other in-tree type.
"""

from __future__ import annotations

from ganymede.jobtypes.base import ReduceState, Verdict, WorkUnits
from ganymede.jobtypes.contained_batch import inputs, plan, run, validate
from ganymede.jobtypes.contained_batch.run import ContainedResult, ContainedTask

__all__ = [
    "ContainedBatch",
    "ContainedTask",
    "ContainedResult",
    "inputs",
    "plan",
    "run",
    "validate",
]


class ContainedBatch:
    """Embarrassingly-parallel batch over a submitter image. All seven protocol
    methods; no optional claim seam."""

    name = "contained_batch"
    version = 1

    # docs/13 §5.2: a type opts in to known-answer probes *by being
    # deterministic*, and this one cannot be. The attribute is read by
    # ``spotcheck.maybe_issue``; its default is off, so a future type is safe
    # until it says otherwise rather than the other way round.
    spot_checkable = False

    # docs/11 §4's split, as a fact the worker can act on rather than a
    # statement in prose. Read by ``Worker.can_honor``: a type that requires an
    # image is refused without one, and a type that does not is refused *with*
    # one -- there is no in-tree body that may run a submitter image unconfined.
    requires_image = True

    # -- spec validation (docs/06 "POST /v1/jobs") ------------------------
    def validate_spec(self, spec) -> None:
        plan.validate_spec(spec)

    # -- plan: one task per shard, never a redundant copy -----------------
    def plan(self, job, conn):
        return plan.plan(job, conn)

    # -- inputs_for: shard GET + output PUT; the image is the payload's ---
    def inputs_for(self, task, store):
        return inputs.inputs_for(task, store)

    # -- run: the contained worker body -----------------------------------
    def run(self, task, task_inputs, on_step=None, should_stop=None, **kw):
        return run.run(task, task_inputs, on_step, should_stop, **kw)

    # -- validate: per-submission structural gate, no compare_digest ------
    def validate(self, task, result, conn, store) -> Verdict:
        return validate.validate(task, result, conn, store)

    # -- reduce: always None -- embarrassingly parallel -------------------
    def reduce(self, job, results, conn, store) -> ReduceState | None:
        return None

    # -- is_complete: not consulted; completion is the dispatcher's -------
    def is_complete(self, job, state) -> bool:
        return True

    # -- credit: trusted work-done units for the 'work' signal ------------
    def credit(self, task, result) -> WorkUnits:
        return WorkUnits("rows", int(getattr(result, "rows", 0)))
