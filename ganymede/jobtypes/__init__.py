# -> ganymede/jobtypes/__init__.py
"""Job-type SDK package (docs/10-jobtype-sdk.md §1-§2).

The generic coordinator imports this package -- the ``REGISTRY`` and
``resolve()`` -- and never a type module directly. First-party types register
on import here.

``collab_lora_finetune`` is the coordinator's own round lifecycle / gates /
combine relocated verbatim behind the ``JobType`` seam (Phase A).
``batch_inference`` (docs/10 §4) is the second, first-party type -- no base
adapter, no ``reduce``, no round -- the proof the seam is not welded to LoRA.
``contained_batch`` (docs/10 §7) is the third, and the one that runs a body
Ganymede did not write: ``jobs.image_id`` is **required** for it and ``NULL``
for the other two. All three are in-tree and versioned by the coordinator
release; what varies is whose code runs inside them.
"""

from __future__ import annotations

from ganymede.jobtypes.base import (
    ClaimRefusal,
    InputRefs,
    JobType,
    ReduceState,
    TaskSpec,
    Verdict,
    WorkUnits,
)
from ganymede.jobtypes.batch_inference import BatchInference
from ganymede.jobtypes.collab_lora_finetune import CollabLoraFinetune
from ganymede.jobtypes.contained_batch import ContainedBatch

__all__ = [
    "REGISTRY",
    "resolve",
    "JobType",
    "TaskSpec",
    "InputRefs",
    "ReduceState",
    "WorkUnits",
    "Verdict",
    "ClaimRefusal",
]

# job_type -> class. First-party, in-tree, versioned by the coordinator release.
REGISTRY: dict[str, type] = {
    CollabLoraFinetune.name: CollabLoraFinetune,
    BatchInference.name: BatchInference,
    ContainedBatch.name: ContainedBatch,
}


def resolve(job_type: str, version: int | None = None):
    """Return a fresh instance of the registered type (docs/10 §2).

    ``version`` is the value ``spec_json.sdk.version`` froze at ``POST
    /v1/jobs``. When given, this asserts ``inst.version >= version`` -- a
    coordinator or worker whose ``REGISTRY`` entry is older than the pinned
    version is out of date, which the claim walk turns into a
    ``job_type_version_unsupported`` refusal (docs/10 §2). ``None`` skips the
    check, for call sites that only need the current implementation.
    """
    try:
        cls = REGISTRY[job_type]
    except KeyError:
        raise KeyError(f"unknown job type: {job_type!r}") from None
    inst = cls()
    if version is not None:
        assert inst.version >= version, (
            f"job type {job_type!r} is v{inst.version}, spec pins v{version}"
        )
    return inst
