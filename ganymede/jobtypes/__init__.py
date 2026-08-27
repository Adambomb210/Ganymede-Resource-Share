# -> ganymede/jobtypes/__init__.py
"""Job-type SDK package (docs/10-jobtype-sdk.md §1-§2).

The generic coordinator imports this package -- the ``REGISTRY`` and
``resolve()`` -- and never a type module directly. First-party types register
on import here.

Phase A ships one type, ``collab_lora_finetune``, whose implementation is the
coordinator's own round lifecycle / gates / combine relocated verbatim behind
the ``JobType`` seam. ``batch_inference`` (docs/10 §4) is a later phase.
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
from ganymede.jobtypes.collab_lora_finetune import CollabLoraFinetune

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
}


def resolve(job_type: str, version: int = 1):
    """Return a fresh instance of the registered type, asserting its version.

    ``spec_json.sdk`` freezes ``{job_type, version}`` at ``POST /v1/jobs``
    (docs/10 §2); a worker/coordinator whose ``REGISTRY`` entry is older than
    that pinned version is out of date. Phase A has one in-tree version (1).
    """
    try:
        cls = REGISTRY[job_type]
    except KeyError:
        raise KeyError(f"unknown job type: {job_type!r}") from None
    inst = cls()
    assert inst.version >= version, (
        f"job type {job_type!r} is v{inst.version}, spec pins v{version}"
    )
    return inst
