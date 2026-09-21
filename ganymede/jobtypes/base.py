"""The job-type contract (docs/10-jobtype-sdk.md §1, docs/05-data-model.md).

Phase A introduces this seam as a *numerically inert relocation*: the
collaborative-LoRA control flow moves behind it verbatim. The generic
coordinator talks to a resolved ``JobType`` instance (see
``ganymede.jobtypes.resolve``) and never imports a type module directly.

``TaskSpec`` is widened here, not redefined -- it gains the generic
``job_id`` / ``input_ref`` fields and defaults the LoRA-specific fields to
``None``, following the shape already frozen on the ``tasks`` table. Every
current call site passes the same fields it always did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class TaskSpec:
    """The coordinator -> worker task descriptor.

    Widened from the LoRA-only shape that lived in ``coordinator/rounds.py``:
    ``job_id`` / ``input_ref`` are the generic handles, and every LoRA-specific
    field now defaults so a non-training type can build one without naming an
    adapter, a run or a round (docs/10 "Spine deviations" #1). No wire or column
    change -- this mirrors the ``tasks`` table's generic ``input_ref_json``
    beside type-specific ``buckets_json`` and its nullable ``run_id`` /
    ``round_idx``. Every current ``collab_lora_finetune`` call site passes the
    same fields by keyword, so the added defaults are inert for it.
    """

    id: str
    # Generic fields (docs/10 "Spine deviations" #1). ``input_ref`` is the
    # type-agnostic input handle -- a shard ref for ``batch_inference``.
    # ``attempt_group`` ties the ``n`` redundant copies of one logical unit
    # together (docs/05 ``tasks.attempt_group``); ``None`` = singleton.
    job_id: str | None = None
    input_ref: str | None = None
    attempt_group: str | None = None
    # LoRA / round fields -- ``None`` for a type that has no run or round.
    run_id: str | None = None
    round_idx: int | None = None
    buckets: list[int] = field(default_factory=list)
    # Total bucket count for the run. The worker receives bucket *indices* and
    # has to turn them into rows itself -- the coordinator never sees the data --
    # which it cannot do without knowing how many buckets the dataset was cut
    # into. Sending the indices alone is not a shard assignment.
    num_buckets: int | None = None
    local_steps: int | None = None
    # Wall-clock safety net (8). The worker stops at whichever comes first, this
    # or lease_expires_at -- they answer different questions: this one is "how
    # long was this work budgeted", the lease is "when does the coordinator stop
    # believing you".
    max_runtime_sec: int | None = None
    lease_expires_at: datetime | None = None
    base_adapter_ref: str | None = None
    base_model: str | None = None
    base_precision: str | None = None
    lora_cfg: dict | None = None
    hyperparams: dict | None = None
    dataset_ref: str | None = None
    # Image tag the worker must be running, or None for no requirement (4.2
    # step 5). A worker that cannot honour it abandons before downloading
    # anything rather than submitting an artifact from the wrong stack.
    required_image: str | None = None
    # The device indices this lease actually holds on its worker (docs/14
    # §5.4) -- what ``_task_payload`` serialises verbatim as ``devices:
    # [int]``. Defaults to empty rather than ``[0]`` so a type that never
    # heard of docs/14 (there is only one today) does not have to name a
    # device it never allocated.
    devices: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class InputRefs:
    """What ``run`` pulls before it starts (docs/10 §1).

    ``artifacts`` maps a logical name to a presigned GET URL; ``params`` carries
    small inline values.
    """

    artifacts: dict[str, str]
    params: dict[str, Any]


@dataclass(frozen=True)
class ReduceState:
    """The reduce-checkpoint record (docs/10 §1, §5).

    For ``collab_lora_finetune``: ``epoch`` is the round index, ``result_ref``
    is the combined adapter (= next round's ``base_adapter_ref``), ``metrics``
    holds divergence / contributor counts. Persisted where the ``rounds`` row
    lives today.
    """

    epoch: int
    result_ref: str | None
    metrics: dict[str, Any]


@dataclass(frozen=True)
class WorkUnits:
    """``credit()``'s return -- an advisory, trusted work-done figure."""

    unit: str  # "tokens" | "rows" | ...
    count: int


@dataclass(frozen=True)
class Verdict:
    """``aggregate.GateResult`` plus ``compare_digest`` (docs/10 §1).

    The type attaches ``compare_digest`` for the coordinator's
    redundant-execution comparison; it never compares across an
    ``attempt_group`` itself.
    """

    accepted: bool
    reason: str | None = None
    detail: str | None = None
    compare_digest: str | None = None


@dataclass(frozen=True)
class ClaimRefusal:
    """Returned by ``shape_claim`` when a machine may not have this task.

    Phase A note: ``collab_lora_finetune`` still signals refusal by raising
    ``coordinator.rounds.NotEligible`` (the ``claim_task`` body moved verbatim),
    exactly as it does today; the generic claim path records that in
    ``worker_eligibility`` unchanged. ``ClaimRefusal`` is frozen here for the
    protocol and for ``batch_inference`` / Phase B.
    """

    reason: str


@runtime_checkable
class JobType(Protocol):
    """The seam every job type implements (method set / signatures per docs/05).

    Phase A ships one implementation, ``CollabLoraFinetune``, whose bodies are
    the relocated coordinator control flow. The ``job``/``JobRow`` arguments are
    Phase B's (there is no ``jobs`` table yet); Phase A drives the type through
    run-centric entry points that delegate to the same moved bodies.
    """

    name: str
    version: int

    def plan(self, job: Any, conn: Any) -> list[TaskSpec]: ...

    def inputs_for(self, task: Any, store: Any) -> InputRefs: ...

    def run(self, task: Any, inputs: InputRefs,
            on_step: Any, should_stop: Any) -> Any: ...

    def validate(self, task: Any, result: Any,
                 conn: Any, store: Any) -> Verdict: ...

    def reduce(self, job: Any, results: list[Any],
               conn: Any, store: Any) -> ReduceState | None: ...

    def is_complete(self, job: Any, state: ReduceState | None) -> bool: ...

    def credit(self, task: Any, result: Any) -> WorkUnits: ...

    # --- optional claim seam (docs/05 "Stage 1 reconciliation", docs/10 §3) ---
    # A type that omits both gets static plan() output and never returns 409.
    # They exist so claim_task's body and the RoundClosed path MOVE VERBATIM.
    #
    # ``free_devices`` / ``gpu_count`` / ``active_task_ids`` are docs/14 §5.3's
    # additive kwargs: the generic walk computes the free set and validates
    # ``gpu_count`` (``devices.allocate`` raises on a non-positive count, and
    # an exception raised mid-walk would be a ``break``, so that check has to
    # happen before this is ever called), then hands both through so the type
    # can call ``ganymede.coordinator.devices.allocate`` itself, inside the
    # same ``immediate()`` block that creates the lease -- the type owns its
    # own transaction (docs/10 §3), so the generic walk cannot allocate on its
    # behalf without reaching into it. ``collab_lora_finetune`` is the one
    # implementation today; a future type adopts the same three kwargs rather
    # than inventing its own allocation seam. All three default to "no
    # devices, one required, nothing known" so a type that predates docs/14
    # needs no change.
    def shape_claim(self, job: Any, profile: Any, settings: Any,
                    conn: Any, now: Any, *, free_devices: list[int],
                    gpu_count: int = 1,
                    active_task_ids: Any = None) -> TaskSpec | ClaimRefusal: ...

    def still_accepting(self, job: Any, task: Any, conn: Any) -> Any: ...
