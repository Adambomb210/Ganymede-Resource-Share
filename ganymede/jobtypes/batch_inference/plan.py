"""``batch_inference`` planning and spec validation (docs/10-jobtype-sdk.md §4).

``plan(job, conn)`` is called **once, at enqueue** -- it has no ``store`` to
enumerate a bucket, so the shard index (refs *and* row counts) is
submitter-declared and shape-checked here by ``validate_spec`` on
``POST /v1/jobs``. There are no rounds: one ``TaskSpec`` per shard, plus ``n``
redundant copies sharing an ``attempt_group`` for a ``fraction`` of shards.

The per-task input descriptor packed into ``TaskSpec.input_ref`` is
self-contained on purpose: the frozen ``inputs_for(task, store)`` signature has
no ``conn`` to read ``jobs.spec_json`` back, so ``plan`` copies every field the
worker and the coordinator's ``validate`` will need (docs/10 §4 lists the shard
ref as ``input_ref``; this is a superset forced by that signature).
"""

from __future__ import annotations

import json
import math
import sqlite3
import uuid

from ganymede.coordinator import store as store_mod
from ganymede.jobtypes.base import TaskSpec

# The output-schema value vocabulary. A submitter names a Python-ish type per
# field; ``validate`` (submit-time, per row) checks each cell against it.
_SCHEMA_TYPES = {"str", "int", "float", "bool"}
_DECODE_MODES = {"greedy", "sample"}


def validate_spec(spec) -> None:
    """Shape-check a ``POST /v1/jobs`` ``spec`` for ``batch_inference``.

    Raises ``ValueError`` with a field-named message. Enforces docs/10 §4's
    object shape; a job that carries ``redundancy`` must decode greedily -- a
    stochastic decode has no exact cross-task comparator.
    """
    if not isinstance(spec, dict):
        raise ValueError("spec must be a JSON object")

    model_ref = spec.get("model_ref")
    if not isinstance(model_ref, str) or not model_ref:
        raise ValueError("spec.model_ref must be a non-empty string")

    shards = spec.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("spec.shards must be a non-empty list")
    for i, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise ValueError(f"spec.shards[{i}] must be an object")
        if not isinstance(shard.get("ref"), str) or not shard["ref"]:
            raise ValueError(f"spec.shards[{i}].ref must be a non-empty string")
        rows = shard.get("rows")
        if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0:
            raise ValueError(f"spec.shards[{i}].rows must be a positive integer")

    if not isinstance(spec.get("output_prefix"), str) or not spec["output_prefix"]:
        raise ValueError("spec.output_prefix must be a non-empty string")
    # A submitter names their inputs by object key, and those keys are handed
    # to ``presign_get`` to mint a URL a worker fetches. Nothing else
    # constrains them, so a key inside the coordinator's own namespace would
    # make the coordinator sign a read of its own internal state on the
    # submitter's behalf -- another run's outer momentum or round base
    # adapter, possibly for a run whose data_classification this submitter has
    # no clearance for. Refused here, at submission, so the submitter is told
    # rather than finding out as a mysterious mid-run failure; the presign
    # helpers in ``inputs.py`` refuse again as defence in depth.
    for i, shard in enumerate(shards):
        if store_mod.is_reserved_key(shard["ref"]):
            raise ValueError(
                f"spec.shards[{i}].ref may not address the coordinator's own "
                f"storage namespace ({', '.join(store_mod.RESERVED_KEY_PREFIXES)})")
    if store_mod.is_reserved_key(model_ref):
        raise ValueError(
            "spec.model_ref may not address the coordinator's own "
            "storage namespace")
    if store_mod.is_reserved_key(spec["output_prefix"]):
        raise ValueError(
            "spec.output_prefix may not write into the coordinator's own "
            "storage namespace")

    template = spec.get("prompt_template")
    if not isinstance(template, str) or "{input}" not in template:
        raise ValueError("spec.prompt_template must be a string containing '{input}'")

    decode = spec.get("decode")
    if not isinstance(decode, dict):
        raise ValueError("spec.decode must be an object")
    if decode.get("mode") not in _DECODE_MODES:
        raise ValueError(f"spec.decode.mode must be one of {sorted(_DECODE_MODES)}")
    mnt = decode.get("max_new_tokens")
    if not isinstance(mnt, int) or isinstance(mnt, bool) or mnt <= 0:
        raise ValueError("spec.decode.max_new_tokens must be a positive integer")

    schema = spec.get("output_schema")
    if not isinstance(schema, dict) or not schema:
        raise ValueError("spec.output_schema must be a non-empty object")
    for name, typ in schema.items():
        if typ not in _SCHEMA_TYPES:
            raise ValueError(
                f"spec.output_schema[{name!r}] must be one of {sorted(_SCHEMA_TYPES)}"
            )

    redundancy = spec.get("redundancy")
    if redundancy is not None:
        if not isinstance(redundancy, dict):
            raise ValueError("spec.redundancy must be an object")
        frac = redundancy.get("fraction")
        if not isinstance(frac, (int, float)) or isinstance(frac, bool) or not 0 < frac <= 1:
            raise ValueError("spec.redundancy.fraction must be in (0, 1]")
        n = redundancy.get("n")
        if not isinstance(n, int) or isinstance(n, bool) or n < 2:
            raise ValueError("spec.redundancy.n must be an integer >= 2")
        sr = redundancy.get("sample_rows")
        if not isinstance(sr, int) or isinstance(sr, bool) or sr < 1:
            raise ValueError("spec.redundancy.sample_rows must be an integer >= 1")
        agree_on = redundancy.get("agree_on")
        if agree_on not in schema:
            raise ValueError(
                "spec.redundancy.agree_on must name a field in output_schema"
            )
        if decode.get("mode") != "greedy":
            raise ValueError(
                "a redundancy job must use decode.mode='greedy' -- a stochastic "
                "decode has no exact cross-task comparator"
            )


def _output_key(output_prefix: str, task_id: str) -> str:
    """Where one shard's output lands. Derived, never taken from a worker."""
    return f"{output_prefix.rstrip('/')}/{task_id}.jsonl"


def _descriptor(spec: dict, shard: dict, task_id: str) -> str:
    """The self-contained per-task input, JSON-encoded for ``input_ref_json``."""
    return json.dumps(
        {
            "shard_ref": shard["ref"],
            "shard_rows": int(shard["rows"]),
            "model_ref": spec["model_ref"],
            "prompt_template": spec["prompt_template"],
            "decode": spec["decode"],
            "output_schema": spec["output_schema"],
            "output_prefix": spec["output_prefix"],
            "output_key": _output_key(spec["output_prefix"], task_id),
        },
        sort_keys=True,
    )


def plan(job: sqlite3.Row, conn: sqlite3.Connection) -> list[TaskSpec]:
    """One ``TaskSpec`` per shard; ``n`` copies sharing an ``attempt_group`` for
    a ``fraction`` of shards. ``conn`` is unused -- the shard index is
    submitter-declared (docs/10 §4) -- and kept for signature parity."""
    spec = json.loads(job["spec_json"])
    shards = spec["shards"]
    redundancy = spec.get("redundancy") or {}
    n = int(redundancy.get("n", 1)) if redundancy else 1
    frac = float(redundancy.get("fraction", 0.0)) if redundancy else 0.0
    # First ``k`` shards get the redundant treatment. Deterministic (not random)
    # so a re-plan of the same spec is identical, and "first k" keeps the
    # sampled set contiguous, which is all a Phase-D scorer needs.
    k = max(1, math.ceil(frac * len(shards))) if frac > 0 and n > 1 else 0

    specs: list[TaskSpec] = []
    for i, shard in enumerate(shards):
        copies = n if i < k else 1
        group = uuid.uuid4().hex if copies > 1 else None
        for _ in range(copies):
            task_id = uuid.uuid4().hex
            specs.append(
                TaskSpec(
                    id=task_id,
                    job_id=job["id"],
                    input_ref=_descriptor(spec, shard, task_id),
                    attempt_group=group,
                )
            )
    return specs
