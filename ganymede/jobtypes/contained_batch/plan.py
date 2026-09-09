"""``contained_batch`` planning and spec validation (docs/10 §7, docs/11 §2).

Structurally ``batch_inference``'s planner: one ``TaskSpec`` per submitter-declared
shard, packed into a self-contained ``input_ref`` descriptor because the frozen
``inputs_for(task, store)`` signature has no ``conn`` to read ``jobs.spec_json``
back.

What differs is what the spec is allowed to say. This type runs *submitter code*,
so the coordinator cannot know whether two runs over one shard produce the same
bytes -- that is a property of an image it did not build. Everything downstream
that assumes reproducibility is therefore refused here rather than left to
misfire later; see ``validate_spec``.
"""

from __future__ import annotations

import json
import sqlite3
import uuid

from ganymede.jobtypes.base import TaskSpec

_SCHEMA_TYPES = {"str", "int", "float", "bool"}


def validate_spec(spec) -> None:
    """Shape-check a ``POST /v1/jobs`` ``spec`` for ``contained_batch``.

    Raises ``ValueError`` with a field-named message.
    """
    if not isinstance(spec, dict):
        raise ValueError("spec must be a JSON object")

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

    schema = spec.get("output_schema")
    if not isinstance(schema, dict) or not schema:
        raise ValueError("spec.output_schema must be a non-empty object")
    for name, typ in schema.items():
        if typ not in _SCHEMA_TYPES:
            raise ValueError(
                f"spec.output_schema[{name!r}] must be one of {sorted(_SCHEMA_TYPES)}"
            )
    if "id" not in schema:
        # ``validate`` aligns the container's output against the input shard by
        # id, and without one there is no way to say a row came back at all --
        # only that the right *number* of rows did.
        raise ValueError("spec.output_schema must include an 'id' field")

    params = spec.get("params", {})
    if not isinstance(params, dict):
        raise ValueError("spec.params must be an object")
    # It is handed to the container as a file, so it has to survive a round trip
    # through JSON. Caught here rather than at stage time, where the failure
    # would be one worker's crash on a job that had already been accepted.
    try:
        json.dumps(params)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"spec.params must be JSON-serializable: {exc}") from None

    if spec.get("redundancy"):
        # docs/10 §4's redundancy compares two machines' outputs for exact
        # equality, and docs/13 §5's spot-checks do the same against an already
        # accepted answer. Both are claims about determinism, and this type
        # cannot make one: the body is an image the coordinator did not build.
        # An honest machine running a job that samples, threads, or stamps a
        # timestamp would be convicted -- and docs/09 §5.1 rates a failed probe
        # the largest single penalty in the system.
        #
        # Refused at submit rather than ignored at plan, because a submitter who
        # asked for redundancy and silently did not get it is owed the error.
        raise ValueError(
            "spec.redundancy is not supported for contained_batch: cross-machine "
            "comparison assumes a deterministic body, and this type runs a "
            "submitter image the coordinator cannot make that claim about"
        )


def _descriptor(spec: dict, shard: dict, task_id: str) -> str:
    """Everything ``inputs_for`` / ``validate`` / the worker need, in one blob."""
    return json.dumps(
        {
            "shard_ref": shard["ref"],
            "shard_rows": int(shard["rows"]),
            "output_key": f"{spec['output_prefix'].rstrip('/')}/{task_id}.jsonl",
            "output_schema": spec["output_schema"],
            "params": spec.get("params", {}),
        },
        sort_keys=True,
    )


def plan(job: sqlite3.Row, conn: sqlite3.Connection) -> list[TaskSpec]:
    """One ``TaskSpec`` per shard. ``conn`` is unused -- the shard index is
    submitter-declared -- and kept for signature parity.

    No ``attempt_group`` is ever set: ``validate_spec`` refuses ``redundancy``,
    so there is never more than one copy of a unit.
    """
    spec = json.loads(job["spec_json"])
    max_runtime = spec.get("max_runtime_sec")
    specs: list[TaskSpec] = []
    for shard in spec["shards"]:
        task_id = uuid.uuid4().hex
        specs.append(
            TaskSpec(
                id=task_id,
                job_id=job["id"],
                input_ref=_descriptor(spec, shard, task_id),
                max_runtime_sec=int(max_runtime) if max_runtime else None,
            )
        )
    return specs
