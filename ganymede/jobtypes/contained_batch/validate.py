"""Submit-time acceptance for ``contained_batch`` (docs/10 §7).

``validate(task, result, conn, store)`` is per-submission and structural:

  a. row count == the shard's declared ``rows``  -> ``"row_count_mismatch"``
  b. every row conforms to ``output_schema``     -> ``"schema_mismatch"``
  c. ids are unique                              -> ``"duplicate_ids"``

**No ``compare_digest``.** This is the one place this type deliberately differs
from ``batch_inference``'s otherwise identical gate. That field is what the
coordinator compares across an ``attempt_group``, and comparing it presumes two
machines running one shard produce the same bytes -- a claim nobody can make
about an image the coordinator did not build. ``plan.validate_spec`` refuses
``redundancy`` for the same reason, so no group ever forms; leaving the digest
off the verdict as well means the assumption cannot be reintroduced by accident
from this side either. The digest is still *reported* in the submission's
metrics, where it is a fingerprint for a human rather than evidence against a
machine.
"""

from __future__ import annotations

import json
from typing import Any

from ganymede.jobtypes.base import Verdict

_PY = {"str": str, "int": (int,), "float": (int, float), "bool": bool}


def _row_ok(row: Any, schema: dict[str, str]) -> bool:
    if not isinstance(row, dict):
        return False
    if set(row.keys()) != set(schema.keys()):
        return False
    for name, typ in schema.items():
        want = _PY.get(typ, object)
        value = row[name]
        if typ == "bool":
            if not isinstance(value, bool):
                return False
        elif isinstance(value, bool) or not isinstance(value, want):
            # bool is an int in Python and is never what a schema meant by one.
            return False
    return True


def validate(task, result, conn, store) -> Verdict:
    """``task``: a ``tasks`` row; ``result``: a ``ContainedResult``."""
    from ganymede.jobtypes.contained_batch.run import parse_jsonl

    desc = json.loads(task["input_ref_json"])
    schema: dict[str, str] = desc["output_schema"]
    expected = int(desc["shard_rows"])

    rows = getattr(result, "rows", None)
    if rows != expected:
        return Verdict(
            accepted=False,
            reason="row_count_mismatch",
            detail=f"shard declared {expected} rows, submission has {rows}",
        )

    # The gate reads the *stored artifact*, not the worker's own count. The
    # worker is the thing being checked, so a verdict that trusted its summary
    # would only be checking its arithmetic.
    try:
        body = parse_jsonl(store.get_bytes(result.output_ref))
    except Exception as exc:  # noqa: BLE001
        return Verdict(accepted=False, reason="unreadable_output",
                       detail=f"could not read {result.output_ref}: {exc}")

    if len(body) != expected:
        return Verdict(
            accepted=False,
            reason="row_count_mismatch",
            detail=f"shard declared {expected} rows, artifact has {len(body)}",
        )

    for i, row in enumerate(body):
        if not _row_ok(row, schema):
            return Verdict(
                accepted=False,
                reason="schema_mismatch",
                detail=f"row {i} does not match output_schema {sorted(schema)}",
            )

    ids = [str(r.get("id")) for r in body]
    if len(set(ids)) != len(ids):
        # A container that emitted one row per input row but repeated an id has
        # answered a different question than the one asked, and nothing
        # downstream that joins on id would notice.
        return Verdict(accepted=False, reason="duplicate_ids",
                       detail="output ids are not unique")

    return Verdict(accepted=True)
