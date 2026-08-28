"""Submit-time acceptance for ``batch_inference`` (docs/10-jobtype-sdk.md §4).

``validate(task, result, conn, store)`` is per-submission and structural:

  a. row count == the shard's declared ``rows``  -> ``"row_count_mismatch"``
  b. every row conforms to ``output_schema``     -> ``"schema_mismatch"``
  c. attach ``compare_digest = result.digest``

It does **not** compare across the ``attempt_group`` -- that is the
coordinator's, driven from ``close.advance_job`` via ``sample_agreement`` below
(kept here because it reads this type's spec shape; invoked, not owned, by the
type). Full probation / sampling scoring is Phase D; this is a basic gate.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from ganymede.jobtypes.base import Verdict
from ganymede.jobtypes.batch_inference.run import parse_jsonl

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
            return False
    return True


def validate(task, result, conn, store) -> Verdict:
    """``task``: a ``tasks`` row; ``result``: an ``InferResult``."""
    desc = json.loads(task["input_ref_json"])
    schema: dict[str, str] = desc["output_schema"]
    expected_rows = int(desc["shard_rows"])

    try:
        raw = store.get_bytes(result.output_ref)
    except Exception:
        return Verdict(False, "missing_artifact", compare_digest=result.digest)

    try:
        rows = parse_jsonl(raw)
    except Exception:
        return Verdict(False, "schema_mismatch", detail="output is not JSONL",
                       compare_digest=result.digest)

    if len(rows) != expected_rows:
        return Verdict(
            False, "row_count_mismatch",
            detail=f"{len(rows)} rows, shard declared {expected_rows}",
            compare_digest=result.digest,
        )

    for i, row in enumerate(rows):
        if not _row_ok(row, schema):
            return Verdict(
                False, "schema_mismatch", detail=f"row {i} does not match output_schema",
                compare_digest=result.digest,
            )

    return Verdict(True, compare_digest=result.digest)


# --------------------------------------------------------------------------
# Coordinator-owned attempt-group agreement (docs/10 §4)
# --------------------------------------------------------------------------


def sample_agreement(
    outputs: Sequence[Sequence[dict[str, Any]]],
    sample_rows: int,
    agree_on: str,
) -> bool:
    """Do the ``n`` result sets agree on ``agree_on`` for a sample of rows?

    Aligns rows by ``id``, samples the first ``sample_rows`` ids present in
    *every* result set, and requires exact equality of the ``agree_on`` field
    across them (exact, because a redundancy job decodes greedily). An empty
    intersection is a disagreement -- the results are not even about the same
    rows.
    """
    if len(outputs) < 2:
        return True
    by_id: list[dict[Any, dict[str, Any]]] = [
        {str(r.get("id")): r for r in result} for result in outputs
    ]
    common = set(by_id[0])
    for m in by_id[1:]:
        common &= set(m)
    if not common:
        return False
    for rid in sorted(common)[: max(1, sample_rows)]:
        values = {json.dumps(m[rid].get(agree_on), sort_keys=True) for m in by_id}
        if len(values) != 1:
            return False
    return True
