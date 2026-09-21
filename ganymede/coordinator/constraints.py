"""The job constraint predicate grammar (docs/07-scheduler.md §2, Decision 15).

``jobs.constraints_json`` is either a **pin** -- ``{"machine_ids": [...]}`` -- or a
**predicate** -- ``field -> condition`` with every field AND-ed. It is set at
``POST /v1/jobs`` and validated there (an unknown field or operator is a
submit-time 422, deliberately unlike ``budget.is_eligible`` which ignores an
unknown ``requires`` key: a job naming an unevaluable field would silently never
place).

Two entry points, kept apart on purpose:

- ``validate(obj)`` -- write-time shape check. Raises ``ConstraintError``; the
  endpoint turns that into 422.
- ``check_constraints(constraints_json, machine_id, profile)`` -- the claim-time
  gate. **Pure**: it reads only ``(machine_id, profile)``, touches no database
  and takes no write lock, which is why it sits in the ``app.py`` walk rather
  than inside ``claim_task``. Returns ``(ok, reason)``; ``reason`` is recorded
  verbatim into ``worker_eligibility`` (docs/07 §3).

Composition with ``budget.is_eligible`` is sequential, not merged (docs/07 §2):
constraints are the job's *targeting*, ``is_eligible`` is the run's *capability
floor*. A pin does not waive the capability gate.
"""

from __future__ import annotations

import json
from typing import Any

# The operator set -- small, total, typed (docs/07 §2). An operator outside this
# set is a submit-time 422.
_NUMERIC_OPS = (">=", ">", "<=", "<")
_EQ_OPS = ("==", "!=")
_MEMBER_OPS = ("in", "not_in")
_OPS = frozenset(_NUMERIC_OPS + _EQ_OPS + _MEMBER_OPS + ("contains",))

# The field resolver map (docs/07 §2, "Field resolver"). Frozen here -- an
# implementer cannot guess it. Every key is a field a predicate may name; a
# field outside this set is a submit-time 422. The value is a callable taking
# the flattened profile view and returning the resolved value, or ``None`` when
# the profile never carried it (fail-closed downstream).
def _probe(profile: dict) -> dict:
    p = profile.get("probe")
    return p if isinstance(p, dict) else {}


def _fingerprint(profile: dict) -> dict:
    fp = profile.get("hardware_fingerprint_json")
    if isinstance(fp, str):
        try:
            fp = json.loads(fp)
        except ValueError:
            fp = {}
    return fp if isinstance(fp, dict) else {}


def _vram_mb(profile: dict) -> float | None:
    # Prefer the probed allocation ceiling over the spec-sheet claim, exactly as
    # ``budget.is_eligible``'s ``min_vram_mb`` does.
    alloc = _probe(profile).get("alloc_max_mb")
    raw = alloc if alloc is not None else profile.get("vram_mb")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _vram_gb(profile: dict) -> float | None:
    mb = _vram_mb(profile)
    return None if mb is None else mb / 1024


def _devices(profile: dict) -> list[dict] | None:
    # docs/14 §2 puts the per-device report on the profile precisely so this
    # stays a dict lookup rather than a DB read -- ``check_constraints`` is
    # pure by contract (module docstring). Worker-side reporting is a later
    # step, so ``None`` here is every machine in the fleet today, not an
    # anomaly.
    ds = profile.get("devices")
    return ds if isinstance(ds, list) and ds else None


def _gpu_count(profile: dict) -> int:
    # docs/14 §5: a pre-009 worker carries no ``devices`` list at all, and
    # that must resolve to 1, not to ``None`` -- the fail-closed rule every
    # other field in this table gets would otherwise make every such machine
    # ineligible for a plain ``gpu_count: 1`` constraint, which is every
    # machine in the fleet until a later step ships per-device reporting.
    # ``vram_gb``'s existing "flat fields stay, populated from device 0"
    # guarantee (docs/14 §2) is exactly why "no report" and "one device" are
    # the same fact here.
    ds = _devices(profile)
    return len(ds) if ds else 1


def _total_vram_gb(profile: dict) -> float | None:
    # Same fallback as ``_gpu_count`` and for the same reason: no ``devices``
    # report means one device, so the "total" is the flat figure ``vram_gb``
    # already reads -- not ``None``, which would fail this field closed for
    # every machine that has not been told about ``devices`` yet.
    ds = _devices(profile)
    if not ds:
        return _vram_gb(profile)
    total_mb = 0.0
    for d in ds:
        if not isinstance(d, dict):
            continue
        try:
            total_mb += float(d.get("vram_mb") or 0)
        except (TypeError, ValueError):
            continue
    return total_mb / 1024


_RESOLVERS: dict[str, Any] = {
    "vram_gb": _vram_gb,
    "vram_mb": _vram_mb,
    "gpu_count": _gpu_count,
    "total_vram_gb": _total_vram_gb,
    "gpu_model": lambda p: p.get("device_name"),
    "backend": lambda p: p.get("backend"),
    "compute_capability": lambda p: p.get("compute_capability"),
    "supports": lambda p: p.get("supports"),
    "bench_score": lambda p: _probe(p).get("bench_score"),
    "driver": lambda p: p.get("driver"),
    "torch_ver": lambda p: p.get("torch_ver"),
    # ``os`` capture lives in the identity / sandbox docs against
    # ``hardware_fingerprint_json`` -- absent today, so a job constraining it
    # fails closed on every machine (not a 422: the field is known, the value is
    # missing).
    "os": lambda p: _probe(p).get("os") or _fingerprint(p).get("os"),
    # ``region`` is declared at enrollment, self-reported and advisory -- never a
    # compliance control. No column carries it yet; the walk may thread it into
    # the profile later, and until then it fails closed.
    "region": lambda p: p.get("region") or _fingerprint(p).get("region"),
}

KNOWN_FIELDS = frozenset(_RESOLVERS)


class ConstraintError(ValueError):
    """A malformed ``constraints_json`` -- the endpoint answers 422."""


# --------------------------------------------------------------------------
# Write-time validation
# --------------------------------------------------------------------------


def validate(obj: Any) -> None:
    """Shape-check a constraint object. Raises ``ConstraintError`` on anything
    the claim-time gate could not evaluate.

    Accepts the two mutually exclusive forms and nothing else: a pin
    (``{"machine_ids": [...]}`` and no sibling key) or a predicate whose every
    field is in ``KNOWN_FIELDS`` and whose every operator is in ``_OPS``.
    """
    if not isinstance(obj, dict):
        raise ConstraintError("constraints must be a JSON object")

    if "machine_ids" in obj:
        if set(obj) != {"machine_ids"}:
            raise ConstraintError(
                "machine_ids is the only key allowed in a pin constraint"
            )
        ids = obj["machine_ids"]
        if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
            raise ConstraintError("machine_ids must be a list of strings")
        return

    for field, condition in obj.items():
        if field not in KNOWN_FIELDS:
            raise ConstraintError(f"unknown constraint field: {field!r}")
        if isinstance(condition, dict):
            if not condition:
                raise ConstraintError(f"{field}: empty condition")
            for op, operand in condition.items():
                if op not in _OPS:
                    raise ConstraintError(f"{field}: unknown operator {op!r}")
                if op in _MEMBER_OPS and not isinstance(operand, list):
                    raise ConstraintError(f"{field} {op}: operand must be an array")
                if op in _NUMERIC_OPS and not _is_number(operand):
                    raise ConstraintError(f"{field} {op}: operand must be a number")
        elif isinstance(condition, (str, int, float, bool)):
            # Bare scalar -- means ``==`` (docs/07 §2; docs/05's frozen example
            # read literally).
            pass
        else:
            raise ConstraintError(
                f"{field}: condition must be a scalar or an {{op: operand}} object"
            )


# --------------------------------------------------------------------------
# Claim-time gate
# --------------------------------------------------------------------------


def check_constraints(
    constraints_json: str | None, machine_id: str, profile: dict
) -> tuple[bool, str | None]:
    """Does this machine satisfy the job's targeting? Pure -- no DB, no lock.

    ``(True, None)`` when it does, else ``(False, reason)`` with ``reason`` one
    of the strings docs/07 §3 freezes, so ``worker_eligibility``'s
    ``fleet_summary`` groups like with like after ``_shape`` strips the digits.
    Fails closed on a missing value for *every* operator, ``!=`` / ``not_in``
    included.
    """
    if not constraints_json:
        return True, None
    try:
        obj = json.loads(constraints_json)
    except ValueError:
        # A job that reached the queue with unparseable constraints should place
        # nowhere rather than everywhere.
        return False, "constraints_json is not valid JSON"
    if not isinstance(obj, dict) or not obj:
        return True, None

    if "machine_ids" in obj:
        pinned = obj["machine_ids"] or []
        if machine_id in pinned:
            return True, None
        return False, f"machine not in pin list ({len(pinned)} pinned)"

    for field, condition in obj.items():
        value = _RESOLVERS.get(field, lambda _p: None)(profile)
        conds = condition if isinstance(condition, dict) else {"==": condition}
        for op, operand in conds.items():
            ok, reason = _holds(field, value, op, operand)
            if not ok:
                return False, reason
    return True, None


def _holds(field: str, value: Any, op: str, operand: Any) -> tuple[bool, str | None]:
    if value is None:
        # Fail closed. Same rationale as ``is_eligible`` / ``clearance_permits``:
        # ``os != windows`` must not place on a machine whose OS was never probed.
        return False, f"{field} missing from probe profile"

    if op in _NUMERIC_OPS:
        num = _as_number(value)
        if num is None:
            return False, f"{field} missing from probe profile"
        cmp = {">=": num >= operand, ">": num > operand,
               "<=": num <= operand, "<": num < operand}[op]
        if cmp:
            return True, None
        return False, f"{field} {_fmt(value)} fails {op} {operand}"

    if op == "==":
        if _scalar_eq(value, operand):
            return True, None
        return False, f"{field} {_fmt(value)} fails == {operand}"
    if op == "!=":
        if not _scalar_eq(value, operand):
            return True, None
        return False, f"{field} {_fmt(value)} fails != {operand}"

    if op == "in":
        if value in operand:
            return True, None
        return False, f"{field} {_fmt(value)} not in job allow-list"
    if op == "not_in":
        if value not in operand:
            return True, None
        return False, f"{field} {_fmt(value)} in job deny-list"

    if op == "contains":
        # ``field`` is a list and the operand appears in it -- mirrors the
        # ``supports`` check in ``is_eligible``. docs/07 §3 does not enumerate a
        # phrasing for this case; this one is consistent with the table.
        if isinstance(value, (list, tuple)) and operand in value:
            return True, None
        return False, f"{field} does not contain {operand}"

    # Unreachable: validate() rejects an unknown operator at submit time.
    return False, f"{field}: unknown operator {op!r}"


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _as_number(x: Any) -> float | None:
    if _is_number(x):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return None
    return None


def _scalar_eq(value: Any, operand: Any) -> bool:
    if _is_number(value) and _is_number(operand):
        return float(value) == float(operand)
    return str(value) == str(operand)


def _fmt(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
