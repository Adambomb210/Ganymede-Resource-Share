"""The job constraint predicate grammar (docs/07-scheduler.md §2, Decision 15).

Pure unit tests -- no DB, no client. The grammar is small and total, so it is
cheap to cover exhaustively: every operator, the bare-scalar shorthand, the
all-AND range, the pin, fail-closed on a missing value for *every* operator, and
each submit-time 422 case.
"""

from __future__ import annotations

import json

import pytest

from ganymede.coordinator import constraints
from ganymede.coordinator.constraints import ConstraintError, check_constraints, validate


# A representative flattened profile (the shape app.py hands the gate).
PROFILE = {
    "backend": "cuda",
    "device_name": "NVIDIA A100",
    "vram_mb": 81920,
    "compute_capability": "8.0",
    "supports": ["bf16", "fp16", "nf4"],
    "driver": "550.54",
    "torch_ver": "2.4.0",
    "probe": {"alloc_max_mb": 80000, "bench_score": 42.0},
}


def _j(obj) -> str:
    return json.dumps(obj)


# --------------------------------------------------------------------------
# Form A -- pin
# --------------------------------------------------------------------------


def test_pin_matches_a_listed_machine():
    ok, reason = check_constraints(_j({"machine_ids": ["m1", "m2"]}), "m2", PROFILE)
    assert ok and reason is None


def test_pin_refuses_an_unlisted_machine_with_the_frozen_string():
    ok, reason = check_constraints(_j({"machine_ids": ["m1", "m2"]}), "m9", PROFILE)
    assert not ok
    assert reason == "machine not in pin list (2 pinned)"


def test_empty_pin_list_matches_nothing():
    ok, reason = check_constraints(_j({"machine_ids": []}), "m1", PROFILE)
    assert not ok
    assert reason == "machine not in pin list (0 pinned)"


def test_pin_with_a_sibling_key_is_a_422():
    with pytest.raises(ConstraintError):
        validate({"machine_ids": ["m1"], "vram_gb": {">=": 24}})


def test_pin_must_be_a_list_of_strings():
    with pytest.raises(ConstraintError):
        validate({"machine_ids": "m1"})
    with pytest.raises(ConstraintError):
        validate({"machine_ids": [1, 2]})


# --------------------------------------------------------------------------
# Form B -- predicate: every operator
# --------------------------------------------------------------------------


# The gate prefers the probed ceiling (alloc_max_mb 80000) over the spec-sheet
# vram_mb, exactly as budget.is_eligible does: 80000 / 1024 = 78.125 GiB.
@pytest.mark.parametrize(
    "cond, ok",
    [
        ({">=": 24}, True),
        ({">=": 78}, True),
        ({">=": 79}, False),
        ({">": 78}, True),
        ({">": 79}, False),
        ({"<=": 78.125}, True),
        ({"<": 78}, False),
        ({"<": 128}, True),
    ],
)
def test_numeric_operators_on_vram_gb(cond, ok):
    got, _ = check_constraints(_j({"vram_gb": cond}), "m1", PROFILE)
    assert got is ok


def test_vram_gb_divides_so_a_24576mb_card_clears_24():
    prof = {**PROFILE, "vram_mb": 24576, "probe": {}}
    ok, _ = check_constraints(_j({"vram_gb": {">=": 24}}), "m1", prof)
    assert ok


def test_range_is_all_and():
    inside = check_constraints(_j({"vram_gb": {">=": 24, "<": 100}}), "m1", PROFILE)[0]
    outside = check_constraints(_j({"vram_gb": {">=": 24, "<": 40}}), "m1", PROFILE)[0]
    assert inside and not outside


def test_eq_and_neq():
    assert check_constraints(_j({"backend": {"==": "cuda"}}), "m1", PROFILE)[0]
    assert not check_constraints(_j({"backend": {"==": "mps"}}), "m1", PROFILE)[0]
    assert check_constraints(_j({"backend": {"!=": "mps"}}), "m1", PROFILE)[0]
    assert not check_constraints(_j({"backend": {"!=": "cuda"}}), "m1", PROFILE)[0]


def test_bare_scalar_means_eq():
    assert check_constraints(_j({"backend": "cuda"}), "m1", PROFILE)[0]
    ok, reason = check_constraints(_j({"backend": "mps"}), "m1", PROFILE)
    assert not ok and reason == "backend cuda fails == mps"


def test_in_and_not_in():
    assert check_constraints(
        _j({"gpu_model": {"in": ["NVIDIA A100", "NVIDIA H100"]}}), "m1", PROFILE
    )[0]
    ok, reason = check_constraints(
        _j({"gpu_model": {"in": ["NVIDIA H100"]}}), "m1", PROFILE
    )
    assert not ok and reason == "gpu_model NVIDIA A100 not in job allow-list"

    assert check_constraints(
        _j({"gpu_model": {"not_in": ["NVIDIA H100"]}}), "m1", PROFILE
    )[0]
    ok, reason = check_constraints(
        _j({"gpu_model": {"not_in": ["NVIDIA A100"]}}), "m1", PROFILE
    )
    assert not ok and reason == "gpu_model NVIDIA A100 in job deny-list"


def test_contains_on_a_list_field():
    assert check_constraints(_j({"supports": {"contains": "nf4"}}), "m1", PROFILE)[0]
    ok, reason = check_constraints(_j({"supports": {"contains": "fp8"}}), "m1", PROFILE)
    assert not ok and "does not contain fp8" in reason


def test_multiple_fields_are_all_and():
    both = check_constraints(
        _j({"backend": "cuda", "vram_gb": {">=": 24}}), "m1", PROFILE
    )[0]
    one_fails = check_constraints(
        _j({"backend": "cuda", "vram_gb": {">=": 999}}), "m1", PROFILE
    )[0]
    assert both and not one_fails


def test_ordering_reason_echoes_the_number_for_shape_grouping():
    # docs/07 §3: the digit-stripped shape must be identical across cards so
    # fleet_summary groups them.
    _, r1 = check_constraints(_j({"vram_gb": {">=": 999}}), "m1",
                             {**PROFILE, "vram_mb": 12288, "probe": {}})
    _, r2 = check_constraints(_j({"vram_gb": {">=": 999}}), "m1",
                             {**PROFILE, "vram_mb": 8192, "probe": {}})
    from ganymede.coordinator.eligibility import _shape
    assert _shape(r1) == _shape(r2)


# --------------------------------------------------------------------------
# Fail closed on a missing value -- every operator, != and not_in included
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cond",
    [
        {">=": 1}, {">": 1}, {"<=": 1}, {"<": 1},
        {"==": "linux"}, {"!=": "windows"},
        {"in": ["linux"]}, {"not_in": ["windows"]},
        {"contains": "x"},
        "linux",
    ],
)
def test_missing_value_fails_closed_for_every_operator(cond):
    # `os` has no source in the profile today -- known field, absent value.
    ok, reason = check_constraints(_j({"os": cond}), "m1", PROFILE)
    assert not ok
    assert reason == "os missing from probe profile"


def test_region_also_fails_closed_when_unset():
    ok, reason = check_constraints(_j({"region": "eu-west"}), "m1", PROFILE)
    assert not ok and reason == "region missing from probe profile"


def test_os_resolves_from_the_probe_when_present():
    prof = {**PROFILE, "probe": {**PROFILE["probe"], "os": "linux"}}
    assert check_constraints(_j({"os": "linux"}), "m1", prof)[0]
    assert not check_constraints(_j({"os": {"!=": "linux"}}), "m1", prof)[0]


# --------------------------------------------------------------------------
# Empty / absent constraints are no constraint
# --------------------------------------------------------------------------


@pytest.mark.parametrize("blank", [None, "", "{}"])
def test_blank_constraints_place_anywhere(blank):
    ok, reason = check_constraints(blank, "m1", PROFILE)
    assert ok and reason is None


def test_unparseable_constraints_place_nowhere():
    ok, reason = check_constraints("{not json", "m1", PROFILE)
    assert not ok


# --------------------------------------------------------------------------
# Submit-time 422s
# --------------------------------------------------------------------------


def test_unknown_field_is_422():
    with pytest.raises(ConstraintError):
        validate({"cores": {">=": 8}})


def test_unknown_operator_is_422():
    with pytest.raises(ConstraintError):
        validate({"vram_gb": {"~=": 24}})


def test_numeric_operator_needs_a_number_operand():
    with pytest.raises(ConstraintError):
        validate({"vram_gb": {">=": "lots"}})


def test_member_operator_needs_an_array_operand():
    with pytest.raises(ConstraintError):
        validate({"gpu_model": {"in": "NVIDIA A100"}})


def test_condition_must_be_scalar_or_op_object():
    with pytest.raises(ConstraintError):
        validate({"gpu_model": ["NVIDIA A100"]})


def test_empty_condition_object_is_422():
    with pytest.raises(ConstraintError):
        validate({"vram_gb": {}})


def test_a_valid_predicate_and_a_valid_pin_both_pass_validate():
    validate({"vram_gb": {">=": 24, "<": 80}, "gpu_model": {"in": ["NVIDIA A100"]},
              "os": "linux", "backend": "cuda"})
    validate({"machine_ids": ["a1b2", "c3d4"]})
    validate({})


def test_every_known_field_is_individually_accepted():
    for field in constraints.KNOWN_FIELDS:
        validate({field: {"==": "x"}} if field not in ("vram_gb", "vram_mb", "bench_score")
                 else {field: {">=": 1}})
