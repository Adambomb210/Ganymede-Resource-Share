"""Tests for ganymede.jobtypes.collab_lora_finetune.aggregate (docs/02-architecture-v2.md section 5)."""

from __future__ import annotations

import io

import pytest
import torch
from safetensors.torch import save as st_save

from ganymede.jobtypes.collab_lora_finetune.aggregate import (
    REJECT_DTYPE_MISMATCH,
    REJECT_KEY_MISMATCH,
    REJECT_NO_STEPS,
    REJECT_NON_FINITE,
    REJECT_NORM_OUTLIER,
    REJECT_NOT_SAFETENSORS,
    REJECT_SHAPE_MISMATCH,
    REJECT_STEPS_INCONSISTENT,
    adapter_divergence,
    check_norms,
    check_structural,
    combine,
    dense_weights,
    load_adapter,
    manifest_of,
)

SHAPES = {
    "a.lora_A": (4, 8),
    "a.lora_B": (8, 4),
    "b.lora_A": (4, 8),
    "b.lora_B": (8, 4),
}


def make_adapter(seed: int, dtype: torch.dtype = torch.float32, scale: float = 1.0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for name, shape in SHAPES.items():
        t = torch.randn(shape, generator=gen) * scale
        out[name] = t.to(dtype)
    return out


def to_bytes(adapter: dict[str, torch.Tensor]) -> bytes:
    return st_save(adapter)


def manifest(adapter: dict[str, torch.Tensor]) -> dict:
    return manifest_of(adapter)


# --- 1. round-trip -----------------------------------------------------------


def test_round_trip_load_and_manifest():
    a = make_adapter(1)
    raw = to_bytes(a)
    loaded = load_adapter(raw)
    assert set(loaded.keys()) == set(a.keys())
    for k in a:
        torch.testing.assert_close(loaded[k], a[k])
    assert manifest_of(loaded) == manifest_of(a)


# --- 2. pickle rejected --------------------------------------------------


def test_pickle_file_rejected_not_safetensors():
    a = make_adapter(2)
    buf = io.BytesIO()
    # Deliberately using torch.save to build a hostile/malformed artifact --
    # this must NOT be executed by check_structural, only rejected.
    torch.save(a, buf)
    raw = buf.getvalue()

    result, out = check_structural(raw, manifest(a), steps_completed=10)
    assert not result.accepted
    assert result.reason == REJECT_NOT_SAFETENSORS
    assert out is None


def test_garbage_bytes_rejected_not_safetensors():
    a = make_adapter(3)
    result, out = check_structural(b"not a safetensors file", manifest(a), steps_completed=10)
    assert not result.accepted
    assert result.reason == REJECT_NOT_SAFETENSORS
    assert out is None


# --- 3. key mismatch -------------------------------------------------------


def test_missing_key_rejected():
    a = make_adapter(4)
    exp = manifest(a)
    incomplete = dict(a)
    del incomplete["b.lora_B"]
    result, out = check_structural(to_bytes(incomplete), exp, steps_completed=10)
    assert result.reason == REJECT_KEY_MISMATCH
    assert "b.lora_B" in result.detail
    assert out is None


def test_extra_key_rejected():
    a = make_adapter(5)
    exp = manifest(a)
    extra = dict(a)
    extra["c.lora_extra"] = torch.randn(2, 2)
    result, out = check_structural(to_bytes(extra), exp, steps_completed=10)
    assert result.reason == REJECT_KEY_MISMATCH
    assert "c.lora_extra" in result.detail


# --- 4. shape mismatch ------------------------------------------------------


def test_wrong_shape_rejected():
    a = make_adapter(6)
    exp = manifest(a)
    bad = dict(a)
    bad["a.lora_A"] = torch.randn(4, 9)  # wrong shape
    result, out = check_structural(to_bytes(bad), exp, steps_completed=10)
    assert result.reason == REJECT_SHAPE_MISMATCH
    assert out is None


# --- 5. dtype mismatch -------------------------------------------------------


def test_wrong_dtype_rejected():
    a = make_adapter(7, dtype=torch.bfloat16)
    exp = manifest(a)
    bad = dict(a)
    bad["a.lora_A"] = bad["a.lora_A"].float()  # fp32 instead of bf16
    result, out = check_structural(to_bytes(bad), exp, steps_completed=10)
    assert result.reason == REJECT_DTYPE_MISMATCH
    assert out is None


# --- 6. non-finite -----------------------------------------------------------


def test_nan_rejected_fp32():
    a = make_adapter(8)
    exp = manifest(a)
    bad = dict(a)
    bad["a.lora_A"] = bad["a.lora_A"].clone()
    bad["a.lora_A"][0, 0] = float("nan")
    result, out = check_structural(to_bytes(bad), exp, steps_completed=10)
    assert result.reason == REJECT_NON_FINITE


def test_inf_rejected_fp32():
    a = make_adapter(9)
    exp = manifest(a)
    bad = dict(a)
    bad["a.lora_A"] = bad["a.lora_A"].clone()
    bad["a.lora_A"][1, 1] = float("inf")
    result, out = check_structural(to_bytes(bad), exp, steps_completed=10)
    assert result.reason == REJECT_NON_FINITE


def test_nan_rejected_bf16():
    a = make_adapter(10, dtype=torch.bfloat16)
    exp = manifest(a)
    bad = dict(a)
    bad["a.lora_A"] = bad["a.lora_A"].clone()
    bad["a.lora_A"][0, 0] = torch.tensor(float("nan"), dtype=torch.bfloat16)
    result, out = check_structural(to_bytes(bad), exp, steps_completed=10)
    assert result.reason == REJECT_NON_FINITE


def test_inf_rejected_bf16():
    a = make_adapter(11, dtype=torch.bfloat16)
    exp = manifest(a)
    bad = dict(a)
    bad["a.lora_A"] = bad["a.lora_A"].clone()
    bad["a.lora_A"][1, 1] = torch.tensor(float("inf"), dtype=torch.bfloat16)
    result, out = check_structural(to_bytes(bad), exp, steps_completed=10)
    assert result.reason == REJECT_NON_FINITE


# --- 7. no steps -------------------------------------------------------------


def test_zero_steps_rejected():
    a = make_adapter(12)
    exp = manifest(a)
    result, out = check_structural(to_bytes(a), exp, steps_completed=0)
    assert result.reason == REJECT_NO_STEPS
    assert out is None


# --- 8. steps inconsistent with heartbeat ------------------------------------


def test_steps_below_heartbeat_rejected():
    a = make_adapter(13)
    exp = manifest(a)
    result, out = check_structural(to_bytes(a), exp, steps_completed=5, heartbeat_steps=20)
    assert result.reason == REJECT_STEPS_INCONSISTENT
    assert "5" in result.detail and "20" in result.detail


def test_steps_implausible_jump_rejected():
    a = make_adapter(14)
    exp = manifest(a)
    # heartbeat=10 -> ceiling is 10*3+100 = 130
    result, out = check_structural(to_bytes(a), exp, steps_completed=131, heartbeat_steps=10)
    assert result.reason == REJECT_STEPS_INCONSISTENT


def test_steps_within_plausible_range_accepted():
    a = make_adapter(15)
    exp = manifest(a)
    result, out = check_structural(to_bytes(a), exp, steps_completed=100, heartbeat_steps=50)
    assert result.accepted
    assert out is not None


# --- 9. gate ordering: structural before numeric -----------------------------


def test_shape_mismatch_takes_priority_over_nan():
    a = make_adapter(16)
    exp = manifest(a)
    bad = dict(a)
    bad["a.lora_A"] = torch.full((4, 9), float("nan"))  # wrong shape AND NaN
    result, out = check_structural(to_bytes(bad), exp, steps_completed=10)
    assert result.reason == REJECT_SHAPE_MISMATCH


# --- 10. check_norms ----------------------------------------------------------


def test_norm_outlier_rejected_among_cohort():
    normal = [make_adapter(100 + i, scale=1.0) for i in range(5)]
    inflated = make_adapter(200, scale=10.0)
    cohort = normal + [inflated]
    results = check_norms(cohort, k=5.0)
    assert len(results) == 6
    assert all(r.accepted for r in results[:5])
    assert not results[5].accepted
    assert results[5].reason == REJECT_NORM_OUTLIER


def test_check_norms_two_adapters_no_rejection():
    cohort = [make_adapter(300, scale=1.0), make_adapter(301, scale=50.0)]
    results = check_norms(cohort, k=5.0)
    assert all(r.accepted for r in results)


def test_check_norms_all_zero_tensor_not_rejected():
    cohort = []
    for i in range(4):
        a = make_adapter(400 + i, scale=1.0)
        a["a.lora_A"] = torch.zeros_like(a["a.lora_A"])  # all-zero across every adapter
        cohort.append(a)
    results = check_norms(cohort, k=5.0)
    assert all(r.accepted for r in results)


# --- 11. dense_weights ---------------------------------------------------------


def test_dense_weights_sum_to_one():
    steps = [10, 20, 30, 40]
    keys = ["t1", "t2"]
    weights = dense_weights(steps, keys)
    total = sum(w["t1"] for w in weights)
    assert total == pytest.approx(1.0, abs=1e-9)
    total2 = sum(w["t2"] for w in weights)
    assert total2 == pytest.approx(1.0, abs=1e-9)


def test_dense_weights_equal_steps_equal_shares():
    steps = [10, 10, 10, 10]
    keys = ["t1"]
    weights = dense_weights(steps, keys)
    for w in weights:
        assert w["t1"] == pytest.approx(0.25, abs=1e-9)


def test_dense_weights_same_scalar_across_tensors():
    steps = [10, 20, 30]
    keys = ["t1", "t2", "t3"]
    weights = dense_weights(steps, keys)
    for w in weights:
        vals = set(w.values())
        assert len(vals) == 1  # identical scalar for every tensor of a given worker


def test_dense_weights_cap_bounds_dominant_worker():
    steps = [1000, 10, 10, 10, 10]  # worker 0 has 100x the others
    keys = ["t1"]
    weights = dense_weights(steps, keys, cap=2.0)
    shares = [w["t1"] for w in weights]
    assert sum(shares) == pytest.approx(1.0, abs=1e-6)
    other_median = sorted(shares[1:])[len(shares[1:]) // 2]
    # dominant worker's final share should be capped, not anywhere near its
    # uncapped ~95% share
    assert shares[0] < 0.9
    assert shares[0] > other_median  # still the largest, just bounded


def test_dense_weights_no_cap_below_three_workers():
    steps = [1000, 10]
    keys = ["t1"]
    uncapped = dense_weights(steps, keys, cap=None)
    capped = dense_weights(steps, keys, cap=2.0)
    # below 3 workers, capping is skipped entirely -- results are identical
    assert uncapped == capped
    assert capped[0]["t1"] == pytest.approx(1000 / 1010, abs=1e-9)


def test_dense_weights_zero_total_raises():
    with pytest.raises(ValueError):
        dense_weights([0, 0, 0], ["t1"])


# --- 12. combine: mean mode equals elementwise mean of adapters --------------


def test_combine_mean_equals_elementwise_mean_of_adapters():
    base = make_adapter(500)
    a1 = make_adapter(501)
    a2 = make_adapter(502)
    keys = list(base.keys())
    weights = [{k: 0.5 for k in keys}, {k: 0.5 for k in keys}]

    result, mom = combine(base, [a1, a2], weights, mode="mean", lr_outer=1.0, beta=0.0)
    assert mom is None

    for k in keys:
        expected = (a1[k].float() + a2[k].float()) / 2
        torch.testing.assert_close(result[k], expected.to(base[k].dtype), atol=1e-6, rtol=1e-5)


# --- 13. combine zero-guard ----------------------------------------------------


def test_combine_zero_weight_tensor_passes_base_through_unchanged():
    base = make_adapter(600)
    a1 = make_adapter(601)
    a2 = make_adapter(602)
    keys = list(base.keys())
    # nobody trained "a.lora_A"
    weights = [
        {k: (0.5 if k != "a.lora_A" else 0.0) for k in keys},
        {k: (0.5 if k != "a.lora_A" else 0.0) for k in keys},
    ]
    result, mom = combine(base, [a1, a2], weights, mode="mean")
    assert torch.equal(result["a.lora_A"], base["a.lora_A"])
    # untouched tensor should be bitwise identical, not merely close
    for k in keys:
        if k != "a.lora_A":
            assert not torch.equal(result[k], base[k])  # sanity: other tensors DID change


def test_combine_zero_weight_tensor_momentum_untouched():
    base = make_adapter(603)
    a1 = make_adapter(604)
    a2 = make_adapter(605)
    keys = list(base.keys())
    weights = [
        {k: (0.5 if k != "a.lora_A" else 0.0) for k in keys},
        {k: (0.5 if k != "a.lora_A" else 0.0) for k in keys},
    ]
    momentum_in = {k: torch.full_like(base[k], 3.0, dtype=torch.float32) for k in keys}
    result, mom_out = combine(base, [a1, a2], weights, mode="diloco", lr_outer=0.5, beta=0.9, momentum=momentum_in)
    assert torch.equal(mom_out["a.lora_A"], momentum_in["a.lora_A"])
    assert torch.equal(result["a.lora_A"], base["a.lora_A"])


# --- 14. combine diloco momentum accumulates -----------------------------------


def test_combine_diloco_momentum_accumulates():
    base = make_adapter(700)
    a1 = make_adapter(701)
    a2 = make_adapter(702)
    keys = list(base.keys())
    weights = [{k: 0.5 for k in keys}, {k: 0.5 for k in keys}]

    result1, mom1 = combine(base, [a1, a2], weights, mode="diloco", lr_outer=0.5, beta=0.9)
    assert mom1 is not None
    for k in keys:
        assert not torch.equal(mom1[k], torch.zeros_like(mom1[k]))

    # Second call re-using the same base/adapters/weights but with the
    # momentum carried forward should move further from base in the same
    # direction as the first step did, because momentum accumulates.
    result2, mom2 = combine(base, [a1, a2], weights, mode="diloco", lr_outer=0.5, beta=0.9, momentum=mom1)

    for k in keys:
        delta1 = (result1[k].float() - base[k].float()).norm()
        delta2 = (result2[k].float() - base[k].float()).norm()
        assert delta2 > delta1


# --- 15. combine preserves keys/shapes/dtypes -----------------------------------


def test_combine_preserves_keys_shapes_dtypes_bf16():
    base = make_adapter(800, dtype=torch.bfloat16)
    a1 = make_adapter(801, dtype=torch.bfloat16)
    a2 = make_adapter(802, dtype=torch.bfloat16)
    keys = list(base.keys())
    weights = [{k: 0.5 for k in keys}, {k: 0.5 for k in keys}]

    result, _ = combine(base, [a1, a2], weights, mode="mean")
    assert set(result.keys()) == set(base.keys())
    for k in keys:
        assert result[k].shape == base[k].shape
        assert result[k].dtype == base[k].dtype


# --- 16. unknown mode raises ------------------------------------------------------


def test_combine_unknown_mode_raises():
    base = make_adapter(900)
    a1 = make_adapter(901)
    keys = list(base.keys())
    weights = [{k: 1.0 for k in keys}]
    with pytest.raises(ValueError):
        combine(base, [a1], weights, mode="bogus")


# --- 17. adapter_divergence -------------------------------------------------------


def test_adapter_divergence_identical_adapters_near_zero():
    a = make_adapter(1000)
    b = {k: v.clone() for k, v in a.items()}
    d = adapter_divergence([a, b])
    assert d == pytest.approx(0.0, abs=1e-5)


def test_adapter_divergence_dissimilar_adapters_clearly_positive():
    a = make_adapter(1001)
    b = make_adapter(1002)
    # Force near-orthogonality isn't guaranteed by different seeds alone in
    # high dimension it's likely, but make it deterministic: negate b.
    b = {k: -v for k, v in a.items()}
    d = adapter_divergence([a, b])
    assert d > 1.5  # negated vectors -> cosine sim near -1 -> distance near 2


def test_adapter_divergence_single_adapter_is_zero():
    a = make_adapter(1003)
    assert adapter_divergence([a]) == 0.0


def test_adapter_divergence_empty_is_zero():
    assert adapter_divergence([]) == 0.0
