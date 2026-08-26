"""Calibration's pure parts: the recommendation, merging, and device reporting."""

from __future__ import annotations

import pytest
import torch

from ganymede.coordinator import budget as budget_mod
from ganymede.trainer import calibrate as C


def test_recommendation_is_the_coordinators_own_arithmetic():
    """Not a reimplementation of it.

    A recommendation that disagreed with what the coordinator actually does would
    be worse than no recommendation: an operator would size a run against a number
    the running system never uses.
    """
    rec = C.recommend_local_steps(
        6.0, target_round_sec=2100, est_download_sec=120,
        est_upload_sec=120, safety_margin_sec=120,
    )
    usable = budget_mod.usable_seconds(2100, 120, 120, 120)
    assert rec["usable_sec"] == usable
    assert rec["local_steps"] == budget_mod.step_budget(6.0, usable)


def test_faster_cards_are_recommended_more_steps():
    slow = C.recommend_local_steps(3.0, target_round_sec=2100)
    fast = C.recommend_local_steps(9.0, target_round_sec=2100)
    assert fast["local_steps"] > slow["local_steps"]


def test_device_description_carries_the_join_key():
    """``name`` is what rounds.claim_task looks throughput up by."""
    info = C.describe_device(torch.device("cpu"))
    assert info["type"] == "cpu"
    assert info["name"].startswith("cpu:")
    assert info["vram_gb"] is None
    assert info["torch"] == torch.__version__


def _calibration(name: str, spm: float, *, base_model="m", dataset="hf://d") -> dict:
    return {
        "run": {"base_model": base_model, "base_precision": "bf16", "dataset_ref": dataset},
        "device": {"name": name},
        "fits": {name: {"bf16": {"ok": True, "max_seq_len": 2048}}},
        "throughput": {name: spm},
        "throughput_detail": {name: {"steps_per_min": spm}},
    }


def test_merging_two_cards_keeps_both():
    merged = C.merge_calibration(_calibration("RTX 3060", 2.5), _calibration("RTX 4090", 7.1))
    assert merged["throughput"] == {"RTX 3060": 2.5, "RTX 4090": 7.1}
    assert set(merged["fits"]) == {"RTX 3060", "RTX 4090"}
    assert set(merged["throughput_detail"]) == {"RTX 3060", "RTX 4090"}


def test_recalibrating_the_same_card_overwrites_it():
    merged = C.merge_calibration(_calibration("RTX 3060", 2.5), _calibration("RTX 3060", 3.1))
    assert merged["throughput"] == {"RTX 3060": 3.1}


@pytest.mark.parametrize("field,value", [("base_model", "other"), ("dataset_ref", "hf://other")])
def test_merging_across_different_runs_is_refused(field, value):
    """Silently merging them would produce a file that sizes rounds for a run
    that was never measured -- and nothing downstream could detect it."""
    incoming = _calibration("RTX 4090", 7.1)
    incoming["run"][field] = value
    with pytest.raises(ValueError, match=field):
        C.merge_calibration(_calibration("RTX 3060", 2.5), incoming)


def test_oom_detection_covers_both_shapes_torch_raises():
    assert C._is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert not C._is_oom(RuntimeError("shape mismatch"))
    assert not C._is_oom(ValueError("out of memory"))  # wrong type: not an OOM
