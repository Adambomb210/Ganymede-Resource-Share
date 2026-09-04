"""Calibration's pure parts: the recommendation, merging, and device reporting."""

from __future__ import annotations

import types

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


# --------------------------------------------------------------------------
# The fit ladder under simulated VRAM constraints (docs/03 pre-rental item 3)
# --------------------------------------------------------------------------


class _SimulatedCard(torch.nn.Module):
    """A fake causal-LM whose forward "OOMs" at seq_len above a cap, standing
    in for a real card whose memory a test can't allocate. Records which rungs
    it was asked to run so a test can assert the ladder *stops* rather than
    retrying higher rungs after a failure."""

    def __init__(self, cap: int):
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)
        self.cap = cap
        self.attempted: list[int] = []

    def forward(self, input_ids, attention_mask=None, labels=None):
        seq = int(input_ids.shape[1])
        self.attempted.append(seq)
        if seq > self.cap:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        out = types.SimpleNamespace()
        out.loss = self.lin(torch.zeros(input_ids.shape[0], seq, 8)).sum()
        return out


@pytest.fixture
def fake_cards(monkeypatch):
    """Point probe_fit's model layer at _SimulatedCard instances. Caps are per
    precision -- nf4 fits deeper rungs than bf16 on the same card."""
    cards: list[_SimulatedCard] = []

    def load_base(base_model, precision, device=None):
        cap = {"bf16": 1024, "nf4": 4096}[precision]
        card = _SimulatedCard(cap)
        cards.append(card)
        return card

    monkeypatch.setattr(C.model_mod, "load_base", load_base)
    monkeypatch.setattr(C.model_mod, "load_tokenizer",
                        lambda base_model: types.SimpleNamespace(pad_token_id=0))
    monkeypatch.setattr(C.model_mod, "attach_lora", lambda model, cfg: model)
    monkeypatch.setattr(C.model_mod, "lora_params",
                        lambda model: list(model.parameters()))
    return cards


def test_ladder_stops_at_the_first_rung_that_ooms(fake_cards):
    """A ladder that retried past a failure would waste minutes per round on
    rungs a cap of 1024 has already ruled out -- and on a CUDA card the retry
    could fragment the allocator badly enough to OOM rungs that would have fit."""
    result = C.probe_fit("m", "bf16", {"rank": 8, "alpha": 16,
                                       "target_modules": []},
                         micro_batch=1, ladder=(512, 1024, 2048, 4096),
                         device=torch.device("cpu"))
    assert result["ok"] and result["max_seq_len"] == 1024
    # 512 and 1024 succeeded, 2048 OOMed, 4096 was never attempted.
    assert fake_cards[0].attempted == [512, 1024, 2048]


def test_nf4_fits_deeper_rungs_than_bf16_on_the_same_card(fake_cards):
    """The reason the probe runs per-precision at all: bf16 and nf4 give
    different answers, and the fits map downstream decides which a run needs."""
    bf16 = C.probe_fit("m", "bf16", {"rank": 8, "alpha": 16,
                                     "target_modules": []},
                       micro_batch=1, ladder=(512, 1024, 2048, 4096),
                       device=torch.device("cpu"))
    nf4 = C.probe_fit("m", "nf4", {"rank": 8, "alpha": 16,
                                   "target_modules": []},
                      micro_batch=1, ladder=(512, 1024, 2048, 4096),
                      device=torch.device("cpu"))
    assert bf16["max_seq_len"] == 1024
    assert nf4["max_seq_len"] == 4096


def test_a_rung_that_spills_past_the_card_is_not_a_fit(fake_cards, monkeypatch):
    """Windows WDDM pages CUDA overflow into shared system RAM, so a rung can
    complete while exceeding the card's physical memory -- measured 45x slower
    than the rung below it (13.85s vs 0.31s/step on a 12GB 3060 at seq 2048,
    ~8GB spilled). The probe stops on such a rung as if it had OOMed, which
    is exactly what Linux would have done with the same allocation: the two
    platforms must give the same answer. The fake torch.cuda surface means
    the test itself runs on any box, card or no card."""
    props = types.SimpleNamespace(total_memory=6 * 2**30, major=8, minor=6)
    monkeypatch.setattr(C.torch.cuda, "get_device_properties", lambda d: props)
    monkeypatch.setattr(C.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(C.torch.cuda, "reset_peak_memory_stats", lambda d: None)

    def fake_max(d=None):
        # Per-rung: rung 512 sits inside the fake 6GB card; anything deeper
        # "completes" while allocating 20GB -- the WDDM-spill shape.
        current = fake_cards[0].attempted[-1]
        return 4 * 2**30 if current <= 512 else 20 * 2**30

    monkeypatch.setattr(C.torch.cuda, "max_memory_allocated", fake_max)
    # The fake card lives on CPU; strip the cuda device off tensor creation so
    # the probe never touches a real card and the test runs on any box.
    real_full = torch.full
    monkeypatch.setattr(
        C.torch, "full",
        lambda size, fill, dtype=None, device=None: real_full(size, fill, dtype=dtype),
    )
    result = C.probe_fit("m", "bf16", {"rank": 8, "alpha": 16,
                                       "target_modules": []},
                         micro_batch=1, ladder=(512, 1024, 2048),
                         device=torch.device("cuda"))
    # rung 512 fit inside the (fake) 6GB; rung 1024 allocated 20GB and was
    # refused; rung 2048 was never attempted.
    assert result["ok"] is True and result["max_seq_len"] == 512
    assert "oversubscribed" in result["error"]
    assert fake_cards[0].attempted == [512, 1024]


def test_first_rung_oom_reports_no_fit(fake_cards, monkeypatch):
    """Nothing fits -- the answer is False/None, not an exception, because a
    calibration probe must always afford an answer downstream code can carry."""
    monkeypatch.setattr(
        C.model_mod, "load_base",
        lambda *a, **k: _SimulatedCard(cap=0),
    )
    result = C.probe_fit("m", "bf16", {"rank": 8, "alpha": 16,
                                       "target_modules": []},
                         micro_batch=1, ladder=(512, 1024),
                         device=torch.device("cpu"))
    assert result["ok"] is False
    assert result["max_seq_len"] is None
