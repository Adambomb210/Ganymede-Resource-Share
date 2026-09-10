"""The baseline's summary statistics -- the part M4's exit criterion reads."""

from __future__ import annotations

import json
import statistics

from ganymede.trainer import baseline as B


def test_cli_refuses_an_explicit_device_it_cannot_see(tmp_path, monkeypatch):
    """docs/03 pre-rental item 2, the debug cost this avoids: on the rental box,
    `main("--device cuda")` exits 2 with the check to run -- not hours into a
    2000-step run on CPU "by accident." The refusal is simulated so the test
    passes on this box whether or not this box has a real card: without the
    monkeypatch, --device cuda just works here."""
    monkeypatch.setattr(B.model_mod.torch.cuda, "is_available", lambda: False)
    p = tmp_path / "run.json"
    p.write_text(json.dumps({}))
    ref = B.main(["--run-config", str(p), "--device", "cuda"])
    assert ref == 2


def _seed_result(seed: int, curve: list[float], grid=(0, 100, 200)) -> dict:
    return {
        "seed": seed,
        "curve": [{"step": s, "loss": l} for s, l in zip(grid, curve)],
        "initial_loss": curve[0],
        "final_loss": curve[-1],
    }


def test_summary_reports_mean_and_spread_not_just_a_number():
    results = [
        _seed_result(1, [2.50, 2.00, 1.80]),
        _seed_result(2, [2.50, 2.02, 1.86]),
        _seed_result(3, [2.50, 1.98, 1.84]),
    ]
    s = B.summarize(results)
    finals = [1.80, 1.86, 1.84]
    assert s["seeds"] == 3
    assert s["final_mean"] == round(statistics.fmean(finals), 5)
    assert s["final_stdev"] == round(statistics.stdev(finals), 5)
    assert s["final_min"] == 1.8
    assert s["final_max"] == 1.86


def test_the_pass_band_is_never_tighter_than_the_observed_range():
    """Three seeds that happen to land close are not evidence of low noise.

    With n=3 a small stdev is as likely to be luck as signal, so a band derived
    from it alone can be tighter than the spread the baseline actually produced
    -- which would fail M4 for a distributed result indistinguishable from a
    baseline rerun.
    """
    tight = [_seed_result(i, [2.5, 2.0, v]) for i, v in enumerate([1.800, 1.801, 1.860])]
    s = B.summarize(tight, k=2.0)
    band = s["tolerance"]["pass_if_final_loss_at_most"]
    assert band >= s["final_max"]
    assert band >= s["tolerance"]["upper_from_stdev"]


def test_a_wide_spread_widens_the_band_through_stdev():
    wide = [_seed_result(i, [2.5, 2.0, v]) for i, v in enumerate([1.60, 1.80, 2.00])]
    s = B.summarize(wide, k=2.0)
    assert s["tolerance"]["upper_from_stdev"] > s["final_max"]
    assert s["tolerance"]["pass_if_final_loss_at_most"] == s["tolerance"]["upper_from_stdev"]


def test_single_seed_gives_a_zero_width_band_and_says_so_in_the_numbers():
    s = B.summarize([_seed_result(1, [2.5, 2.0, 1.8])])
    assert s["seeds"] == 1
    assert s["final_stdev"] == 0.0
    # Degenerate but honest: with no spread the only defensible band is the point
    # itself, which is exactly why the CLI warns against running one seed.
    assert s["tolerance"]["pass_if_final_loss_at_most"] == 1.8


def test_curves_are_summarized_point_by_point():
    results = [_seed_result(1, [2.5, 2.0, 1.8]), _seed_result(2, [2.5, 2.1, 1.9])]
    s = B.summarize(results)
    assert [p["step"] for p in s["curve"]] == [0, 100, 200]
    assert s["curve"][1]["mean"] == 2.05
    assert s["curve"][2]["min"] == 1.8 and s["curve"][2]["max"] == 1.9


def test_mismatched_eval_grids_produce_no_curve_rather_than_a_wrong_one():
    """Averaging a step-100 loss with a step-250 loss is not an approximation of
    anything; the summary drops the curve instead of inventing one."""
    results = [_seed_result(1, [2.5, 2.0, 1.8]), _seed_result(2, [2.5, 2.1, 1.9], grid=(0, 250, 500))]
    s = B.summarize(results)
    assert s["curve"] == []
    assert s["final_mean"] > 0  # the final-loss statistics are still valid


def test_the_summary_carries_timing_when_the_seeds_recorded_it():
    """M4b asks whether the distributed run beats one GPU on **wall-clock**, so
    the baseline has to say when it reached each loss, not just that it got
    there. Mean seconds per curve point is what the comparison needs."""
    results = [
        {"seed": i, "final_loss": 1.8, "initial_loss": 2.5,
         "curve": [{"step": st, "loss": 2.0, "train_sec": st * mult}
                   for st in (0, 100, 200)]}
        for i, mult in ((1, 1.0), (2, 2.0))
    ]
    s = B.summarize(results)
    assert [p["train_sec_mean"] for p in s["curve"]] == [0.0, 150.0, 300.0]


def test_a_baseline_written_before_timing_existed_summarizes_without_it():
    """`baseline.json` is a checked-in artifact produced by a GPU run. Reading
    an older one must not raise -- the harness reports that criterion as
    unevaluated, which is not the same as failed."""
    s = B.summarize([_seed_result(1, [2.5, 2.0, 1.8]),
                     _seed_result(2, [2.5, 2.1, 1.81])])
    assert s["curve"], "the curve should still be summarized"
    assert all("train_sec_mean" not in p for p in s["curve"])
