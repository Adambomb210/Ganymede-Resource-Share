"""Tests for ganymede.coordinator.budget (docs/02-architecture-v2.md 3.5, 6.8, 6.10)."""

from __future__ import annotations

import pytest

from ganymede.coordinator.budget import (
    CLEARANCE_ORDER,
    Budget,
    bucket_count,
    clearance_permits,
    is_eligible,
    meets_floor,
    plan_budget,
    resolve_throughput,
    step_budget,
    steps_for_buckets,
    usable_seconds,
)


# --------------------------------------------------------------------------- #
# usable_seconds
# --------------------------------------------------------------------------- #


def test_usable_seconds_subtracts_all_overheads():
    assert usable_seconds(
        remaining_sec=900, est_download_sec=60, est_upload_sec=30, safety_margin_sec=60
    ) == (900 - 60 - 30 - 60)


def test_usable_seconds_floors_at_zero():
    assert (
        usable_seconds(
            remaining_sec=100, est_download_sec=60, est_upload_sec=30, safety_margin_sec=60
        )
        == 0
    )
    assert (
        usable_seconds(
            remaining_sec=0, est_download_sec=60, est_upload_sec=30, safety_margin_sec=60
        )
        == 0
    )


# --------------------------------------------------------------------------- #
# resolve_throughput
# --------------------------------------------------------------------------- #


def test_resolve_throughput_prefers_measured():
    assert resolve_throughput(measured=12.0, calibrated=8.0, cold_start=3.0) == (12.0, "measured")


def test_resolve_throughput_measured_zero_falls_through_to_calibrated():
    assert resolve_throughput(measured=0.0, calibrated=8.0, cold_start=3.0) == (8.0, "calibrated")


def test_resolve_throughput_measured_none_falls_through_to_calibrated():
    assert resolve_throughput(measured=None, calibrated=8.0, cold_start=3.0) == (
        8.0,
        "calibrated",
    )


def test_resolve_throughput_calibrated_zero_falls_through_to_cold_start():
    assert resolve_throughput(measured=None, calibrated=0.0, cold_start=3.0) == (
        3.0,
        "cold_start",
    )


def test_resolve_throughput_both_absent_falls_to_cold_start():
    assert resolve_throughput(measured=None, calibrated=None, cold_start=3.0) == (
        3.0,
        "cold_start",
    )


def test_resolve_throughput_nonpositive_cold_start_raises():
    with pytest.raises(ValueError):
        resolve_throughput(measured=None, calibrated=None, cold_start=0.0)
    with pytest.raises(ValueError):
        resolve_throughput(measured=None, calibrated=None, cold_start=-1.0)


# --------------------------------------------------------------------------- #
# step_budget
# --------------------------------------------------------------------------- #


def test_step_budget_basic():
    assert step_budget(throughput_steps_per_min=10.0, usable_sec=600) == 100


def test_step_budget_zero_usable():
    assert step_budget(throughput_steps_per_min=10.0, usable_sec=0) == 0


def test_step_budget_nonpositive_throughput_raises():
    with pytest.raises(ValueError):
        step_budget(throughput_steps_per_min=0.0, usable_sec=600)
    with pytest.raises(ValueError):
        step_budget(throughput_steps_per_min=-5.0, usable_sec=600)


# --------------------------------------------------------------------------- #
# bucket_count
# --------------------------------------------------------------------------- #


def test_bucket_count_scales_roughly_linearly_with_local_steps():
    small = bucket_count(
        local_steps=100, samples_per_step=8, samples_per_bucket=50, total_buckets=1000
    )
    large = bucket_count(
        local_steps=200, samples_per_step=8, samples_per_bucket=50, total_buckets=1000
    )
    assert large == pytest.approx(2 * small, rel=0.1)


def test_bucket_count_never_exceeds_total_buckets():
    n = bucket_count(
        local_steps=1_000_000, samples_per_step=8, samples_per_bucket=1, total_buckets=64
    )
    assert n == 64


def test_bucket_count_never_below_one_for_positive_steps():
    n = bucket_count(local_steps=1, samples_per_step=1, samples_per_bucket=10_000, total_buckets=64)
    assert n == 1


def test_bucket_count_zero_for_zero_steps():
    assert (
        bucket_count(local_steps=0, samples_per_step=8, samples_per_bucket=50, total_buckets=64)
        == 0
    )


def test_bucket_count_guards_division_by_zero():
    with pytest.raises(ValueError):
        bucket_count(local_steps=100, samples_per_step=8, samples_per_bucket=0, total_buckets=64)
    with pytest.raises(ValueError):
        bucket_count(
            local_steps=100,
            samples_per_step=8,
            samples_per_bucket=50,
            total_buckets=64,
            target_passes=0,
        )


# --------------------------------------------------------------------------- #
# The heterogeneity case -- M1 exit criterion
# --------------------------------------------------------------------------- #


def test_heterogeneous_workers_get_proportional_steps_and_buckets():
    """Two workers with 3x different throughput, same remaining_sec: budgets and
    bucket counts should both come out at ~3:1, so the fast worker gets more
    data proportional to its extra steps rather than over-epoching a fixed shard.
    """
    remaining_sec = 1200
    est_download_sec, est_upload_sec, safety_margin_sec = 60, 30, 60
    samples_per_step, samples_per_bucket, total_buckets = 8, 50, 100000

    usable = usable_seconds(remaining_sec, est_download_sec, est_upload_sec, safety_margin_sec)

    slow_throughput = 5.0
    fast_throughput = 15.0  # 3x

    slow_steps = step_budget(slow_throughput, usable)
    fast_steps = step_budget(fast_throughput, usable)

    slow_buckets = bucket_count(slow_steps, samples_per_step, samples_per_bucket, total_buckets)
    fast_buckets = bucket_count(fast_steps, samples_per_step, samples_per_bucket, total_buckets)

    step_ratio = fast_steps / slow_steps
    bucket_ratio = fast_buckets / slow_buckets

    assert step_ratio == pytest.approx(3.0, rel=0.05)
    assert bucket_ratio == pytest.approx(3.0, rel=0.05)


# --------------------------------------------------------------------------- #
# plan_budget
# --------------------------------------------------------------------------- #


def test_plan_budget_returns_none_when_below_min_usable_sec():
    # 3 minutes remaining, 5-minute minimum usable -> worker gets 204.
    result = plan_budget(
        remaining_sec=180,
        measured=None,
        calibrated=None,
        cold_start=3.0,
        samples_per_step=8,
        samples_per_bucket=50,
        total_buckets=64,
        est_download_sec=10,
        est_upload_sec=5,
        safety_margin_sec=5,
        min_usable_sec=300,
    )
    assert result is None


def test_plan_budget_normal_case_measured():
    result = plan_budget(
        remaining_sec=900,
        measured=20.0,
        calibrated=8.0,
        cold_start=3.0,
        samples_per_step=8,
        samples_per_bucket=50,
        total_buckets=64,
        est_download_sec=60,
        est_upload_sec=30,
        safety_margin_sec=60,
        min_usable_sec=300,
    )
    assert isinstance(result, Budget)
    assert result.throughput_source == "measured"
    assert result.throughput_steps_per_min == 20.0
    assert result.usable_sec == 900 - 60 - 30 - 60
    assert result.local_steps == step_budget(20.0, result.usable_sec)
    assert 1 <= result.n_buckets <= 64


def test_plan_budget_normal_case_calibrated():
    result = plan_budget(
        remaining_sec=900,
        measured=None,
        calibrated=8.0,
        cold_start=3.0,
        samples_per_step=8,
        samples_per_bucket=50,
        total_buckets=64,
        est_download_sec=60,
        est_upload_sec=30,
        safety_margin_sec=60,
        min_usable_sec=300,
    )
    assert result is not None
    assert result.throughput_source == "calibrated"
    assert result.throughput_steps_per_min == 8.0


def test_plan_budget_normal_case_cold_start():
    result = plan_budget(
        remaining_sec=900,
        measured=None,
        calibrated=None,
        cold_start=3.0,
        samples_per_step=8,
        samples_per_bucket=50,
        total_buckets=64,
        est_download_sec=60,
        est_upload_sec=30,
        safety_margin_sec=60,
        min_usable_sec=300,
    )
    assert result is not None
    assert result.throughput_source == "cold_start"
    assert result.throughput_steps_per_min == 3.0


def test_plan_budget_returns_none_when_local_steps_zero():
    # Usable time clears the min_usable_sec bar but throughput is so low that
    # the resulting step budget rounds down to zero.
    result = plan_budget(
        remaining_sec=310,
        measured=None,
        calibrated=None,
        cold_start=0.01,
        samples_per_step=8,
        samples_per_bucket=50,
        total_buckets=64,
        est_download_sec=1,
        est_upload_sec=1,
        safety_margin_sec=1,
        min_usable_sec=300,
    )
    assert result is None


# --------------------------------------------------------------------------- #
# meets_floor
# --------------------------------------------------------------------------- #


def test_meets_floor_below_floor_fails():
    # 5% of median against a 10% floor.
    assert meets_floor(candidate_steps=5, median_steps=100, floor_frac=0.10) is False


def test_meets_floor_above_floor_passes():
    assert meets_floor(candidate_steps=50, median_steps=100, floor_frac=0.10) is True


def test_meets_floor_zero_median_always_passes():
    assert meets_floor(candidate_steps=0, median_steps=0, floor_frac=0.10) is True
    assert meets_floor(candidate_steps=1, median_steps=-5, floor_frac=0.10) is True


# --------------------------------------------------------------------------- #
# is_eligible
# --------------------------------------------------------------------------- #


def test_is_eligible_missing_capability_rejected():
    profile = {"backend": "cuda", "supports": ["bf16", "fp16"]}
    requires = {"supports": ["nf4"]}
    ok, reason = is_eligible(profile, requires)
    assert ok is False
    assert "nf4" in reason


def test_is_eligible_sufficient_vram():
    profile = {"vram_mb": 24000}
    requires = {"min_vram_mb": 12000}
    ok, reason = is_eligible(profile, requires)
    assert ok is True
    assert reason is None


def test_is_eligible_insufficient_vram():
    profile = {"vram_mb": 8192}
    requires = {"min_vram_mb": 12000}
    ok, reason = is_eligible(profile, requires)
    assert ok is False
    assert "8192" in reason and "12000" in reason


def test_is_eligible_backend_not_in_allowed_list():
    profile = {"backend": "mps"}
    requires = {"backends": ["cuda", "rocm"]}
    ok, reason = is_eligible(profile, requires)
    assert ok is False
    assert "mps" in reason


def test_is_eligible_empty_requires_accepts_everything():
    profile = {"backend": "mps", "vram_mb": 8, "supports": []}
    ok, reason = is_eligible(profile, {})
    assert ok is True
    assert reason is None


def test_is_eligible_prefers_probe_alloc_max_over_reported_vram():
    requires = {"min_vram_mb": 12000}

    # Reported vram_mb claims enough, but the probe measured less -> rejected.
    profile_overclaims = {
        "vram_mb": 16000,
        "probe": {"alloc_max_mb": 8000},
    }
    ok, reason = is_eligible(profile_overclaims, requires)
    assert ok is False
    assert "8000" in reason

    # Reported vram_mb understates, but the probe measured enough -> accepted.
    profile_underclaims = {
        "vram_mb": 8000,
        "probe": {"alloc_max_mb": 16000},
    }
    ok, reason = is_eligible(profile_underclaims, requires)
    assert ok is True
    assert reason is None


def test_is_eligible_missing_referenced_field_rejected_not_raised():
    profile = {"backend": "cuda"}  # no vram_mb, no probe at all
    requires = {"min_vram_mb": 12000}
    ok, reason = is_eligible(profile, requires)
    assert ok is False
    assert reason is not None


# --------------------------------------------------------------------------- #
# clearance_permits
# --------------------------------------------------------------------------- #


def test_clearance_permits_full_truth_table():
    expected = {
        ("open", "open"): True,
        ("open", "internal"): False,
        ("open", "restricted"): False,
        ("internal", "open"): True,
        ("internal", "internal"): True,
        ("internal", "restricted"): False,
        ("restricted", "open"): True,
        ("restricted", "internal"): True,
        ("restricted", "restricted"): True,
    }
    for (contributor, run_class), want in expected.items():
        assert clearance_permits(contributor, run_class) is want, (contributor, run_class)


def test_clearance_permits_unknown_strings_fail_closed():
    assert clearance_permits("nonsense", "open") is False
    assert clearance_permits("open", "nonsense") is False
    assert clearance_permits("nonsense", "nonsense") is False


def test_clearance_order_matches_spec():
    assert CLEARANCE_ORDER == ("open", "internal", "restricted")


# --------------------------------------------------------------------------
# Setup time is a fixed cost, not a rate
# --------------------------------------------------------------------------


def test_setup_time_is_reserved_like_download_and_upload():
    """Loading the base model is a per-task cost, so it comes off usable time.

    Folding it into throughput instead would make the error depend on round
    length -- small on long rounds, large on short ones, and never visible as an
    error anywhere.
    """
    without = usable_seconds(1800, 60, 30, 60)
    with_setup = usable_seconds(1800, 60, 30, 60, est_setup_sec=120)
    assert without - with_setup == 120


def test_unreserved_setup_overcommits_the_worker():
    """The concrete failure, with the numbers a live M2 round actually produced.

    Setup was 110 s against 46 s of training on a 0.6B model with a warm cache.
    Budget without reserving it and the worker is handed steps it cannot reach:
    it stops at its deadline part-way through and delivers short every round,
    with nothing anywhere reporting a problem.
    """
    throughput = 47.9  # steps/min, as measured
    remaining, setup = 600, 110

    naive = step_budget(throughput, usable_seconds(remaining, 5, 5, 10))
    honest = step_budget(
        throughput, usable_seconds(remaining, 5, 5, 10, est_setup_sec=setup)
    )

    assert naive > honest
    # The worker's real training window is what is left after setup, so the
    # naive budget is unreachable by roughly the setup time's worth of steps.
    reachable = step_budget(throughput, remaining - 5 - 5 - 10 - setup)
    assert honest == reachable
    assert naive - honest == pytest.approx(throughput * setup / 60, abs=1)


def test_plan_budget_threads_setup_through():
    plan = plan_budget(
        remaining_sec=1800, measured=10.0, calibrated=None, cold_start=3.0,
        samples_per_step=8, samples_per_bucket=222, total_buckets=64,
        est_download_sec=60, est_upload_sec=30, safety_margin_sec=60,
        min_usable_sec=300, est_setup_sec=300,
    )
    assert plan is not None
    assert plan.usable_sec == 1800 - 60 - 30 - 60 - 300


def test_setup_can_exhaust_a_short_round_entirely():
    """A worker joining with four minutes left cannot load an 8B model and train.

    Refusing with a 204 is the right answer: the alternative is a lease held for
    a round's remainder that produces nothing.
    """
    plan = plan_budget(
        remaining_sec=240, measured=10.0, calibrated=None, cold_start=3.0,
        samples_per_step=8, samples_per_bucket=222, total_buckets=64,
        est_download_sec=60, est_upload_sec=30, safety_margin_sec=60,
        min_usable_sec=60, est_setup_sec=300,
    )
    assert plan is None


# --------------------------------------------------------------------------
# The budget never asks for more passes than the data supports
# --------------------------------------------------------------------------


@pytest.fixture
def plan_kwargs() -> dict:
    """A plan_budget call with everything but the axis under test pinned.

    Overheads at zero and a long round, so ``usable_sec`` is a round number and
    each test below varies exactly one thing.
    """
    return {
        "remaining_sec": 600,
        "measured": None,
        "calibrated": None,
        "cold_start": 30.0,
        "samples_per_step": 8,
        "samples_per_bucket": 234,
        "total_buckets": 64,
        "est_download_sec": 0,
        "est_upload_sec": 0,
        "safety_margin_sec": 0,
        "min_usable_sec": 60,
    }



def test_a_budget_bigger_than_the_whole_dataset_is_cut_to_the_data(plan_kwargs):
    """Observed under M4a on a small run: a measured throughput produced 14,268
    steps over 352 training samples -- about 81 passes on a run configured for
    one. bucket_count caps the shard at the run's total and cannot see that the
    budget beside it still asks for more."""
    plan = plan_budget(**{
        **plan_kwargs,
        "measured": 6000.0,          # a very fast worker on a very small run
        "samples_per_step": 2,
        "samples_per_bucket": 11,
        "total_buckets": 32,
        "target_passes": 1.0,
    })

    assert plan is not None
    assert plan.n_buckets == 32                       # everything there is
    assert plan.local_steps == 32 * 11 // 2           # one pass over it, not eighty
    assert plan.data_limited is True


def test_a_budget_the_data_can_support_is_left_alone(plan_kwargs):
    """The clamp must be a no-op in the ordinary case, or every budget in every
    healthy run gets quietly rounded down."""
    plan = plan_budget(**{
        **plan_kwargs,
        "measured": 30.0,
        "samples_per_step": 8,
        "samples_per_bucket": 234,
        "total_buckets": 64,
    })

    assert plan is not None
    assert plan.n_buckets < 64
    assert plan.data_limited is False
    assert plan.local_steps == step_budget(30.0, plan.usable_sec)


def test_target_passes_above_one_raises_the_ceiling_it_clamps_to(plan_kwargs):
    """A run that deliberately asks for three passes should get three, not one:
    the clamp enforces the configured number, it does not impose its own."""
    common = {
        **plan_kwargs, "measured": 6000.0, "samples_per_step": 2,
        "samples_per_bucket": 11, "total_buckets": 32,
    }
    one = plan_budget(**{**common, "target_passes": 1.0})
    three = plan_budget(**{**common, "target_passes": 3.0})

    assert three.local_steps == 3 * one.local_steps


def test_a_dataset_too_small_for_a_single_step_is_a_204_not_a_zero_step_lease(
    plan_kwargs
):
    """Leasing a shard for zero work marks it spoken for and produces nothing."""
    plan = plan_budget(**{
        **plan_kwargs,
        "measured": 600.0,
        "samples_per_step": 64,
        "samples_per_bucket": 1,
        "total_buckets": 8,
    })

    assert plan is None


def test_steps_for_buckets_inverts_bucket_count(plan_kwargs):
    """The round trip has to be lossless in the direction that matters: convert
    a budget to buckets and back, and you must not get less than you started
    with, or the clamp would bite in the healthy case."""
    for steps in (1, 7, 33, 250, 1000):
        n = bucket_count(steps, samples_per_step=4, samples_per_bucket=97, total_buckets=1024)
        assert steps_for_buckets(n, 97, 4) >= steps
