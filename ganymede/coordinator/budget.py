"""Per-worker step budgeting and eligibility (docs/02-architecture-v2.md 3.5, 6.8, 6.10).

The core idea (3.5): **fix the round deadline, vary the work per worker.** Uniform
step counts mean the slowest card sets everyone's deadline; a 3090 paired with a
4090 straggles every round and either blocks the close or gets truncated, wasting a
fraction of its work forever. Instead each worker gets a step budget sized so it
finishes near the same wall-clock deadline as everyone else, and aggregation later
weights each worker by the steps it actually completed (5.2).

This module is pure arithmetic and policy -- no I/O, no torch, no numpy -- which is
what makes it cheap to test exhaustively. It is deliberately ignorant of SQLite,
HTTP, and the task/round lifecycle in db.py; callers translate its return values
(notably ``plan_budget``'s ``None``) into coordinator behaviour like a 204 response.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Budget:
    local_steps: int
    n_buckets: int
    throughput_steps_per_min: float
    throughput_source: str  # "measured" | "calibrated" | "cold_start"
    usable_sec: int
    # True when the run's whole dataset was smaller than the worker's time-based
    # budget, so the budget was cut down to what the data supports. Not an error
    # -- it is the honest answer -- but it means this machine could have done
    # more work than the run has data for, which is a fact about the run's
    # shape that an operator wants to know. See plan_budget.
    data_limited: bool = False


def usable_seconds(
    remaining_sec: int,
    est_download_sec: int,
    est_upload_sec: int,
    safety_margin_sec: int,
    est_setup_sec: int = 0,
) -> int:
    """Seconds of actual training time available to a worker claiming right now.

    The input is *time remaining in the round*, not the round's full duration --
    this is the whole point (3.2, "claim against time remaining, not a full
    round"). A worker joining twelve minutes before the deadline gets twelve
    minutes of work to budget from, not a full round's worth; sizing against the
    full round would hand it work it cannot finish before the round closes
    underneath it, wasting the effort entirely.

    ``est_setup_sec`` covers loading the base model and attaching the adapter,
    and it belongs here for the same reason download and upload do: it is a
    fixed cost per task, not a rate, so folding it into throughput would make
    the error depend on round length. It is not a rounding error. A live M2
    round measured 110 s of setup against 46 s of training on a 0.6B model with
    a warm cache -- the setup was 2.4x the work. An 8B model on a cold cache is
    minutes. Budget without reserving it and every worker is handed more steps
    than it can reach, stops at its deadline part-way through, and delivers
    short every round with nothing reporting a problem.
    """
    usable = (remaining_sec - est_download_sec - est_upload_sec
              - safety_margin_sec - est_setup_sec)
    return max(0, usable)


def resolve_throughput(
    measured: float | None,
    calibrated: float | None,
    cold_start: float,
) -> tuple[float, str]:
    """Three-tier throughput ladder (3.5), in strict order of preference.

    1. measured  -- observed steps/min for this (run, gpu_model), from submitted
       metrics on prior rounds.
    2. calibrated -- calibration.json for the run, keyed by GPU class (M0).
    3. cold_start -- conservative default for an unseen GPU class on an
       uncalibrated run; deliberately low, corrected after one round.
    """
    if measured is not None and measured > 0:
        return measured, "measured"
    if calibrated is not None and calibrated > 0:
        return calibrated, "calibrated"
    if cold_start <= 0:
        # Not a runtime condition -- a non-positive cold-start constant is a
        # deployment misconfiguration (see COLD_START_STEPS_PER_MIN in config.py),
        # and guessing a budget from it would silently produce empty rounds.
        raise ValueError(f"cold_start throughput must be positive, got {cold_start}")
    return cold_start, "cold_start"


def step_budget(throughput_steps_per_min: float, usable_sec: int) -> int:
    """local_steps_i = throughput_i * usable_minutes, floored to an integer step count."""
    if throughput_steps_per_min <= 0:
        raise ValueError(
            f"throughput_steps_per_min must be positive, got {throughput_steps_per_min}"
        )
    steps = math.floor(throughput_steps_per_min * usable_sec / 60)
    return max(0, steps)


def bucket_count(
    local_steps: int,
    samples_per_step: int,
    samples_per_bucket: int,
    total_buckets: int,
    target_passes: float = 1.0,
) -> int:
    """How many dataset buckets a worker with this step budget should receive.

    If every worker got the same number of buckets but a different step budget,
    the fast worker would simply do more epochs over the same small shard --
    overfitting it. That's worse than useless: it earns a large aggregation
    weight (5.2 weights by steps completed) for a *narrower* view of the data
    than the slow worker beside it. So bucket count scales with the budget: a
    worker gets roughly enough data that its step budget amounts to
    ``target_passes`` passes over it, converting hardware heterogeneity into
    data coverage rather than weight concentration on a repeated shard.
    """
    if samples_per_bucket <= 0:
        raise ValueError(f"samples_per_bucket must be positive, got {samples_per_bucket}")
    if target_passes <= 0:
        raise ValueError(f"target_passes must be positive, got {target_passes}")
    if local_steps <= 0:
        # No work, no data -- there is nothing for this worker to be assigned.
        return 0
    samples_needed = local_steps * samples_per_step / target_passes
    n = math.ceil(samples_needed / samples_per_bucket)
    return max(1, min(n, total_buckets))


def steps_for_buckets(
    n_buckets: int,
    samples_per_bucket: int,
    samples_per_step: int,
    target_passes: float = 1.0,
) -> int:
    """How many steps ``n_buckets`` of data justify -- the inverse of bucket_count.

    Needed because ``bucket_count`` clamps at the run's total: a worker that
    wants more data than the run *has* is given everything, and the step budget
    that asked for more than everything then no longer matches the shard it got.
    """
    if samples_per_step <= 0:
        raise ValueError(f"samples_per_step must be positive, got {samples_per_step}")
    return int(n_buckets * samples_per_bucket * target_passes // samples_per_step)


def meets_floor(candidate_steps: int, median_steps: float, floor_frac: float) -> bool:
    """Minimum viable throughput gate (3.5).

    Below some speed a worker costs more than it contributes: it consumes a
    lease slot and a full ~25 MB round trip to add noise-level weight. A worker
    that cannot reach ``floor_frac`` of the cohort's median budget is not
    offered work for this run (it stays eligible for others).

    With no cohort yet (median_steps <= 0, e.g. the very first worker to claim
    against a run), there is nothing to compare against -- always pass. Never
    exclude the machine that would otherwise define the median.
    """
    if median_steps <= 0:
        return True
    return candidate_steps >= floor_frac * median_steps


def plan_budget(
    remaining_sec: int,
    measured: float | None,
    calibrated: float | None,
    cold_start: float,
    samples_per_step: int,
    samples_per_bucket: int,
    total_buckets: int,
    est_download_sec: int,
    est_upload_sec: int,
    safety_margin_sec: int,
    min_usable_sec: int,
    target_passes: float = 1.0,
    est_setup_sec: int = 0,
) -> Budget | None:
    """Compose the pieces above into the budget offered to a claiming worker.

    Returns ``None`` when no budget should be offered at all; the caller (the
    coordinator's claim endpoint) turns that into ``204 No Content`` plus a
    ``Retry-After`` pointing past the round boundary (3.2).
    """
    usable = usable_seconds(remaining_sec, est_download_sec, est_upload_sec,
                            safety_margin_sec, est_setup_sec)

    # Deliberate refusal, not a failure: handing a worker three minutes of a
    # fifteen-minute task wastes its bandwidth on a download/upload round trip
    # that produces nothing usable. Better to send it away with a Retry-After
    # than to let it churn for nothing.
    if usable < min_usable_sec:
        return None

    throughput, source = resolve_throughput(measured, calibrated, cold_start)
    local_steps = step_budget(throughput, usable)
    if local_steps <= 0:
        return None

    n_buckets = bucket_count(
        local_steps, samples_per_step, samples_per_bucket, total_buckets, target_passes
    )

    # Close the loop that bucket_count's clamp opens. bucket_count sizes the
    # shard to the budget and then caps it at the run's total, so a worker fast
    # enough to want more data than the run has is handed every bucket -- and
    # the budget that asked for more than everything is left untouched beside
    # it. The two then disagree, and the worker quietly does many more passes
    # over its shard than `target_passes` asked for.
    #
    # Observed under M4a, and not marginally: on a small dataset a measured
    # throughput produced a budget of 14,268 steps over 352 training samples --
    # about 81 passes on a run configured for one. That is precisely the
    # overfitting bucket_count exists to prevent, arriving through the one path
    # bucket_count cannot see. Worse, every worker in the round is handed the
    # same whole dataset, so the shards stop being shards and aggregation
    # weights three copies of one piece of work as three contributions.
    #
    # Clamping here is a no-op whenever the cap did not bite: n_buckets was
    # derived from local_steps, so converting it back always returns at least
    # local_steps. It only does something in the case it is there for.
    supported = steps_for_buckets(n_buckets, samples_per_bucket, samples_per_step,
                                  target_passes)
    data_limited = supported < local_steps
    if data_limited:
        local_steps = supported
        # A run whose entire dataset is smaller than one optimizer step has
        # nothing to hand out; better to say so with a 204 than to lease a
        # shard for zero work.
        if local_steps <= 0:
            return None

    return Budget(
        local_steps=local_steps,
        n_buckets=n_buckets,
        throughput_steps_per_min=throughput,
        throughput_source=source,
        usable_sec=usable,
        data_limited=data_limited,
    )


def is_eligible(profile: dict, requires: dict) -> tuple[bool, str | None]:
    """Capability filtering (6.8, 6.9): does this worker's self-reported profile
    satisfy the run's requirement block? All ``requires`` keys are optional --
    an absent key means no constraint on that axis.

    Returns ``(True, None)`` if eligible, else ``(False, reason)`` naming the
    first unmet requirement. A profile missing a field that a requirement
    references is treated as not meeting it, not as a crash -- registration
    always succeeds (6.9) even for a sparse or degraded profile, and ineligibility
    is the correct, quiet outcome rather than an exception bubbling into the
    claim path.
    """
    supports = requires.get("supports")
    if supports:
        have = profile.get("supports") or []
        for cap in supports:
            if cap not in have:
                return False, f"missing capability: {cap}"

    backends = requires.get("backends")
    if backends:
        backend = profile.get("backend")
        if backend not in backends:
            return False, f"backend {backend} not in {backends}"

    min_vram_mb = requires.get("min_vram_mb")
    if min_vram_mb is not None:
        # Prefer the probed allocation ceiling over the self-reported vram_mb: a
        # reported figure is a claim (spec-sheet or OS-reported), a probed
        # allocation ceiling is a measurement of what the machine actually
        # reserved in a self-test (6.9) -- unified memory shared with the OS, or
        # a desktop card already driving displays, both make the claim optimistic.
        probe = profile.get("probe") or {}
        alloc_max_mb = probe.get("alloc_max_mb")
        effective_vram = alloc_max_mb if alloc_max_mb is not None else profile.get("vram_mb")
        if effective_vram is None or effective_vram < min_vram_mb:
            return False, f"vram_mb {effective_vram} < {min_vram_mb}"

    min_alloc_max_mb = requires.get("min_alloc_max_mb")
    if min_alloc_max_mb is not None:
        probe = profile.get("probe") or {}
        alloc_max_mb = probe.get("alloc_max_mb")
        if alloc_max_mb is None or alloc_max_mb < min_alloc_max_mb:
            return False, f"alloc_max_mb {alloc_max_mb} < {min_alloc_max_mb}"

    min_bench_score = requires.get("min_bench_score")
    if min_bench_score is not None:
        probe = profile.get("probe") or {}
        bench_score = probe.get("bench_score")
        if bench_score is None or bench_score < min_bench_score:
            return False, f"bench_score {bench_score} < {min_bench_score}"

    return True, None


CLEARANCE_ORDER = ("open", "internal", "restricted")


def clearance_permits(contributor_clearance: str, run_classification: str) -> bool:
    """Second eligibility axis: data sensitivity (6.10).

    ``eligible = capability_match AND contributor.clearance >= run.data_classification``,
    where ">=" means "at least as trusted" over CLEARANCE_ORDER's increasing scale.

    An unrecognized string on either side fails closed (returns False) rather
    than raising or defaulting to permissive -- every eligible contributor
    receives the dataset in plaintext, including personal laptops nobody
    administers, so a typo'd or unknown classification must never be treated as
    satisfied. Silently excluding a contributor is the safe failure mode here;
    silently admitting one to data it shouldn't see is not.
    """
    try:
        contributor_idx = CLEARANCE_ORDER.index(contributor_clearance)
        run_idx = CLEARANCE_ORDER.index(run_classification)
    except ValueError:
        return False
    return contributor_idx >= run_idx
