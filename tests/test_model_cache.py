"""The process-lifetime model cache (``ganymede/trainer/modelcache.py``).

docs/03 flagged the uncached reload as worth knowing before M4b: 110 s on a 1.7B
model, and a worker takes several tasks per round when its budget is small
relative to the round.

The tempting test here is "``load_base`` was called once", and it is nearly
worthless -- it asserts the speedup and none of the safety, which is the same
shape of blind spot the ``FakeWorker`` had. Reuse is only correct if a task run
against a *reused* stack produces exactly what it would have produced against a
fresh one, so that is what most of this file asserts, byte for byte.
"""

from __future__ import annotations

import pytest
import torch

from ganymede.jobtypes.collab_lora_finetune import aggregate
from ganymede.trainer import model as M
from ganymede.trainer import train as T
from ganymede.trainer.modelcache import ModelCache
from scripts.newrun import build_seed_adapter

CPU = torch.device("cpu")

# Dropout on purpose. It is the live consumer of the global RNG inside
# ``train_loop``, so it is what makes the cached and uncached paths able to
# disagree at all -- with the repo default of 0.0 every test here would pass
# against a cache that quietly shifted the RNG position.
DROPPY_LORA = {"rank": 4, "alpha": 8, "dropout": 0.1,
               "target_modules": ["q_proj", "v_proj"]}


@pytest.fixture
def make_task(tiny_model_dir):
    def _make(**overrides):
        hp = {
            "lr": 1e-3, "seq_len": 32, "micro_batch": 2, "grad_accum": 2,
            "eval_size": 40, "data_seed": 5, "gradient_checkpointing": False,
        }
        hp.update(overrides.pop("hyperparams", {}))
        payload = {
            "task_id": "t1", "run_id": "r1", "round_idx": 0,
            "base_model": tiny_model_dir, "base_precision": "fp32",
            "lora_cfg": dict(DROPPY_LORA), "dataset_ref": "hf://unused",
            "buckets": [0, 1], "num_buckets": 8, "hyperparams": hp,
            "local_steps": 4, "seed": 1234, "max_runtime_sec": 600,
        }
        payload.update(overrides)
        return T.Task.from_payload(payload)
    return _make


@pytest.fixture
def seed_bytes(tiny_model_dir):
    return aggregate.save_adapter(build_seed_adapter(tiny_model_dir, DROPPY_LORA))


@pytest.fixture
def other_adapter(tiny_model_dir, seed_bytes):
    """A *different* adapter to start from, so "task 2 inherited task 1's
    weights" is a failure this file can actually see."""
    adapter = aggregate.load_adapter(seed_bytes)
    return aggregate.save_adapter(
        {k: v + 0.05 for k, v in adapter.items()}
    )


# ---------------------------------------------------------------------------
# The claim that matters
# ---------------------------------------------------------------------------


def test_a_task_on_a_reused_stack_trains_what_it_would_have_trained_fresh(
    make_task, seed_bytes, other_adapter, tiny_rows
):
    """The whole safety argument, asserted byte for byte.

    Task A runs first and leaves the cached stack holding *its* adapter, its
    gradients and its RNG history. Task B then runs on that stack and must
    produce exactly the adapter it would have produced on a model loaded from
    disk a moment earlier.

    B starts from a different adapter than A precisely so that inheriting A's
    weights would change the answer. If ``load_lora_state`` ever stopped being
    strict in both directions, or the reset were moved to a call site and
    forgotten, this is the test that fails.
    """
    task_a = make_task(task_id="a", buckets=[0, 1], seed=11)
    task_b = make_task(task_id="b", buckets=[2, 3], seed=22)

    cache = ModelCache()
    T.run_task(task_a, seed_bytes, rows=tiny_rows, device=CPU, cache=cache)
    cached_b = T.run_task(task_b, other_adapter, rows=tiny_rows, device=CPU, cache=cache)

    fresh_b = T.run_task(task_b, other_adapter, rows=tiny_rows, device=CPU)

    assert cache.stats() == {"hits": 1, "misses": 1, "resident": 1}
    assert cached_b.adapter_bytes == fresh_b.adapter_bytes
    assert cached_b.metrics["steps"] == fresh_b.metrics["steps"]


def test_the_cache_does_not_move_the_rng_position(make_task, seed_bytes, tiny_rows):
    """A regression test for a bug this change would otherwise have introduced.

    ``attach_lora`` draws from the global RNG -- peft initialises LoRA-A, and
    ``init_from`` then overwrites it, but the draw still advances the stream. So
    with the seed set *before* the setup, a cache hit (which skips
    ``attach_lora``) would leave ``train_loop`` starting from a different RNG
    position and training a different adapter. Not a crash and not a gate
    failure: just a worker whose output silently depends on whether it happened
    to be the first task in the process.

    Seeding immediately before ``train_loop`` is what makes the two identical,
    and this asserts it with dropout live.
    """
    task = make_task()
    cache = ModelCache()

    T.run_task(make_task(task_id="warm", seed=7), seed_bytes,
               rows=tiny_rows, device=CPU, cache=cache)
    on_hit = T.run_task(task, seed_bytes, rows=tiny_rows, device=CPU, cache=cache)
    uncached = T.run_task(task, seed_bytes, rows=tiny_rows, device=CPU)

    assert cache.hits == 1
    assert on_hit.adapter_bytes == uncached.adapter_bytes


def test_an_early_stop_does_not_leak_a_gradient_into_the_next_task(
    make_task, seed_bytes, other_adapter, tiny_rows
):
    """The one path where a stale ``.grad`` could reach a live backward.

    A task told to stop leaves whatever gradients its last step produced on the
    very tensors the next task inherits. ``train_loop`` opens each step with
    ``zero_grad(set_to_none=True)``, so this is already covered two modules
    away; the cache zeroes on reset as well rather than depending on that, and
    this pins the behaviour from the outside.
    """
    steps = iter(range(100))
    stop_after_two = lambda: next(steps) >= 2  # noqa: E731

    cache = ModelCache()
    stopped = T.run_task(make_task(task_id="a", local_steps=50), seed_bytes,
                         should_stop=stop_after_two, rows=tiny_rows,
                         device=CPU, cache=cache)
    assert 0 < stopped.metrics["steps"] < 50

    task_b = make_task(task_id="b", seed=99)
    after = T.run_task(task_b, other_adapter, rows=tiny_rows, device=CPU, cache=cache)
    fresh = T.run_task(task_b, other_adapter, rows=tiny_rows, device=CPU)
    assert after.adapter_bytes == fresh.adapter_bytes


def test_setup_sec_says_whether_it_skipped_the_load(make_task, seed_bytes, tiny_rows):
    """``safety_margin_sec`` is meant to be set from observed ``setup_sec``
    (docs/02), and a cache makes that figure bimodal -- the first task in a
    process pays the full load and every task after it pays almost nothing.
    An operator averaging the two would under-size the margin for exactly the
    task that needs it, so the metric says which kind it was."""
    cache = ModelCache()
    cold = T.run_task(make_task(), seed_bytes, rows=tiny_rows, device=CPU, cache=cache)
    warm = T.run_task(make_task(), seed_bytes, rows=tiny_rows, device=CPU, cache=cache)
    none = T.run_task(make_task(), seed_bytes, rows=tiny_rows, device=CPU)

    assert cold.metrics["setup_cached"] is False
    assert warm.metrics["setup_cached"] is True
    assert none.metrics["setup_cached"] is False


# ---------------------------------------------------------------------------
# Keying
# ---------------------------------------------------------------------------


def test_a_different_lora_cfg_reloads_rather_than_reusing(make_task, tiny_model_dir, tiny_rows):
    """Two runs can share a base model and disagree about the adapter shape.

    Reusing the stack across that difference would mean training against the
    wrong structure. It cannot happen quietly in either direction: the config is
    part of the key so it misses, and if it somehow did not, ``load_lora_state``
    is strict and raises.
    """
    wide = {**DROPPY_LORA, "rank": 8}
    cache = ModelCache(capacity=2)

    for cfg in (DROPPY_LORA, wide):
        seed = aggregate.save_adapter(build_seed_adapter(tiny_model_dir, cfg))
        T.run_task(make_task(lora_cfg=dict(cfg)), seed,
                   rows=tiny_rows, device=CPU, cache=cache)

    assert cache.stats() == {"hits": 0, "misses": 2, "resident": 2}


def test_gradient_checkpointing_is_part_of_the_key(make_task, seed_bytes, tiny_rows):
    """It comes from the task's ``hp``, not the run, so two tasks against one
    base can disagree -- and a stack whose hooks were attached for the other
    answer would train under a setting nobody asked for."""
    cache = ModelCache(capacity=2)
    for flag in (False, True):
        T.run_task(make_task(hyperparams={"gradient_checkpointing": flag}),
                   seed_bytes, rows=tiny_rows, device=CPU, cache=cache)
    assert cache.misses == 2


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------


def test_capacity_is_never_exceeded(make_task, tiny_model_dir, tiny_rows):
    """docs/02 §6.2 puts base models at ~16 GB. A cache that held two on a box
    sized for one would turn a saved minute into an OOM, so the default holds
    exactly one and the eviction happens *before* the replacement is built."""
    cache = ModelCache()
    for rank in (4, 8, 16):
        cfg = {**DROPPY_LORA, "rank": rank}
        seed = aggregate.save_adapter(build_seed_adapter(tiny_model_dir, cfg))
        T.run_task(make_task(lora_cfg=dict(cfg)), seed,
                   rows=tiny_rows, device=CPU, cache=cache)
        assert cache.stats()["resident"] == 1

    assert cache.stats() == {"hits": 0, "misses": 3, "resident": 1}


def test_eviction_runs_before_the_replacement_is_built(
    make_task, tiny_model_dir, tiny_rows, monkeypatch
):
    """The ordering is the whole point of the eviction, so it is asserted rather
    than assumed: holding the outgoing model while the incoming one allocates
    needs room for two.

    Observed by looking at what is resident at the moment ``load_base`` is
    entered, which is the only place the two orderings differ.
    """
    cache = ModelCache()
    resident_during_load = []
    real_load = M.load_base

    def watching_load(*a, **kw):
        resident_during_load.append(cache.stats()["resident"])
        return real_load(*a, **kw)

    monkeypatch.setattr(M, "load_base", watching_load)
    for rank in (4, 8):
        cfg = {**DROPPY_LORA, "rank": rank}
        seed = aggregate.save_adapter(build_seed_adapter(tiny_model_dir, cfg))
        T.run_task(make_task(lora_cfg=dict(cfg)), seed,
                   rows=tiny_rows, device=CPU, cache=cache)

    # Nothing resident on the first load, and nothing on the second either --
    # the entry from the first was dropped before the second allocated.
    assert resident_during_load == [0, 0]


# ---------------------------------------------------------------------------
# No cache means no change
# ---------------------------------------------------------------------------


def test_without_a_cache_nothing_is_kept_alive(
    make_task, seed_bytes, tiny_rows, monkeypatch
):
    """``cache=None`` has to be literally today's code path. A one-shot CLI has
    nothing to reuse a model across and should not pay to keep one resident."""
    loads = []
    real_load = M.load_base
    monkeypatch.setattr(
        M, "load_base",
        lambda *a, **kw: (loads.append(a[:2]), real_load(*a, **kw))[1],
    )
    T.run_task(make_task(), seed_bytes, rows=tiny_rows, device=CPU)
    T.run_task(make_task(), seed_bytes, rows=tiny_rows, device=CPU)

    assert len(loads) == 2
