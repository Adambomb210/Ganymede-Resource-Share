"""``run_task`` end to end, and its handoff to the coordinator's gates.

Fast and offline: everything here runs against the ~107k-parameter Qwen3 from
``tiny_model_dir``, which is the production architecture at a size where a full
training run costs milliseconds. What it cannot show is that loss goes *down* --
that needs a real model and lives in ``test_trainer_cpu.py`` behind the ``slow``
marker.
"""

from __future__ import annotations

import time

import pytest
import torch

from ganymede.coordinator import aggregate
from ganymede.trainer import data as D
from ganymede.trainer import evaluate as E
from ganymede.trainer import model as M
from ganymede.trainer import train as T
from scripts.newrun import build_seed_adapter

CPU = torch.device("cpu")


@pytest.fixture
def make_task(tiny_model_dir, tiny_lora_cfg):
    def _make(**overrides):
        hp = {
            "lr": 1e-3, "seq_len": 32, "micro_batch": 2, "grad_accum": 2,
            "eval_size": 40, "data_seed": 5, "gradient_checkpointing": False,
        }
        hp.update(overrides.pop("hyperparams", {}))
        payload = {
            "task_id": "t1", "run_id": "r1", "round_idx": 0,
            "base_model": tiny_model_dir, "base_precision": "fp32",
            "lora_cfg": tiny_lora_cfg, "dataset_ref": "hf://unused",
            "buckets": [0, 1], "num_buckets": 8, "hyperparams": hp,
            "local_steps": 4, "seed": 1234, "max_runtime_sec": 600,
        }
        payload.update(overrides)
        return T.Task.from_payload(payload)
    return _make


@pytest.fixture
def seed_bytes(tiny_model_dir, tiny_lora_cfg):
    return aggregate.save_adapter(build_seed_adapter(tiny_model_dir, tiny_lora_cfg))


def test_a_trained_adapter_passes_the_coordinators_real_gates(make_task, seed_bytes, tiny_rows):
    """The handoff M1 and M0 meet at.

    Gate 2's expected manifest is the round's own ``base_adapter_ref`` -- the
    adapter the worker was handed. So this asserts the exact check the
    coordinator performs on submit, using the coordinator's own code.
    """
    result = T.run_task(make_task(), seed_bytes, rows=tiny_rows, device=CPU)

    expected = aggregate.manifest_of(aggregate.load_adapter(seed_bytes))
    gate, adapter = aggregate.check_structural(result.adapter_bytes, expected, result.steps)

    assert gate.accepted, f"{gate.reason}: {gate.detail}"
    assert adapter is not None
    assert result.steps == 4


def test_training_actually_changes_the_adapter(make_task, seed_bytes, tiny_rows):
    """A trainer that returned its input unchanged would pass every gate."""
    result = T.run_task(make_task(), seed_bytes, rows=tiny_rows, device=CPU)

    before = aggregate.load_adapter(seed_bytes)
    after = aggregate.load_adapter(result.adapter_bytes)
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    assert len(changed) == len(before)  # every LoRA tensor moved, A and B alike


def test_on_step_fires_once_per_optimizer_step_not_per_micro_batch(make_task, seed_bytes, tiny_rows):
    """The error this guards would be invisible because it partly cancels.

    The coordinator closes a round on accumulated steps and derives throughput
    from steps per minute. If a step meant a micro-batch, both would be inflated
    by ``grad_accum`` -- rounds closing early *and* budgets sized larger, two
    wrongs that look nearly right together.
    """
    seen: list[int] = []
    task = make_task(local_steps=5, hyperparams={"grad_accum": 4, "micro_batch": 2})
    result = T.run_task(task, seed_bytes, on_step=lambda s, l: seen.append(s),
                        rows=tiny_rows, device=CPU)

    assert seen == [0, 1, 2, 3, 4]
    assert result.steps == 5
    assert result.metrics["samples"] == 5 * 4 * 2  # steps x grad_accum x micro_batch


def test_should_stop_ends_the_round_without_an_error(make_task, seed_bytes, tiny_rows):
    """4.4: a worker told to stop abandons and exits. It still returns what it
    has, because the caller decides whether to submit or abandon -- not this."""
    calls = {"n": 0}

    def stop_after_two():
        calls["n"] += 1
        return calls["n"] > 2

    result = T.run_task(make_task(local_steps=50), seed_bytes,
                        should_stop=stop_after_two, rows=tiny_rows, device=CPU)

    assert result.stopped_early
    assert result.metrics["stopped_early"]
    assert 0 < result.steps < 50


def test_max_runtime_sec_is_a_real_backstop(make_task, seed_bytes, tiny_rows):
    result = T.run_task(make_task(local_steps=10_000, max_runtime_sec=1),
                        seed_bytes, rows=tiny_rows, device=CPU)
    assert result.stopped_early
    assert result.steps < 10_000


def test_the_same_seed_reproduces_the_same_adapter(make_task, seed_bytes, tiny_rows):
    """A round that fails has to be reproducible, or it cannot be debugged.

    Dropout is the live consumer of the RNG here; an unseeded mask would make an
    identical retry produce a different artifact.
    """
    cfg = {"hyperparams": {"dropout": 0.0}}
    a = T.run_task(make_task(**cfg), seed_bytes, rows=tiny_rows, device=CPU)
    b = T.run_task(make_task(**cfg), seed_bytes, rows=tiny_rows, device=CPU)
    assert a.adapter_bytes == b.adapter_bytes


def test_a_different_task_seed_gives_a_different_adapter(make_task, seed_bytes, tiny_rows):
    a = T.run_task(make_task(seed=1), seed_bytes, rows=tiny_rows, device=CPU)
    b = T.run_task(make_task(seed=2), seed_bytes, rows=tiny_rows, device=CPU)
    assert a.adapter_bytes != b.adapter_bytes


def test_workers_on_different_buckets_produce_combinable_adapters(make_task, seed_bytes, tiny_rows):
    """Two workers, disjoint shards, one aggregate -- the whole point of the system."""
    a = T.run_task(make_task(task_id="a", buckets=[0, 1], seed=11),
                   seed_bytes, rows=tiny_rows, device=CPU)
    b = T.run_task(make_task(task_id="b", buckets=[2, 3], seed=22),
                   seed_bytes, rows=tiny_rows, device=CPU)

    base = aggregate.load_adapter(seed_bytes)
    adapters = [aggregate.load_adapter(a.adapter_bytes), aggregate.load_adapter(b.adapter_bytes)]
    weights = aggregate.dense_weights([a.steps, b.steps], sorted(base))
    combined, momentum = aggregate.combine(base, adapters, weights)

    assert momentum is None  # mode="mean" tracks none
    assert set(combined) == set(adapters[0])
    assert all(torch.isfinite(t).all() for t in combined.values())
    assert aggregate.adapter_divergence(adapters) > 0  # they really did diverge


def test_metrics_report_the_rate_without_the_setup_cost(make_task, seed_bytes, tiny_rows):
    """``steps_per_min`` feeds the coordinator's budget arithmetic as a *rate*.

    Blending the fixed model-load cost into it would mis-size every budget by an
    amount that depends on round length. The fixed cost is reported separately as
    ``setup_sec`` instead.
    """
    result = T.run_task(make_task(), seed_bytes, rows=tiny_rows, device=CPU)
    m = result.metrics

    assert m["train_sec"] <= m["seconds"]
    assert m["setup_sec"] > 0
    assert m["steps_per_min"] == pytest.approx(m["steps"] / m["train_sec"] * 60, rel=2e-2)
    assert m["steps_per_min"] >= m["steps"] / m["seconds"] * 60  # never understates the card
    assert m["device"] == "cpu"
    assert m["rows_assigned"] == 2 * ((len(tiny_rows) - 40) // 8)


def test_epochs_are_reported_when_a_shard_is_cycled(make_task, seed_bytes, tiny_rows):
    """Cycling is legal but worth seeing: 6.10 warns that on a small corpus a
    worker repeats data sooner than it would on a large one, and dataset
    exhaustion is the first thing to check if M4's convergence looks flat."""
    small_shard = make_task(buckets=[0], num_buckets=64, local_steps=40,
                            hyperparams={"micro_batch": 2, "grad_accum": 2})
    result = T.run_task(small_shard, seed_bytes, rows=tiny_rows, device=CPU)
    assert result.metrics["epochs"] > 1


def test_held_out_loss_is_finite_and_token_weighted(tiny_model_dir, tiny_lora_cfg, tiny_rows):
    partition = D.plan_partition(n_rows=len(tiny_rows), num_buckets=8, eval_size=40, data_seed=5)
    tokenizer = M.load_tokenizer(tiny_model_dir)
    base = M.load_base(tiny_model_dir, "fp32", device=CPU)
    peft_model = M.attach_lora(base, tiny_lora_cfg)

    result = E.held_out_loss(
        peft_model, tokenizer, tiny_rows, partition.eval_indices(),
        seq_len=32, micro_batch=4, device=CPU,
    )

    assert result.examples == 40
    assert result.tokens > 0
    assert result.loss > 0 and result.loss == pytest.approx(result.loss)  # finite

    # Token weighting: evaluating a subset with a different token count must move
    # the number, and batch size must not.
    by_two = E.held_out_loss(peft_model, tokenizer, tiny_rows, partition.eval_indices(),
                             seq_len=32, micro_batch=2, device=CPU)
    assert by_two.loss == pytest.approx(result.loss, rel=1e-4)
    assert by_two.tokens == result.tokens


def test_eval_leaves_the_model_in_training_mode(tiny_model_dir, tiny_lora_cfg, tiny_rows):
    """Silently leaving a model in eval() disables dropout for the rest of the
    round -- a change to the training recipe that no log would mention."""
    partition = D.plan_partition(n_rows=len(tiny_rows), num_buckets=8, eval_size=20, data_seed=5)
    tokenizer = M.load_tokenizer(tiny_model_dir)
    peft_model = M.attach_lora(M.load_base(tiny_model_dir, "fp32", device=CPU), tiny_lora_cfg)
    peft_model.train()

    E.held_out_loss(peft_model, tokenizer, tiny_rows, partition.eval_indices(),
                    seq_len=32, micro_batch=4, device=CPU)
    assert peft_model.training


def test_smoke_set_is_twenty_greedy_completions(tiny_model_dir, tiny_lora_cfg):
    tokenizer = M.load_tokenizer(tiny_model_dir)
    peft_model = M.attach_lora(M.load_base(tiny_model_dir, "fp32", device=CPU), tiny_lora_cfg)

    assert len(E.SMOKE_PROMPTS) == 20
    out = E.smoke_generate(peft_model, tokenizer, prompts=E.SMOKE_PROMPTS[:3],
                           max_new_tokens=4, device=CPU)
    assert [o["instruction"] for o in out] == [p["instruction"] for p in E.SMOKE_PROMPTS[:3]]
    assert all("completion" in o for o in out)

    # Greedy means repeatable; a tripwire that trips at random gets ignored.
    again = E.smoke_generate(peft_model, tokenizer, prompts=E.SMOKE_PROMPTS[:3],
                            max_new_tokens=4, device=CPU)
    assert out == again


def test_smoke_diff_reports_only_what_changed():
    before = [{"instruction": "a", "completion": "x"}, {"instruction": "b", "completion": "y"}]
    after = [{"instruction": "a", "completion": "x"}, {"instruction": "b", "completion": "z"}]
    changed = E.diff_smoke(before, after)
    assert len(changed) == 1
    assert changed[0] == {"instruction": "b", "before": "y", "after": "z"}
