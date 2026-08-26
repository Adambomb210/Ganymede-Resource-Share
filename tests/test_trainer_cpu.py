"""The claims that need a real model, on CPU.

Marked ``slow``: this downloads ``Qwen/Qwen3-0.6B-Base`` and Dolly 15k and takes
minutes. Everything structural is already covered offline in milliseconds by the
tiny-model suite; what is left here is the one thing a 107k-parameter model
cannot show -- that the trainer actually *learns*.

``Qwen3-0.6B-Base`` rather than the bring-up model ``Qwen3-1.7B-Base`` because it
is small enough to train on CPU, which is the documented reason it is kept
around (docs/03-roadmap.md, *Model*). The architecture is identical, so
everything this proves transfers.

**These numbers are not calibration data.** Steps per minute on CPU says nothing
about a 3060, and no part of the system should read a throughput figure produced
here.
"""

from __future__ import annotations

import pytest
import torch

from ganymede.coordinator import aggregate
from ganymede.trainer import data as D
from ganymede.trainer import evaluate as E
from ganymede.trainer import model as M
from ganymede.trainer import train as T
from scripts.newrun import build_seed_adapter

pytestmark = pytest.mark.slow

BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
DATASET = "hf://databricks/databricks-dolly-15k"
LORA_CFG = {"rank": 8, "alpha": 16, "dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}
CPU = torch.device("cpu")

HYPERPARAMS = {
    "lr": 2e-4, "seq_len": 512, "micro_batch": 2, "grad_accum": 4,
    "eval_size": 750, "data_seed": 20260826, "gradient_checkpointing": False,
}


@pytest.fixture(scope="module")
def dolly():
    try:
        return D.resolve_dataset(DATASET)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"dataset unreachable: {exc}")


def test_dolly_is_the_size_the_run_config_was_sized_for(dolly):
    """15,011 rows is what ``--dataset-rows`` was set from, and it decides every
    bucket boundary. An unpinned hub dataset that gained or lost rows would
    repartition the whole run silently, so this is worth asserting out loud."""
    assert len(dolly) == 15011
    assert set(dolly[0]) >= {"instruction", "context", "response"}


def test_the_bringup_bucketing_is_what_the_roadmap_says(dolly):
    p = D.plan_partition(n_rows=len(dolly), num_buckets=64, eval_size=750, data_seed=0)
    assert p.samples_per_bucket == 222       # (15011 - 750) // 64
    assert 100 <= p.samples_per_bucket <= 500  # 6.10's band
    assert p.dropped == 53                   # < 0.4% of the corpus


def test_training_lowers_held_out_loss(dolly):
    """The claim everything else is in service of.

    Held-out loss rather than training loss: training loss at these batch sizes
    is noisy enough that a 20-step window can descend or ascend by chance, and
    the held-out split is the instrument M4 actually uses.
    """
    task = T.Task(
        task_id="cpu-descent", run_id="cpu", round_idx=0,
        base_model=BASE_MODEL, base_precision="fp32", lora_cfg=LORA_CFG,
        dataset_ref=DATASET, buckets=[3, 17, 42], num_buckets=64,
        hyperparams=HYPERPARAMS, local_steps=24, seed=7,
    )
    seed_bytes = aggregate.save_adapter(build_seed_adapter(BASE_MODEL, LORA_CFG))
    partition = D.plan_partition(
        n_rows=len(dolly), num_buckets=64,
        eval_size=HYPERPARAMS["eval_size"], data_seed=HYPERPARAMS["data_seed"],
    )
    held_out = partition.eval_indices()[:64]

    tokenizer = M.load_tokenizer(BASE_MODEL)

    def held_out_loss_of(adapter_bytes: bytes) -> float:
        base = M.load_base(BASE_MODEL, "fp32", device=CPU)
        peft_model = M.attach_lora(
            base, LORA_CFG, init_from=aggregate.load_adapter(adapter_bytes)
        )
        return E.held_out_loss(
            peft_model, tokenizer, dolly, held_out,
            seq_len=HYPERPARAMS["seq_len"], micro_batch=2, device=CPU,
        ).loss

    before = held_out_loss_of(seed_bytes)
    result = T.run_task(task, seed_bytes, rows=dolly, device=CPU)
    after = held_out_loss_of(result.adapter_bytes)

    assert result.steps == 24
    assert after < before, f"held-out loss did not improve: {before:.4f} -> {after:.4f}"


def test_a_real_submission_passes_the_real_gates(dolly):
    """The M0/M1 handoff on the actual bring-up architecture, not a stand-in."""
    task = T.Task(
        task_id="cpu-gates", run_id="cpu", round_idx=0,
        base_model=BASE_MODEL, base_precision="fp32", lora_cfg=LORA_CFG,
        dataset_ref=DATASET, buckets=[5], num_buckets=64,
        hyperparams=HYPERPARAMS, local_steps=2, seed=9,
    )
    seed_bytes = aggregate.save_adapter(build_seed_adapter(BASE_MODEL, LORA_CFG))
    result = T.run_task(task, seed_bytes, rows=dolly, device=CPU)

    expected = aggregate.manifest_of(aggregate.load_adapter(seed_bytes))
    gate, adapter = aggregate.check_structural(result.adapter_bytes, expected, result.steps)

    assert gate.accepted, f"{gate.reason}: {gate.detail}"
    assert len(adapter) == 224  # 28 layers x 4 modules x (A, B)
    assert all(torch.isfinite(t).all() for t in adapter.values())


def test_the_smoke_set_generates_on_the_real_tokenizer(dolly):
    """The round-0 smoke output is an M0 exit criterion; this proves the path
    works against a real tokenizer and a real chat-template-free base model."""
    base = M.load_base(BASE_MODEL, "fp32", device=CPU)
    peft_model = M.attach_lora(
        base, LORA_CFG, init_from=build_seed_adapter(BASE_MODEL, LORA_CFG)
    )
    out = E.smoke_generate(
        peft_model, M.load_tokenizer(BASE_MODEL),
        prompts=E.SMOKE_PROMPTS[:2], max_new_tokens=12, device=CPU,
    )
    assert len(out) == 2
    assert any(o["completion"].strip() for o in out)
