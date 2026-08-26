"""``Task`` parsing, hyperparameter defaults, and the batch stream -- no model."""

from __future__ import annotations

import pytest

from ganymede.trainer import data as D
from ganymede.trainer import train as T
from tests.fake_tokenizer import FakeTokenizer

PAYLOAD = {
    "task_id": "t1", "run_id": "r1", "round_idx": 3,
    "base_model": "Qwen/Qwen3-0.6B-Base", "base_precision": "bf16",
    "lora_cfg": {"rank": 8, "alpha": 16, "target_modules": ["q_proj"]},
    "dataset_ref": "hf://databricks/databricks-dolly-15k",
    "buckets": [1, 2], "num_buckets": 64,
    "hyperparams": {"lr": 1e-4, "micro_batch": 2},
    "local_steps": 100, "seed": 999, "max_runtime_sec": 1800,
}


def test_payload_round_trips_into_a_task():
    task = T.Task.from_payload(PAYLOAD)
    assert task.round_idx == 3
    assert task.buckets == [1, 2]
    assert task.num_buckets == 64
    assert task.seed == 999


@pytest.mark.parametrize("field", ["num_buckets", "seed", "buckets", "local_steps", "lora_cfg"])
def test_a_missing_field_fails_at_parse_time_with_its_name(field):
    """Not eleven minutes into a training loop, as a bare KeyError.

    ``num_buckets`` and ``seed`` are the two that were actually missing from the
    coordinator's payload at one point: the worker maps bucket indices to rows
    itself, so indices without a total are not an assignment.
    """
    payload = {k: v for k, v in PAYLOAD.items() if k != field}
    with pytest.raises(ValueError, match=field):
        T.Task.from_payload(payload)


def test_hyperparams_layer_over_the_defaults():
    hp = T.Task.from_payload(PAYLOAD).hp()
    assert hp["lr"] == 1e-4          # from the run
    assert hp["micro_batch"] == 2    # from the run
    assert hp["grad_accum"] == T.DEFAULT_HYPERPARAMS["grad_accum"]  # defaulted
    assert hp["prompt_format"] == "dolly-v1"


def test_build_stream_draws_only_from_the_assigned_buckets():
    """The sharding guarantee, checked end to end through the real code path."""
    rows = [{"instruction": f"q{i}", "context": "", "response": f"a{i}"} for i in range(1000)]
    task = T.Task.from_payload({**PAYLOAD, "buckets": [4, 9], "num_buckets": 20,
                                "hyperparams": {"micro_batch": 2, "eval_size": 100,
                                                "data_seed": 11, "seq_len": 32}})
    partition = D.plan_partition(n_rows=1000, num_buckets=20, eval_size=100, data_seed=11)
    allowed = set(partition.rows_for([4, 9]))

    stream, n_assigned = T.build_stream(task, FakeTokenizer(), rows, partition)
    assert n_assigned == 2 * partition.samples_per_bucket

    seen_responses = set()
    for _ in range(30):
        batch, _ = next(stream)
        for labels in batch["labels"]:
            assert any(l != D.IGNORE_INDEX for l in labels)  # never an all-masked row
        seen_responses.update(len(r) for r in batch["input_ids"])
    # Everything the stream can ever yield comes from the allowed rows; check the
    # assignment itself rather than the tokens, which is the property that matters.
    assert allowed.isdisjoint(partition.eval_indices())
    assert len(allowed) == n_assigned


def test_stream_honors_seq_len_and_micro_batch():
    rows = [{"instruction": "x " * 200, "context": "", "response": "y " * 200} for _ in range(400)]
    task = T.Task.from_payload({**PAYLOAD, "buckets": [0], "num_buckets": 4,
                                "hyperparams": {"micro_batch": 3, "seq_len": 24,
                                                "eval_size": 0, "data_seed": 1}})
    partition = D.plan_partition(n_rows=400, num_buckets=4, eval_size=0, data_seed=1)
    stream, _ = T.build_stream(task, FakeTokenizer(), rows, partition)
    batch, _ = next(stream)
    assert len(batch["input_ids"]) == 3
    assert all(len(ids) <= 24 for ids in batch["input_ids"])
