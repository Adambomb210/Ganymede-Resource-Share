"""Bucketing, formatting, masking and batching -- all without a model.

The bucketing tests carry more weight than their size suggests. The coordinator
never sees the dataset; it sends bucket *indices* and trusts every worker to turn
them into the same rows. There is no cross-check anywhere in the system that
would catch two workers disagreeing -- both would train, both would submit, both
would pass every acceptance gate, and the run would silently be training on the
wrong sharding. These tests are that cross-check.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from ganymede.trainer import data as D
from tests.fake_tokenizer import FakeTokenizer


# --------------------------------------------------------------------------
# Partitioning
# --------------------------------------------------------------------------


def test_eval_split_never_appears_in_any_bucket():
    """The property the whole M4 comparison rests on.

    If a held-out row is also a training row, held-out loss measures memorization
    and the distributed-vs-single-node comparison is meaningless -- while looking
    better than a correct one, which is the worst way for it to fail.
    """
    p = D.plan_partition(n_rows=1000, num_buckets=16, eval_size=100, data_seed=7)
    held = set(p.eval_indices())
    assert len(held) == 100
    for b in range(p.num_buckets):
        assert held.isdisjoint(p.bucket_indices(b))


def test_buckets_are_disjoint_equal_and_cover_the_rest():
    p = D.plan_partition(n_rows=1000, num_buckets=16, eval_size=100, data_seed=7)
    assert p.samples_per_bucket == 900 // 16  # 56
    seen: set[int] = set()
    for b in range(16):
        rows = p.bucket_indices(b)
        assert len(rows) == p.samples_per_bucket
        assert seen.isdisjoint(rows)
        seen.update(rows)
    assert len(seen) == 16 * p.samples_per_bucket
    assert p.dropped == 900 - 16 * p.samples_per_bucket == 4
    assert seen.isdisjoint(p.eval_indices())


def test_partition_is_stable_across_processes():
    """``random.Random`` is used precisely so this holds.

    A worker on Windows and the coordinator on Linux must agree, and they never
    exchange the assignment to check. Anything with per-process entropy -- str
    hashing, dict order, torch's RNG stream across versions -- would pass a
    single-process test and fail in the fleet.
    """
    code = (
        "from ganymede.trainer import data as D;"
        "p = D.plan_partition(n_rows=997, num_buckets=13, eval_size=61, data_seed=42);"
        "print(p.bucket_indices(5)[:8], p.eval_indices()[:5])"
    )
    runs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
        for _ in range(2)
    }
    assert len(runs) == 1

    local = D.plan_partition(n_rows=997, num_buckets=13, eval_size=61, data_seed=42)
    assert str(local.bucket_indices(5)[:8]) in next(iter(runs))


def test_data_seed_changes_the_assignment():
    a = D.plan_partition(n_rows=1000, num_buckets=10, eval_size=50, data_seed=1)
    b = D.plan_partition(n_rows=1000, num_buckets=10, eval_size=50, data_seed=2)
    assert a.bucket_indices(0) != b.bucket_indices(0)
    assert a.samples_per_bucket == b.samples_per_bucket  # sizes are structural


def test_rows_for_concatenates_in_bucket_order():
    p = D.plan_partition(n_rows=500, num_buckets=10, eval_size=0, data_seed=3)
    assert p.rows_for([2, 5]) == p.bucket_indices(2) + p.bucket_indices(5)


def test_bucket_index_out_of_range_is_an_error():
    p = D.plan_partition(n_rows=500, num_buckets=10, eval_size=0, data_seed=3)
    with pytest.raises(ValueError, match="out of range"):
        p.bucket_indices(10)
    with pytest.raises(ValueError, match="out of range"):
        p.bucket_indices(-1)


def test_partition_refuses_impossible_splits():
    with pytest.raises(ValueError, match="empty"):
        D.plan_partition(n_rows=0, num_buckets=4, eval_size=0, data_seed=0)
    with pytest.raises(ValueError, match="fewer than one row"):
        D.plan_partition(n_rows=100, num_buckets=64, eval_size=50, data_seed=0)


def test_suggest_num_buckets_targets_the_documented_band():
    # 6.10's band is 100-500 samples per bucket.
    for n in (5_000, 15_011, 200_000, 940_000):
        b = D.suggest_num_buckets(n)
        assert D.MIN_BUCKETS <= b <= D.MAX_BUCKETS
        assert 100 <= n // b <= 500
    assert D.suggest_num_buckets(50) == D.MIN_BUCKETS  # tiny sets clamp up
    assert D.suggest_num_buckets(10**9) == D.MAX_BUCKETS


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------


def test_dolly_format_omits_an_empty_context_block():
    prompt, completion = D.get_format("dolly-v1")(
        {"instruction": "Say hi", "context": "", "response": "hi"}
    )
    assert "### Input:" not in prompt
    assert prompt.endswith("### Response:\n")
    assert completion == "hi"


def test_dolly_format_includes_context_when_present():
    prompt, _ = D.get_format("dolly-v1")(
        {"instruction": "Summarize", "context": "Some text.", "response": "x"}
    )
    assert "### Input:\nSome text." in prompt
    assert prompt.index("### Input:") < prompt.index("### Response:")


def test_unknown_format_names_what_is_available():
    with pytest.raises(ValueError, match="dolly-v1"):
        D.get_format("nope")


# --------------------------------------------------------------------------
# Masking and truncation
# --------------------------------------------------------------------------


def test_completion_only_masks_exactly_the_prompt():
    tok = FakeTokenizer()
    ex = D.encode_example(tok, "a b c", "d e", seq_len=64)
    n_completion = 3  # "d", "e", eos
    assert ex["labels"][-n_completion:] == ex["input_ids"][-n_completion:]
    assert set(ex["labels"][:-n_completion]) == {D.IGNORE_INDEX}
    assert len(ex["labels"]) == len(ex["input_ids"])


def test_full_sequence_loss_when_completion_only_is_off():
    tok = FakeTokenizer()
    ex = D.encode_example(tok, "a b", "c", seq_len=64, completion_only=False)
    assert ex["labels"] == ex["input_ids"]
    assert D.IGNORE_INDEX not in ex["labels"]


def test_truncation_sacrifices_the_prompt_not_the_completion():
    """The bug this guards is a NaN, and a very confusing one.

    Right-truncating the concatenation would cut the completion off a long
    example, leaving every label masked. Cross-entropy over zero unmasked tokens
    is NaN, one NaN micro-batch poisons the whole accumulated optimizer step, and
    the resulting loss curve gives no hint that the cause was an input too long.
    """
    tok = FakeTokenizer()
    long_prompt = " ".join(f"w{i}" for i in range(500))
    ex = D.encode_example(tok, long_prompt, "keep me", seq_len=16)

    assert len(ex["input_ids"]) <= 16
    supervised = [l for l in ex["labels"] if l != D.IGNORE_INDEX]
    assert len(supervised) == 3  # "keep", "me", eos
    assert supervised == ex["input_ids"][-3:]


def test_completion_longer_than_seq_len_still_has_supervised_tokens():
    tok = FakeTokenizer()
    ex = D.encode_example(tok, "prompt", " ".join(f"t{i}" for i in range(100)), seq_len=8)
    assert len(ex["input_ids"]) == 8
    assert all(l != D.IGNORE_INDEX for l in ex["labels"])


def test_no_bos_duplication_when_bos_equals_eos():
    tok = FakeTokenizer(bos_token_id=2, eos_token_id=2)
    ex = D.encode_example(tok, "a", "b", seq_len=16)
    assert ex["input_ids"].count(2) == 1  # the trailing eos only


def test_seq_len_must_be_positive():
    with pytest.raises(ValueError):
        D.encode_example(FakeTokenizer(), "a", "b", seq_len=0)


# --------------------------------------------------------------------------
# Collation
# --------------------------------------------------------------------------


def test_collate_pads_to_the_batch_max_and_masks_the_padding():
    batch = D.collate(
        [
            {"input_ids": [5, 6, 7], "labels": [-100, 6, 7]},
            {"input_ids": [8], "labels": [8]},
        ],
        pad_id=0,
    )
    assert batch["input_ids"] == [[5, 6, 7], [8, 0, 0]]
    assert batch["attention_mask"] == [[1, 1, 1], [1, 0, 0]]
    # Padded label positions must be IGNORE_INDEX, not pad_id: pad_id is a real
    # token id, and supervising it teaches the model to emit padding.
    assert batch["labels"] == [[-100, 6, 7], [8, -100, -100]]


def test_collate_rejects_an_empty_batch():
    with pytest.raises(ValueError):
        D.collate([], pad_id=0)


# --------------------------------------------------------------------------
# The micro-batch stream
# --------------------------------------------------------------------------


def test_stream_is_deterministic_for_a_seed_and_differs_across_seeds():
    rows = list(range(20))
    a = [b for b, _ in D.micro_batches(rows, 4, seed=1, max_batches=5)]
    b = [b for b, _ in D.micro_batches(rows, 4, seed=1, max_batches=5)]
    c = [b for b, _ in D.micro_batches(rows, 4, seed=2, max_batches=5)]
    assert a == b
    assert a != c


def test_stream_cycles_rather_than_running_out():
    """Stopping at the end of the shard would silently under-deliver the budget.

    The coordinator closes a round on accumulated steps; a worker that quietly
    delivered fewer would stall the round with no error anywhere.
    """
    rows = list(range(8))
    batches = list(D.micro_batches(rows, 4, seed=0, max_batches=6))
    assert len(batches) == 6
    epochs = [e for _, e in batches]
    assert epochs == [0, 0, 1, 1, 2, 2]


def test_each_pass_is_reshuffled():
    rows = list(range(12))
    batches = [b for b, _ in D.micro_batches(rows, 3, seed=5, max_batches=8)]
    first_pass, second_pass = batches[:4], batches[4:]
    assert sorted(x for b in first_pass for x in b) == sorted(rows)
    assert first_pass != second_pass


def test_a_partial_trailing_micro_batch_is_dropped():
    # 10 rows at micro_batch 4 yields two full batches; the remaining 2 roll into
    # the next pass rather than forming a short batch, which would make the
    # effective batch size -- and so the effective learning rate -- vary by step.
    batches = list(D.micro_batches(list(range(10)), 4, seed=0, max_batches=2))
    assert all(len(b) == 4 for b, _ in batches)


def test_a_shard_smaller_than_one_micro_batch_raises_rather_than_hanging():
    """The failure mode this replaces is the nastiest kind: not a crash.

    Partial batches are dropped, so a too-small shard yields nothing -- and the
    stream cycles, so "nothing" is an infinite loop incrementing an epoch
    counter. The worker would hang holding its lease until it expired and look
    like a slow straggler rather than a bug.
    """
    with pytest.raises(ValueError, match="smaller than a single batch"):
        next(D.micro_batches([1, 2, 3], 4, seed=0))


def test_stream_rejects_nonsense():
    with pytest.raises(ValueError):
        next(D.micro_batches([1, 2], 0, seed=0))
    with pytest.raises(ValueError):
        next(D.micro_batches([], 2, seed=0))


# --------------------------------------------------------------------------
# dataset_ref
# --------------------------------------------------------------------------


def test_unknown_dataset_scheme_is_rejected_by_name():
    with pytest.raises(ValueError, match="hf://"):
        D.resolve_dataset("databricks/databricks-dolly-15k")


def test_s3_datasets_say_they_are_unimplemented_rather_than_failing_obscurely():
    with pytest.raises(NotImplementedError, match="undecided"):
        D.resolve_dataset("s3://ganymede/data/whatever")
