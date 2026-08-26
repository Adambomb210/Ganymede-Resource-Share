"""Dataset resolution, bucketing, prompt formatting and batching.

Nothing here touches a model, so all of it is testable without a weight download.
That matters more than it sounds: the bucketing scheme is the one part of the
system where the coordinator and every worker must agree *without ever exchanging
the data*, and a silent disagreement produces a run that trains fine and means
nothing.

The contract, in one paragraph
------------------------------
The coordinator never sees the dataset. It hands a worker a list of **bucket
indices** and the run's ``num_buckets``; the worker turns those into rows by
itself. So the mapping from bucket index to rows must be a pure function of
values every party already has: ``(dataset_ref, data_seed, eval_size,
num_buckets)``. It must not depend on row *content*, on the machine, on the
Python version, or on iteration order of anything.

Why permute-then-slice rather than ``hash(row) % num_buckets``
--------------------------------------------------------------
Both give a stable assignment. Permute-then-slice wins on three counts:

1. **Bucket sizes are exact**, so ``samples_per_bucket`` is a derived fact rather
   than an estimate -- and the whole step-budget calculation in
   ``coordinator/budget.py`` is arithmetic over that number. An approximate
   bucket size makes every budget approximate.
2. **It does not depend on row text.** A dataset revision that fixes a typo in
   one row would, under content hashing, potentially move that row to a
   different bucket and change nothing else -- a change invisible in every log.
3. A bucket is an index range, so materializing one is a slice rather than a
   full-table scan.

The permutation uses ``random.Random``, not ``torch.randperm`` or numpy. CPython's
Mersenne Twister and the shuffle algorithm over it are documented and stable
across versions and platforms; torch's RNG stream carries no such promise across
versions, and a worker on a pinned-by-digest container and a worker on a native
Windows install do not have the same torch build (4.1).

The eval split is carved out **before** bucketing
--------------------------------------------------
``plan_partition`` permutes first, takes the first ``eval_size`` rows as held-out,
and buckets only what remains. Held-out rows therefore cannot appear in any
bucket, which is the property the M4 comparison rests on -- see docs/03-roadmap.md,
*Eval metric*. Doing it the other way round (bucket everything, then sample an
eval set) leaks, and the leak shows up as a suspiciously good eval curve that
nobody can explain.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

# Rows per bucket we aim for when suggesting a bucket count (docs/03-roadmap.md,
# *Bucket count should scale with dataset size*). Below ~100 a worker's shard is
# statistical noise rather than a sample; far above ~500 the bucket stops being a
# useful unit of assignment.
TARGET_SAMPLES_PER_BUCKET = 200
MIN_BUCKETS = 8
MAX_BUCKETS = 4096

# Masked-out label value; `transformers` cross-entropy ignores exactly this.
IGNORE_INDEX = -100


# --------------------------------------------------------------------------
# Partitioning
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Partition:
    """A dataset's deterministic split into a held-out set and equal buckets.

    Construct with :func:`plan_partition`; the fields are derived, not chosen.
    """

    n_rows: int
    eval_size: int
    num_buckets: int
    samples_per_bucket: int
    data_seed: int
    _order: tuple[int, ...]

    @property
    def dropped(self) -> int:
        """Rows in neither the eval split nor any bucket.

        ``n_train % num_buckets`` rows are dropped so that every bucket is
        *exactly* the same size. That equality is what makes a step budget
        computed from ``samples_per_bucket`` exact rather than approximate, and
        the cost is at most ``num_buckets - 1`` rows -- 63 of 14,261 for the
        bring-up run, or 0.4%.
        """
        return self.n_rows - self.eval_size - self.num_buckets * self.samples_per_bucket

    def eval_indices(self) -> list[int]:
        return list(self._order[: self.eval_size])

    def bucket_indices(self, bucket: int) -> list[int]:
        if not 0 <= bucket < self.num_buckets:
            raise ValueError(f"bucket {bucket} out of range for {self.num_buckets} buckets")
        start = self.eval_size + bucket * self.samples_per_bucket
        return list(self._order[start : start + self.samples_per_bucket])

    def rows_for(self, buckets: Sequence[int]) -> list[int]:
        """Row indices for an assignment of several buckets, in bucket order."""
        out: list[int] = []
        for b in buckets:
            out.extend(self.bucket_indices(b))
        return out


def suggest_num_buckets(n_train_rows: int, target: int = TARGET_SAMPLES_PER_BUCKET) -> int:
    """Bucket count for a dataset of this size -- a suggestion, not a rule.

    ``num_buckets`` is fixed per run at prep time and never changes mid-run
    (6.10), so it is a run-config value a human sets. This exists so that value
    has a defensible default instead of being invented per run.
    """
    return max(MIN_BUCKETS, min(MAX_BUCKETS, n_train_rows // target))


def plan_partition(
    n_rows: int,
    num_buckets: int,
    eval_size: int,
    data_seed: int,
) -> Partition:
    """The one function that decides which rows are which. Pure and total.

    Every worker on a run calls this with identical arguments and gets an
    identical answer, which is the entire mechanism by which sharding works
    without the coordinator ever seeing the data.
    """
    if n_rows <= 0:
        raise ValueError("dataset is empty")
    if num_buckets < 1:
        raise ValueError("num_buckets must be >= 1")
    if eval_size < 0:
        raise ValueError("eval_size must be >= 0")

    n_train = n_rows - eval_size
    if n_train < num_buckets:
        raise ValueError(
            f"{n_rows} rows minus {eval_size} held out leaves {n_train} for "
            f"{num_buckets} buckets -- fewer than one row each"
        )

    order = list(range(n_rows))
    random.Random(data_seed).shuffle(order)

    return Partition(
        n_rows=n_rows,
        eval_size=eval_size,
        num_buckets=num_buckets,
        samples_per_bucket=n_train // num_buckets,
        data_seed=data_seed,
        _order=tuple(order),
    )


# --------------------------------------------------------------------------
# Prompt formatting
# --------------------------------------------------------------------------

Formatter = Callable[[dict[str, Any]], tuple[str, str]]


def _dolly_v1(row: dict[str, Any]) -> tuple[str, str]:
    """Databricks Dolly 15k -> (prompt, completion).

    A **base** model has no chat template, so the format is ours to define --
    which is exactly why it is named and versioned rather than inlined. The
    baseline (M0) and the distributed run (M4) must be formatted identically or
    the comparison measures the format change instead of the aggregation, and a
    format that lives as a f-string in two files will eventually differ in one.

    Bumping this format means bumping the name and re-running the baseline.
    """
    instruction = (row.get("instruction") or "").strip()
    context = (row.get("context") or "").strip()
    response = (row.get("response") or "").strip()

    parts = ["### Instruction:", instruction, ""]
    if context:
        parts += ["### Input:", context, ""]
    parts += ["### Response:", ""]
    return "\n".join(parts), response


FORMATS: dict[str, Formatter] = {"dolly-v1": _dolly_v1}


def get_format(name: str) -> Formatter:
    try:
        return FORMATS[name]
    except KeyError:
        raise ValueError(f"unknown prompt_format {name!r}; have {sorted(FORMATS)}") from None


# --------------------------------------------------------------------------
# Tokenization and masking
# --------------------------------------------------------------------------


def encode_example(
    tokenizer,
    prompt: str,
    completion: str,
    seq_len: int,
    completion_only: bool = True,
) -> dict[str, list[int]]:
    """Tokenize one example into ``input_ids`` and ``labels``.

    ``completion_only`` masks the prompt out of the loss, which is standard SFT
    practice: the model is not being taught to produce the instruction.

    Truncation cuts the **prompt** from the left, never the completion from the
    left. Right-truncating the concatenation -- the obvious implementation -- can
    remove the completion entirely on a long-context example, leaving every label
    masked. Cross-entropy over zero unmasked tokens is a NaN, and one NaN in one
    micro-batch poisons the accumulated gradient for the whole optimizer step.
    That failure is silent in a loss curve that has already gone to NaN for a
    reason nobody will guess.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]

    eos = tokenizer.eos_token_id
    if eos is not None:
        completion_ids = completion_ids + [eos]

    bos = tokenizer.bos_token_id
    prefix = [bos] if bos is not None and bos != eos else []

    # The completion is the supervised part, so it is the part that survives.
    if len(completion_ids) >= seq_len:
        completion_ids = completion_ids[:seq_len]
        prefix, prompt_ids = [], []
    else:
        room = seq_len - len(completion_ids) - len(prefix)
        if room < len(prompt_ids):
            prompt_ids = prompt_ids[len(prompt_ids) - room :] if room > 0 else []

    input_ids = prefix + prompt_ids + completion_ids
    if completion_only:
        labels = [IGNORE_INDEX] * (len(input_ids) - len(completion_ids)) + list(completion_ids)
    else:
        labels = list(input_ids)

    return {"input_ids": input_ids, "labels": labels}


def collate(examples: Sequence[dict[str, list[int]]], pad_id: int) -> dict[str, list[list[int]]]:
    """Right-pad a micro-batch to its longest member.

    Padding to the batch max rather than to ``seq_len`` is a real speedup on
    Dolly, where the median example is far shorter than the limit. Padded label
    positions get ``IGNORE_INDEX`` so they contribute nothing to the loss, and
    ``attention_mask`` keeps them out of attention.

    Returns plain lists; :mod:`ganymede.trainer.train` turns them into tensors on
    the right device. Keeping tensors out of this module is what lets the whole
    masking story be tested without torch.
    """
    if not examples:
        raise ValueError("empty micro-batch")
    width = max(len(e["input_ids"]) for e in examples)
    return {
        "input_ids": [e["input_ids"] + [pad_id] * (width - len(e["input_ids"])) for e in examples],
        "attention_mask": [[1] * len(e["input_ids"]) + [0] * (width - len(e["input_ids"])) for e in examples],
        "labels": [e["labels"] + [IGNORE_INDEX] * (width - len(e["labels"])) for e in examples],
    }


# --------------------------------------------------------------------------
# The stream a training loop consumes
# --------------------------------------------------------------------------


def micro_batches(
    rows: Sequence[int],
    micro_batch: int,
    seed: int,
    max_batches: int | None = None,
) -> Iterator[tuple[list[int], int]]:
    """Yield ``(row_indices, epoch)`` forever, reshuffling between passes.

    The stream **cycles** rather than stopping at the end of the assigned rows.
    A worker is budgeted a number of optimizer steps, and stopping early because
    the shard ran out would silently under-deliver against that budget -- the
    round would close short and nobody would see why. ``plan_budget`` sizes the
    bucket assignment to cover the budget in ``target_passes``, so cycling is the
    exception rather than the rule; when it does happen the returned epoch counter
    makes it visible in the submission metrics.

    Each pass is shuffled with a different derived seed, so a second pass over
    the same rows is not the same sequence of batches.
    """
    if micro_batch < 1:
        raise ValueError("micro_batch must be >= 1")
    if not rows:
        raise ValueError("no rows assigned")
    if len(rows) < micro_batch:
        # Partial batches are dropped (see below), so a shard smaller than one
        # micro-batch yields nothing at all -- and because the stream cycles,
        # "nothing at all" is an infinite loop incrementing an epoch counter,
        # not an exception. A worker would hang holding a lease until it
        # expired, and the round would look like a straggler rather than a bug.
        raise ValueError(
            f"{len(rows)} rows assigned but micro_batch is {micro_batch}: "
            f"the shard is smaller than a single batch"
        )

    epoch = 0
    emitted = 0
    while True:
        order = list(rows)
        random.Random(_epoch_seed(seed, epoch)).shuffle(order)
        for i in range(0, len(order) - micro_batch + 1, micro_batch):
            yield order[i : i + micro_batch], epoch
            emitted += 1
            if max_batches is not None and emitted >= max_batches:
                return
        epoch += 1


def _epoch_seed(seed: int, epoch: int) -> int:
    # Plain arithmetic rather than hashing: it must be reproducible across
    # processes, and Python salts str hashing per process by default.
    return (seed * 1_000_003 + epoch) % (2**31)


# --------------------------------------------------------------------------
# dataset_ref resolution
# --------------------------------------------------------------------------


def resolve_dataset(dataset_ref: str) -> list[dict[str, Any]]:
    """Materialize a ``dataset_ref`` into rows.

    Two schemes, dispatched on prefix:

    ``hf://<repo>[@<revision>][:<split>]``
        A HuggingFace hub dataset, e.g. ``hf://databricks/databricks-dolly-15k``.
        Split defaults to ``train``. **Pin the revision for anything you intend to
        compare across time** -- an unpinned hub dataset can gain or lose rows,
        which silently repartitions every bucket in the run.

    ``s3://<bucket>/<prefix>``
        Reserved for self-hosted datasets (the shape 8's example uses). Not
        implemented yet; it needs a decided on-disk layout, and inventing one
        before there is a dataset to put in it would be guessing.
    """
    if dataset_ref.startswith("hf://"):
        from datasets import load_dataset

        spec = dataset_ref[len("hf://") :]
        split = "train"
        if ":" in spec:
            spec, split = spec.rsplit(":", 1)
        revision = None
        if "@" in spec:
            spec, revision = spec.split("@", 1)
        ds = load_dataset(spec, split=split, revision=revision)
        return [dict(row) for row in ds]

    if dataset_ref.startswith("s3://"):
        raise NotImplementedError(
            "s3:// datasets are not implemented; the on-disk layout is undecided "
            "(see resolve_dataset's docstring)"
        )

    raise ValueError(f"unrecognized dataset_ref {dataset_ref!r}: expected hf:// or s3://")
