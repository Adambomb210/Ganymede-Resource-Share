"""``run_task`` -- what the worker calls, and the sync boundary it owns (4.3).

Two things here are load-bearing and easy to get subtly wrong.

**A "step" is an optimizer step, not a micro-batch.** The coordinator's whole
step-budget calculation reads ``samples_per_step = micro_batch * grad_accum``
(``coordinator/budget.py``), and the round closes on accumulated steps (3.2). If
this loop counted micro-batches, every budget in the system would be off by the
accumulation factor -- rounds would close early, throughput measurements would be
inflated by the same factor, and the two errors would partly cancel, which is the
worst possible outcome because it would look nearly right.

**The inner optimizer is fresh every round.** That is standard DiLoCo, not an
oversight: momentum lives in the *outer* step on the coordinator (5.2). It has a
pleasant operational consequence -- nothing optimizer-shaped needs to survive
preemption, so a worker that dies mid-round costs exactly the steps it had done
and nothing more (4.4).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch

from ganymede.trainer import data as data_mod
from ganymede.trainer import model as model_mod

# Defaults for hyperparams a run config omits. Named here rather than scattered
# through ``hp.get(...)`` calls so that "what does this run actually use" is one
# place to look, and so the trainer's defaults and the coordinator's defaults can
# be compared without reading both loops.
DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "lr": 2e-4,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "seq_len": 1024,
    "micro_batch": 2,
    "grad_accum": 8,
    "prompt_format": "dolly-v1",
    "completion_only": True,
    "data_seed": 0,
    "eval_size": 750,
    "gradient_checkpointing": None,  # None -> on for CUDA, off elsewhere
}


@dataclass(frozen=True)
class Task:
    """The claim payload (8), parsed once.

    Constructed from the coordinator's JSON by :meth:`from_payload` so that a
    missing field fails here, with a name, rather than as a ``KeyError`` eleven
    minutes into a training loop.
    """

    task_id: str
    run_id: str
    round_idx: int
    base_model: str
    base_precision: str
    lora_cfg: dict[str, Any]
    dataset_ref: str
    buckets: list[int]
    num_buckets: int
    hyperparams: dict[str, Any]
    local_steps: int
    seed: int
    max_runtime_sec: int = 3600

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Task":
        required = (
            "task_id", "run_id", "round_idx", "base_model", "base_precision",
            "lora_cfg", "dataset_ref", "buckets", "num_buckets", "local_steps", "seed",
        )
        missing = [k for k in required if k not in payload]
        if missing:
            raise ValueError(f"task payload is missing {missing}")
        return cls(
            task_id=payload["task_id"],
            run_id=payload["run_id"],
            round_idx=int(payload["round_idx"]),
            base_model=payload["base_model"],
            base_precision=payload["base_precision"],
            lora_cfg=payload["lora_cfg"],
            dataset_ref=payload["dataset_ref"],
            buckets=list(payload["buckets"]),
            num_buckets=int(payload["num_buckets"]),
            hyperparams=payload.get("hyperparams") or {},
            local_steps=int(payload["local_steps"]),
            seed=int(payload["seed"]),
            max_runtime_sec=int(payload.get("max_runtime_sec", 3600)),
        )

    def hp(self) -> dict[str, Any]:
        return {**DEFAULT_HYPERPARAMS, **self.hyperparams}


@dataclass
class TrainResult:
    """4.3 sketches a 3-tuple return; a record is the same thing with names.

    ``steps`` is the count actually completed, which is <= ``task.local_steps``
    whenever the run was cut short by ``should_stop`` or ``max_runtime_sec``. The
    coordinator needs the real number: it is what the round accumulates toward
    closing, and what the throughput estimate for this worker is derived from.
    """

    adapter_bytes: bytes
    steps: int
    metrics: dict[str, Any] = field(default_factory=dict)
    stopped_early: bool = False


def _noop_on_step(step: int, loss: float) -> None:
    pass


def _never_stop() -> bool:
    return False


def build_stream(
    task: Task,
    tokenizer,
    rows: Sequence[dict[str, Any]],
    partition: data_mod.Partition,
):
    """Turn the task's bucket assignment into an endless stream of micro-batches.

    Split out from :func:`run_task` so a caller can inspect what a task would
    actually train on without loading a model -- which is how the bucketing tests
    stay fast, and how ``ganymede-calibrate`` reuses the exact production path
    instead of an approximation of it.
    """
    hp = task.hp()
    formatter = data_mod.get_format(hp["prompt_format"])
    row_ids = partition.rows_for(task.buckets)

    def batches():
        for indices, epoch in data_mod.micro_batches(row_ids, int(hp["micro_batch"]), task.seed):
            encoded = []
            for i in indices:
                prompt, completion = formatter(rows[i])
                encoded.append(
                    data_mod.encode_example(
                        tokenizer, prompt, completion,
                        seq_len=int(hp["seq_len"]),
                        completion_only=bool(hp["completion_only"]),
                    )
                )
            yield data_mod.collate(encoded, tokenizer.pad_token_id), epoch

    return batches(), len(row_ids)


@dataclass
class LoopOutcome:
    steps: int
    losses: list[float]
    tokens: int
    samples: int
    epoch: int
    stopped_early: bool
    train_seconds: float


def train_loop(
    *,
    peft_model,
    stream,
    params: list[torch.nn.Parameter],
    opt: torch.optim.Optimizer,
    steps: int,
    grad_accum: int,
    max_grad_norm: float,
    device: torch.device,
    on_step: Callable[[int, float], None] = _noop_on_step,
    should_stop: Callable[[], bool] = _never_stop,
    deadline: float | None = None,
    step_offset: int = 0,
) -> LoopOutcome:
    """The optimizer loop itself, extracted so it has exactly one implementation.

    ``run_task`` and ``ganymede-baseline`` both drive this. That is the point:
    the baseline exists to be *compared against* a distributed run, so any
    difference between how the baseline trains and how a worker trains lands
    directly in the number M4 rests on. Two loops that were written to be
    identical will not stay identical; one loop cannot drift from itself.

    The baseline calls it in segments (train a while, evaluate, continue) with a
    persistent model and optimizer, which is why the model is passed in already
    built rather than constructed here.
    """
    losses: list[float] = []
    tokens = 0
    samples = 0
    epoch = 0
    step = 0
    stopped_early = False
    loop_started = time.monotonic()

    while step < steps:
        if should_stop() or (deadline is not None and time.monotonic() > deadline):
            stopped_early = True
            break

        opt.zero_grad(set_to_none=True)
        accumulated = 0.0
        for _ in range(grad_accum):
            batch, epoch = next(stream)
            input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
            attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long, device=device)
            labels = torch.tensor(batch["labels"], dtype=torch.long, device=device)

            out = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # Scale before backward so the accumulated gradient is the mean over
            # the effective batch rather than its sum -- otherwise the effective
            # learning rate scales with grad_accum, and a run's lr would quietly
            # mean something different on a 24 GB card than on a 12 GB one.
            (out.loss / grad_accum).backward()

            accumulated += out.loss.item() / grad_accum
            tokens += int(attention_mask.sum().item())
            samples += input_ids.shape[0]

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        opt.step()

        losses.append(accumulated)
        on_step(step_offset + step, accumulated)
        step += 1

    return LoopOutcome(
        steps=step, losses=losses, tokens=tokens, samples=samples,
        epoch=epoch, stopped_early=stopped_early,
        train_seconds=time.monotonic() - loop_started,
    )


def run_task(
    task: Task,
    base_adapter_bytes: bytes,
    on_step: Callable[[int, float], None] = _noop_on_step,
    should_stop: Callable[[], bool] = _never_stop,
    *,
    rows: Sequence[dict[str, Any]] | None = None,
    device: torch.device | None = None,
) -> TrainResult:
    """Train ``task.local_steps`` optimizer steps on the assigned buckets.

    ``rows`` is an injection point for tests and for a worker that has already
    materialized the dataset; left as ``None`` it resolves ``task.dataset_ref``.
    """
    from ganymede.jobtypes.collab_lora_finetune.aggregate import load_adapter, save_adapter

    started = time.monotonic()
    hp = task.hp()
    device = device or model_mod.pick_device()

    # Seeded before anything that draws: dropout is the live consumer here, and
    # an unseeded dropout mask makes an otherwise identical retry of a task
    # produce a different adapter, which would make a failed round impossible to
    # reproduce.
    torch.manual_seed(task.seed)

    if rows is None:
        rows = data_mod.resolve_dataset(task.dataset_ref)

    partition = data_mod.plan_partition(
        n_rows=len(rows),
        num_buckets=task.num_buckets,
        eval_size=int(hp["eval_size"]),
        data_seed=int(hp["data_seed"]),
    )

    tokenizer = model_mod.load_tokenizer(task.base_model)
    base = model_mod.load_base(task.base_model, task.base_precision, device=device)

    checkpointing = hp["gradient_checkpointing"]
    if checkpointing is None:
        checkpointing = device.type == "cuda"
    if checkpointing:
        # Order matters: peft's inputs come from a frozen embedding, so without
        # enable_input_require_grads the checkpointed segment has no input
        # requiring grad and torch silently produces no gradient at all.
        base.enable_input_require_grads()
        base.gradient_checkpointing_enable()

    peft_model = model_mod.attach_lora(base, task.lora_cfg, init_from=load_adapter(base_adapter_bytes))
    peft_model.train()

    params = model_mod.lora_params(peft_model)
    opt = torch.optim.AdamW(params, lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]))

    setup_seconds = time.monotonic() - started
    stream, n_rows_assigned = build_stream(task, tokenizer, rows, partition)
    grad_accum = max(1, int(hp["grad_accum"]))
    max_grad_norm = float(hp["max_grad_norm"])

    outcome = train_loop(
        peft_model=peft_model,
        stream=stream,
        params=params,
        opt=opt,
        steps=task.local_steps,
        grad_accum=grad_accum,
        max_grad_norm=max_grad_norm,
        device=device,
        on_step=on_step,
        should_stop=should_stop,
        deadline=started + task.max_runtime_sec,
    )
    step = outcome.steps
    losses = outcome.losses
    stopped_early = outcome.stopped_early

    elapsed = time.monotonic() - started
    adapter = model_mod.lora_state(peft_model)

    metrics = {
        "steps": step,
        "seconds": round(elapsed, 2),
        # Training-only rate, excluding model load. This is the number the
        # coordinator folds into its throughput estimate (closer.close_round),
        # and it must be a *rate* rather than a blend of a rate and a fixed
        # cost: plan_budget multiplies it by usable seconds, so a blended
        # figure would mis-size every budget by an amount that depends on how
        # long the round is -- small error on long rounds, large on short ones,
        # and never visible as an error at all. The fixed setup cost is
        # reported separately as `setup_sec`; on the coordinator side it is
        # `safety_margin_sec` that has to cover it, so set that setting from
        # observed `setup_sec` rather than from a guess.
        "steps_per_min": (
            round(step / outcome.train_seconds * 60, 3) if outcome.train_seconds > 0 else 0.0
        ),
        "setup_sec": round(setup_seconds, 3),
        "train_sec": round(outcome.train_seconds, 3),
        "tokens": outcome.tokens,
        "samples": outcome.samples,
        "rows_assigned": n_rows_assigned,
        "epochs": outcome.epoch + 1 if step else 0,
        "loss_first": round(losses[0], 5) if losses else None,
        "loss_last": round(losses[-1], 5) if losses else None,
        "loss_mean": round(sum(losses) / len(losses), 5) if losses else None,
        # The trailing mean is what a caller should actually compare across
        # rounds; a single final step is noisy enough at these batch sizes to
        # move by more than a round's real progress.
        "loss_tail_mean": round(sum(losses[-10:]) / len(losses[-10:]), 5) if losses else None,
        "stopped_early": stopped_early,
        "device": device.type,
        # closer.close_round folds throughput back in only when *both*
        # steps_per_min and gpu_model are present; without this key the
        # measured-throughput feedback loop never engages and every budget in
        # the run stays at the cold-start guess.
        "gpu_model": model_mod.device_name(device),
        "base_precision": task.base_precision,
    }

    return TrainResult(
        adapter_bytes=save_adapter(adapter),
        steps=step,
        metrics=metrics,
        stopped_early=stopped_early,
    )
