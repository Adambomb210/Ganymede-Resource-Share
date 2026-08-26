"""``ganymede-calibrate`` -- measure a run on one GPU (docs/03-roadmap.md, M0).

The roadmap's framing is the important part: because Ganymede will run many
models and fine-tunes rather than one, calibration is **per-run work that recurs**
on every new base model, dataset or hardware mix. So it is a tool you run per run,
not a script you run once and delete. Everything below follows from that -- the
output is a machine-readable artifact the coordinator stores, not a paragraph in
a README that goes stale the first time someone changes ``seq_len``.

Its ``calibration.json`` feeds three consumers:

- **Round sizing** (3.3): ``throughput[gpu_model]`` is read by
  ``coordinator/rounds.py`` when a worker of that model claims, so a 3090 and a
  4090 in the same round both finish near the deadline instead of the 3090
  straggling every time.
- **Capability filtering** (6.2): ``fits`` says which precisions and sequence
  lengths this card can actually honor.
- **The M4 comparison**: the calibrated single-node figures are what the
  distributed run has to beat.

Measured on the production path, deliberately
---------------------------------------------
Throughput is measured by driving ``train.train_loop`` -- the same loop a worker
runs -- rather than a stripped-down benchmark. A benchmark that omits gradient
clipping, or accumulates differently, or skips the tokenizer, measures a program
that nobody runs. The whole value of the number is that it predicts the real
thing.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from typing import Any

import torch

from ganymede.coordinator import budget as budget_mod
from ganymede.trainer import data as data_mod
from ganymede.trainer import model as model_mod
from ganymede.trainer import train as train_mod

CALIBRATION_VERSION = 1

# Ladder for the fit probe. Powers of two because attention kernels and KV
# allocation are happiest there, and because a max_seq_len of 1,732 is a number
# no one will ever be able to justify in a run config.
SEQ_LEN_LADDER = (512, 1024, 2048, 4096, 8192)

# Steps discarded before timing. The first steps of any run are unrepresentative
# -- CUDA context creation, kernel autotuning, allocator growth, and (on a cold
# page cache) the tail of the model load are all still happening. Timing them
# understates the card, and understating throughput means under-budgeting every
# worker of that model for the entire run.
DEFAULT_WARMUP_STEPS = 3
DEFAULT_MEASURE_STEPS = 12


def describe_device(device: torch.device) -> dict[str, Any]:
    """What the coordinator keys throughput on, plus enough to debug a surprise.

    ``name`` must match what a worker reports as ``device_name`` in its
    ``compute_profile`` (6.9), because that string is the join key between this
    file and ``rounds.claim_task``. Both read ``torch.cuda.get_device_name``, so
    they agree by construction rather than by convention.
    """
    info: dict[str, Any] = {
        "type": device.type,
        "torch": torch.__version__,
        "platform": f"{platform.system()} {platform.machine()}",
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info["name"] = torch.cuda.get_device_name(device)
        info["vram_gb"] = round(props.total_memory / 1024**3, 2)
        info["capability"] = f"{props.major}.{props.minor}"
        info["driver"] = torch.version.cuda
    else:
        info["name"] = f"cpu:{platform.processor() or platform.machine()}"
        info["vram_gb"] = None
    return info


def _peak_vram_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return round(torch.cuda.max_memory_allocated(device) / 1024**3, 3)


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def probe_fit(
    base_model: str,
    precision: str,
    lora_cfg: dict[str, Any],
    *,
    micro_batch: int,
    ladder: tuple[int, ...] = SEQ_LEN_LADDER,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Largest ``seq_len`` on this card at this precision, by trying them.

    One forward *and backward* per rung, not just a forward: activation memory
    for the backward pass is the larger term, and a probe that only measures the
    forward will happily report a sequence length that OOMs three steps into a
    real round -- after the worker has downloaded the model and claimed a lease.

    Walks the ladder upward and stops at the first failure. Every rung reloads
    nothing, but each rung's allocator peak is reset, so ``peak_vram_gb`` is that
    rung's cost rather than the high-water mark of everything before it.
    """
    device = device or model_mod.pick_device()
    result: dict[str, Any] = {"ok": False, "max_seq_len": None, "peak_vram_gb": None}

    try:
        tokenizer = model_mod.load_tokenizer(base_model)
        base = model_mod.load_base(base_model, precision, device=device)
        peft_model = model_mod.attach_lora(base, lora_cfg)
        peft_model.train()
    except Exception as exc:  # noqa: BLE001 - a failure to load is a fit answer
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    params = model_mod.lora_params(peft_model)
    pad = tokenizer.pad_token_id or 0

    for seq_len in ladder:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        try:
            ids = torch.full((micro_batch, seq_len), pad, dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            labels = ids.clone()
            out = peft_model(input_ids=ids, attention_mask=mask, labels=labels)
            out.loss.backward()
            for p in params:
                p.grad = None
            result["ok"] = True
            result["max_seq_len"] = seq_len
            result["peak_vram_gb"] = _peak_vram_gb(device)
        except Exception as exc:  # noqa: BLE001
            if not _is_oom(exc):
                result["error"] = f"{type(exc).__name__}: {exc}"
            break

    del peft_model, base
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def measure_throughput(
    run_cfg: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    measure_steps: int = DEFAULT_MEASURE_STEPS,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Steps per minute on the production loop, warmup excluded."""
    device = device or model_mod.pick_device()
    hp = {**train_mod.DEFAULT_HYPERPARAMS, **run_cfg.get("hyperparams", {})}
    grad_accum = max(1, int(hp["grad_accum"]))
    micro_batch = int(hp["micro_batch"])

    partition = data_mod.plan_partition(
        n_rows=len(rows),
        num_buckets=int(run_cfg["num_buckets"]),
        eval_size=int(hp["eval_size"]),
        data_seed=int(hp["data_seed"]),
    )

    # Calibration trains on the whole training set rather than an assignment,
    # since it is measuring the card and not a shard.
    probe_task = train_mod.Task(
        task_id="calibration", run_id=run_cfg.get("run_id", "calibration"), round_idx=0,
        base_model=run_cfg["base_model"], base_precision=run_cfg["base_precision"],
        lora_cfg=run_cfg["lora_cfg"], dataset_ref=run_cfg["dataset_ref"],
        buckets=list(range(int(run_cfg["num_buckets"]))),
        num_buckets=int(run_cfg["num_buckets"]),
        hyperparams=hp, local_steps=warmup_steps + measure_steps,
        seed=int(run_cfg.get("calibration_seed", 0)),
    )

    tokenizer = model_mod.load_tokenizer(probe_task.base_model)
    load_started = time.monotonic()
    base = model_mod.load_base(probe_task.base_model, probe_task.base_precision, device=device)
    checkpointing = hp["gradient_checkpointing"]
    if checkpointing is None:
        checkpointing = device.type == "cuda"
    if checkpointing:
        base.enable_input_require_grads()
        base.gradient_checkpointing_enable()
    peft_model = model_mod.attach_lora(base, probe_task.lora_cfg)
    peft_model.train()
    setup_sec = time.monotonic() - load_started

    params = model_mod.lora_params(peft_model)
    opt = torch.optim.AdamW(params, lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]))
    stream, _ = train_mod.build_stream(probe_task, tokenizer, rows, partition)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_mod.train_loop(
        peft_model=peft_model, stream=stream, params=params, opt=opt,
        steps=warmup_steps, grad_accum=grad_accum,
        max_grad_norm=float(hp["max_grad_norm"]), device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    measured = train_mod.train_loop(
        peft_model=peft_model, stream=stream, params=params, opt=opt,
        steps=measure_steps, grad_accum=grad_accum,
        max_grad_norm=float(hp["max_grad_norm"]), device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    seconds = measured.train_seconds
    steps_per_min = measured.steps / seconds * 60 if seconds > 0 else 0.0

    return {
        "steps_per_min": round(steps_per_min, 3),
        "tokens_per_sec": round(measured.tokens / seconds, 1) if seconds > 0 else 0.0,
        "samples_per_step": micro_batch * grad_accum,
        "seq_len": int(hp["seq_len"]),
        "measured_steps": measured.steps,
        "warmup_steps": warmup_steps,
        "seconds": round(seconds, 2),
        "setup_sec": round(setup_sec, 2),
        "peak_vram_gb": _peak_vram_gb(device),
        "gradient_checkpointing": bool(checkpointing),
        "loss_mean": round(sum(measured.losses) / len(measured.losses), 5) if measured.losses else None,
    }


def recommend_local_steps(
    steps_per_min: float,
    *,
    target_round_sec: int,
    est_download_sec: int = 120,
    est_upload_sec: int = 120,
    safety_margin_sec: int = 120,
) -> dict[str, Any]:
    """What ``local_steps`` a round of this length should offer this card.

    Delegates to the coordinator's own ``budget`` module rather than
    reimplementing the arithmetic. If the recommendation and the coordinator ever
    disagree, the recommendation is worse than useless -- it is a number an
    operator would trust while the running system quietly used a different one.
    """
    usable = budget_mod.usable_seconds(
        target_round_sec, est_download_sec, est_upload_sec, safety_margin_sec
    )
    return {
        "target_round_sec": target_round_sec,
        "usable_sec": usable,
        "local_steps": budget_mod.step_budget(steps_per_min, usable),
    }


def calibrate(
    run_cfg: dict[str, Any],
    *,
    target_round_sec: int = 2100,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    measure_steps: int = DEFAULT_MEASURE_STEPS,
    probe_precisions: tuple[str, ...] = ("bf16", "nf4"),
    device: torch.device | None = None,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Produce a complete ``calibration.json`` for one run on this machine."""
    device = device or model_mod.pick_device()
    hp = {**train_mod.DEFAULT_HYPERPARAMS, **run_cfg.get("hyperparams", {})}
    rows = rows if rows is not None else data_mod.resolve_dataset(run_cfg["dataset_ref"])

    device_info = describe_device(device)

    fits = {}
    for precision in probe_precisions:
        fits[precision] = probe_fit(
            run_cfg["base_model"], precision, run_cfg["lora_cfg"],
            micro_batch=int(hp["micro_batch"]), device=device,
        )

    throughput = measure_throughput(
        run_cfg, rows, warmup_steps=warmup_steps,
        measure_steps=measure_steps, device=device,
    )

    return {
        "version": CALIBRATION_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run": {
            "base_model": run_cfg["base_model"],
            "base_precision": run_cfg["base_precision"],
            "lora_cfg": run_cfg["lora_cfg"],
            "dataset_ref": run_cfg["dataset_ref"],
            "num_buckets": run_cfg["num_buckets"],
            "dataset_rows": len(rows),
        },
        "hyperparams": hp,
        "device": device_info,
        # Keyed by device *then* precision, like `throughput`: a 3060 and a 4090
        # do not fit the same sequence lengths, so a fleet's calibration file has
        # to carry a fit answer per card rather than one for the run.
        "fits": {device_info["name"]: fits},
        # Keyed by device name because that is what a claiming worker reports and
        # what rounds.claim_task looks up. One file per card class; merging
        # several cards' files is a dict update, which is the whole reason the
        # shape is a map rather than a scalar.
        "throughput": {device_info["name"]: throughput["steps_per_min"]},
        "throughput_detail": {device_info["name"]: throughput},
        "recommended": recommend_local_steps(
            throughput["steps_per_min"], target_round_sec=target_round_sec
        ),
    }


def merge_calibration(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Fold a second card's calibration into an existing file.

    A heterogeneous fleet (6.8) needs one ``calibration.json`` covering several
    cards, and the cards are calibrated on different machines at different times.
    Merging is a dict update over the two keyed maps; everything else is taken
    from ``existing``, since the run config had better be identical and a
    mismatch there is an operator error worth refusing.
    """
    for field in ("base_model", "base_precision", "dataset_ref"):
        if existing["run"][field] != incoming["run"][field]:
            raise ValueError(
                f"refusing to merge calibrations for different runs: "
                f"{field} is {existing['run'][field]!r} vs {incoming['run'][field]!r}"
            )
    merged = dict(existing)
    merged["throughput"] = {**existing["throughput"], **incoming["throughput"]}
    merged["throughput_detail"] = {
        **existing.get("throughput_detail", {}),
        **incoming.get("throughput_detail", {}),
    }
    merged["fits"] = {**existing.get("fits", {}), **incoming.get("fits", {})}
    return merged


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ganymede-calibrate",
        description="Measure a run config on this machine and emit calibration.json.",
    )
    p.add_argument("--run-config", required=True, help="JSON file: base_model, base_precision, lora_cfg, dataset_ref, num_buckets, hyperparams")
    p.add_argument("--out", default="calibration.json")
    p.add_argument("--target-round-sec", type=int, default=2100, help="round length to size local_steps for (default 35 min)")
    p.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    p.add_argument("--measure-steps", type=int, default=DEFAULT_MEASURE_STEPS)
    p.add_argument("--probe-precisions", default="bf16,nf4", help="comma-separated; empty to skip the fit probe")
    p.add_argument("--device", default=None, help="override device selection, e.g. cuda:1 or cpu")
    p.add_argument("--merge-into", default=None, help="existing calibration.json to fold this card's numbers into")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    with open(args.run_config) as fh:
        run_cfg = json.load(fh)

    device = torch.device(args.device) if args.device else model_mod.pick_device()
    precisions = tuple(p for p in args.probe_precisions.split(",") if p.strip())

    if device.type != "cuda":
        print(
            f"warning: calibrating on {device.type}, not a GPU. The harness runs and the "
            f"output is well-formed, but the throughput figure describes this machine and "
            f"must not be used to size rounds for a fleet of GPUs.",
            file=sys.stderr,
        )

    result = calibrate(
        run_cfg,
        target_round_sec=args.target_round_sec,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        probe_precisions=precisions,
        device=device,
    )

    if args.merge_into:
        with open(args.merge_into) as fh:
            result = merge_calibration(json.load(fh), result)

    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")

    name = result["device"]["name"]
    print(f"calibrated {run_cfg['base_model']} on {name}")
    for precision, fit in result["fits"][name].items():
        if fit.get("ok"):
            print(f"  fits {precision}: max_seq_len {fit['max_seq_len']}, peak {fit['peak_vram_gb']} GB")
        else:
            print(f"  fits {precision}: no ({fit.get('error', 'out of memory at the smallest rung')})")
    detail = result["throughput_detail"][name]
    print(f"  throughput: {detail['steps_per_min']} steps/min, {detail['tokens_per_sec']} tok/s")
    print(f"  recommended local_steps: {result['recommended']['local_steps']} "
          f"for a {args.target_round_sec}s round")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
