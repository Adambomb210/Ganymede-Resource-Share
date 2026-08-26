"""``ganymede-baseline`` -- the single-node reference M4 is measured against.

Why this exists at all (docs/03-roadmap.md, *Why the baseline can't be dropped*):
a distributed run that is quietly **worse than one GPU** looks exactly like one
that is working. The loss curve descends in both cases. Nothing errors. Without a
reference point you find out months later, or never -- and 5.2's research risk
(DiLoCo's outer step is characterized over full-parameter training, not LoRA
adapters) makes that a live possibility rather than a hypothetical.

Multi-seed, because "matches the baseline" needs a tolerance
------------------------------------------------------------
One baseline number gives you a judgement call: is 1.87 vs 1.84 a regression or
noise? Nobody knows, and the answer will be decided by whoever wants which
outcome. Running 2-3 seeds converts the question into a measurement -- the
run-to-run spread **is** the tolerance, and a distributed result inside that band
is indistinguishable from a rerun.

Two decisions in here worth stating plainly
-------------------------------------------
**Every seed starts from the same adapter -- the run's actual round-0 seed
adapter.** Varying the initialization too would widen the band, and a wider band
is a *weaker* test. The distributed run has exactly one initialization (the seed
adapter is minted once by ``newrun.py``), so the noise the baseline should
characterize is data order and dropout, which is what varying the training seed
varies.

**The baseline drives ``train.train_loop``**, the same loop a worker runs, in
segments with evaluation between them. Any difference between how the baseline
trains and how a worker trains lands directly in the number M4 rests on.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any

import torch

from ganymede.trainer import data as data_mod
from ganymede.trainer import evaluate as eval_mod
from ganymede.trainer import model as model_mod
from ganymede.trainer import train as train_mod

BASELINE_VERSION = 1

# Multiplier on the seed-to-seed standard deviation for the pass band. Two
# sigmas over a 3-seed sample is not a rigorous confidence interval and is not
# claimed as one; it is a stated, checkable convention, which is the property
# that matters here.
DEFAULT_TOLERANCE_K = 2.0


def run_baseline_seed(
    run_cfg: dict[str, Any],
    rows: list[dict[str, Any]],
    seed_adapter: dict[str, torch.Tensor],
    *,
    seed: int,
    total_steps: int,
    eval_every: int,
    eval_examples: int | None,
    device: torch.device | None = None,
    on_step=None,
    collect_smoke: bool = False,
) -> dict[str, Any]:
    """One single-node run: train ``total_steps``, evaluating along the way."""
    device = device or model_mod.pick_device()
    hp = {**train_mod.DEFAULT_HYPERPARAMS, **run_cfg.get("hyperparams", {})}
    grad_accum = max(1, int(hp["grad_accum"]))
    num_buckets = int(run_cfg["num_buckets"])

    torch.manual_seed(seed)

    partition = data_mod.plan_partition(
        n_rows=len(rows), num_buckets=num_buckets,
        eval_size=int(hp["eval_size"]), data_seed=int(hp["data_seed"]),
    )

    # The whole training set, which is what "single node" means: the baseline is
    # not sharded, so it holds every bucket.
    task = train_mod.Task(
        task_id=f"baseline-{seed}", run_id=run_cfg.get("run_id", "baseline"), round_idx=0,
        base_model=run_cfg["base_model"], base_precision=run_cfg["base_precision"],
        lora_cfg=run_cfg["lora_cfg"], dataset_ref=run_cfg["dataset_ref"],
        buckets=list(range(num_buckets)), num_buckets=num_buckets,
        hyperparams=hp, local_steps=total_steps, seed=seed,
    )

    tokenizer = model_mod.load_tokenizer(task.base_model)
    base = model_mod.load_base(task.base_model, task.base_precision, device=device)
    checkpointing = hp["gradient_checkpointing"]
    if checkpointing is None:
        checkpointing = device.type == "cuda"
    if checkpointing:
        base.enable_input_require_grads()
        base.gradient_checkpointing_enable()

    peft_model = model_mod.attach_lora(base, task.lora_cfg, init_from=seed_adapter)
    peft_model.train()

    params = model_mod.lora_params(peft_model)
    opt = torch.optim.AdamW(params, lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]))
    stream, _ = train_mod.build_stream(task, tokenizer, rows, partition)

    def evaluate_now() -> dict[str, Any]:
        return eval_mod.held_out_loss(
            peft_model, tokenizer, rows, partition.eval_indices(),
            seq_len=int(hp["seq_len"]), micro_batch=int(hp["micro_batch"]),
            prompt_format=hp["prompt_format"], completion_only=bool(hp["completion_only"]),
            device=device, limit=eval_examples,
        ).as_dict()

    curve: list[dict[str, Any]] = []
    # Step 0 is the seed adapter, which is a no-op on the base model (lora_B is
    # exactly zero), so this point is the untuned base model's held-out loss.
    # Recording it is what makes the rest of the curve interpretable.
    curve.append({"step": 0, **evaluate_now()})

    done = 0
    train_seconds = 0.0
    losses: list[float] = []
    while done < total_steps:
        chunk = min(eval_every, total_steps - done)
        outcome = train_mod.train_loop(
            peft_model=peft_model, stream=stream, params=params, opt=opt,
            steps=chunk, grad_accum=grad_accum,
            max_grad_norm=float(hp["max_grad_norm"]), device=device,
            on_step=on_step or train_mod._noop_on_step, step_offset=done,
        )
        done += outcome.steps
        train_seconds += outcome.train_seconds
        losses.extend(outcome.losses)
        curve.append({"step": done, **evaluate_now()})
        if outcome.steps < chunk:  # nothing left to draw from; should not happen
            break

    smoke = None
    if collect_smoke:
        smoke = eval_mod.smoke_generate(
            peft_model, tokenizer, prompt_format=hp["prompt_format"], device=device
        )

    return {
        "seed": seed,
        "curve": curve,
        "initial_loss": curve[0]["loss"],
        "final_loss": curve[-1]["loss"],
        "steps": done,
        "train_sec": round(train_seconds, 2),
        "steps_per_min": round(done / train_seconds * 60, 3) if train_seconds > 0 else 0.0,
        "train_loss_tail_mean": round(sum(losses[-20:]) / len(losses[-20:]), 5) if losses else None,
        "smoke": smoke,
    }


def summarize(seed_results: list[dict[str, Any]], k: float = DEFAULT_TOLERANCE_K) -> dict[str, Any]:
    """Turn per-seed curves into the band M4 is checked against."""
    finals = [r["final_loss"] for r in seed_results]
    mean = statistics.fmean(finals)
    stdev = statistics.stdev(finals) if len(finals) > 1 else 0.0

    # Align the curves point-by-point. They share a step grid by construction
    # (same total_steps, same eval_every), so a mismatch means someone merged
    # incompatible runs and the summary would be meaningless rather than merely
    # imprecise.
    grids = {tuple(p["step"] for p in r["curve"]) for r in seed_results}
    curve_summary: list[dict[str, Any]] = []
    if len(grids) == 1:
        for i, step in enumerate(next(iter(grids))):
            at = [r["curve"][i]["loss"] for r in seed_results]
            curve_summary.append({
                "step": step,
                "mean": round(statistics.fmean(at), 5),
                "stdev": round(statistics.stdev(at), 5) if len(at) > 1 else 0.0,
                "min": round(min(at), 5),
                "max": round(max(at), 5),
            })

    return {
        "seeds": len(finals),
        "final_mean": round(mean, 5),
        "final_stdev": round(stdev, 5),
        "final_min": round(min(finals), 5),
        "final_max": round(max(finals), 5),
        "curve": curve_summary,
        "tolerance": {
            "k": k,
            "upper_from_stdev": round(mean + k * stdev, 5),
            "observed_max": round(max(finals), 5),
            # The looser of the two, deliberately. With three seeds a small
            # standard deviation is not evidence that the noise is small -- it
            # is as likely to be three draws that happened to land close. A
            # distributed result inside the range the baseline *actually
            # produced* from noise alone cannot honestly be called a
            # regression, so the observed range is a floor on the band.
            "pass_if_final_loss_at_most": round(max(mean + k * stdev, max(finals)), 5),
        },
    }


def run_baseline(
    run_cfg: dict[str, Any],
    *,
    seeds: list[int],
    total_steps: int,
    eval_every: int,
    eval_examples: int | None = None,
    device: torch.device | None = None,
    rows: list[dict[str, Any]] | None = None,
    seed_adapter: dict[str, torch.Tensor] | None = None,
    tolerance_k: float = DEFAULT_TOLERANCE_K,
    verbose: bool = True,
) -> dict[str, Any]:
    device = device or model_mod.pick_device()
    rows = rows if rows is not None else data_mod.resolve_dataset(run_cfg["dataset_ref"])

    if seed_adapter is None:
        from scripts.newrun import build_seed_adapter

        seed_adapter = build_seed_adapter(run_cfg["base_model"], run_cfg["lora_cfg"])

    hp = {**train_mod.DEFAULT_HYPERPARAMS, **run_cfg.get("hyperparams", {})}
    results = []
    for seed in seeds:
        if verbose:
            print(f"seed {seed}: {total_steps} steps on {device.type}", flush=True)
        results.append(
            run_baseline_seed(
                run_cfg, rows, seed_adapter, seed=seed, total_steps=total_steps,
                eval_every=eval_every, eval_examples=eval_examples, device=device,
                collect_smoke=(seed == seeds[0]),
            )
        )
        if verbose:
            r = results[-1]
            print(f"  {r['initial_loss']:.4f} -> {r['final_loss']:.4f} "
                  f"({r['steps_per_min']} steps/min)", flush=True)

    return {
        "version": BASELINE_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run": {
            "base_model": run_cfg["base_model"],
            "base_precision": run_cfg["base_precision"],
            "lora_cfg": run_cfg["lora_cfg"],
            "dataset_ref": run_cfg["dataset_ref"],
            "num_buckets": run_cfg["num_buckets"],
            "dataset_rows": len(rows),
        },
        "protocol": {
            "total_steps": total_steps,
            "eval_every": eval_every,
            "eval_size": int(hp["eval_size"]),
            "eval_examples": eval_examples,
            "samples_per_step": int(hp["micro_batch"]) * int(hp["grad_accum"]),
            "prompt_format": hp["prompt_format"],
            "seeds": seeds,
            "device": device.type,
        },
        "hyperparams": hp,
        "per_seed": results,
        "summary": summarize(results, k=tolerance_k),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ganymede-baseline",
        description="Single-node multi-seed baseline; emits baseline.json.",
    )
    p.add_argument("--run-config", required=True)
    p.add_argument("--out", default="baseline.json")
    p.add_argument("--smoke-out", default=None, help="also write the seed-0 generation smoke set here")
    p.add_argument("--seeds", default="1,2,3", help="comma-separated training seeds (2-3 is the point)")
    p.add_argument("--steps", type=int, default=2000, help="optimizer steps per seed")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-examples", type=int, default=None, help="cap held-out examples per eval (default: the whole split)")
    p.add_argument("--tolerance-k", type=float, default=DEFAULT_TOLERANCE_K)
    p.add_argument("--device", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    with open(args.run_config) as fh:
        run_cfg = json.load(fh)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if len(seeds) < 2:
        print(
            "warning: a single seed gives a number, not a tolerance. M4's exit "
            "criterion needs a spread -- use 2-3 seeds.",
            flush=True,
        )

    device = torch.device(args.device) if args.device else model_mod.pick_device()
    result = run_baseline(
        run_cfg, seeds=seeds, total_steps=args.steps, eval_every=args.eval_every,
        eval_examples=args.eval_examples, device=device, tolerance_k=args.tolerance_k,
    )

    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")

    if args.smoke_out:
        smoke = next((r["smoke"] for r in result["per_seed"] if r["smoke"]), None)
        with open(args.smoke_out, "w") as fh:
            json.dump(smoke, fh, indent=2)
            fh.write("\n")

    s = result["summary"]
    print(f"baseline over {s['seeds']} seeds: final held-out loss "
          f"{s['final_mean']} +/- {s['final_stdev']} "
          f"(range {s['final_min']}-{s['final_max']})")
    print(f"  M4 passes if the distributed run's final loss is at most "
          f"{s['tolerance']['pass_if_final_loss_at_most']}")
    print(f"  wrote {args.out}" + (f" and {args.smoke_out}" if args.smoke_out else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
