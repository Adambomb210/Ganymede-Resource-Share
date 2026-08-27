"""Phase A golden fixture — Part B: the fixed-seed CUDA loss trace.

Fixture 1 of docs/10-jobtype-sdk.md's inertness checklist, and the trace
`04-platform-expansion.md`'s M4b guardrail names as the Phase A entry criterion.

Runs `ganymede.trainer.train.run_task` on the real bring-up model
(Qwen3-1.7B-Base, bf16, gradient checkpointing) for 300 optimizer steps with a
fixed seed, on the 3060, and records every per-step training loss plus the
trained adapter's digest and its verdict through the real §5.1 gates.

After the Phase A move, re-run and compare: per-step loss within the tolerance
band recorded in the file (CUDA kernels are not bitwise deterministic); the gate
verdict and structural manifest exactly.

Writes tests/golden/phase_a_losstrace.json.  ~25 min on a contended 3060.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import platform
import time

import torch

from ganymede.coordinator import aggregate as A
from ganymede.trainer import data as D
from ganymede.trainer import train as T
from scripts.newrun import build_seed_adapter

OUT = pathlib.Path("tests/golden/phase_a_losstrace.json")

BASE_MODEL = "Qwen/Qwen3-1.7B-Base"
DATASET = "hf://databricks/databricks-dolly-15k"
LORA_CFG = {"rank": 16, "alpha": 32, "dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}
# From configs/bringup-1.7b.json, verbatim.
HYPERPARAMS = {
    "lr": 2e-4, "weight_decay": 0.0, "max_grad_norm": 1.0, "seq_len": 1024,
    "micro_batch": 2, "grad_accum": 8, "prompt_format": "dolly-v1",
    "completion_only": True, "eval_size": 750, "data_seed": 20260826,
    "samples_per_bucket": 222, "target_passes": 1.0,
    "gradient_checkpointing": None,   # -> on for CUDA
}
STEPS = 300
SEED = 20260827
BUCKETS = list(range(24))


def main() -> None:
    assert torch.cuda.is_available(), "this fixture is the CUDA trace; run it on the 3060"
    # Trim run-to-run kernel-selection variance without full deterministic mode
    # (which raises on some attention ops). The tolerance band still applies.
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    dev = torch.device("cuda")
    rows = D.resolve_dataset(DATASET)
    seed_adapter = build_seed_adapter(BASE_MODEL, LORA_CFG)
    seed_bytes = A.save_adapter(seed_adapter)
    seed_manifest = A.manifest_of(seed_adapter)

    task = T.Task(
        task_id="golden-phase-a", run_id="golden", round_idx=0,
        base_model=BASE_MODEL, base_precision="bf16", lora_cfg=LORA_CFG,
        dataset_ref=DATASET, buckets=BUCKETS, num_buckets=64,
        hyperparams=HYPERPARAMS, local_steps=STEPS, seed=SEED,
        max_runtime_sec=7200,
    )

    trace: list[dict] = []
    def on_step(step: int, loss: float) -> None:
        trace.append({"step": step, "loss": loss})

    t0 = time.monotonic()
    result = T.run_task(task, seed_bytes, on_step=on_step, rows=rows, device=dev)
    wall = time.monotonic() - t0

    gate, adapter = A.check_structural(result.adapter_bytes, seed_manifest, result.steps)
    peak_alloc_gb = round(torch.cuda.max_memory_allocated(0) / 1e9, 3)

    out = {
        "meta": {
            "purpose": "Phase A numerical-inertness golden fixture — the fixed-seed loss trace",
            "captured_against": "a1b4e36",
            "checklist": "docs/10-jobtype-sdk.md inertness checklist fixture 1; 04 M4b guardrail",
            "reproduce": "re-run tests/golden/capture_losstrace.py after the Phase A move",
            "seed": SEED, "steps": STEPS, "buckets": BUCKETS,
            "base_model": BASE_MODEL, "base_precision": "bf16",
            "lora_cfg": LORA_CFG, "hyperparams": HYPERPARAMS,
            "torch": torch.__version__,
            "device_name": torch.cuda.get_device_name(0),
            "platform": platform.platform(),
            "tf32_disabled": True, "cudnn_benchmark": False,
            "tolerance_band": {
                "per_step_loss_rel_err": 1e-2,
                "cumulative_mean_loss_rel_err": 1e-3,
                "basis": "conservative, not measured. bf16 + non-deterministic "
                         "attention/matmul kernels drift run-to-run even with "
                         "TF32 and cudnn.benchmark off. A verbatim function move "
                         "on the same torch/GPU lands well inside 1e-2 per step; "
                         "a real body change (reordered reduction, changed loss "
                         "scaling, mis-threaded seed) diverges 10-100%+. Tighten "
                         "to a measured band by diffing two fresh runs if the "
                         "gate ever needs it.",
            },
        },
        "run": {
            "steps_completed": result.steps,
            "stopped_early": result.stopped_early,
            "wall_seconds": round(wall, 2),
            "steps_per_min": round(result.steps / (wall / 60.0), 3),
            "metrics": result.metrics,
            "final_train_loss": trace[-1]["loss"] if trace else None,
            "mean_loss_all_steps": (sum(p["loss"] for p in trace) / len(trace)) if trace else None,
            "mean_loss_last_50": (sum(p["loss"] for p in trace[-50:]) / len(trace[-50:])) if trace else None,
            "peak_alloc_gb": peak_alloc_gb,
        },
        "gate": {
            "accepted": gate.accepted, "reason": gate.reason, "detail": gate.detail,
            "adapter_key_count": len(adapter) if adapter else None,
            "trained_adapter_digest": hashlib.sha256(result.adapter_bytes).hexdigest(),
            "seed_adapter_digest": hashlib.sha256(seed_bytes).hexdigest(),
            "manifest_key_count": len(seed_manifest),
        },
        "loss_trace": trace,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")
    print(f"  {result.steps} steps in {wall/60:.1f} min "
          f"({out['run']['steps_per_min']} steps/min), "
          f"loss {trace[0]['loss']:.4f} -> {trace[-1]['loss']:.4f}, "
          f"gate: {'accepted' if gate.accepted else gate.reason}")


if __name__ == "__main__":
    main()
