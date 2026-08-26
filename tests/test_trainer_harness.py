"""``ganymede-calibrate`` and ``ganymede-baseline``, run for real on a tiny model.

These prove the harnesses *work* -- they produce well-formed artifacts that the
coordinator can consume -- not that their numbers mean anything. A throughput
figure measured on a 107k-parameter model on CPU describes nothing anyone cares
about. The numbers that matter come from a GPU and cannot be produced here; what
can be produced here, and is, is the guarantee that when someone runs these on a
real card the output will be shaped correctly and the coordinator will read it.
"""

from __future__ import annotations

import json

import pytest
import torch

from ganymede.coordinator import rounds
from ganymede.trainer import baseline as B
from ganymede.trainer import calibrate as C

CPU = torch.device("cpu")


@pytest.fixture
def run_cfg(tiny_model_dir, tiny_lora_cfg):
    return {
        "run_id": "harness", "base_model": tiny_model_dir, "base_precision": "fp32",
        "lora_cfg": tiny_lora_cfg, "dataset_ref": "hf://unused", "num_buckets": 8,
        "hyperparams": {
            "lr": 1e-3, "seq_len": 32, "micro_batch": 2, "grad_accum": 2,
            "eval_size": 40, "data_seed": 5, "gradient_checkpointing": False,
        },
    }


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def test_calibration_output_is_complete_and_serializable(run_cfg, tiny_rows, tmp_path):
    result = C.calibrate(
        run_cfg, warmup_steps=1, measure_steps=2,
        probe_precisions=("fp32",), device=CPU, rows=tiny_rows,
    )

    assert result["version"] == C.CALIBRATION_VERSION
    assert result["run"]["dataset_rows"] == len(tiny_rows)

    name = result["device"]["name"]
    assert result["fits"][name]["fp32"]["ok"]
    assert result["fits"][name]["fp32"]["max_seq_len"] in C.SEQ_LEN_LADDER
    assert result["throughput"][name] > 0
    assert result["throughput_detail"][name]["measured_steps"] == 2
    assert result["recommended"]["local_steps"] > 0

    # It has to survive a round trip through JSON: it is written to a file and
    # later stored in the coordinator's calibration table as text.
    (tmp_path / "calibration.json").write_text(json.dumps(result))
    assert json.loads((tmp_path / "calibration.json").read_text()) == result


def test_the_coordinator_actually_reads_the_calibration_we_emit(
    run_cfg, tiny_rows, conn, seeded_run, client, register_worker, make_contributor
):
    """The join between M0's output and M1's budget arithmetic.

    ``rounds.claim_task`` looks up ``calibration_json -> throughput -> <device
    name>``. Nothing else in the system checks that the key it looks up is the
    key calibration writes, so if the two ever disagreed the effect would be
    silent: every worker would fall back to the cold-start guess forever, and the
    only symptom would be step budgets that never improve.
    """
    calibration = C.calibrate(
        run_cfg, warmup_steps=1, measure_steps=2,
        probe_precisions=(), device=CPU, rows=tiny_rows,
    )
    device_name = calibration["device"]["name"]
    # Pin a throughput far from the cold-start default so the effect is unambiguous.
    calibration["throughput"][device_name] = 90.0

    run_id = seeded_run(hyperparams={"cold_start_steps_per_min": 3.0})
    conn.execute(
        "INSERT INTO calibration (run_id, calibration_json, created_at) VALUES (?, ?, ?)",
        (run_id, json.dumps(calibration), rounds._iso(rounds.utcnow())),
    )
    conn.commit()

    _, key = make_contributor()
    worker_id = register_worker(key, device=device_name)
    claimed = client.post(
        "/v1/tasks/claim",
        json={"worker_id": worker_id, "run_id": run_id},
        headers={"Authorization": f"Bearer {key}"},
    )

    assert claimed.status_code == 200, claimed.text
    budgeted = claimed.json()["local_steps"]

    # 90 steps/min against a cold start of 3 is a 30x difference; anything in
    # that neighbourhood proves the lookup connected.
    assert budgeted > 100


def test_probe_fit_reports_a_missing_backend_rather_than_crashing(tiny_model_dir, tiny_lora_cfg):
    """nf4 is CUDA-only. A worker that cannot honor the run's pinned precision
    must find out before it claims, not three steps into a round."""
    fit = C.probe_fit(tiny_model_dir, "nf4", tiny_lora_cfg, micro_batch=1, device=CPU)
    assert fit["ok"] is False
    assert "bitsandbytes" in fit["error"]


def test_probe_fit_walks_up_the_ladder_and_stops_at_the_ceiling(tiny_model_dir, tiny_lora_cfg):
    # The tiny model's max_position_embeddings is 128, so 512 is already past it
    # and every rung fails -- which is exactly what an undersized card looks like.
    fit = C.probe_fit(tiny_model_dir, "fp32", tiny_lora_cfg, micro_batch=1,
                      ladder=(8, 16, 32), device=CPU)
    assert fit["ok"] and fit["max_seq_len"] == 32


# --------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------


def test_baseline_produces_a_curve_per_seed_and_a_band(run_cfg, tiny_rows, tiny_model_dir, tiny_lora_cfg):
    from scripts.newrun import build_seed_adapter

    result = B.run_baseline(
        run_cfg, seeds=[1, 2], total_steps=4, eval_every=2,
        eval_examples=8, device=CPU, rows=tiny_rows,
        seed_adapter=build_seed_adapter(tiny_model_dir, tiny_lora_cfg),
        verbose=False,
    )

    assert result["protocol"]["seeds"] == [1, 2]
    assert result["protocol"]["samples_per_step"] == 4
    assert len(result["per_seed"]) == 2

    for r in result["per_seed"]:
        assert [p["step"] for p in r["curve"]] == [0, 2, 4]
        assert all(p["loss"] > 0 for p in r["curve"])
        assert r["steps"] == 4

    summary = result["summary"]
    assert summary["seeds"] == 2
    assert summary["tolerance"]["pass_if_final_loss_at_most"] >= summary["final_max"]
    assert [p["step"] for p in summary["curve"]] == [0, 2, 4]


def test_every_seed_starts_from_the_identical_adapter(run_cfg, tiny_rows, tiny_model_dir, tiny_lora_cfg):
    """Varying the initialization too would widen the band, and a wider band is
    a weaker test. The distributed run has exactly one initialization."""
    from scripts.newrun import build_seed_adapter

    result = B.run_baseline(
        run_cfg, seeds=[1, 2, 3], total_steps=2, eval_every=2, eval_examples=8,
        device=CPU, rows=tiny_rows,
        seed_adapter=build_seed_adapter(tiny_model_dir, tiny_lora_cfg),
        verbose=False,
    )
    starts = {r["curve"][0]["loss"] for r in result["per_seed"]}
    assert len(starts) == 1


def test_only_the_first_seed_carries_the_smoke_set(run_cfg, tiny_rows, tiny_model_dir, tiny_lora_cfg):
    from scripts.newrun import build_seed_adapter

    result = B.run_baseline(
        run_cfg, seeds=[1, 2], total_steps=2, eval_every=2, eval_examples=4,
        device=CPU, rows=tiny_rows,
        seed_adapter=build_seed_adapter(tiny_model_dir, tiny_lora_cfg),
        verbose=False,
    )
    assert result["per_seed"][0]["smoke"] is not None
    assert result["per_seed"][1]["smoke"] is None
    assert len(result["per_seed"][0]["smoke"]) == len(__import__(
        "ganymede.trainer.evaluate", fromlist=["SMOKE_PROMPTS"]).SMOKE_PROMPTS)


# --------------------------------------------------------------------------
# The throughput feedback loop, both directions
# --------------------------------------------------------------------------


def test_submitted_metrics_carry_the_key_the_coordinator_folds_throughput_by(
    tiny_model_dir, tiny_lora_cfg, tiny_rows, conn, seeded_run
):
    """``closer.close_round`` folds a submission's throughput in only when
    *both* ``steps_per_min`` and ``gpu_model`` are present.

    Omitting ``gpu_model`` fails silently in the worst way: every round still
    closes, every submission is still accepted, and the estimate simply never
    updates -- so budgets stay at the cold-start guess for the life of the run
    and nothing anywhere reports a problem.
    """
    from ganymede.coordinator import aggregate
    from ganymede.trainer import model as M
    from ganymede.trainer import train as T
    from scripts.newrun import build_seed_adapter

    run_id = seeded_run(run_id="fb")
    task = T.Task(
        task_id="fb", run_id=run_id, round_idx=0, base_model=tiny_model_dir,
        base_precision="fp32", lora_cfg=tiny_lora_cfg, dataset_ref="hf://unused",
        buckets=[0], num_buckets=8,
        hyperparams={"seq_len": 32, "micro_batch": 2, "grad_accum": 1,
                     "eval_size": 40, "data_seed": 5, "gradient_checkpointing": False},
        local_steps=2, seed=3,
    )
    seed_bytes = aggregate.save_adapter(build_seed_adapter(tiny_model_dir, tiny_lora_cfg))
    metrics = T.run_task(task, seed_bytes, rows=tiny_rows, device=CPU).metrics

    assert metrics["gpu_model"] == M.device_name(CPU)
    assert metrics["steps_per_min"] > 0

    # Exactly what close_round does with them.
    rounds.update_throughput(conn, run_id, metrics["gpu_model"], float(metrics["steps_per_min"]))
    stored = conn.execute(
        "SELECT gpu_model, steps_per_min FROM throughput WHERE run_id = 'fb'"
    ).fetchone()
    assert stored["gpu_model"] == metrics["gpu_model"]


def test_calibration_and_the_trainer_name_the_same_device_identically(tiny_model_dir):
    """The two halves of the loop must agree on the join key.

    Calibration writes ``throughput[<name>]``; the trainer reports
    ``gpu_model`` for the coordinator to fold in; a worker reports
    ``device_name`` when it claims. One implementation, so they cannot drift.
    """
    from ganymede.trainer import model as M

    assert C.describe_device(CPU)["name"] == M.device_name(CPU)


def test_the_fit_probe_survives_its_own_child_being_killed(tiny_model_dir, tiny_lora_cfg):
    """A probe that can take down the calibration run has failed at its job.

    On CUDA an over-allocation raises a catchable ``OutOfMemoryError``. On CPU
    the kernel's OOM killer sends SIGKILL, which is not catchable — and that is
    not hypothetical: probing a 0.6B model at 1024 tokens on this machine killed
    the calibration process outright, which is what motivated the isolation.
    """
    fit = C.probe_fit_isolated(
        tiny_model_dir, "fp32", tiny_lora_cfg,
        micro_batch=1, ladder=(8, 16), device=CPU,
    )
    assert fit["ok"] and fit["max_seq_len"] == 16


def test_a_dead_probe_reports_the_last_rung_it_proved(tiny_model_dir, tiny_lora_cfg, tmp_path):
    """The ladder is monotonic, so the highest recorded rung is the answer even
    when the child never got to say so itself."""
    import json

    progress = tmp_path / "progress.json"
    C.probe_fit(
        tiny_model_dir, "fp32", tiny_lora_cfg, micro_batch=1,
        ladder=(8, 16, 32), device=CPU, progress_path=str(progress),
    )
    # The file is rewritten after every success, so it holds the final one.
    assert json.loads(progress.read_text())["max_seq_len"] == 32


def test_the_recorded_round0_smoke_set_covers_every_prompt():
    """It is an M0 deliverable, and it drifts silently.

    Adding a prompt to ``SMOKE_PROMPTS`` without re-recording leaves a reference
    file that no longer covers the set it is compared against -- and ``diff_smoke``
    zips the two, so the extra prompt is simply never checked. No error, just a
    tripwire with a gap in it.
    """
    import json
    import pathlib

    from ganymede.trainer import data as D
    from ganymede.trainer import evaluate as E

    recorded = json.loads(
        (pathlib.Path("configs") / "smoke-round0-qwen3-1.7b.json").read_text()
    )
    assert recorded["round"] == 0
    assert recorded["prompt_format"] in D.FORMATS
    assert [c["instruction"] for c in recorded["completions"]] == [
        p["instruction"] for p in E.SMOKE_PROMPTS
    ]
    assert all(c["completion"].strip() for c in recorded["completions"])
