"""Phase A golden fixtures — Part A: the deterministic CPU math.

Captures, on today's code (a1b4e36), the exact outputs of every aggregate / gate
function that Phase A relocates verbatim into
`ganymede/jobtypes/collab_lora_finetune/`. After the move, re-run and diff:
`combine` must be bit-identical, `check_*` verdict slugs and details must not
drift, `should_close` decisions must match over the replayed timeline.

Fixtures 2-5 of docs/10-jobtype-sdk.md's inertness checklist. Fixture 1 (the
CUDA loss trace) is golden_losstrace.py.

Writes tests/golden/phase_a_aggregate.json.
"""
from __future__ import annotations

import base64
import hashlib
import json
import pathlib
import sys

import torch

from ganymede.jobtypes.collab_lora_finetune import aggregate as A
from ganymede.jobtypes.collab_lora_finetune import plan as R
from ganymede.coordinator.db import connect, init_schema, immediate

OUT = pathlib.Path("tests/golden/phase_a_aggregate.json")
SEED = 20260827


def _digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _adapter(shape_spec, gen, scale=1.0):
    """A synthetic multi-tensor adapter. Not real LoRA shapes — the functions
    under test are pure per-tensor math and only care about a dict of tensors
    with a zero tensor and norm spread present."""
    return {name: (torch.randn(*shp, generator=gen) * scale).float()
            for name, shp in shape_spec}


SHAPES = [
    ("l0.q.A", (8, 32)), ("l0.q.B", (32, 8)),
    ("l0.k.A", (8, 32)), ("l0.k.B", (32, 8)),
    ("l1.v.A", (8, 32)), ("l1.v.B", (32, 8)),
    ("l1.o.A", (8, 32)), ("l1.o.B", (32, 8)),
    ("untouched", (16, 16)),   # left exactly == base by every worker -> zero-guard
]


def build():
    g = torch.Generator().manual_seed(SEED)
    base = _adapter(SHAPES, g)
    fx: dict = {"meta": {
        "purpose": "Phase A numerical-inertness golden fixtures (aggregate + gates + should_close)",
        "captured_against": "a1b4e36",
        "seed": SEED,
        "torch": torch.__version__,
        "checklist": "docs/10-jobtype-sdk.md inertness checklist, fixtures 2-5",
        "reproduce": "re-run tests/golden/capture_aggregate.py after the Phase A move; diff this file",
        "exact_equality_required": [
            "fixture_2.combine_adapter_digest", "fixture_2.dense_weights",
            "fixture_2.adapter_divergence", "fixture_3.check_norms",
            "fixture_3.dense_weights_capped", "fixture_3.combine_zero_guard_digest",
            "fixture_4.momentum_digest_reduce1", "fixture_4.momentum_digest_reduce2",
            "fixture_4.combine_digest_reduce2", "fixture_5.*.reason",
            "fixture_5.*.detail", "fixture_6.should_close",
        ],
        "tolerance": "none — CPU fp32, reduction order preserved by a verbatim move",
    }}

    # ---- fixture 2: reduce, small cohort (n=2) ----------------------------
    g2 = torch.Generator().manual_seed(SEED + 1)
    w2a = _adapter(SHAPES, g2, scale=0.10)
    w2b = _adapter(SHAPES, g2, scale=0.10)
    # both leave "untouched" exactly at base
    for w in (w2a, w2b):
        w["untouched"] = base["untouched"].clone()
    steps2 = [120, 80]
    keys = [n for n, _ in SHAPES]
    dw2 = A.dense_weights(steps2, keys, cap=2.0)
    comb2, mom2 = A.combine(base, [w2a, w2b], dw2, mode="mean", lr_outer=1.0, beta=0.0)
    fx["fixture_2"] = {
        "n": 2, "steps": steps2,
        "dense_weights": [round(list(d.values())[0], 12) for d in dw2],
        "adapter_divergence": A.adapter_divergence([w2a, w2b]),
        "combine_adapter_digest": _digest(A.save_adapter(comb2)),
        "combine_momentum_is_none": mom2 is None,
        "per_tensor_norms_combined": {k: round(torch.linalg.norm(v.float()).item(), 8)
                                      for k, v in comb2.items()},
    }

    # ---- fixture 3: reduce, gate-4 cohort (n=4, one norm outlier) ---------
    g3 = torch.Generator().manual_seed(SEED + 2)
    cohort = [_adapter(SHAPES, g3, scale=0.10) for _ in range(4)]
    for w in cohort:
        w["untouched"] = base["untouched"].clone()
    # make worker index 2 a norm outlier on one tensor (> 5x median)
    cohort[2]["l0.q.A"] = cohort[2]["l0.q.A"] * 50.0
    norm_verdicts = A.check_norms(cohort, k=5.0)
    steps3 = [200, 210, 5000, 190]   # worker 2 also over-reports steps -> cap engages
    dw3 = A.dense_weights(steps3, keys, cap=2.0)
    comb3, _ = A.combine(base, cohort, dw3, mode="mean", lr_outer=1.0, beta=0.0)
    # zero-guard: explicit per-tensor weights where nobody trained "untouched"
    # (w_sum == 0.0 for that key) -> combine must clone base through exactly.
    trained_keys = [k for k in keys if k != "untouched"]
    zg_weights = [{k: dw3[i][k] for k in trained_keys} for i in range(4)]
    comb3_zg, _ = A.combine(base, cohort, zg_weights, mode="mean", lr_outer=1.0, beta=0.0)
    fx["fixture_3"] = {
        "n": 4, "steps": steps3,
        "check_norms": [{"accepted": v.accepted, "reason": v.reason, "detail": v.detail}
                        for v in norm_verdicts],
        "dense_weights_capped": [round(list(d.values())[0], 12) for d in dw3],
        "adapter_divergence": A.adapter_divergence(cohort),
        "combine_digest": _digest(A.save_adapter(comb3)),
        "combine_zero_guard_digest": _digest(A.save_adapter(comb3_zg)),
        "zero_guard_untouched_is_exact_base": bool(
            torch.equal(comb3_zg["untouched"], base["untouched"])),
    }

    # ---- fixture 4: two consecutive reduces, beta != 0 -------------------
    g4 = torch.Generator().manual_seed(SEED + 3)
    r1 = [_adapter(SHAPES, g4, scale=0.08) for _ in range(3)]
    r2 = [_adapter(SHAPES, g4, scale=0.08) for _ in range(3)]
    for w in r1 + r2:
        w["untouched"] = base["untouched"].clone()
    dw_r1 = A.dense_weights([100, 100, 100], keys, cap=2.0)
    a1, m1 = A.combine(base, r1, dw_r1, mode="diloco", lr_outer=0.7, beta=0.9)
    dw_r2 = A.dense_weights([100, 100, 100], keys, cap=2.0)
    a2, m2 = A.combine(a1, r2, dw_r2, mode="diloco", lr_outer=0.7, beta=0.9, momentum=m1)
    fx["fixture_4"] = {
        "mode": "diloco", "lr_outer": 0.7, "beta": 0.9,
        "momentum_digest_reduce1": _digest(A.save_adapter(m1)),
        "momentum_digest_reduce2": _digest(A.save_adapter(m2)),
        "combine_digest_reduce1": _digest(A.save_adapter(a1)),
        "combine_digest_reduce2": _digest(A.save_adapter(a2)),
    }

    # ---- fixture 5: structural gates, malformed + valid -----------------
    expected = A.manifest_of(base)
    valid_bytes = A.save_adapter({k: v.clone() for k, v in base.items()})

    wrong_keys = {k: v for k, v in base.items() if k != "untouched"}
    wrong_keys["surprise"] = torch.zeros(4, 4)
    wrong_shape = {k: v.clone() for k, v in base.items()}
    wrong_shape["l0.q.A"] = torch.zeros(8, 33)
    nan_ad = {k: v.clone() for k, v in base.items()}
    nan_ad["l1.v.B"] = nan_ad["l1.v.B"].clone(); nan_ad["l1.v.B"][0, 0] = float("nan")

    cases = {
        "valid":              (valid_bytes,                       50, None),
        "not_safetensors":    (b"this is a pickle, not a tensor", 50, None),
        "truncated":          (valid_bytes[: len(valid_bytes) // 2], 50, None),
        "key_mismatch":       (A.save_adapter(wrong_keys),        50, None),
        "shape_mismatch":     (A.save_adapter(wrong_shape),       50, None),
        "non_finite":         (A.save_adapter(nan_ad),            50, None),
        "no_steps":           (valid_bytes,                        0, None),
        "steps_inconsistent": (valid_bytes,                       10, 400),
    }
    f5: dict = {}
    for name, (raw, steps, hb) in cases.items():
        gate, adapter = A.check_structural(raw, expected, steps, heartbeat_steps=hb)
        f5[name] = {"accepted": gate.accepted, "reason": gate.reason,
                    "detail": gate.detail, "adapter_returned": adapter is not None}
    fx["fixture_5"] = f5

    # ---- fixture 6: should_close over an injected now timeline ----------
    from datetime import datetime, timedelta, timezone
    conn = connect(":memory:")
    init_schema(conn)
    t0 = datetime(2026, 8, 27, 12, 0, 0, tzinfo=timezone.utc)
    with immediate(conn):
        conn.execute(
            "INSERT INTO runs (id, status, base_model, base_precision, lora_cfg_json, "
            "dataset_ref, hyperparams_json, target_rounds, num_buckets, created_at) "
            "VALUES ('r','active','m','bf16','{}','d','{}',20,64,?)", (t0.isoformat(),))
        conn.execute(
            "INSERT INTO rounds (run_id, idx, base_adapter_ref, status, target_steps, "
            "min_round_sec, max_round_sec, opened_at) "
            "VALUES ('r',0,'base','open',100,300,900,?)", (t0.isoformat(),))
        # one task + submission so round_progress sees steps + a submission
        conn.execute(
            "INSERT INTO tasks (id, run_id, round_idx, buckets_json, local_steps, status, created_at) "
            "VALUES ('t1','r',0,'[1]',60,'submitted',?)", (t0.isoformat(),))
        conn.execute(
            "INSERT INTO submissions (task_id, artifact_ref, steps_completed, received_at) "
            "VALUES ('t1','k',60,?)", (t0.isoformat(),))
    timeline = []
    for dt_sec, label in [(120, "before min"), (301, "past min, under target steps"),
                          (901, "past backstop")]:
        close, reason = R.should_close(conn, "r", 0, now=t0 + timedelta(seconds=dt_sec))
        timeline.append({"elapsed_sec": dt_sec, "note": label,
                         "close": close, "reason": reason})
    # now bump steps over target and re-check the min/target branch
    with immediate(conn):
        conn.execute("UPDATE submissions SET steps_completed = 150 WHERE task_id='t1'")
    for dt_sec, label in [(250, "steps>=target but under min"),
                          (350, "steps>=target and past min")]:
        close, reason = R.should_close(conn, "r", 0, now=t0 + timedelta(seconds=dt_sec))
        timeline.append({"elapsed_sec": dt_sec, "note": label,
                         "close": close, "reason": reason})
    fx["fixture_6"] = {"should_close": timeline}

    return fx


if __name__ == "__main__":
    fx = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fx, indent=2, sort_keys=True))
    print(f"wrote {OUT}")
    print(json.dumps(fx, indent=2, sort_keys=True))
