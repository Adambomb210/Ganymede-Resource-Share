"""Tests for the scripts/ admin CLI (docs/03-roadmap.md M1).

Everything except test_build_seed_adapter_real_model runs offline against a
tiny local `Qwen3Config` (see the `tiny_base_model` fixture) so the
newrun.py-driven tests are fast and don't depend on HuggingFace being
reachable. The 224-tensor shape check against the real production model
(Qwen/Qwen3-1.7B-Base) is the one test that needs the network, and it skips
rather than fails when the Hub can't be reached.
"""

from __future__ import annotations

import json
import re
import sqlite3

import pytest
import torch

from ganymede.coordinator import rounds
from ganymede.coordinator.aggregate import load_adapter, save_adapter
from ganymede.coordinator.auth import hash_key
from ganymede.coordinator.store import adapter_key, base_adapter_key, momentum_key
from scripts import backup, gc, issue_key, newrun
from tests.conftest import FakeStore

TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj"


@pytest.fixture(scope="module")
def tiny_base_model(tmp_path_factory) -> str:
    """A microscopic, entirely local Qwen3 config.

    Enough for AutoModelForCausalLM.from_config to build a real module graph
    on the meta device -- so build_seed_adapter sees genuine peft-wrapped
    q/k/v/o_proj layers -- without ever touching the network, unlike the real
    `Qwen/Qwen3-1.7B-Base` id used in test_build_seed_adapter_real_model.
    """
    from transformers import Qwen3Config

    cfg = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64,
    )
    d = tmp_path_factory.mktemp("tiny-base-model")
    cfg.save_pretrained(d)
    return str(d)


def _newrun_argv(run_id: str, base_model: str, num_buckets: int = 5,
                 dataset_rows: int = 1000, eval_size: int = 100) -> list[str]:
    return [
        "--run-id", run_id,
        "--base-model", base_model,
        "--base-precision", "bf16",
        "--dataset", "dolly15k",
        "--num-buckets", str(num_buckets),
        "--dataset-rows", str(dataset_rows),
        "--eval-size", str(eval_size),
        "--target-rounds", "3",
        "--target-steps", "100",
        "--min-round-sec", "0",
        "--max-round-sec", "60",
        "--lora-r", "4",
        "--lora-alpha", "8",
        "--lora-dropout", "0.0",
        "--target-modules", TARGET_MODULES,
    ]


# --------------------------------------------------------------------------
# build_seed_adapter
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_build_seed_adapter_real_model() -> None:
    """The exact shape named in the spec: 224 LoRA tensors for
    Qwen3-1.7B-Base at r=16 over q/k/v/o_proj, B==0, A!=0, everything finite.
    """
    lora_cfg = {"rank": 16, "alpha": 32, "dropout": 0.05,
                "target_modules": TARGET_MODULES.split(",")}
    try:
        adapter = newrun.build_seed_adapter("Qwen/Qwen3-1.7B-Base", lora_cfg)
    except Exception as exc:  # network-dependent: config.json fetch from the Hub
        pytest.skip(f"HuggingFace Hub unreachable: {exc}")

    assert len(adapter) == 224
    for name, t in adapter.items():
        assert torch.isfinite(t).all(), f"{name} has non-finite values"
        if "lora_B" in name:
            assert torch.count_nonzero(t).item() == 0, f"{name} should be exactly zero"
        else:
            assert torch.count_nonzero(t).item() > 0, f"{name} should be non-zero"


def test_seed_adapter_roundtrips_through_safetensors(tiny_base_model: str) -> None:
    lora_cfg = {"rank": 4, "alpha": 8, "dropout": 0.0,
                "target_modules": TARGET_MODULES.split(",")}
    adapter = newrun.build_seed_adapter(tiny_base_model, lora_cfg, seed=42)

    restored = load_adapter(save_adapter(adapter))

    assert restored.keys() == adapter.keys()
    for name, tensor in adapter.items():
        assert restored[name].dtype == tensor.dtype
        assert torch.equal(restored[name], tensor.contiguous())


# --------------------------------------------------------------------------
# newrun.main
# --------------------------------------------------------------------------


def test_newrun_creates_run_buckets_round_and_adapter(conn, store, settings, tiny_base_model) -> None:
    run_id = "run-newrun-1"
    rc = newrun.main(_newrun_argv(run_id, tiny_base_model, num_buckets=5),
                      settings=settings, store=store)
    assert rc == 0

    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    assert row is not None
    assert row["status"] == "active"
    assert row["current_round"] == 0

    n_buckets = conn.execute(
        "SELECT COUNT(*) AS c FROM buckets WHERE run_id = ?", (run_id,)
    ).fetchone()["c"]
    assert n_buckets == 5

    rnd = conn.execute(
        "SELECT * FROM rounds WHERE run_id = ? AND idx = 0", (run_id,)
    ).fetchone()
    assert rnd is not None
    assert rnd["status"] == "open"
    assert rnd["base_adapter_ref"] == base_adapter_key(run_id, 0)

    assert base_adapter_key(run_id, 0) in store.objects


def test_newrun_refuses_to_clobber_existing_run(conn, store, settings, tiny_base_model) -> None:
    run_id = "run-newrun-dup"
    argv = _newrun_argv(run_id, tiny_base_model)
    assert newrun.main(argv, settings=settings, store=store) == 0

    objects_before = dict(store.objects)
    buckets_before = conn.execute(
        "SELECT COUNT(*) AS c FROM buckets WHERE run_id = ?", (run_id,)
    ).fetchone()["c"]
    round_before = dict(conn.execute(
        "SELECT * FROM rounds WHERE run_id = ? AND idx = 0", (run_id,)
    ).fetchone())

    rc = newrun.main(argv, settings=settings, store=store)
    assert rc != 0

    assert store.objects == objects_before
    buckets_after = conn.execute(
        "SELECT COUNT(*) AS c FROM buckets WHERE run_id = ?", (run_id,)
    ).fetchone()["c"]
    assert buckets_after == buckets_before
    round_after = dict(conn.execute(
        "SELECT * FROM rounds WHERE run_id = ? AND idx = 0", (run_id,)
    ).fetchone())
    assert round_after == round_before


def test_newrun_dry_run_writes_nothing(conn, store, settings, tiny_base_model) -> None:
    run_id = "run-newrun-dry"
    argv = _newrun_argv(run_id, tiny_base_model) + ["--dry-run"]
    rc = newrun.main(argv, settings=settings, store=store)
    assert rc == 0
    assert conn.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone() is None
    assert base_adapter_key(run_id, 0) not in store.objects


# --------------------------------------------------------------------------
# issue_key.main
# --------------------------------------------------------------------------


def test_issue_key_stores_only_hash(conn, settings, capsys) -> None:
    rc = issue_key.main(["--name", "alice", "--clearance", "internal"], settings=settings)
    assert rc == 0

    out = capsys.readouterr().out
    key = re.search(r"^key: (\S+)$", out, re.MULTILINE).group(1)
    cid = re.search(r"^id: (\S+)$", out, re.MULTILINE).group(1)

    row = conn.execute("SELECT * FROM contributors WHERE id = ?", (cid,)).fetchone()
    assert row is not None
    assert row["clearance"] == "internal"
    assert row["enabled"] == 1
    assert row["key_hash"] == hash_key(key)
    # The plaintext key must not be recoverable from any stored column.
    for col in row.keys():
        assert key not in str(row[col])


def test_issue_key_revoke_disables_but_keeps_row(conn, settings, capsys) -> None:
    issue_key.main(["--name", "bob"], settings=settings)
    cid = re.search(r"^id: (\S+)$", capsys.readouterr().out, re.MULTILINE).group(1)

    rc = issue_key.main(["--revoke", cid], settings=settings)
    assert rc == 0

    row = conn.execute("SELECT * FROM contributors WHERE id = ?", (cid,)).fetchone()
    assert row is not None  # revocation never deletes the row
    assert row["enabled"] == 0


# --------------------------------------------------------------------------
# gc.main
# --------------------------------------------------------------------------


def test_gc_deletes_only_old_worker_submissions(conn, store, settings) -> None:
    run_id = "gc-run"
    now = rounds._iso(rounds.utcnow())
    mref = momentum_key(run_id)
    conn.execute(
        """INSERT INTO runs
             (id, status, base_model, base_precision, lora_cfg_json, dataset_ref,
              hyperparams_json, current_round, target_rounds, combine_mode,
              lr_outer, outer_beta, outer_momentum_ref, requires_json,
              data_classification, num_buckets, created_at)
           VALUES (?, 'active', 'm', 'bf16', '{}', 'd', '{}', 3, 10,
                   'mean', 1.0, 0.0, ?, '{}', 'open', 4, ?)""",
        (run_id, mref, now),
    )
    store.put_bytes(mref, b"momentum")

    base_refs = {idx: base_adapter_key(run_id, idx) for idx in range(4)}
    for idx, ref in base_refs.items():
        store.put_bytes(ref, f"base-{idx}".encode())

    # Rounds 0-2 closed (each one's result_adapter_ref is the next round's base
    # -- same convention closer.py uses), round 3 still open.
    for idx in range(3):
        conn.execute(
            """INSERT INTO rounds
                 (run_id, idx, base_adapter_ref, status, target_steps,
                  min_round_sec, max_round_sec, opened_at, closed_at, result_adapter_ref)
               VALUES (?, ?, ?, 'closed', 10, 0, 60, ?, ?, ?)""",
            (run_id, idx, base_refs[idx], now, now, base_refs[idx + 1]),
        )
        for t in range(2):
            store.put_bytes(adapter_key(run_id, idx, f"task-{idx}-{t}"), b"submission")
    conn.execute(
        """INSERT INTO rounds
             (run_id, idx, base_adapter_ref, status, target_steps,
              min_round_sec, max_round_sec, opened_at)
           VALUES (?, 3, ?, 'open', 10, 0, 60, ?)""",
        (run_id, base_refs[3], now),
    )

    protected = set(base_refs.values()) | {mref}
    submission_keys = [k for k in store.objects if "/submissions/" in k]
    assert len(submission_keys) == 6

    # Neither a bare dry-run-flag nor a --yes without... wait, --yes alone
    # deletes; check that omitting --yes never deletes, and --dry-run always
    # wins even together with --yes.
    rc = gc.main(["--run-id", run_id, "--keep-rounds", "1"], settings=settings, store=store)
    assert rc == 0
    assert len([k for k in store.objects if "/submissions/" in k]) == 6

    rc = gc.main(["--run-id", run_id, "--keep-rounds", "1", "--yes", "--dry-run"],
                 settings=settings, store=store)
    assert rc == 0
    assert len([k for k in store.objects if "/submissions/" in k]) == 6

    rc = gc.main(["--run-id", run_id, "--keep-rounds", "1", "--yes"],
                 settings=settings, store=store)
    assert rc == 0

    remaining = [k for k in store.objects if "/submissions/" in k]
    # keep-rounds=1: only the most recent closed round (idx 2) keeps its grace
    # window; rounds 0 and 1's submissions are gc'd.
    assert len(remaining) == 2
    assert all(f"/rounds/{2:05d}/submissions/" in k for k in remaining)

    for ref in protected:
        assert ref in store.objects, f"{ref} must never be gc'd"


# --------------------------------------------------------------------------
# backup.main
# --------------------------------------------------------------------------


def test_backup_refuses_same_source_and_dest(settings, capsys) -> None:
    rc = backup.main(
        [
            "--dest-endpoint", settings.storage.endpoint_url,
            "--dest-bucket", settings.storage.bucket,
            "--dest-access-key", "some-other-key",
            "--dest-secret-key", "some-other-secret",
        ],
        settings=settings,
    )
    assert rc != 0
    err = capsys.readouterr().err.lower()
    assert "backup" in err


def test_backup_produces_restorable_sqlite_snapshot(conn, store, settings, tmp_path) -> None:
    dest = FakeStore()
    conn.execute(
        """INSERT INTO contributors (id, name, key_hash, enabled, clearance, created_at)
           VALUES (?, ?, ?, 1, 'open', ?)""",
        ("c1", "alice", "somehash", rounds._iso(rounds.utcnow())),
    )

    rc = backup.main(
        [
            "--dest-endpoint", "http://backup.test:9000",
            "--dest-bucket", "ganymede-backup",
            "--dest-access-key", "backup-key",
            "--dest-secret-key", "backup-secret",
        ],
        settings=settings, store=store, dest_store=dest,
    )
    assert rc == 0

    db_keys = [k for k in dest.objects if k.endswith("coordinator.db")]
    assert len(db_keys) == 1

    restored_path = tmp_path / "restored.db"
    restored_path.write_bytes(dest.objects[db_keys[0]])
    restored = sqlite3.connect(str(restored_path))
    try:
        row = restored.execute(
            "SELECT name FROM contributors WHERE id = ?", ("c1",)
        ).fetchone()
    finally:
        restored.close()
    assert row is not None
    assert row[0] == "alice"


def test_backup_copies_latest_result_and_momentum_refs(conn, store, settings) -> None:
    dest = FakeStore()
    run_id = "backup-run"
    now = rounds._iso(rounds.utcnow())
    mref = momentum_key(run_id)
    store.put_bytes(mref, b"momentum-bytes")

    conn.execute(
        """INSERT INTO runs
             (id, status, base_model, base_precision, lora_cfg_json, dataset_ref,
              hyperparams_json, current_round, target_rounds, combine_mode,
              lr_outer, outer_beta, outer_momentum_ref, requires_json,
              data_classification, num_buckets, created_at)
           VALUES (?, 'active', 'm', 'bf16', '{}', 'd', '{}', 1, 10,
                   'mean', 1.0, 0.0, ?, '{}', 'open', 4, ?)""",
        (run_id, mref, now),
    )
    base0 = base_adapter_key(run_id, 0)
    result0 = base_adapter_key(run_id, 1)
    store.put_bytes(base0, b"base-0")
    store.put_bytes(result0, b"result-0")
    conn.execute(
        """INSERT INTO rounds
             (run_id, idx, base_adapter_ref, status, target_steps,
              min_round_sec, max_round_sec, opened_at, closed_at, result_adapter_ref)
           VALUES (?, 0, ?, 'closed', 10, 0, 60, ?, ?, ?)""",
        (run_id, base0, now, now, result0),
    )

    rc = backup.main(
        [
            "--dest-endpoint", "http://backup.test:9000",
            "--dest-bucket", "ganymede-backup",
            "--dest-access-key", "backup-key",
            "--dest-secret-key", "backup-secret",
            "--run-id", run_id,
        ],
        settings=settings, store=store, dest_store=dest,
    )
    assert rc == 0
    assert dest.objects.get(result0) == b"result-0"
    assert dest.objects.get(mref) == b"momentum-bytes"


# --------------------------------------------------------------------------
# The data contract newrun derives (see ganymede/trainer/data.py)
# --------------------------------------------------------------------------


def test_newrun_derives_samples_per_bucket_from_the_real_partition(
    tiny_base_model, settings, store, tmp_path
):
    """The coordinator's budget arithmetic reads this number, and the workers get
    their rows from ``plan_partition``. Deriving it by *calling* plan_partition
    rather than by reimplementing its arithmetic is what keeps them equal."""
    from ganymede.trainer import data as data_mod

    argv = _newrun_argv("spb", tiny_base_model, num_buckets=7,
                        dataset_rows=1000, eval_size=90)
    assert newrun.main(argv, settings=settings, store=store) == 0

    conn = sqlite3.connect(settings.db_path)
    hp = json.loads(conn.execute(
        "SELECT hyperparams_json FROM runs WHERE id = 'spb'").fetchone()[0])

    expected = data_mod.plan_partition(
        n_rows=1000, num_buckets=7, eval_size=90, data_seed=0
    ).samples_per_bucket
    assert hp["samples_per_bucket"] == expected == (1000 - 90) // 7


def test_newrun_stores_everything_a_worker_needs_to_rebuild_the_partition(
    tiny_base_model, settings, store
):
    """``eval_size``, ``data_seed`` and ``prompt_format`` travel to every worker
    inside ``hyperparams``. Without them a worker cannot reconstruct the split,
    and the coordinator has no way to send the data itself."""
    argv = _newrun_argv("contract", tiny_base_model) + [
        "--data-seed", "4242", "--prompt-format", "dolly-v1",
    ]
    assert newrun.main(argv, settings=settings, store=store) == 0

    conn = sqlite3.connect(settings.db_path)
    hp = json.loads(conn.execute(
        "SELECT hyperparams_json FROM runs WHERE id = 'contract'").fetchone()[0])
    assert hp["eval_size"] == 100
    assert hp["data_seed"] == 4242
    assert hp["prompt_format"] == "dolly-v1"


def test_explicit_hyperparams_win_over_the_derived_defaults(tiny_base_model, settings, store):
    argv = _newrun_argv("override", tiny_base_model) + [
        "--hyperparams", json.dumps({"samples_per_bucket": 11, "lr": 3e-4}),
    ]
    assert newrun.main(argv, settings=settings, store=store) == 0

    conn = sqlite3.connect(settings.db_path)
    hp = json.loads(conn.execute(
        "SELECT hyperparams_json FROM runs WHERE id = 'override'").fetchone()[0])
    assert hp["samples_per_bucket"] == 11
    assert hp["lr"] == 3e-4


def test_newrun_refuses_a_partition_that_cannot_exist(tiny_base_model, settings, store, capsys):
    """More buckets than training rows is a run that could never dispatch a task;
    better to say so at creation than to fail every claim."""
    argv = _newrun_argv("impossible", tiny_base_model, num_buckets=500,
                        dataset_rows=200, eval_size=100)
    assert newrun.main(argv, settings=settings, store=store) == 1
    assert "fewer than one row" in capsys.readouterr().err


# --------------------------------------------------------------------------
# --from-config: one file for calibrate, baseline and newrun
# --------------------------------------------------------------------------


def _run_config(base_model: str, **overrides) -> dict:
    cfg = {
        "run_id": "fromcfg",
        "base_model": base_model,
        "base_precision": "bf16",
        "lora_cfg": {"rank": 4, "alpha": 8, "dropout": 0.05,
                     "target_modules": TARGET_MODULES.split(",")},
        "dataset_ref": "hf://databricks/databricks-dolly-15k",
        "dataset_rows": 1000,
        "num_buckets": 5,
        "hyperparams": {"lr": 3e-4, "micro_batch": 2, "grad_accum": 4,
                        "eval_size": 100, "data_seed": 77, "prompt_format": "dolly-v1"},
        "target_rounds": 3, "target_steps": 100,
        "min_round_sec": 0, "max_round_sec": 60,
        "classification": "open",
    }
    cfg.update(overrides)
    return cfg


def test_a_run_config_can_replace_every_flag(tiny_base_model, settings, store, tmp_path):
    """Calibrating one configuration and creating a run from another is a
    mistake with no symptom -- the run trains, its budgets are just sized for a
    model nobody measured. One file for all three tools removes the chance."""
    path = tmp_path / "run.json"
    path.write_text(json.dumps(_run_config(tiny_base_model)))

    assert newrun.main(["--from-config", str(path)], settings=settings, store=store) == 0

    conn = sqlite3.connect(settings.db_path)
    row = conn.execute(
        "SELECT base_model, base_precision, num_buckets, lora_cfg_json, hyperparams_json "
        "FROM runs WHERE id = 'fromcfg'").fetchone()
    assert row[0] == tiny_base_model
    assert row[1] == "bf16"
    assert row[2] == 5
    assert json.loads(row[3])["rank"] == 4
    hp = json.loads(row[4])
    assert hp["lr"] == 3e-4
    assert hp["data_seed"] == 77
    assert hp["samples_per_bucket"] == (1000 - 100) // 5


def test_explicit_flags_beat_the_config_file(tiny_base_model, settings, store, tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(_run_config(tiny_base_model)))

    assert newrun.main(
        ["--from-config", str(path), "--run-id", "flagwins", "--num-buckets", "10"],
        settings=settings, store=store,
    ) == 0

    conn = sqlite3.connect(settings.db_path)
    n = conn.execute("SELECT num_buckets FROM runs WHERE id = 'flagwins'").fetchone()[0]
    assert n == 10
    hp = json.loads(conn.execute(
        "SELECT hyperparams_json FROM runs WHERE id = 'flagwins'").fetchone()[0])
    assert hp["samples_per_bucket"] == (1000 - 100) // 10  # rederived, not copied


def test_a_stale_samples_per_bucket_in_the_file_is_ignored(tiny_base_model, settings, store, tmp_path):
    """It is a derived value. Trusting the file would let an edit to
    ``num_buckets`` that forgot to update it mis-size every budget in the run."""
    cfg = _run_config(tiny_base_model, run_id="stale")
    cfg["hyperparams"]["samples_per_bucket"] = 99999
    path = tmp_path / "run.json"
    path.write_text(json.dumps(cfg))

    assert newrun.main(["--from-config", str(path)], settings=settings, store=store) == 0

    conn = sqlite3.connect(settings.db_path)
    hp = json.loads(conn.execute(
        "SELECT hyperparams_json FROM runs WHERE id = 'stale'").fetchone()[0])
    assert hp["samples_per_bucket"] == (1000 - 100) // 5


def test_missing_settings_are_listed_by_flag_name(settings, store, capsys):
    assert newrun.main(["--run-id", "nope"], settings=settings, store=store) == 1
    err = capsys.readouterr().err
    assert "--base-model" in err and "--dataset-rows" in err


def test_an_unreadable_config_fails_before_touching_the_database(settings, store, capsys):
    assert newrun.main(["--from-config", "/nonexistent/run.json"],
                       settings=settings, store=store) == 1
    assert "--from-config" in capsys.readouterr().err


def test_the_checked_in_bringup_config_is_self_consistent():
    """``samples_per_bucket`` in the file is documentation of a derived value.
    If it drifts from what plan_partition computes, the file is stale and a
    reader would be misled about how the corpus is actually split."""
    import pathlib

    from ganymede.trainer import data as data_mod

    for name in ("bringup-1.7b.json", "cpu-probe-0.6b.json"):
        cfg = json.loads((pathlib.Path("configs") / name).read_text())
        partition = data_mod.plan_partition(
            n_rows=cfg["dataset_rows"], num_buckets=cfg["num_buckets"],
            eval_size=cfg["hyperparams"]["eval_size"],
            data_seed=cfg["hyperparams"]["data_seed"],
        )
        assert cfg["hyperparams"]["samples_per_bucket"] == partition.samples_per_bucket, name
        # 6.10's band: below ~100 a worker's shard is statistical noise.
        assert 100 <= partition.samples_per_bucket <= 500, name
