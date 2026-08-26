"""Shared fixtures for the coordinator suite.

The integration tests run against an **in-memory store** rather than a real
MinIO container. That is deliberate: ``tests/test_store.py`` already proves the
S3 layer against real MinIO, including the presigned-URL signing footgun, so
re-paying container startup for every state-machine test would buy nothing but
slower feedback. The fake implements exactly the portable subset ``Store``
exposes, so a divergence between them shows up as an AttributeError rather than
as a silently different behaviour.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import torch
from fastapi.testclient import TestClient

from ganymede.coordinator import rounds
from ganymede.coordinator.aggregate import save_adapter
from ganymede.coordinator.app import create_app
from ganymede.coordinator.auth import generate_key, hash_key
from ganymede.coordinator.config import Settings, StorageConfig
from ganymede.coordinator.db import connect, init_schema
from ganymede.coordinator.store import ObjectNotFound, base_adapter_key


class FakeStore:
    """In-memory stand-in for Store, matching its public surface exactly."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.cfg = StorageConfig(
            endpoint_url="http://storage.test:9000", bucket="ganymede",
            region="us-east-1", access_key="k", secret_key="s",
        )

    def ensure_bucket(self) -> None:
        pass

    def _expiry(self, expires_in: int | None) -> datetime:
        return datetime.now(timezone.utc) + timedelta(
            seconds=expires_in or self.cfg.presign_expiry_sec
        )

    def presign_put(self, key: str, expires_in: int | None = None):
        return f"{self.cfg.endpoint_url}/{self.cfg.bucket}/{key}?sig=put", self._expiry(expires_in)

    def presign_get(self, key: str, expires_in: int | None = None):
        return f"{self.cfg.endpoint_url}/{self.cfg.bucket}/{key}?sig=get", self._expiry(expires_in)

    def put_bytes(self, key: str, data: bytes) -> None:
        self.objects[key] = data

    def get_bytes(self, key: str) -> bytes:
        try:
            return self.objects[key]
        except KeyError as exc:
            raise ObjectNotFound(key) from exc

    def head(self, key: str):
        if key not in self.objects:
            return None
        return {"size": len(self.objects[key]), "etag": "x",
                "last_modified": datetime.now(timezone.utc)}

    def delete(self, key: str) -> None:
        self.objects.pop(key, None)

    def list_prefix(self, prefix: str) -> list[str]:
        return sorted(k for k in self.objects if k.startswith(prefix))


def make_adapter(scale: float = 1.0, dtype: torch.dtype = torch.float32,
                 seed: int | None = None) -> dict[str, torch.Tensor]:
    """A miniature stand-in for a LoRA adapter: two layers, A and B each."""
    g = torch.Generator().manual_seed(seed if seed is not None else 0)
    return {
        f"layers.{i}.{proj}.lora_{ab}.weight": (
            torch.randn(4, 8, generator=g) * scale
        ).to(dtype)
        for i in range(2)
        for proj in ("q_proj", "v_proj")
        for ab in ("A", "B")
    }


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        db_path=str(tmp_path / "coordinator.db"),
        storage=StorageConfig(endpoint_url="http://storage.test:9000", bucket="ganymede",
                              region="us-east-1", access_key="k", secret_key="s"),
        coordinator_host="http://coordinator.test",
        require_tls=False,          # the suite speaks plain HTTP; TLS has its own test
        lease_duration_sec=900,
        heartbeat_interval_sec=60,
        min_usable_sec=300,
        est_download_sec=60,
        est_upload_sec=30,
        # Zero in the suite so the existing budget expectations stay readable;
        # test_budget.py covers the reservation itself directly.
        est_setup_sec=0,
        safety_margin_sec=60,
        norm_reject_k=5.0,
        dominance_cap=2.0,
        throughput_floor_frac=0.10,
        gc_keep_rounds=3,
    )


@pytest.fixture
def conn(settings):
    c = connect(settings.db_path)
    init_schema(c)
    yield c
    c.close()


@pytest.fixture
def make_contributor(conn):
    """Create a contributor and return (contributor_id, plaintext_key)."""
    def _make(name: str = "tester", clearance: str = "open", enabled: bool = True):
        cid, key = uuid.uuid4().hex, generate_key()
        conn.execute(
            """INSERT INTO contributors (id, name, key_hash, enabled, clearance, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (cid, name, hash_key(key), 1 if enabled else 0, clearance,
             rounds._iso(rounds.utcnow())),
        )
        return cid, key
    return _make


@pytest.fixture
def app(settings, store, conn):
    return create_app(settings, store)


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def seeded_run(conn, store):
    """An active run with round 0 open and its base adapter in the store."""
    def _seed(
        run_id: str = "run1",
        num_buckets: int = 64,
        target_steps: int = 100,
        min_round_sec: int = 0,
        max_round_sec: int = 3600,
        target_rounds: int = 3,
        combine_mode: str = "mean",
        lr_outer: float = 1.0,
        outer_beta: float = 0.0,
        requires: dict | None = None,
        classification: str = "open",
        hyperparams: dict | None = None,
        required_image: str | None = None,
    ) -> str:
        hp = {"micro_batch": 8, "grad_accum": 1, "samples_per_bucket": 234,
              "target_passes": 1.0, "cold_start_steps_per_min": 30.0}
        hp.update(hyperparams or {})
        conn.execute(
            """INSERT INTO runs
                 (id, status, base_model, base_precision, lora_cfg_json, dataset_ref,
                  hyperparams_json, current_round, target_rounds, combine_mode,
                  lr_outer, outer_beta, requires_json, data_classification,
                  num_buckets, required_image, created_at)
               VALUES (?, 'active', 'Qwen/Qwen3-1.7B-Base', 'bf16', ?, 'dolly15k',
                       ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, json.dumps({"r": 16, "alpha": 32}), json.dumps(hp), target_rounds,
             combine_mode, lr_outer, outer_beta, json.dumps(requires or {}),
             classification, num_buckets, required_image,
             rounds._iso(rounds.utcnow())),
        )
        for b in range(num_buckets):
            conn.execute(
                "INSERT INTO buckets (run_id, bucket_idx, times_trained) VALUES (?, ?, 0)",
                (run_id, b),
            )
        base_ref = base_adapter_key(run_id, 0)
        store.put_bytes(base_ref, save_adapter(make_adapter(scale=0.01, seed=0)))
        rounds.open_round(conn, run_id, 0, base_ref, target_steps,
                          min_round_sec, max_round_sec)
        return run_id
    return _seed


@pytest.fixture
def register_worker(client):
    """Register a worker and return its id."""
    def _register(key: str, backend: str = "cuda", device: str = "RTX 3060",
                  vram_mb: int = 12288, supports: list[str] | None = None,
                  probe: dict | None = None, image_tag: str | None = None) -> str:
        resp = client.post(
            "/v1/workers/register",
            headers={"Authorization": f"Bearer {key}"},
            json={"image_tag": image_tag, "compute_profile": {
                "backend": backend, "device_name": device, "vram_mb": vram_mb,
                "supports": supports if supports is not None else ["bf16", "fp16", "nf4"],
                "probe": probe or {"alloc_max_mb": vram_mb - 1000, "bench_score": 40.0},
            }},
        )
        assert resp.status_code == 200, resp.text
        return resp.json()["worker_id"]
    return _register


# --------------------------------------------------------------------------
# A real-but-tiny model, so the trainer is testable offline in milliseconds
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """A genuine Qwen3 checkpoint of ~107k parameters, with a real tokenizer.

    This is the difference between a trainer test suite people run and one they
    skip. The alternative -- ``Qwen/Qwen3-0.6B-Base`` -- needs a network round
    trip and about a hundred seconds per load, which pushes every model-touching
    test behind the ``slow`` marker and therefore out of the normal loop.

    It is the *same architecture* the bring-up run uses (``Qwen3ForCausalLM``,
    uniform full-attention layers), so the LoRA target modules, the parameter
    naming, and the shapes of everything are the production ones. What it is not
    is a model that can learn anything, so nothing here asserts about loss going
    down; that claim needs the real model and lives in the slow suite.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3Config

    path = tmp_path_factory.mktemp("tiny-qwen3")

    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    vocab.update({f"w{i}": i + 4 for i in range(200)})
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>",
        pad_token="<pad>", bos_token="<bos>", eos_token="<eos>",
    ).save_pretrained(path)

    AutoModelForCausalLM.from_config(
        Qwen3Config(
            vocab_size=len(vocab), hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, max_position_embeddings=128,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
        )
    ).save_pretrained(path)

    return str(path)


@pytest.fixture
def tiny_lora_cfg():
    return {"rank": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]}


@pytest.fixture
def tiny_rows():
    """Synthetic rows in Dolly's shape, drawn from the tiny tokenizer's vocabulary."""
    import random

    rng = random.Random(0)
    def words(n):
        return " ".join(f"w{rng.randrange(200)}" for _ in range(n))

    return [
        {"instruction": words(6), "context": words(4) if i % 3 == 0 else "", "response": words(5)}
        for i in range(400)
    ]
