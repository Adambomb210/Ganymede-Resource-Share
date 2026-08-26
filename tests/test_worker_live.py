"""One real worker against one real coordinator (M2's first exit criterion).

Everything else in the suite tests a seam: ``test_worker_loop`` drives the loop
against a stub client, ``test_integration`` drives the coordinator with a fake
worker. Neither exercises the join -- the real ``urllib`` client against the real
FastAPI app, real presigned URLs against real MinIO, the real trainer producing
an artifact the real gates accept.

Marked ``slow``: it needs Docker for MinIO and starts uvicorn as a subprocess.
The model is the ~107k-parameter Qwen3 from ``tiny_model_dir``, so the *training*
costs seconds; what takes the time is standing the stack up.

This is a scaled-down copy of a round run by hand during M2: a containerized
``ganymede/worker-llm`` completed 207 steps against this same stack under a
read-only rootfs with ``--cap-drop=ALL``. What could not be captured here is the
container itself -- that lives in CI (.github/workflows/ci.yml), which builds the
real Dockerfiles and smoke-tests them on Linux.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from tests.test_store import (  # noqa: F401 - imported for the fixture itself
    PUBLIC_ENDPOINT,
    ROOT_PASSWORD,
    ROOT_USER,
    minio_container,
)

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for(url: str, timeout: float = 60.0) -> bool:
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            pass
        time.sleep(0.5)
    return False


@pytest.fixture
def live_stack(minio_container, tmp_path, tiny_model_dir, tiny_lora_cfg, tiny_rows):
    """A real coordinator process, a real bucket, a real run, and a real key."""
    from ganymede.coordinator.auth import generate_key, hash_key
    from ganymede.coordinator.db import connect, init_schema
    from ganymede.coordinator import rounds

    port = _free_port()
    db_path = str(tmp_path / "live.db")
    bucket = f"ganymede-live-{uuid.uuid4().hex[:8]}"
    dataset = str(tmp_path / "rows.json")
    Path(dataset).write_text(json.dumps(tiny_rows))

    env = {
        **os.environ,
        "GANYMEDE_DB": db_path,
        "STORAGE_HOST": PUBLIC_ENDPOINT,
        "S3_BUCKET": bucket,
        "S3_ACCESS_KEY": ROOT_USER,
        "S3_SECRET_KEY": ROOT_PASSWORD,
        "COORDINATOR_HOST": f"http://127.0.0.1:{port}",
        "GANYMEDE_REQUIRE_TLS": "0",
        # Small enough that a short round still offers a usable budget.
        "GANYMEDE_MIN_USABLE_SEC": "5",
        "GANYMEDE_EST_DOWNLOAD_SEC": "1",
        "GANYMEDE_EST_UPLOAD_SEC": "1",
        "GANYMEDE_EST_SETUP_SEC": "1",
        "GANYMEDE_SAFETY_MARGIN_SEC": "1",
        "PYTHONPATH": str(REPO_ROOT),
    }

    key = generate_key()
    argv = [
        "--run-id", "live", "--base-model", tiny_model_dir, "--base-precision", "fp32",
        "--dataset", f"file://{dataset}", "--dataset-rows", str(len(tiny_rows)),
        "--eval-size", "40", "--num-buckets", "8",
        "--target-rounds", "2", "--target-steps", "3",
        "--min-round-sec", "0", "--max-round-sec", "600",
        "--lora-r", str(tiny_lora_cfg["rank"]), "--lora-alpha", str(tiny_lora_cfg["alpha"]),
        "--lora-dropout", "0.0",
        "--target-modules", ",".join(tiny_lora_cfg["target_modules"]),
        "--hyperparams", json.dumps({
            "seq_len": 32, "micro_batch": 2, "grad_accum": 1,
            "gradient_checkpointing": False, "cold_start_steps_per_min": 60.0,
        }),
    ]

    from ganymede.coordinator.config import Settings
    from ganymede.coordinator.store import Store
    from scripts import newrun

    saved = dict(os.environ)
    os.environ.update({k: v for k, v in env.items() if k != "PYTHONPATH"})
    try:
        settings = Settings.from_env()
        store = Store(settings.storage)
        assert newrun.main(argv, settings=settings, store=store) == 0

        conn = connect(db_path)
        init_schema(conn)
        conn.execute(
            """INSERT INTO contributors (id, name, key_hash, enabled, clearance, created_at)
               VALUES (?, 'live', ?, 1, 'open', ?)""",
            (uuid.uuid4().hex, hash_key(key), rounds._iso(rounds.utcnow())),
        )
        conn.commit()
        conn.close()
    finally:
        os.environ.clear()
        os.environ.update(saved)

    # A log file rather than subprocess.PIPE. An unread pipe holds about 64 KB
    # and uvicorn logs a line per request; once it fills, the coordinator blocks
    # on its next write and stops serving, which looks from the outside like a
    # hung worker. This test is short enough not to reach that, but the fixture
    # it is a template for is not (test_worker_concurrency), so it does not
    # model the hazard.
    server_log = tmp_path / "coordinator.log"
    server_handle = server_log.open("w")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "ganymede.coordinator.app:bootstrap",
         "--factory", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(REPO_ROOT), env=env,
        stdout=server_handle, stderr=subprocess.STDOUT, text=True,
    )
    try:
        if not _wait_for(f"http://127.0.0.1:{port}/healthz"):
            server.terminate()
            server.wait(timeout=5)
            server_handle.close()
            pytest.skip(f"coordinator did not start: {server_log.read_text()[-2000:]}")
        yield {"url": f"http://127.0.0.1:{port}", "key": key,
               "db": db_path, "rows": tiny_rows}
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)
        server_handle.close()


def _worker(live_stack, tmp_path, **config):
    from ganymede.worker.client import CoordinatorClient
    from ganymede.worker.control import ControlFiles
    from ganymede.worker.loop import Worker, WorkerConfig
    from ganymede.worker.probe import run_probe

    cfg = WorkerConfig(coordinator_url=live_stack["url"], key=live_stack["key"],
                       once=True, **config)
    return Worker(
        config=cfg,
        client=CoordinatorClient(cfg.coordinator_url, cfg.key),
        control=ControlFiles(tmp_path / "state", install_signal_handlers=False),
        profile=run_probe("cpu", skip_bench=True, skip_alloc=True),
    )


def _rounds(db_path: str) -> list[dict]:
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM rounds ORDER BY idx")]
    finally:
        conn.close()


def test_a_real_worker_completes_a_real_round(live_stack, tmp_path):
    """Register, claim, download, train, upload, submit -- over real HTTP.

    The presigned URL round trip is the part that cannot be faked: it is signed
    against a hostname, fetched by a client with no coordinator credentials, and
    is the one place M1's storage footgun (6.6) would bite.

    The dataset is resolved for real too, from the ``file://`` ref in the run
    config -- the worker never receives rows, only a ref and bucket indices.
    """
    worker = _worker(live_stack, tmp_path)
    assert worker.run() == 0
    assert worker.rounds_done == 1

    rounds_after = _rounds(live_stack["db"])
    assert rounds_after[0]["status"] == "closed"
    assert rounds_after[0]["result_adapter_ref"]
    assert rounds_after[0]["distinct_contributors"] == 1
    # The close published the next round's base adapter, which is what makes a
    # run a sequence rather than a single step.
    assert len(rounds_after) == 2
    assert rounds_after[1]["status"] == "open"
    assert rounds_after[1]["base_adapter_ref"] == rounds_after[0]["result_adapter_ref"]


def test_the_measured_throughput_lands_under_the_key_the_next_claim_reads(
    live_stack, tmp_path
):
    """The feedback loop, end to end.

    The worker names its device, the trainer reports that name as ``gpu_model``,
    the closer folds throughput in under it, and the next claim looks it up by
    the same string. Every one of those is a separate call site, and a mismatch
    is silent -- budgets simply never improve.
    """
    import sqlite3

    from ganymede.device import device_name
    import torch

    assert _worker(live_stack, tmp_path).run() == 0

    conn = sqlite3.connect(live_stack["db"])
    conn.row_factory = sqlite3.Row
    try:
        recorded = conn.execute("SELECT gpu_model, steps_per_min FROM throughput").fetchall()
    finally:
        conn.close()

    assert len(recorded) == 1
    assert recorded[0]["gpu_model"] == device_name(torch.device("cpu"))
    assert recorded[0]["steps_per_min"] > 0


def test_a_worker_on_the_wrong_image_gets_no_work_and_takes_no_lease(
    live_stack, tmp_path
):
    """Observed live: without this filter on the coordinator side, an ineligible
    worker is handed a lease it must immediately abandon -- once per poll
    interval, marking shards spoken for and churning bucket counters. Seven
    leases in under three minutes, all abandoned."""
    import sqlite3

    conn = sqlite3.connect(live_stack["db"])
    conn.execute("UPDATE runs SET required_image = 'ganymede/worker-llm:v9'")
    conn.commit()
    conn.close()

    worker = _worker(live_stack, tmp_path, image_tag="ganymede/worker-llm:v1")
    assert worker.run() == 0
    assert worker.rounds_done == 0

    conn = sqlite3.connect(live_stack["db"])
    try:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
    finally:
        conn.close()
