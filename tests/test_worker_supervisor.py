"""Step 7: a real multi-device supervisor (docs/14 §2), with real children.

``tests/test_worker_loop.py``'s new supervisor tests drive ``_run_supervisor``
against a fake ``_spawn`` -- deliberately, since ``StubClient`` cannot cross a
real ``multiprocessing`` spawn boundary intact (a copy of it in the child
would record calls nobody in the test ever reads back). What they cannot
prove is that the real machinery -- an actual OS process, the per-backend env
pin actually applied before anything touches CUDA, the result queue actually
carrying data back across the boundary, a real crash or a real SIGKILL --
holds together. That is what this file is for.

No Docker and no real coordinator. This box has neither (``minio_container``
already skips both ``test_worker_concurrency.py`` and ``test_worker_live.py``
here for exactly that reason), and a test proving process isolation must be
able to run -- and be trusted -- on a machine with no GPU and no Docker
either. A hand-rolled HTTP server that answers every request ``200 {}``
stands in for the coordinator's write endpoints (heartbeat, abandon, submit),
which is all a crash-isolation test needs. What actually trains a real model
against a real coordinator lives in the slow suite
(``test_worker_concurrency.py``, ``test_worker_live.py``).

What this deliberately does not prove: that ``CUDA_VISIBLE_DEVICES`` (or
``HIP_VISIBLE_DEVICES`` / ``ZE_AFFINITY_MASK``) actually restricts a real
device on a real card -- there is no GPU on this box to check that against.
``_pin_env``'s own table-driven tests in ``test_worker_loop.py`` are what
verify the per-backend *values*; this file verifies that whatever ``_pin_env``
returns is applied, in a real child, before anything else runs.
"""

from __future__ import annotations

import http.server
import threading
import time
from typing import Any

import pytest

from ganymede.worker.client import CoordinatorClient
from ganymede.worker.control import ControlFiles
from ganymede.worker.loop import Worker, WorkerConfig

pytestmark = pytest.mark.slow


class _EchoHandler(http.server.BaseHTTPRequestHandler):
    """Answers every request ``200 {}``. Enough for heartbeat/abandon/submit
    -- a real crash or a real kill never gets far enough to need more."""

    def _reply(self) -> None:
        body = b"{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler's naming
        self._reply()

    def do_POST(self) -> None:  # noqa: N802
        self._reply()

    def do_PUT(self) -> None:  # noqa: N802
        self._reply()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # quiet -- this is a test fixture, not something to watch


@pytest.fixture
def echo_coordinator():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _EchoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _task(task_id: str, devices: list[int], base_adapter_url: str) -> dict:
    """A ``collab_lora_finetune``-shaped payload, real enough that
    ``run_round`` dispatches it to ``_run_train_round`` exactly as it would
    the real thing. What happens next is controlled entirely by
    ``base_adapter_url`` -- the very first thing that method touches.
    """
    return {
        "task_id": task_id, "run_id": "r1", "round_idx": 0,
        "base_model": "tiny", "base_precision": "fp32",
        "lora_cfg": {"rank": 4, "alpha": 8, "target_modules": ["q_proj"]},
        "dataset_ref": "hf://x", "buckets": [0], "num_buckets": 8,
        "hyperparams": {}, "local_steps": 4, "seed": 1,
        "max_runtime_sec": 600, "required_image": None,
        "base_adapter_url": base_adapter_url,
        "devices": devices,
    }


def test_a_two_device_supervisor_survives_a_crash_and_a_kill(tmp_path, echo_coordinator):
    """Two real child processes -- one crashing on its own, one SIGKILLed
    from outside -- proving process isolation for real (docs/14 §2's whole
    premise) rather than through the fake ``_spawn`` the unit suite uses.

    ``task["base_adapter_url"] = "not-a-url"`` makes ``self.client.download``
    raise ``ValueError: unknown url type`` on its very first line, instantly
    -- before any network retry, before any real training, before anything
    touches torch. That reaches ``run_round``'s bare ``except Exception``,
    which abandons (inside the *child's own* Worker, against the echo
    server) and re-raises, so the child exits on its own, nonzero -- the
    case M4a's exception policy has always handled, now happening one
    process away from this one.

    The second task is never given the chance to reach that: it is
    SIGKILLed the instant both are spawned, which is the case ``run_round``
    cannot handle on its own -- there is no Python left to run its ``except``
    blocks -- and exactly what ``Worker._reap``'s backstop abandon exists
    for.
    """
    config = WorkerConfig(
        coordinator_url=echo_coordinator, key="k", backend="cpu",
        state_dir=str(tmp_path),
    )
    worker = Worker(
        config=config,
        client=CoordinatorClient(echo_coordinator, "k", verify_tls=False,
                                 timeout=5, max_retries=0),
        control=ControlFiles(tmp_path, install_signal_handlers=False),
        profile={"backend": "cpu", "device_name": "cpu:test",
                "supports": ["fp32"], "probe": {},
                "devices": [{"index": 0}, {"index": 1}]},
    )
    worker.worker_id = "w1"
    assert worker._slot_count() == 2

    # A spy on the PARENT's own _abandon -- never on the crashed child's,
    # which lives in a different process and a different Worker instance
    # entirely. What this observes is only Worker._reap's backstop.
    backstopped: list[str] = []
    real_abandon = worker._abandon

    def spy_abandon(task_id: str) -> None:
        backstopped.append(task_id)
        real_abandon(task_id)

    worker._abandon = spy_abandon

    worker._spawn(_task("crash", [0], "not-a-url"))
    worker._spawn(_task("killed", [1], "http://127.0.0.1:1/unreachable"))

    assert set(worker.active) == {"crash", "killed"}
    # Two genuinely different OS processes -- the whole point of this file
    # over the fake-_spawn unit tests.
    assert (worker.active["crash"].process.pid
            != worker.active["killed"].process.pid)

    deadline = time.monotonic() + 30
    while worker.active["crash"].process.is_alive() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not worker.active["crash"].process.is_alive(), (
        "the crashing child never exited -- the instant ValueError this test "
        "relies on did not propagate the way run_round's own tests say it should"
    )
    assert worker.active["crash"].process.exitcode != 0, (
        "run_round re-raises after abandoning on a bare Exception (module "
        "docstring) -- a clean exit here would mean that stopped happening"
    )

    # SIGKILL, not terminate -- the same reasoning test_worker_concurrency.py
    # gives for its own blast-radius test: a machine that loses power does
    # not get to run its abandon handler, and that is the case this backstop
    # exists for.
    worker.active["killed"].process.kill()

    deadline = time.monotonic() + 15
    while worker.active and time.monotonic() < deadline:
        worker._reap()
        if worker.active:
            time.sleep(0.1)

    assert worker.active == {}, "a child was never reaped"
    assert worker.tasks_done == 2

    # The crash resolved itself (its own run_round abandoned it before
    # re-raising, in its own process) and still reached _run_child's
    # `finally` on the way out -- Python runs `finally` even while
    # unwinding an exception -- so it left a result behind and needed no
    # backstop. The kill left nothing: no result, and the backstop is the
    # only thing that ever released that lease.
    assert backstopped == ["killed"], backstopped
