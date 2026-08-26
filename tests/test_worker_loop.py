"""The entrypoint loop (§4.2): declining, dropping, abandoning, and idling.

These drive the real ``Worker`` against a stub client. The trainer is stubbed
too — ``tests/test_worker_live.py`` covers the real one against a real
coordinator; what is under test here is the decision-making, which is where the
loop can be wrong in ways that cost a round rather than crash.
"""

from __future__ import annotations

import threading

import pytest

from ganymede.worker import loop as loop_mod
from ganymede.worker.client import CoordinatorError, LeaseLost, RoundClosed
from ganymede.worker.control import ControlFiles
from ganymede.worker.loop import Heartbeater, Worker, WorkerConfig


class StubClient:
    """Records calls; answers from a script."""

    def __init__(self, tasks=None, submit_response=None):
        self.tasks = list(tasks or [])
        self.submit_response = submit_response or {"accepted": True}
        self.calls: list[tuple] = []
        self.heartbeat_raises: Exception | None = None
        self.upload_raises: Exception | None = None

    def register(self, profile, image_tag=None):
        self.calls.append(("register", image_tag))
        return {"worker_id": "w1", "heartbeat_interval_sec": 5}

    def claim(self, worker_id, **kwargs):
        self.calls.append(("claim", kwargs.get("run_id")))
        if self.tasks:
            return self.tasks.pop(0), 0
        return None, 1

    def heartbeat(self, task_id, steps, loss=None):
        self.calls.append(("heartbeat", task_id, steps))
        if self.heartbeat_raises:
            raise self.heartbeat_raises
        return {}

    def download(self, url):
        self.calls.append(("download", url))
        return b"adapter"

    def upload_url(self, task_id):
        self.calls.append(("upload_url", task_id))
        return {"url": "http://storage/put", "key": f"runs/r/{task_id}"}

    def upload(self, url, data, content_type="application/octet-stream"):
        self.calls.append(("upload", len(data)))
        if self.upload_raises is not None:
            raise self.upload_raises

    def submit(self, task_id, key, steps, tokens_seen=0, metrics=None):
        self.calls.append(("submit", task_id, steps, metrics))
        return self.submit_response

    def abandon(self, task_id):
        self.calls.append(("abandon", task_id))
        return {"ok": True}

    def kinds(self) -> list[str]:
        return [c[0] for c in self.calls]


TASK = {
    "task_id": "t1", "run_id": "r1", "round_idx": 0,
    "base_model": "tiny", "base_precision": "fp32",
    "lora_cfg": {"rank": 4, "alpha": 8, "target_modules": ["q_proj"]},
    "dataset_ref": "hf://x", "buckets": [0], "num_buckets": 8,
    "hyperparams": {}, "local_steps": 4, "seed": 1,
    "max_runtime_sec": 600, "required_image": None,
    "base_adapter_url": "http://storage/get",
}

PROFILE = {"backend": "cpu", "device_name": "cpu:test", "supports": ["fp32", "bf16"],
           "probe": {}, "vram_mb": 8000}


def make_worker(tmp_path, client=None, **config_kwargs) -> Worker:
    return Worker(
        config=WorkerConfig(coordinator_url="http://c", key="k", **config_kwargs),
        client=client or StubClient(),
        control=ControlFiles(tmp_path, install_signal_handlers=False),
        profile=dict(PROFILE),
    )


# --------------------------------------------------------------------------
# Step 5: can we honor this task?
# --------------------------------------------------------------------------


def test_a_supported_task_is_honored(tmp_path):
    assert make_worker(tmp_path).can_honor(TASK) == (True, None)


def test_an_unsupported_precision_is_declined(tmp_path):
    """Silently training at a different precision would break 5.2's
    shared-frozen-base assumption without erroring anywhere."""
    worker = make_worker(tmp_path)
    honored, reason = worker.can_honor({**TASK, "base_precision": "nf4"})
    assert not honored
    assert loop_mod.DECLINE_PRECISION in reason
    assert "nf4" in reason


def test_a_mismatched_image_is_declined(tmp_path):
    worker = make_worker(tmp_path, image_tag="ganymede/worker-llm:v2")
    honored, reason = worker.can_honor({**TASK, "required_image": "ganymede/worker-llm:v3"})
    assert not honored
    assert loop_mod.DECLINE_IMAGE in reason


def test_a_native_worker_is_declined_from_a_container_only_run(tmp_path):
    """This is how 6.10 holds a restricted run to the container path: a native
    install has no image tag to match."""
    worker = make_worker(tmp_path, image_tag=None)
    honored, _ = worker.can_honor({**TASK, "required_image": "ganymede/worker-llm:v3"})
    assert not honored


def test_a_run_with_no_image_requirement_accepts_a_native_worker(tmp_path):
    worker = make_worker(tmp_path, image_tag=None)
    assert worker.can_honor({**TASK, "required_image": None})[0]


def test_declining_abandons_before_downloading_anything(tmp_path):
    """The expensive version of this failure is discovering it after a multi-GB
    base-model download, with the shard already marked leased."""
    client = StubClient(tasks=[{**TASK, "base_precision": "nf4"}])
    worker = make_worker(tmp_path, client=client, once=True)
    worker.worker_id = "w1"

    monkey_idle(worker)
    worker.run()

    assert "abandon" in client.kinds()
    assert "download" not in client.kinds()


def monkey_idle(worker):
    worker._idle = lambda seconds=0: None


# --------------------------------------------------------------------------
# Heartbeats
# --------------------------------------------------------------------------


@pytest.fixture
def fast_heartbeat(monkeypatch):
    """Drop the interval floor so a beat lands immediately.

    The floor itself is real and tested separately; here it would only mean
    every heartbeat test waits five seconds to observe a decision made in
    microseconds.
    """
    monkeypatch.setattr(loop_mod, "MIN_HEARTBEAT_INTERVAL_SEC", 0.01)


def test_a_409_on_heartbeat_marks_the_work_for_dropping(fast_heartbeat):
    """3.2: a closed round's work is dropped, not argued with."""
    client = StubClient()
    client.heartbeat_raises = RoundClosed("closed")
    beat = Heartbeater(client, "t1", interval_sec=0).start()
    beat._thread.join(timeout=5)

    assert beat.round_closed
    assert beat.should_drop()


def test_a_410_on_heartbeat_marks_the_lease_lost(fast_heartbeat):
    client = StubClient()
    client.heartbeat_raises = LeaseLost("expired")
    beat = Heartbeater(client, "t1", interval_sec=0).start()
    beat._thread.join(timeout=5)

    assert beat.lease_lost and beat.should_drop()


def test_a_transient_heartbeat_failure_does_not_drop_the_work(fast_heartbeat):
    """The client already retried, and the lease has slack. Losing it entirely
    surfaces as a 410 on a later beat -- treating one failed beat as fatal would
    throw away a round for a blip."""
    client = StubClient()
    client.heartbeat_raises = CoordinatorError(500, "boom", "http://c")
    beat = Heartbeater(client, "t1", interval_sec=0).start()

    import time

    time.sleep(0.2)
    beat.stop()
    assert not beat.should_drop()


def test_the_heartbeat_interval_has_a_floor():
    """A zero or tiny interval from a misconfigured coordinator would turn the
    heartbeat thread into a busy loop against the API, from every worker at once."""
    assert Heartbeater(StubClient(), "t1", interval_sec=0).interval \
        == loop_mod.MIN_HEARTBEAT_INTERVAL_SEC
    assert loop_mod.MIN_HEARTBEAT_INTERVAL_SEC >= 5


# --------------------------------------------------------------------------
# One round
# --------------------------------------------------------------------------


@pytest.fixture
def stub_trainer(monkeypatch):
    """Replaces run_task, since what is under test is the loop's decisions."""
    import types

    class Result:
        def __init__(self, steps=4, stopped_early=False):
            self.adapter_bytes = b"trained"
            self.steps = steps
            self.stopped_early = stopped_early
            self.metrics = {"steps": steps, "tokens": 128, "steps_per_min": 12.0}

    state = {"result": Result(), "on_step": None, "should_stop": None}

    def fake_run_task(task, base_adapter, on_step=None, should_stop=None, **kwargs):
        state["on_step"] = on_step
        state["should_stop"] = should_stop
        if on_step:
            on_step(0, 1.5)
        return state["result"]

    module = types.ModuleType("ganymede.trainer.train")
    module.run_task = fake_run_task

    class Task:
        def __init__(self, **kw):
            self.__dict__.update(kw)

        @classmethod
        def from_payload(cls, payload):
            return cls(base_model=payload["base_model"], **{})

    module.Task = Task
    monkeypatch.setitem(__import__("sys").modules, "ganymede.trainer.train", module)
    state["Result"] = Result
    return state


def test_a_finished_round_uploads_then_submits(tmp_path, stub_trainer):
    client = StubClient()
    worker = make_worker(tmp_path, client=client)
    response = worker.run_round(TASK)

    assert response == {"accepted": True}
    kinds = client.kinds()
    assert kinds.index("download") < kinds.index("upload_url") < kinds.index("upload") < kinds.index("submit")


def test_submitted_metrics_carry_the_measured_transfer_times(tmp_path, stub_trainer):
    """6.9: transfer rates are measured, not asked for. After one round the
    coordinator knows this machine's real bandwidth, which beats a number a
    contributor would have to look up and would often get wrong."""
    client = StubClient()
    make_worker(tmp_path, client=client).run_round(TASK)

    metrics = next(c[3] for c in client.calls if c[0] == "submit")
    assert "download_sec" in metrics and "upload_sec" in metrics
    assert metrics["artifact_bytes"] == len(b"trained")
    assert metrics["backend"] == "cpu"


def test_a_closed_round_drops_the_work_without_submitting(tmp_path, stub_trainer, fast_heartbeat):
    client = StubClient()
    client.heartbeat_raises = RoundClosed("closed")
    worker = make_worker(tmp_path, client=client)
    worker.heartbeat_interval = 0

    import time

    def slow_run_task(task, base_adapter, on_step=None, should_stop=None, **kwargs):
        time.sleep(0.3)  # let one heartbeat land
        return stub_trainer["Result"]()

    __import__("sys").modules["ganymede.trainer.train"].run_task = slow_run_task

    assert worker.run_round(TASK) is None
    assert "submit" not in client.kinds()


def test_being_told_to_stop_abandons_rather_than_racing_to_upload(tmp_path, stub_trainer):
    """4.4: Docker's default stop grace is 10 s and a half-uploaded artifact is
    worse than none. Abandoning releases the shard for immediate re-lease --
    strictly better for the swarm than a partial artifact of uncertain quality."""
    client = StubClient()
    worker = make_worker(tmp_path, client=client)
    worker.control.request_stop()

    assert worker.run_round(TASK) is None
    assert "abandon" in client.kinds()
    assert "submit" not in client.kinds()


def test_zero_steps_abandons_instead_of_submitting_nothing(tmp_path, stub_trainer):
    """An empty submission would consume a gate check and an aggregation slot to
    contribute nothing; gate 5 rejects it anyway (`no_steps`)."""
    stub_trainer["result"] = stub_trainer["Result"](steps=0)
    client = StubClient()

    assert make_worker(tmp_path, client=client).run_round(TASK) is None
    assert "abandon" in client.kinds()
    assert "submit" not in client.kinds()


def test_a_trainer_crash_releases_the_lease_before_propagating(tmp_path, stub_trainer):
    """Otherwise the shard stays leased until expiry for a worker that has
    already died -- a round's worth of a bucket nobody is training."""
    def explode(*args, **kwargs):
        raise RuntimeError("cuda assert")

    __import__("sys").modules["ganymede.trainer.train"].run_task = explode
    client = StubClient()

    with pytest.raises(RuntimeError):
        make_worker(tmp_path, client=client).run_round(TASK)
    assert "abandon" in client.kinds()


def test_a_failed_abandon_never_takes_the_worker_down(tmp_path, stub_trainer):
    """Abandoning is a courtesy that shortens the next worker's wait; the lease
    expires on its own regardless."""
    client = StubClient()
    client.abandon = lambda task_id: (_ for _ in ()).throw(CoordinatorError(500, "x", "u"))
    worker = make_worker(tmp_path, client=client)
    worker.control.request_stop()

    assert worker.run_round(TASK) is None  # no exception


# --------------------------------------------------------------------------
# The outer loop
# --------------------------------------------------------------------------


def test_no_work_is_normal_and_the_worker_keeps_polling(tmp_path):
    client = StubClient(tasks=[])
    worker = make_worker(tmp_path, client=client, once=True)
    monkey_idle(worker)

    assert worker.run() == 0
    assert client.kinds().count("claim") == 1


def test_a_stop_file_ends_the_loop_before_claiming(tmp_path):
    client = StubClient(tasks=[TASK])
    worker = make_worker(tmp_path, client=client)
    worker.control.request_stop()

    assert worker.run() == 0
    assert "claim" not in client.kinds()


def test_a_pause_file_keeps_the_worker_alive_but_idle(tmp_path, monkeypatch):
    """7.1: stay installed, take no new work."""
    client = StubClient(tasks=[TASK])
    worker = make_worker(tmp_path, client=client)
    worker.control.request_pause()

    slept = []
    monkeypatch.setattr(loop_mod.time, "sleep", lambda s: (slept.append(s), worker.control.request_stop()))

    assert worker.run() == 0
    assert "claim" not in client.kinds()
    assert slept == [loop_mod.PAUSE_POLL_SEC]


def test_a_claim_failure_is_survivable(tmp_path):
    """A coordinator restart must not take the fleet down with it (6.4)."""
    client = StubClient()
    client.claim = lambda *a, **k: (_ for _ in ()).throw(CoordinatorError(0, "refused", "u"))
    worker = make_worker(tmp_path, client=client)

    calls = {"n": 0}

    def idle(seconds=0):
        calls["n"] += 1
        if calls["n"] >= 2:
            worker.control.request_stop()

    worker._idle = idle
    assert worker.run() == 0
    assert calls["n"] >= 2


def _task_in_round(idx: int, task_id: str | None = None) -> dict:
    return {**TASK, "round_idx": idx, "task_id": task_id or f"t{idx}"}


def test_max_rounds_stops_a_long_running_worker(tmp_path, stub_trainer):
    client = StubClient(tasks=[_task_in_round(0), _task_in_round(1), _task_in_round(2)])
    worker = make_worker(tmp_path, client=client, max_rounds=2)
    monkey_idle(worker)

    assert worker.run() == 0
    assert worker.rounds_done == 2
    assert worker.tasks_done == 2


def test_several_tasks_inside_one_round_count_as_one_round(tmp_path, stub_trainer):
    """A worker that finishes its step budget while the round is still open
    claims again -- routinely several times over. ``--max-rounds 2`` has to
    mean two rounds, not two claims.

    It meant two claims until M4a, where three workers each took twenty tiny
    cold-start tasks inside round 0, hit ``--max-rounds 20``, and exited before
    the round they were all working had closed. The run never advanced and
    nothing reported a problem.
    """
    same_round = [_task_in_round(0, f"t0-{i}") for i in range(4)]
    client = StubClient(tasks=[*same_round, _task_in_round(1)])
    worker = make_worker(tmp_path, client=client, max_rounds=2)
    monkey_idle(worker)

    assert worker.run() == 0
    assert worker.rounds_done == 2      # round 0 and round 1
    assert worker.tasks_done == 5       # but five claims to get there


def test_a_round_whose_work_was_dropped_still_counts_as_worked(tmp_path, monkeypatch):
    """Otherwise ``--max-rounds`` quietly means "until N rounds happen to go
    your way" -- and a worker on an unlucky stretch never stops."""
    from ganymede.worker import loop as loop_mod

    client = StubClient(tasks=[_task_in_round(0), _task_in_round(1)])
    worker = make_worker(tmp_path, client=client, max_rounds=2)
    monkey_idle(worker)
    # Every round drops its work, as a round closing underneath the worker does.
    monkeypatch.setattr(loop_mod.Worker, "run_round", lambda self, task: None)

    assert worker.run() == 0
    assert worker.rounds_done == 2


def test_the_worker_remembers_which_base_models_it_has_cached(tmp_path, stub_trainer):
    """6.2: given two eligible runs the coordinator prefers the one whose base
    model this worker already holds -- seconds instead of a 16 GB download."""
    client = StubClient(tasks=[TASK])
    worker = make_worker(tmp_path, client=client, once=True)
    monkey_idle(worker)
    worker.run()

    assert "tiny" in worker.cached_base_models


def test_config_reads_the_documented_environment(monkeypatch):
    monkeypatch.setenv("GANYMEDE_COORDINATOR_URL", "https://c.example")
    monkeypatch.setenv("GANYMEDE_KEY", "secret")
    monkeypatch.setenv("GANYMEDE_IMAGE_TAG", "ganymede/worker-llm:v3")
    config = WorkerConfig.from_env()

    assert config.coordinator_url == "https://c.example"
    assert config.key == "secret"
    assert config.image_tag == "ganymede/worker-llm:v3"


def test_a_missing_key_fails_at_startup_with_a_name(monkeypatch, tmp_path):
    monkeypatch.delenv("GANYMEDE_KEY", raising=False)
    with pytest.raises(SystemExit, match="GANYMEDE_KEY"):
        Worker.create(WorkerConfig(coordinator_url="http://c", key=""))


def test_a_storage_outage_costs_the_round_not_the_worker(tmp_path, stub_trainer):
    """§6.4: an unreachable object store is something a worker rides out.

    Observed in M4a: MinIO went away mid-round and all three workers exited on
    the spot, on an unhandled exception out of the upload. On an unscheduled
    volunteer fleet that is the worst available failure -- the machines that
    were contributing stop permanently, and nobody is watching to notice.
    """
    from ganymede.worker.client import CoordinatorError

    client = StubClient(tasks=[_task_in_round(0), _task_in_round(1)])
    client.upload_raises = CoordinatorError(0, "http://storage/put", "connection refused")
    worker = make_worker(tmp_path, client=client, max_rounds=2)
    monkey_idle(worker)

    # Survives both failed rounds and exits on its own terms, not on a traceback.
    assert worker.run() == 0
    assert worker.rounds_done == 2
    # And gave each shard back rather than sitting on a lease it could not use.
    assert [c for c in client.calls if c[0] == "abandon"] != []


def test_an_unexpected_error_still_stops_the_worker(tmp_path, stub_trainer):
    """The other half of the split. A storage outage is infrastructure and
    passes; a bug in the worker is not something it can retry its way out of,
    and a machine failing every round forever while holding leases is worse
    than one the host agent restarts."""
    client = StubClient(tasks=[_task_in_round(0)])
    client.upload_raises = ValueError("something is genuinely wrong")
    worker = make_worker(tmp_path, client=client, max_rounds=2)
    monkey_idle(worker)

    with pytest.raises(ValueError, match="genuinely wrong"):
        worker.run()
