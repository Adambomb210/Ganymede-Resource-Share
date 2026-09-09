"""The entrypoint loop (§4.2): declining, dropping, abandoning, and idling.

These drive the real ``Worker`` against a stub client. The trainer is stubbed
too — ``tests/test_worker_live.py`` covers the real one against a real
coordinator; what is under test here is the decision-making, which is where the
loop can be wrong in ways that cost a round rather than crash.
"""

from __future__ import annotations

import json
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
        self.heartbeat_body: dict = {}
        self.upload_raises: Exception | None = None
        self.claims: list[dict] = []

    def register(self, profile, image_tag=None):
        self.calls.append(("register", image_tag))
        return {"worker_id": "w1", "heartbeat_interval_sec": 5}

    def claim(self, worker_id, **kwargs):
        self.calls.append(("claim", kwargs.get("run_id")))
        self.claims.append(dict(kwargs))
        if self.tasks:
            return self.tasks.pop(0), 0
        return None, 1

    def heartbeat(self, task_id, steps, loss=None):
        self.calls.append(("heartbeat", task_id, steps))
        if self.heartbeat_raises:
            raise self.heartbeat_raises
        return dict(self.heartbeat_body)

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


def test_a_cancel_on_the_heartbeat_is_latched(fast_heartbeat):
    """docs/11 §3 step 2: the cancel arrives on a heartbeat response and
    nowhere else. Latched rather than read once -- the beat that carries it is
    not the beat anyone is looking at."""
    client = StubClient()
    client.heartbeat_body = {"cancel": "soft"}
    beat = Heartbeater(client, "t1", interval_sec=0).start()

    import time

    time.sleep(0.2)
    beat.stop()
    assert beat.cancelled() == "soft"
    assert beat.should_drop()


def test_the_cancel_handler_runs_on_the_thread_that_heard_it(fast_heartbeat):
    """A hard cancel that waits for the training loop to come round is not a
    hard cancel, so the container is signalled from the heartbeat thread."""
    client = StubClient()
    client.heartbeat_body = {"cancel": "hard"}
    seen: list[str] = []
    beat = Heartbeater(client, "t1", interval_sec=0)
    beat.on_cancel = seen.append
    beat.start()

    import time

    time.sleep(0.2)
    beat.stop()
    assert seen[:1] == ["hard"]


def test_the_handler_fires_once_however_many_beats_carry_it(fast_heartbeat):
    client = StubClient()
    client.heartbeat_body = {"cancel": "soft"}
    seen: list[str] = []
    beat = Heartbeater(client, "t1", interval_sec=0)
    beat.on_cancel = seen.append
    beat.start()

    import time

    time.sleep(0.3)
    beat.stop()
    assert len(seen) == 1


def test_a_handler_that_raises_does_not_kill_the_heartbeat_thread(fast_heartbeat):
    """The cancel is still latched, and the loop still abandons. A failure to
    stop a container must not also cost the shard's clean release."""
    client = StubClient()
    client.heartbeat_body = {"cancel": "hard"}

    def explode(mode):
        raise RuntimeError("docker is wedged")

    beat = Heartbeater(client, "t1", interval_sec=0)
    beat.on_cancel = explode
    beat.start()

    import time

    time.sleep(0.2)
    beat.stop()
    assert beat.cancelled() == "hard"


def test_the_lease_crumb_is_written_on_each_beat(fast_heartbeat, tmp_path):
    """What the host agent reads to tell a live lease from a wedged worker
    (docs/11 §3)."""
    from ganymede.worker import sandbox

    client = StubClient()
    beat = Heartbeater(client, "t1", interval_sec=0, crumb_root=tmp_path,
                       container="ganymede-job-t1").start()

    import time

    time.sleep(0.2)
    beat.stop()
    crumb = sandbox.read_lease_crumb(tmp_path)
    assert crumb["task_id"] == "t1"
    assert crumb["container"] == "ganymede-job-t1"


def test_an_unwritable_crumb_does_not_cost_a_renewed_lease(fast_heartbeat, tmp_path,
                                                            monkeypatch):
    """Best-effort by construction: the beat succeeded, and a bookkeeping file
    that could not be written must not turn that into a dropped task."""
    from ganymede.worker import sandbox

    def explode(*args, **kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(sandbox, "write_lease_crumb", explode)
    client = StubClient()
    beat = Heartbeater(client, "t1", interval_sec=0, crumb_root=tmp_path).start()

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

    state = {"result": Result(), "on_step": None, "should_stop": None, "kwargs": {}}

    def fake_run_task(task, base_adapter, on_step=None, should_stop=None, **kwargs):
        state["on_step"] = on_step
        state["should_stop"] = should_stop
        state["kwargs"] = kwargs
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


def test_a_paused_worker_gives_the_card_back(tmp_path, monkeypatch):
    """docs/02 §7.1 on the pause sentinel: *create the file and Ganymede stops
    taking the GPU*.

    Staying installed and claiming nothing is not enough once a worker keeps a
    model resident between tasks -- a paused worker still holding 16 GB is
    taking the GPU in the only sense the contributor cares about, and the kill
    switch that makes the whole ask reasonable would not have done what it says
    on the tin.
    """
    from ganymede.trainer.modelcache import ModelCache

    worker = make_worker(tmp_path, client=StubClient(tasks=[TASK]))
    worker.model_cache = ModelCache()
    worker.model_cache._models[("base", "tiny", "fp32", "cuda")] = object()
    worker.control.request_pause()
    monkeypatch.setattr(loop_mod.time, "sleep",
                        lambda s: worker.control.request_stop())

    assert worker.run() == 0
    assert worker.model_cache.stats()["resident"] == 0


def test_a_storage_outage_keeps_the_model_warm(tmp_path, monkeypatch):
    """The other side of the line above. A worker backing off a coordinator or
    object-store outage (M4a's lesson) has not been asked to stand down -- it is
    waiting out someone else's problem and will want the model back shortly, so
    dropping it there would turn a blip into a reload."""
    from ganymede.trainer.modelcache import ModelCache

    client = StubClient()
    client.claim = lambda *a, **k: (_ for _ in ()).throw(CoordinatorError(0, "down", "u"))
    worker = make_worker(tmp_path, client=client)
    worker.model_cache = ModelCache()
    worker.model_cache._models[("base", "tiny", "fp32", "cuda")] = object()

    calls = {"n": 0}

    def idle(seconds=0):
        calls["n"] += 1
        if calls["n"] >= 2:
            worker.control.request_stop()

    worker._idle = idle
    assert worker.run() == 0
    assert worker.model_cache.stats()["resident"] == 1


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


def test_the_worker_advertises_the_base_models_it_actually_loaded(tmp_path, stub_trainer):
    """6.2: given two eligible runs the coordinator prefers the one whose base
    model this worker already holds -- seconds instead of a 16 GB download.

    The set is now the model cache's, which is populated when a load *returns*.
    It used to be filled from the parsed payload before the load ran, so a
    worker whose 16 GB download died half way still advertised affinity for the
    model it did not have -- and the coordinator, believing it, preferentially
    sent it more of the same. ``batch_inference`` never fed the old set at all.
    """
    from ganymede.trainer.modelcache import ModelCache

    client = StubClient(tasks=[TASK])
    worker = make_worker(tmp_path, client=client, once=True)
    monkey_idle(worker)

    # Nothing has loaded yet, so a worker advertises nothing.
    worker.model_cache = ModelCache()
    assert worker._cached_base_models() == set()

    # A load that returned is what puts a ref in the set.
    worker.model_cache.loaded.add("tiny")
    assert worker._cached_base_models() == {"tiny"}

    worker.run()
    assert client.claims[0]["cached_base_models"] == ["tiny"]


def test_a_stubbed_trainer_loads_nothing_and_so_advertises_nothing(tmp_path, stub_trainer):
    """The other half of the claim above, and the reason the test beside it has
    to reach into the cache: a worker only advertises a model it really loaded,
    and ``stub_trainer`` never loads one."""
    client = StubClient(tasks=[TASK])
    worker = make_worker(tmp_path, client=client, once=True)
    monkey_idle(worker)
    worker.run()

    assert worker._cached_base_models() == set()


def test_both_bodies_are_handed_the_same_process_lifetime_cache(tmp_path, stub_trainer):
    """docs/03's pre-M4b note: the point of the cache is that it outlives the
    task. A cache built per task would save nothing and still hold a model."""
    client = StubClient(tasks=[_task_in_round(0), _task_in_round(1)])
    worker = make_worker(tmp_path, client=client, max_rounds=2)
    monkey_idle(worker)
    worker.run()

    assert worker.tasks_done == 2
    assert worker.model_cache is not None
    # The same object the second task was handed -- not one built per task,
    # which would save nothing and still hold a model.
    assert stub_trainer["kwargs"]["cache"] is worker.model_cache


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


# --------------------------------------------------------------------------
# batch_inference: the second body (docs/10 §4, Phase E)
#
# Until Phase E this loop *was* the training body. A ``batch_inference`` claim
# reached ``task["base_adapter_url"]``, raised KeyError, and -- because
# ``run_round`` re-raises after abandoning -- took the worker down on the first
# one. The coordinator had been able to plan, serve, validate and close these
# jobs since Phase A's move; nothing on this side could run one, and the whole
# ``test_batch_inference`` suite was green throughout, because it drives the
# coordinator with a FakeWorker. That gap is what these tests are for.
# --------------------------------------------------------------------------


BATCH_TASK = {
    "task_id": "b1", "job_id": "j1", "job_type": "batch_inference",
    "run_id": None, "round_idx": None,
    "image_ref": None, "image_digest": None, "image_pull_url": None,
    "input_ref": json.dumps({
        "shard_ref": "shards/0", "shard_rows": 4, "model_ref": "hf://tiny",
        "prompt_template": "{input}", "decode": {"mode": "greedy", "max_new_tokens": 8},
        "output_schema": {"id": "str", "output": "str"},
        "output_key": "out/j1/b1.jsonl",
    }),
    "attempt_group": None,
    "artifacts": {"model": "hf://tiny", "shard": "http://storage/get?sig=shard"},
    "params": {
        "shard_ref": "shards/0", "shard_rows": 4,
        "output_put_url": "http://storage/put?sig=out", "output_key": "out/j1/b1.jsonl",
        "decode": {"mode": "greedy", "max_new_tokens": 8},
        "prompt_template": "{input}", "output_schema": {"id": "str", "output": "str"},
    },
    "sdk": {"job_type": "batch_inference", "version": 1},
    "max_runtime_sec": 600, "lease_expires_at": None,
    "heartbeat_interval_sec": 5, "required_image": None,
}


@pytest.fixture
def stub_infer(monkeypatch):
    """Replaces the type's ``run``. What is under test is the loop's decisions;
    the real body has its own suite in ``test_batch_inference``."""
    from ganymede.jobtypes.batch_inference import run as run_mod

    state = {"rows": 4, "should_stop": None, "signals": [], "calls": 0}

    def fake_run(task, inputs, on_step=None, should_stop=None, **kwargs):
        state["calls"] += 1
        state["task"] = task
        state["inputs"] = inputs
        state["should_stop"] = should_stop
        signal = should_stop() if should_stop else None
        state["signals"].append(signal)
        if on_step:
            on_step(state["rows"], 0.0)
        if signal == "hard":
            # What the real body does: abort with nothing uploaded.
            raise RuntimeError("batch_inference run aborted by a hard stop")
        rows = 1 if signal == "soft" else state["rows"]
        return run_mod.InferResult(
            rows=rows, output_ref="out/j1/b1.jsonl", digest="d" * 64,
            seconds=1.5, metrics={"rows": rows},
        )

    monkeypatch.setattr(run_mod, "run", fake_run)
    return state


def _latch(monkeypatch, **attrs):
    """Start the heartbeat with state already latched.

    ``heartbeat_body`` alone would not do it: the thread sleeps a full interval
    before its first beat, and the body asks ``should_stop`` immediately. What a
    cancel *arrives* as is covered by the Heartbeats section above; what these
    tests are about is the loop's reaction to it having arrived."""
    original = loop_mod.Heartbeater.start

    def start_latched(self):
        for name, value in attrs.items():
            setattr(self, name, value)
        return original(self)

    monkeypatch.setattr(loop_mod.Heartbeater, "start", start_latched)


def test_a_batch_task_no_longer_kills_the_worker(tmp_path, stub_infer):
    """The regression. Before the dispatch existed this raised KeyError on
    ``base_adapter_url`` after abandoning the lease, and ``run()`` re-raised."""
    client = StubClient()
    worker = make_worker(tmp_path, client=client)
    response = worker.run_round(BATCH_TASK)

    assert response == {"accepted": True}
    assert "abandon" not in client.kinds()


def test_the_worker_submits_the_derived_key_and_never_uploads_twice(tmp_path, stub_infer):
    """``run`` already PUT the output to the presigned URL the claim carried, so
    the loop asks for the upload slot and uses only its *key* -- the
    coordinator's derivation of the one place a submission may land. Submitting
    ``params["output_key"]`` on faith is what the made-up-key guard refuses."""
    client = StubClient()
    worker = make_worker(tmp_path, client=client)
    worker.run_round(BATCH_TASK)

    assert client.kinds().count("upload_url") == 1
    assert "upload" not in client.kinds()
    submit = next(c for c in client.calls if c[0] == "submit")
    assert submit[1] == "b1"
    assert submit[2] == 4  # rows are this type's "steps" on the wire


def test_the_submission_carries_the_digest(tmp_path, stub_infer):
    """The one metric here that is not diagnostics: ``compare_digest`` is what
    the coordinator's attempt-group agreement is computed from. Drop it and
    every redundant group reads as a disagreement -- silently, because a missing
    digest arrives as an empty string and two empties still differ from a real
    one."""
    client = StubClient()
    make_worker(tmp_path, client=client).run_round(BATCH_TASK)

    metrics = next(c for c in client.calls if c[0] == "submit")[3]
    assert metrics["digest"] == "d" * 64
    assert metrics["rows"] == 4
    # ``_infer_result_for`` on the coordinator reads exactly this key back.
    assert metrics["seconds"] == 1.5


def test_a_soft_cancel_abandons_rather_than_submitting_a_partial(
        tmp_path, stub_infer, monkeypatch):
    """A soft stop flushes what it has (docs/10 §4), which is fewer rows than
    the shard declared -- and ``validate`` rejects on the row count. Submitting
    one would spend an attempt to be told no.

    The ``signals`` assertion is the load-bearing half: ``should_drop()`` is
    true whenever a cancel is latched, so asking it before ``cancelled()``
    would fold this into "hard" and delete the soft path without failing
    anything else here."""
    _latch(monkeypatch, cancel_mode="soft")
    client = StubClient()
    worker = make_worker(tmp_path, client=client)
    worker.run_round(BATCH_TASK)

    assert stub_infer["signals"] == ["soft"], "a bool callback would have said 'hard'"
    assert "submit" not in client.kinds()
    assert "abandon" in client.kinds()


def test_a_hard_cancel_abandons_without_crashing_the_worker(
        tmp_path, stub_infer, monkeypatch):
    """``run`` raises on a hard stop. That is an abort we asked for, and it must
    not reach ``run_round``'s ``except Exception``, which re-raises."""
    _latch(monkeypatch, cancel_mode="hard")
    client = StubClient()
    worker = make_worker(tmp_path, client=client)

    assert worker.run_round(BATCH_TASK) is None
    assert stub_infer["signals"] == ["hard"]
    assert "submit" not in client.kinds()
    assert "abandon" in client.kinds()


def test_a_run_that_fails_on_its_own_still_reaches_the_handler(tmp_path, monkeypatch):
    """The other half of the same ``except RuntimeError``. Nothing asked this
    run to stop, so the abort is a real failure and must not be swallowed."""
    from ganymede.jobtypes.batch_inference import run as run_mod

    def explodes(task, inputs, on_step=None, should_stop=None, **kwargs):
        raise RuntimeError("the model would not load")

    monkeypatch.setattr(run_mod, "run", explodes)
    client = StubClient()
    with pytest.raises(RuntimeError, match="would not load"):
        make_worker(tmp_path, client=client).run_round(BATCH_TASK)
    assert "abandon" in client.kinds()


def test_a_dropped_lease_neither_submits_nor_abandons(tmp_path, stub_infer, monkeypatch):
    """The lease is already gone; there is nothing to give back."""
    _latch(monkeypatch, lease_lost=True)
    client = StubClient()
    assert make_worker(tmp_path, client=client).run_round(BATCH_TASK) is None
    assert "submit" not in client.kinds()
    assert "abandon" not in client.kinds()


def test_zero_rows_is_abandoned_rather_than_submitted(tmp_path, stub_infer):
    stub_infer["rows"] = 0
    client = StubClient()
    assert make_worker(tmp_path, client=client).run_round(BATCH_TASK) is None
    assert "submit" not in client.kinds()
    assert "abandon" in client.kinds()


def test_an_unknown_job_type_is_declined_rather_than_run(tmp_path):
    """A worker can be older than the coordinator. A type this build has no body
    for must be refused at step 5, where the reason is reported -- not
    discovered inside ``run_round``, which is where it used to become a
    KeyError."""
    worker = make_worker(tmp_path)
    honored, reason = worker.can_honor({**BATCH_TASK, "job_type": "dataset_map"})
    assert not honored
    assert loop_mod.DECLINE_JOBTYPE in reason
    assert "dataset_map" in reason


def test_a_task_that_pins_an_image_is_declined_by_either_body(tmp_path):
    """A task naming an image wants its body run *inside* it (docs/11 §2), and
    no body in this build does that -- ``sandbox.JobContainer`` is built and
    nothing here calls it. Running the in-tree body instead would not fail,
    which is the problem: the confinement the image exists to provide would be
    silently absent.

    Not theoretical. ``POST /v1/jobs`` accepts an ``image_id`` on any job type,
    the claim walk serves such a job to any worker reporting a container
    runtime, and ``required_image`` -- the field the check above looks at -- is
    a different field and null on that payload."""
    worker = make_worker(tmp_path)
    for task in (BATCH_TASK, TASK):
        honored, reason = worker.can_honor({**task, "image_ref": "img1",
                                            "image_pull_url": "http://s/img"})
        assert not honored
        assert loop_mod.DECLINE_CONTAINED in reason
        assert "img1" in reason


def test_a_payload_with_no_job_type_is_still_trained(tmp_path, stub_trainer):
    """A task row planned before docs/10 §1's dispatcher carries no job_type.
    It is collab_lora_finetune, and the default must not decline it."""
    worker = make_worker(tmp_path)
    assert worker.can_honor({k: v for k, v in TASK.items() if k != "job_type"}) == (True, None)
    assert worker.run_round(TASK)["accepted"] is True


def test_max_rounds_counts_shards_for_a_roundless_type(tmp_path, stub_infer):
    """batch_inference has no rounds (docs/10 §5), so there is no unit coarser
    than the task. ``int(task["round_idx"])`` on a None was the second way a
    batch claim killed a worker -- after ``run_round``, in ``run`` itself."""
    tasks = [{**BATCH_TASK, "task_id": f"b{i}"} for i in range(3)]
    client = StubClient(tasks=tasks)
    worker = make_worker(tmp_path, client=client, max_rounds=2)
    monkey_idle(worker)

    assert worker.run() == 0
    assert worker.rounds_done == 2
    assert worker.tasks_done == 2
