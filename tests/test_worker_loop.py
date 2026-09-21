"""The entrypoint loop (§4.2): declining, dropping, abandoning, and idling.

These drive the real ``Worker`` against a stub client. The trainer is stubbed
too — ``tests/test_worker_live.py`` covers the real one against a real
coordinator; what is under test here is the decision-making, which is where the
loop can be wrong in ways that cost a round rather than crash.
"""

from __future__ import annotations

import json
import queue
import threading

import pytest

from ganymede.worker import loop as loop_mod
from ganymede.worker import sandbox as sandbox_mod
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

    # `node_id` is accepted rather than swallowed by **kwargs on purpose:
    # this stub failing loudly when the real client's signature changed is
    # the suite noticing, and a stand-in that accepts anything notices
    # nothing.
    def register(self, profile, image_tag=None, node_id=None):
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
    crumb = sandbox.read_lease_crumb(tmp_path, "t1")
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


def test_skip_alloc_reaches_the_probe():
    """``skip_alloc`` was accepted by ``probe.run_probe`` since docs/14 §2's
    predecessor but had no way to reach it from a real worker start -- a
    4-card inventory probe is now four ceiling searches, and an operator who
    has already characterized the box needs to be able to skip all of them."""
    worker = Worker.create(WorkerConfig(
        coordinator_url="http://c", key="k", backend="cpu",
        skip_alloc=True, skip_bench=True,
    ))
    assert worker.profile["probe"]["method"] == "skipped"
    assert worker.profile["probe"]["alloc_max_mb"] is None


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


# --------------------------------------------------------------------------
# contained_batch (docs/10 §7, docs/11 §4)
# --------------------------------------------------------------------------

CONTAINED_TASK = {
    **{k: v for k, v in TASK.items() if k not in ("base_adapter_url",)},
    "task_id": "c1",
    "job_type": "contained_batch",
    "run_id": None,
    "round_idx": None,
    "image_ref": "img1",
    "image_digest": "sha256:" + "ab" * 32,
    "image_pull_url": "http://store/img1",
    "artifacts": {"shard": "http://store/shard"},
    "params": {"shard_ref": "in/0", "shard_rows": 2,
               "output_key": "out/j1/c1.jsonl", "output_schema": {"id": "str"},
               "output_put_url": "http://store/put", "params": {}},
}


def test_a_contained_task_is_honored_when_the_machine_has_a_runtime(tmp_path):
    worker = make_worker(tmp_path)
    worker.profile["container_runtime"] = "docker"
    assert worker.can_honor(CONTAINED_TASK) == (True, None)


def test_a_contained_task_without_an_image_is_declined(tmp_path):
    """The inverse of the refusal below, and the more dangerous direction. A
    contained type with no image has nothing to run, and the failure without
    this check is not a crash but an empty output that ``validate`` rejects on
    row count -- an attempt spent, and a reason naming the wrong thing."""
    worker = make_worker(tmp_path)
    worker.profile["container_runtime"] = "docker"
    honored, reason = worker.can_honor({**CONTAINED_TASK, "image_ref": None})
    assert not honored
    assert loop_mod.DECLINE_IMAGE_REQUIRED in reason


def test_a_contained_task_is_declined_by_a_machine_with_no_runtime(tmp_path):
    """The coordinator refuses this at claim off the profile this worker
    registered (docs/11 §4). Step 5 is where a *stale* profile surfaces -- a
    daemon that stopped since registration -- which is the case every other
    check here exists for."""
    worker = make_worker(tmp_path)
    worker.profile["container_runtime"] = None
    honored, reason = worker.can_honor(CONTAINED_TASK)
    assert not honored
    assert loop_mod.DECLINE_NO_RUNTIME in reason


def test_an_in_tree_type_still_refuses_an_image(tmp_path):
    """docs/11 §4's other direction, which ``contained_batch`` must not have
    loosened. ``POST /v1/jobs`` accepts an ``image_id`` on any job type, and an
    in-tree body handed one would run the task to completion, successfully,
    with exactly the confinement the image exists to provide absent."""
    worker = make_worker(tmp_path)
    worker.profile["container_runtime"] = "docker"
    for task in (BATCH_TASK, TASK):
        honored, reason = worker.can_honor({**task, "image_ref": "img1"})
        assert not honored
        assert loop_mod.DECLINE_CONTAINED in reason


@pytest.fixture
def stub_contained(monkeypatch):
    """Replaces ``contained_batch``'s ``run``; the loop's decisions are what is
    under test here, and the body itself is covered in test_contained_batch."""
    from ganymede.jobtypes.contained_batch import run as cb_run

    class Result:
        def __init__(self, rows=2):
            self.rows = rows
            self.digest = "d" * 64
            self.output_ref = "out/j1/c1.jsonl"
            self.seconds = 1.0
            self.exit_code = 0
            self.metrics = {"rows": rows}

    state = {"result": Result(), "raises": None, "signals": []}

    def fake_run(task, inputs, on_step=None, should_stop=None, **kw):
        if should_stop is not None:
            sig = should_stop()
            if sig:
                state["signals"].append(sig)
                raise cb_run.ContainedCancelled(sig)
        if state["raises"] is not None:
            raise state["raises"]
        if on_step:
            on_step(state["result"].rows, 0.0)
        return state["result"]

    monkeypatch.setattr(cb_run, "run", fake_run)
    return state


def _contained_worker(tmp_path, client, **kw):
    worker = make_worker(tmp_path, client=client, once=True, **kw)
    worker.profile["container_runtime"] = "docker"
    monkey_idle(worker)
    return worker


def test_a_contained_shard_is_submitted_through_the_same_path_as_a_batch_one(
    tmp_path, stub_contained
):
    """``ContainedResult`` reuses ``InferResult``'s field set so ``_submit_shard``
    carries it unchanged -- the cheapest real evidence the seam generalises. If
    a third type had needed a fourth submit path, that would have been the
    finding."""
    client = StubClient(tasks=[CONTAINED_TASK])
    worker = _contained_worker(tmp_path, client)
    assert worker.run() == 0

    assert [c[0] for c in client.calls] == [
        "register", "claim", "upload_url", "submit"]
    submit = [c for c in client.calls if c[0] == "submit"][0]
    assert submit[1] == "c1" and submit[2] == 2
    # The key is the coordinator's derivation, never params["output_key"].
    assert submit[3]["digest"] == "d" * 64
    assert not any(c[0] == "upload" for c in client.calls), "uploaded twice"


def test_run_contained_passes_the_workers_backend_not_the_tasks(
    tmp_path, monkeypatch
):
    """docs/14 §2: the container pin needs the *machine's* backend, which
    lives on ``self.profile`` -- the task payload carries only ``devices``
    (the per-lease half). ``_run_contained`` must supply both to
    ``contained_batch.run.run`` rather than leave the backend for
    ``sandbox.device_argv`` to guess at."""
    from ganymede.jobtypes.contained_batch import run as cb_run

    captured = {}

    def fake_run(task, inputs, on_step=None, should_stop=None, **kw):
        captured.update(kw)
        return type("R", (), {
            "rows": 1, "digest": "d" * 64, "output_ref": "out/j1/c1.jsonl",
            "seconds": 1.0, "exit_code": 0, "metrics": {"rows": 1},
        })()

    monkeypatch.setattr(cb_run, "run", fake_run)

    client = StubClient(tasks=[CONTAINED_TASK])
    worker = _contained_worker(tmp_path, client)
    worker.profile["backend"] = "cuda"
    assert worker.run() == 0

    assert captured.get("backend") == "cuda"


def test_a_contained_tasks_heartbeat_carries_its_container_name(
    tmp_path, stub_contained, monkeypatch
):
    """Without this, ``Heartbeater._crumb()`` always writes ``container:
    null`` and ``host.agent.reap_orphaned_jobs`` bails out immediately on it
    (``if not container: return []``) -- the wedged-worker backstop would
    never fire for a contained job. ``run_round`` must wire the name in for
    ``contained_batch``, deterministically from the task id, and nothing else
    needs to change for the other two job types to keep getting ``None``."""
    from ganymede.worker import sandbox

    real_init = Heartbeater.__init__
    captured: list[str | None] = []

    def spy_init(self, *args, **kwargs):
        captured.append(kwargs.get("container"))
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(Heartbeater, "__init__", spy_init)

    client = StubClient(tasks=[CONTAINED_TASK])
    worker = _contained_worker(tmp_path, client, job_scratch=str(tmp_path))
    assert worker.run() == 0

    # ``test_the_lease_crumb_is_written_on_each_beat`` already covers that a
    # ``Heartbeater`` given a container writes it into the crumb; what was
    # actually broken is that ``run_round`` never gave it one for real work,
    # which is what this pins -- at the constructor, not by racing the
    # background thread's own timer for a tick.
    assert captured == [sandbox.container_name_for("c1")]


def test_a_non_contained_tasks_heartbeat_carries_no_container(
    tmp_path, stub_trainer, monkeypatch
):
    """The inverse: ``collab_lora_finetune`` and ``batch_inference`` have no
    container to name, and ``None`` must stay correct for them."""
    captured: list[str | None] = []
    real_init = Heartbeater.__init__

    def spy_init(self, *args, **kwargs):
        captured.append(kwargs.get("container"))
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(Heartbeater, "__init__", spy_init)

    worker = make_worker(tmp_path, job_scratch=str(tmp_path))
    assert worker.run_round(TASK)["accepted"] is True
    assert captured == [None]


def test_the_heartbeat_thread_is_wired_to_cancel_the_job_container(
    tmp_path, stub_contained, monkeypatch
):
    """Defect 1: ``Heartbeater.on_cancel`` used to be assigned only in tests --
    in production nothing ever set it, so a latched cancel sat inert until
    ``_supervise``'s own poll came round. Exercised directly against
    ``_run_contained`` (rather than through a live thread) to avoid making the
    test a race against the real heartbeat interval; what it checks is that
    the handler ``_run_contained`` wires in actually reaches the container,
    which is the part that was missing."""
    from ganymede.worker import sandbox

    calls: list[tuple] = []

    class FakeJobContainer:
        def __init__(self, task_id, config):
            calls.append(("init", task_id))

        def cancel(self, mode, grace_sec=None):
            calls.append(("cancel", mode))

    monkeypatch.setattr(sandbox, "JobContainer", FakeJobContainer)
    monkeypatch.setenv("GANYMEDE_JOB_SCRATCH", str(tmp_path))

    client = StubClient()
    worker = _contained_worker(tmp_path, client, job_scratch=str(tmp_path))
    beat = Heartbeater(client, "c1", interval_sec=0)

    worker._run_contained(CONTAINED_TASK, beat, 0.0)

    assert beat.on_cancel is not None
    beat.on_cancel("hard")
    assert ("init", "c1") in calls
    assert ("cancel", "hard") in calls


def test_cancelling_before_the_container_exists_does_not_reach_the_worker(
    tmp_path, stub_contained, monkeypatch
):
    """The scenario the fix is really for: a cancel can land during the
    archive pull, before ``jt.run`` has called ``start`` -- there is no
    should_stop check during that download at all. ``JobContainer.cancel``
    against an unknown name is a nonzero exit, not a raise (``sandbox.py``'s
    kill path), so the handler must swallow whatever the real sandbox module
    does here without disturbing the round; this pins that a genuine
    ``SandboxError`` (an unset ``GANYMEDE_JOB_SCRATCH``, say) is caught inside
    the handler itself rather than left to ``Heartbeater``'s best-effort
    log-and-swallow around it."""
    monkeypatch.delenv("GANYMEDE_JOB_SCRATCH", raising=False)

    client = StubClient()
    worker = _contained_worker(tmp_path, client, job_scratch=str(tmp_path))
    beat = Heartbeater(client, "c1", interval_sec=0)

    worker._run_contained(CONTAINED_TASK, beat, 0.0)

    assert beat.on_cancel is not None
    beat.on_cancel("hard")  # must not raise despite no GANYMEDE_JOB_SCRATCH


@pytest.mark.parametrize("mode", ["soft", "hard"])
def test_a_cancelled_contained_task_abandons_and_never_submits(
    tmp_path, stub_contained, monkeypatch, mode
):
    """And the mode reaches the type unchanged. For every previous type soft and
    hard collapsed (docs/11 §3); folding them here would SIGKILL a job that was
    promised a grace period to checkpoint in."""
    client = StubClient(tasks=[CONTAINED_TASK])
    worker = _contained_worker(tmp_path, client)
    _latch(monkeypatch, cancel_mode=mode)

    assert worker.run() == 0
    assert stub_contained["signals"] == [mode]
    assert ("abandon", "c1") in client.calls
    assert not any(c[0] == "submit" for c in client.calls)


def test_a_broken_image_abandons_the_task_and_keeps_the_worker(
    tmp_path, stub_contained
):
    """The submitter's code exited non-zero. That is a verdict on the job, not
    on this machine, and it must not end the worker."""
    from ganymede.jobtypes.contained_batch.run import ContainedFailure

    client = StubClient(tasks=[CONTAINED_TASK])
    worker = _contained_worker(tmp_path, client)
    stub_contained["raises"] = ContainedFailure(3)

    assert worker.run() == 0
    assert ("abandon", "c1") in client.calls
    assert not any(c[0] == "submit" for c in client.calls)


def test_a_dead_docker_daemon_does_not_kill_the_worker(tmp_path, stub_contained):
    """``SandboxError`` is a ``RuntimeError`` subclass, so without an explicit
    handler it reaches ``run_round``'s ``except Exception`` -- which abandons
    **and re-raises**. A machine whose daemon stopped would exit permanently and
    quietly, which is exactly the M4a failure the handlers were written to
    prevent, arriving by a new route."""
    client = StubClient(tasks=[CONTAINED_TASK])
    worker = _contained_worker(tmp_path, client)
    stub_contained["raises"] = sandbox_mod.SandboxError("daemon is not running")

    assert worker.run() == 0
    assert ("abandon", "c1") in client.calls


def test_a_bad_archive_abandons_without_backing_off(tmp_path, stub_contained):
    """docs/11 §2.3: abandon, reason ``image_digest_mismatch``, re-queue. Not a
    verdict on the job and not on this host -- the bytes in flight were wrong
    and the next worker may pull them intact -- so unlike a dead daemon this one
    does not back off."""
    client = StubClient(tasks=[CONTAINED_TASK])
    worker = _contained_worker(tmp_path, client)
    idled = []
    worker._idle = lambda s=0: idled.append(s)
    stub_contained["raises"] = sandbox_mod.DigestMismatch("archive hashed x")

    assert worker.run() == 0
    assert ("abandon", "c1") in client.calls
    assert idled == [], "a bad archive is not a reason to stop claiming"


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


# --------------------------------------------------------------------------
# Step 7: a multi-device host runs a supervisor (docs/14 §2)
#
# Two kinds of test below. ``_pin_env`` and ``_slot_count`` are pure
# functions, tested directly. The supervisor loop itself is tested against a
# fake ``_spawn`` rather than real ``multiprocessing`` -- ``StubClient``
# cannot cross a real spawn boundary intact (a copy of it in the child would
# record calls nobody in this test ever reads back), so everything that
# checks *decisions* (claiming up to slot count, ``active_task_ids``, crash
# isolation, draining on stop) is written against a substitute that stays
# in-process, exactly the reasoning ``Worker._spawn``'s own docstring gives.
# A real multi-process run lives in ``tests/test_worker_supervisor.py``.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("backend,devices,expected", [
    ("cuda", [2], {"CUDA_VISIBLE_DEVICES": "2"}),
    ("cuda", [0, 3], {"CUDA_VISIBLE_DEVICES": "0,3"}),
    ("rocm", [1], {"HIP_VISIBLE_DEVICES": "1", "CUDA_VISIBLE_DEVICES": "1"}),
    ("xpu", [0], {"ZE_AFFINITY_MASK": "0"}),
    ("mps", [0], {}),
    ("cpu", [0], {}),
    ("cpu", [0, 1], {}),  # GANYMEDE_CPU_SLOTS > 1: no token exists to pin with.
])
def test_pin_env_matches_docs_14s_table(backend, devices, expected):
    # ``env={}`` rather than the ambient one: these assert the *table*, and a
    # developer who happens to have CUDA_VISIBLE_DEVICES set should not see
    # them fail. The composition that variable triggers is tested separately
    # below.
    assert loop_mod._pin_env(backend, devices, env={}) == expected


def test_pin_env_is_a_noop_with_nothing_to_pin():
    """No devices on the lease -- unreachable in practice (a lease always
    holds at least one), but the function must not invent a pin for a device
    that was never named."""
    assert loop_mod._pin_env("cuda", [], env={}) == {}


def test_pin_env_refuses_an_unrecognised_backend_with_devices_to_pin():
    """docs/14 §2: a backend with no known pinning form refuses to launch
    rather than falling back to 'all devices' -- written about the container
    pin column, but the reasoning is about the backend having no known pin at
    all, so it is applied here too (see the step report)."""
    assert loop_mod._pin_env("some_future_backend", [0, 1], env={}) is None


# --------------------------------------------------------------------------
# A visibility restriction already on the worker (docs/14 §2, review-added)
# --------------------------------------------------------------------------


def test_pin_env_composes_through_an_ambient_cuda_restriction():
    """``CUDA_VISIBLE_DEVICES`` does not nest: a process that sets it has the
    value read against the box's *full physical* device list, not against
    whatever an ancestor had already narrowed things to.

    ``probe.run_probe`` enumerates ``range(torch.cuda.device_count())``, which
    *is* narrowed -- so a worker launched with ``CUDA_VISIBLE_DEVICES=4,5,6,7``
    reports its four cards as 0-3, and those are the indices the coordinator
    allocates. Writing a lease's ``[2]`` straight through would land the child
    on physical card 2, a card deliberately withheld from Ganymede, instead of
    card 6.
    """
    env = {"CUDA_VISIBLE_DEVICES": "4,5,6,7"}
    assert loop_mod._pin_env("cuda", [2], env=env) == {"CUDA_VISIBLE_DEVICES": "6"}
    assert loop_mod._pin_env("cuda", [0, 3], env=env) == {"CUDA_VISIBLE_DEVICES": "4,7"}


def test_pin_env_composition_is_positional_so_uuid_lists_work_too():
    """An ambient list may name devices by UUID. A local index refers to a
    *position* in that list, so nothing needs to parse the entries."""
    env = {"CUDA_VISIBLE_DEVICES": "GPU-aaa,GPU-bbb,GPU-ccc"}
    assert loop_mod._pin_env("cuda", [1], env=env) == {"CUDA_VISIBLE_DEVICES": "GPU-bbb"}


def test_pin_env_refuses_a_local_index_the_ambient_list_cannot_resolve():
    """Out of range against the ambient list means there is no card this lease
    could legally run on -- refuse, the same as an unpinnable backend, rather
    than guess and land on hardware the worker does not hold."""
    assert loop_mod._pin_env("cuda", [4], env={"CUDA_VISIBLE_DEVICES": "4,5"}) is None


def test_pin_env_is_unchanged_when_nothing_is_ambient():
    """The ordinary deployment: no restriction set, every local index already
    physical, byte-for-byte today's value."""
    for var in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ZE_AFFINITY_MASK"):
        assert loop_mod._pin_env("cuda", [1, 2], env={var: ""}) ==             {"CUDA_VISIBLE_DEVICES": "1,2"}


def test_pin_env_composes_rocm_through_either_variable_name():
    """A ROCm box may carry the restriction under HIP's name or CUDA's; both
    resolve, and both names are written on the way out (a PyTorch ROCm build
    answers to either)."""
    assert loop_mod._pin_env("rocm", [1], env={"HIP_VISIBLE_DEVICES": "8,9"}) ==         {"HIP_VISIBLE_DEVICES": "9", "CUDA_VISIBLE_DEVICES": "9"}
    assert loop_mod._pin_env("rocm", [1], env={"CUDA_VISIBLE_DEVICES": "8,9"}) ==         {"HIP_VISIBLE_DEVICES": "9", "CUDA_VISIBLE_DEVICES": "9"}


def test_pin_env_composes_xpu_through_ze_affinity_mask():
    assert loop_mod._pin_env("xpu", [1], env={"ZE_AFFINITY_MASK": "2,3"}) ==         {"ZE_AFFINITY_MASK": "3"}


def test_slot_count_defaults_to_one_with_no_devices_reported(tmp_path):
    """A pre-multi-GPU profile (or a probe that failed to enumerate) --
    exactly today's single-device worker."""
    assert make_worker(tmp_path)._slot_count() == 1


def test_slot_count_is_one_for_a_single_reported_device(tmp_path):
    worker = make_worker(tmp_path)
    worker.profile["devices"] = [{"index": 0}]
    assert worker._slot_count() == 1


def test_slot_count_matches_the_reported_device_count(tmp_path):
    worker = make_worker(tmp_path)
    worker.profile["devices"] = [{"index": 0}, {"index": 1}, {"index": 2}]
    assert worker._slot_count() == 3


def test_a_single_device_worker_never_spawns_a_child(tmp_path, stub_trainer):
    """Backward compatibility, proved by construction rather than merely
    observed: ``run`` dispatches to ``_run_single`` at slot count 1, which is
    ``run``'s own old body, byte for byte, and never calls ``_spawn`` at
    all."""
    tasks = [_task_in_round(0), _task_in_round(1)]
    client = StubClient(tasks=tasks)
    worker = make_worker(tmp_path, client=client, max_rounds=2)
    monkey_idle(worker)
    spawned = []
    worker._spawn = lambda task: spawned.append(task["task_id"])

    assert worker.run() == 0
    assert spawned == []
    assert worker.rounds_done == 2
    assert worker.tasks_done == 2


class FakeProcess:
    """Stands in for ``multiprocessing.Process``: a real OS process is never
    started, so ``StubClient`` calls made through it stay visible to the
    test, the same way `_run_single`'s always have been."""

    def __init__(self):
        self._alive = True
        self.exitcode: int | None = None

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        pass

    def finish(self, exitcode: int = 0) -> None:
        self._alive = False
        self.exitcode = exitcode


class FakeQueue:
    """Stands in for the ``multiprocessing.Queue`` a real child reports
    through (``_run_child``'s ``finally``)."""

    def __init__(self):
        self._items: list = []

    def get_nowait(self):
        if not self._items:
            raise queue.Empty()
        return self._items.pop(0)

    def put(self, item) -> None:
        self._items.append(item)


def make_supervised_worker(tmp_path, client, devices: int = 2, **config_kwargs) -> Worker:
    """A ``Worker`` with a multi-device profile and a fake ``_spawn``, so the
    supervisor loop (``_run_supervisor``) runs for real while every child it
    creates is a ``FakeProcess``/``FakeQueue`` pair the test fully controls.

    ``worker._spawned`` records every task id ``_spawn`` was asked to start,
    in order -- the cheapest possible check that the loop tried to fill every
    slot it was supposed to.
    """
    profile = {**PROFILE, "devices": [{"index": i} for i in range(devices)]}
    worker = Worker(
        config=WorkerConfig(coordinator_url="http://c", key="k", **config_kwargs),
        client=client,
        control=ControlFiles(tmp_path, install_signal_handlers=False),
        profile=profile,
    )
    worker.worker_id = "w1"
    worker._spawned = []

    def fake_spawn(task):
        worker._spawned.append(task["task_id"])
        worker.active[task["task_id"]] = loop_mod._Child(task, FakeProcess(), FakeQueue())

    worker._spawn = fake_spawn
    return worker


def test_a_multi_device_worker_claims_up_to_its_slot_count_concurrently(tmp_path, monkeypatch):
    """Two devices, two tasks claimed and started before either has to
    finish -- and the second claim reports the first as already held
    (docs/14 §5.1's ``active_task_ids``), which is only reachable once a
    worker can hold more than one lease at a time."""
    tasks = [_task_in_round(0, "t0"), _task_in_round(0, "t1")]
    client = StubClient(tasks=tasks)
    worker = make_supervised_worker(tmp_path, client, devices=2)

    def fast_forward(seconds=0):
        # Reached only once both slots are full and there is nothing left to
        # claim -- finish both children (one reporting a loaded base model,
        # to pin down the affinity-hint plumbing too) and stop.
        items = list(worker.active.items())
        for task_id, child in items:
            child.process.finish()
            child.queue.put({"loaded": ["tiny"] if task_id == "t0" else []})
        worker.control.request_stop()

    monkeypatch.setattr(loop_mod.time, "sleep", fast_forward)

    assert worker.run() == 0
    assert worker._spawned == ["t0", "t1"]
    assert [c["active_task_ids"] for c in client.claims] == [[], ["t0"]]
    assert worker.tasks_done == 2
    # Both tasks are round 0 of the same run -- one round, two tasks.
    assert worker.rounds_done == 1
    # The affinity hint one child reported is folded in at reap time.
    assert worker.cached_base_models == {"tiny"}


def test_a_crashed_childs_backstop_abandon_never_touches_its_sibling(tmp_path, monkeypatch):
    """Item 6: a child that exits without ever reaching ``_run_child``'s
    ``finally`` -- a SIGKILL, an OS OOM kill, a crash below Python's own
    exception handling -- must not leave the parent believing its lease is
    still live. Its still-training sibling must be completely unaffected."""
    tasks = [_task_in_round(0, "crashed"), _task_in_round(0, "sibling")]
    client = StubClient(tasks=tasks)
    worker = make_supervised_worker(tmp_path, client, devices=2)

    calls = {"n": 0}

    def fast_forward(seconds=0):
        calls["n"] += 1
        if calls["n"] == 1:
            # Only "crashed" finishes, and without ever writing to its
            # queue -- exactly what a child that never reached the
            # `finally` looks like from the supervisor's side.
            worker.active["crashed"].process.finish(exitcode=-9)
        else:
            if "sibling" in worker.active:
                worker.active["sibling"].process.finish()
                worker.active["sibling"].queue.put({"loaded": []})
            worker.control.request_stop()

    monkeypatch.setattr(loop_mod.time, "sleep", fast_forward)

    assert worker.run() == 0
    assert worker._spawned == ["crashed", "sibling"]
    assert ("abandon", "crashed") in client.calls
    assert ("abandon", "sibling") not in client.calls
    assert worker.tasks_done == 2


def test_a_declined_task_is_abandoned_without_ever_being_spawned(tmp_path, monkeypatch):
    """The supervisor's own ``can_honor`` gate still runs before ``_spawn`` --
    a stale profile must not cost a real child process, just as it costs
    nothing in ``_run_single`` today."""
    tasks = [{**_task_in_round(0), "base_precision": "nf4"}]
    client = StubClient(tasks=tasks)
    worker = make_supervised_worker(tmp_path, client, devices=2)
    monkeypatch.setattr(loop_mod.time, "sleep",
                        lambda s: worker.control.request_stop())

    assert worker.run() == 0
    assert worker._spawned == []
    assert "abandon" in client.kinds()


def test_max_rounds_drains_in_flight_children_before_returning(tmp_path, monkeypatch):
    """``--max-rounds`` must not exit with a lease still training -- that
    would look, from outside, exactly like the abandoned-mid-round case."""
    tasks = [_task_in_round(0, "t0")]
    client = StubClient(tasks=tasks)
    worker = make_supervised_worker(tmp_path, client, devices=2, max_rounds=1)

    def fast_forward(seconds=0):
        child = worker.active.get("t0")
        if child is not None and child.process.is_alive():
            child.process.finish()
            child.queue.put({"loaded": []})

    monkeypatch.setattr(loop_mod.time, "sleep", fast_forward)

    assert worker.run() == 0
    assert worker.active == {}
    assert worker.tasks_done == 1


def test_stop_stops_new_claims_but_waits_for_an_in_flight_child(tmp_path, monkeypatch):
    """4.4's guarantee, generalised: the supervisor must not exit while a
    child is still training just because a second, empty poll came back
    first."""
    tasks = [_task_in_round(0, "t0")]
    client = StubClient(tasks=tasks)
    worker = make_supervised_worker(tmp_path, client, devices=2)

    def fast_forward(seconds=0):
        child = worker.active.get("t0")
        if child is not None and child.process.is_alive():
            worker.control.request_stop()
            child.process.finish()
            child.queue.put({"loaded": []})

    monkeypatch.setattr(loop_mod.time, "sleep", fast_forward)

    assert worker.run() == 0
    assert worker.tasks_done == 1
    assert ("abandon", "t0") not in client.calls
    assert client.kinds().count("claim") == 2  # t0, then a 204 with a free slot


def test_pause_stops_new_claims_and_then_stop_drains_what_is_left(tmp_path, monkeypatch):
    """7.1: stay installed, take no new work -- generalised the same way
    stop is. Nothing here releases a GPU the parent never held (docs/14 §2's
    per-child model cache); a paused supervisor's job is only to stop
    claiming and keep reaping."""
    tasks = [_task_in_round(0, "t0")]
    client = StubClient(tasks=tasks)
    worker = make_supervised_worker(tmp_path, client, devices=2)
    worker.control.request_pause()

    calls = {"n": 0}

    def fast_forward(seconds=0):
        calls["n"] += 1
        if calls["n"] == 1:
            worker.control.clear()
        elif calls["n"] == 2:
            worker.control.request_stop()
        child = worker.active.get("t0")
        if child is not None and child.process.is_alive():
            child.process.finish()
            child.queue.put({"loaded": []})

    monkeypatch.setattr(loop_mod.time, "sleep", fast_forward)

    assert worker.run() == 0
    # register, then a pause-poll sleep before the first claim is even
    # attempted -- pause holds new work, exactly as it does at slot count 1.
    assert client.calls[0][0] == "register"
    assert client.kinds().count("claim") == 2  # t0, then a trailing 204
    assert calls["n"] >= 2, "the pause branch's own sleep never ran"
    assert worker._spawned == ["t0"]
    assert worker.tasks_done == 1
