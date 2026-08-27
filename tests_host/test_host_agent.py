"""The host agent's tick (docs/02-architecture-v2.md 7).

These are the M3 exit criteria, stated as tests:

    machine idles                 -> worker starts within one timer interval
    touch the pause sentinel      -> the running container stops, none start
    manifest tag bump             -> agent pulls and restarts on the new tag

Each is a statement about *one tick*, which is why `tick` returns a
`TickResult` rather than only logging what it did.
"""

from __future__ import annotations

from ganymede.host import agent as agent_mod
from ganymede.host import idle as idle_mod
from ganymede.host import manifest as manifest_mod
from ganymede.host import runtime as runtime_mod
from ganymede.host.config import HostConfig


class FakeRuntime:
    def __init__(self, running: bool = False, image: str | None = None):
        self._status = runtime_mod.RuntimeStatus(running=running, image=image)
        self.started: list[str] = []
        self.pulled: list[str] = []
        self.stops = 0

    def status(self):
        return self._status

    def ensure_image(self, image):
        self.pulled.append(image)
        return True

    def start(self, image, env):
        self.started.append(image)
        self._status = runtime_mod.RuntimeStatus(running=True, image=image)

    def stop(self):
        self.stops += 1
        self._status = runtime_mod.RuntimeStatus(running=False, image=self._status.image)


class FakeBackend:
    def __init__(self, idle: bool, reason: str = ""):
        self._report = idle_mod.IdleReport(idle=idle, reason=reason or ("idle" if idle else "busy"))

    def is_idle(self):
        return self._report.idle

    def report(self, **kw):
        return self._report


class BoolOnlyBackend:
    """7.1 anticipates `vast` and `tensordock` backends that are one API call
    and a bool. They must work here without carrying a reason string."""

    def __init__(self, idle: bool):
        self._idle = idle

    def is_idle(self):
        return self._idle


def _config(**kw) -> HostConfig:
    base = {"coordinator_url": "https://c.example", "key": "k", "cache_cap_gb": 0}
    base.update(kw)
    return HostConfig(**base)


def _fetch(*images: str | None):
    def fetch(config):
        return manifest_mod.parse({
            "api_version": "v1",
            "runs": [
                {"run_id": f"r{i}", "base_model": "Qwen/Q", "required_image": img}
                for i, img in enumerate(images)
            ],
        })
    return fetch


# --------------------------------------------------------------------------
# Exit criterion 1: machine idles -> worker starts
# --------------------------------------------------------------------------


def test_an_idle_machine_starts_a_worker_on_the_required_image():
    rt = FakeRuntime(running=False)
    result = agent_mod.tick(_config(), runtime=rt, backend=FakeBackend(True),
                            fetch=_fetch("ganymede/worker-llm:v3"))
    assert result.action == "started"
    assert rt.started == ["ganymede/worker-llm:v3"]


def test_a_busy_machine_starts_nothing():
    rt = FakeRuntime(running=False)
    result = agent_mod.tick(_config(), runtime=rt, backend=FakeBackend(False, "nvidia-smi: 1 process"),
                            fetch=_fetch("img:v1"))
    assert result.action == "idle-skip"
    assert rt.started == []


def test_a_worker_already_on_the_right_image_is_left_alone():
    """Restarting it would throw away the round in progress for nothing."""
    rt = FakeRuntime(running=True, image="img:v1")
    result = agent_mod.tick(_config(), runtime=rt, backend=FakeBackend(True), fetch=_fetch("img:v1"))
    assert result.action == "running"
    assert rt.stops == 0
    assert rt.started == []


def test_a_backend_that_only_implements_the_protocol_still_works():
    rt = FakeRuntime()
    result = agent_mod.tick(_config(), runtime=rt, backend=BoolOnlyBackend(True), fetch=_fetch("img:v1"))
    assert result.action == "started"


# --------------------------------------------------------------------------
# Exit criterion 2: pause stops a running worker
# --------------------------------------------------------------------------


def test_pause_stops_a_worker_that_is_already_running(tmp_path):
    """7.1's kill switch has to take the GPU back *now*. An agent that only
    declined to start new workers would answer "after this round", which is not
    what someone who wants to play a game is asking for."""
    cfg = _config(state_dir=str(tmp_path))
    (tmp_path / "pause").touch()

    rt = FakeRuntime(running=True, image="img:v1")
    result = agent_mod.tick(cfg, runtime=rt, backend=idle_mod.LocalIdleBackend(cfg),
                            fetch=_fetch("img:v1"))
    assert result.action == "stopped"
    assert rt.stops == 1


def test_pause_prevents_a_new_worker_from_starting(tmp_path):
    cfg = _config(state_dir=str(tmp_path))
    (tmp_path / "pause").touch()

    rt = FakeRuntime(running=False)
    result = agent_mod.tick(cfg, runtime=rt, backend=idle_mod.LocalIdleBackend(cfg),
                            fetch=_fetch("img:v1"))
    assert result.action == "idle-skip"
    assert rt.started == []


def test_the_machine_becoming_busy_stops_the_worker_too():
    rt = FakeRuntime(running=True, image="img:v1")
    result = agent_mod.tick(_config(), runtime=rt, backend=FakeBackend(False), fetch=_fetch("img:v1"))
    assert result.action == "stopped"
    assert rt.stops == 1


# --------------------------------------------------------------------------
# Exit criterion 3: a tag bump restarts on the new tag, with no manual step
# --------------------------------------------------------------------------


def test_a_manifest_tag_bump_pulls_and_restarts_with_no_manual_step():
    rt = FakeRuntime(running=True, image="ganymede/worker-llm:v1")
    result = agent_mod.tick(_config(), runtime=rt, backend=FakeBackend(True),
                            fetch=_fetch("ganymede/worker-llm:v2"))
    assert result.action == "started"
    assert rt.stops == 1
    assert rt.pulled == ["ganymede/worker-llm:v2"]
    assert rt.started == ["ganymede/worker-llm:v2"]


def test_a_local_pin_survives_a_manifest_that_wants_something_else():
    rt = FakeRuntime(running=True, image="ganymede/worker-llm:v1")
    result = agent_mod.tick(_config(image_tag="ganymede/worker-llm:v1"), runtime=rt,
                            backend=FakeBackend(True), fetch=_fetch("ganymede/worker-llm:v9"))
    assert result.action == "running"
    assert rt.stops == 0


# --------------------------------------------------------------------------
# Failure behaviour -- 4.2's split, applied to the host
# --------------------------------------------------------------------------


def test_an_unreachable_coordinator_is_a_quiet_skip_not_a_crash():
    """It is Tuesday and the coordinator is restarting. The timer fires again
    in fifteen minutes; a host agent that gave up permanently would be a
    machine that silently stopped contributing."""
    def fetch(config):
        raise manifest_mod.ManifestError("https://c.example/v1/manifest", "unreachable")

    rt = FakeRuntime(running=False)
    result = agent_mod.tick(_config(), runtime=rt, backend=FakeBackend(True), fetch=fetch)
    assert result.action == "noop"
    assert result.ok
    assert rt.started == []


def test_an_unreachable_coordinator_does_not_stop_a_working_worker():
    """The worker has its own retry and backoff (4.2); killing it because the
    *host* could not reach the coordinator would turn a blip into a lost round."""
    def fetch(config):
        raise manifest_mod.ManifestError("https://c.example/v1/manifest", "unreachable")

    rt = FakeRuntime(running=True, image="img:v1")
    result = agent_mod.tick(_config(), runtime=rt, backend=FakeBackend(True), fetch=fetch)
    assert result.action == "running"
    assert rt.stops == 0


def test_no_active_runs_starts_nothing():
    """A worker with no run to work on would poll 204 forever and hold the GPU
    warm for nothing."""
    rt = FakeRuntime(running=False)
    result = agent_mod.tick(_config(), runtime=rt, backend=FakeBackend(True), fetch=_fetch())
    assert result.action == "noop"
    assert rt.started == []


def test_runs_that_name_no_image_leave_a_running_worker_alone():
    rt = FakeRuntime(running=True, image="img:v1")
    result = agent_mod.tick(_config(), runtime=rt, backend=FakeBackend(True), fetch=_fetch(None))
    assert result.action == "running"
    assert rt.stops == 0


def test_a_missing_key_is_the_one_fatal_configuration_error():
    """No amount of waiting fixes it, so it exits non-zero and lands in
    `systemctl status`, which is where a contributor looks first."""
    result = agent_mod.tick(HostConfig(coordinator_url="https://c.example"),
                            runtime=FakeRuntime(), backend=FakeBackend(True), fetch=_fetch("i:v1"))
    assert result.action == "error"
    assert not result.ok
    assert "key" in result.detail


def test_a_docker_daemon_that_will_not_answer_is_an_error_not_a_hang():
    class Broken(FakeRuntime):
        def status(self):
            raise runtime_mod.CommandFailed(["docker", "inspect"], -1, "timed out after 30s")

    result = agent_mod.tick(_config(), runtime=Broken(), backend=FakeBackend(True),
                            fetch=_fetch("img:v1"))
    assert result.action == "error"


def test_a_start_that_fails_is_reported_rather_than_raised():
    class WontStart(FakeRuntime):
        def start(self, image, env):
            raise runtime_mod.CommandFailed(["docker", "run"], 125, "no such device")

    result = agent_mod.tick(_config(), runtime=WontStart(), backend=FakeBackend(True),
                            fetch=_fetch("img:v1"))
    assert result.action == "error"
    assert "no such device" in result.detail


# --------------------------------------------------------------------------
# The worker's environment
# --------------------------------------------------------------------------


def test_the_container_path_passes_only_the_two_settings_it_needs():
    """Cache and state are fixed by the mounts in `runtime.run_argv`; telling
    the container about host paths it cannot see would be wrong."""
    env = agent_mod.worker_env(_config())
    assert set(env) == {"GANYMEDE_COORDINATOR_URL", "GANYMEDE_KEY"}


def test_the_native_path_is_told_where_the_host_put_things(tmp_path):
    env = agent_mod.worker_env(_config(runtime="native", state_dir=str(tmp_path),
                                       cache_dir=str(tmp_path / "hf")))
    assert env["GANYMEDE_STATE"] == str(tmp_path)
    assert env["GANYMEDE_CACHE_DIR"] == str(tmp_path / "hf")


# --------------------------------------------------------------------------
# Cache eviction happens at the one moment it is safe
# --------------------------------------------------------------------------


def test_eviction_runs_before_a_start_and_protects_the_active_runs_models(tmp_path, monkeypatch):
    """Evicting a base model this machine is about to be handed work for means
    re-downloading ~16 GB before anything useful happens (6.7)."""
    seen = {}

    def fake_evict(cache_dir, cap_bytes, *, protect=frozenset(), dry_run=False):
        seen["protect"] = protect
        return agent_mod.cache_mod.EvictionResult([], 0, 0, 0, dry_run)

    monkeypatch.setattr(agent_mod.cache_mod, "evict_to_cap", fake_evict)
    agent_mod.tick(_config(cache_cap_gb=50, cache_dir=str(tmp_path)), runtime=FakeRuntime(),
                   backend=FakeBackend(True), fetch=_fetch("img:v1"))
    assert seen["protect"] == frozenset({"Qwen/Q"})


def test_a_cache_that_cannot_be_pruned_does_not_block_work(tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(agent_mod.cache_mod, "evict_to_cap", boom)
    rt = FakeRuntime()
    result = agent_mod.tick(_config(cache_cap_gb=50, cache_dir=str(tmp_path)), runtime=rt,
                            backend=FakeBackend(True), fetch=_fetch("img:v1"))
    assert result.action == "started"


def test_a_zero_cap_disables_eviction_entirely(tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("should not have been called")

    monkeypatch.setattr(agent_mod.cache_mod, "evict_to_cap", boom)
    agent_mod.tick(_config(cache_cap_gb=0, cache_dir=str(tmp_path)), runtime=FakeRuntime(),
                   backend=FakeBackend(True), fetch=_fetch("img:v1"))
