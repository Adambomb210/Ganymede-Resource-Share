"""Starting and stopping the worker (docs/02-architecture-v2.md 7 step 4, 4.6).

No Docker daemon and no real subprocesses: both runtimes take an injected
boundary. The assertions that earn their place are about the **composed argv**.
A dropped `--read-only` or a `-e GANYMEDE_KEY=<token>` changes nothing about
how the worker behaves, produces no error, and would survive indefinitely --
the only place that class of regression is visible is the argv itself.
"""

from __future__ import annotations

import json

import pytest

from ganymede.host import runtime as runtime_mod
from ganymede.host.config import HostConfig


class FakeDocker:
    """Records argv, answers from a queue of scripted results."""

    def __init__(self, *, inspect: object = None, fail: set[str] | None = None):
        self.calls: list[list[str]] = []
        self._inspect = inspect
        self._fail = fail or set()

    def __call__(self, argv, *, timeout, check, env=None):
        self.calls.append(list(argv))
        verb = argv[1] if len(argv) > 1 else ""
        if verb in self._fail:
            if check:
                raise runtime_mod.CommandFailed(argv, 1, "scripted failure")
            return runtime_mod._Completed(1, "", "scripted failure")
        if verb == "inspect":
            if self._inspect is None:
                return runtime_mod._Completed(1, "", "No such object")
            return runtime_mod._Completed(0, json.dumps(self._inspect), "")
        if verb == "image":
            return runtime_mod._Completed(0 if self._inspect is not None else 1, "[]", "")
        return runtime_mod._Completed(0, "", "")

    def argv_for(self, verb: str) -> list[str] | None:
        for call in self.calls:
            if len(call) > 1 and call[1] == verb:
                return call
        return None


def _inspect(running: bool, image: str = "ganymede/worker-llm:v1", status: str = "running"):
    return [{"State": {"Running": running, "Status": status, "ExitCode": 0},
             "Config": {"Image": image}}]


def _config(**kw) -> HostConfig:
    base = {"coordinator_url": "https://c.example", "key": "sekrit"}
    base.update(kw)
    return HostConfig(**base)


# --------------------------------------------------------------------------
# The argv -- 4.6's isolation baseline
# --------------------------------------------------------------------------


def test_every_hardening_flag_from_the_spec_is_present():
    argv = runtime_mod.DockerRuntime(_config()).run_argv("img:v1", {})
    for flag in ("--cap-drop=ALL", "--security-opt=no-new-privileges", "--read-only", "--detach"):
        assert flag in argv, flag
    assert argv[-1] == "img:v1"


def test_the_writable_mounts_read_only_makes_necessary_are_all_there():
    """transformers, Triton and torch.compile all write caches; without these
    the container is hardened into being unable to train."""
    argv = runtime_mod.DockerRuntime(_config()).run_argv("img:v1", {})
    pairs = list(zip(argv, argv[1:]))
    assert ("--tmpfs", "/tmp") in pairs
    assert ("--tmpfs", "/run/ganymede") in pairs


def test_the_state_volume_is_mounted_so_the_pause_switch_can_reach_inside():
    """7.1's kill switch is a file. Under --read-only it has nowhere to land
    without this mount, and the contributor's only remaining stop mechanism
    would be a signal -- which 4.4 is explicit must not be what correctness
    rests on."""
    argv = runtime_mod.DockerRuntime(_config()).run_argv("img:v1", {})
    assert ("-v", "ganymede-state:/var/lib/ganymede") in list(zip(argv, argv[1:]))


def test_the_hf_cache_volume_is_mounted_so_a_base_model_is_pulled_once():
    argv = runtime_mod.DockerRuntime(_config()).run_argv("img:v1", {})
    assert ("-v", "ganymede-hf:/cache/hf") in list(zip(argv, argv[1:]))


def test_resource_limits_come_from_config():
    cfg = _config(memory="8g", cpus="2", pids_limit=64, gpus="device=0")
    argv = runtime_mod.DockerRuntime(cfg).run_argv("img:v1", {})
    pairs = list(zip(argv, argv[1:]))
    assert ("--memory", "8g") in pairs
    assert ("--cpus", "2") in pairs
    assert ("--pids-limit", "64") in pairs
    assert ("--gpus", "device=0") in pairs


def test_the_bearer_key_never_appears_in_the_argv():
    """A value passed as `-e NAME=value` lands in the host process table and in
    `docker inspect` output for the life of the container. GANYMEDE_KEY is a
    bearer token (6.3); it travels by name only, and Docker reads the value
    from the agent's own environment."""
    argv = runtime_mod.DockerRuntime(_config()).run_argv(
        "img:v1", {"GANYMEDE_KEY": "sekrit", "GANYMEDE_COORDINATOR_URL": "https://c.example"}
    )
    assert "sekrit" not in " ".join(argv)
    assert "-e" in argv
    assert "GANYMEDE_KEY" in argv
    assert not any("=" in a and a.startswith("GANYMEDE_") for a in argv)


def test_a_stop_timeout_long_enough_to_abandon_a_lease():
    """Docker's default 10 s is not enough for a worker to notice, abandon and
    exit (4.4)."""
    argv = runtime_mod.DockerRuntime(_config()).run_argv("img:v1", {})
    assert ("--stop-timeout", "120") in list(zip(argv, argv[1:]))


def test_root_does_not_pass_user_and_so_leaves_the_images_own_user_in_place(monkeypatch):
    """`--user 0:0` would run the worker as root inside the container, undoing
    the USER the image already set for itself."""
    monkeypatch.setattr(runtime_mod.os, "getuid", lambda: 0, raising=False)
    argv = runtime_mod.DockerRuntime(_config()).run_argv("img:v1", {})
    assert "--user" not in argv


def test_a_rootless_contributor_keeps_their_own_uid(monkeypatch):
    monkeypatch.setattr(runtime_mod.os, "getuid", lambda: 1000, raising=False)
    monkeypatch.setattr(runtime_mod.os, "getgid", lambda: 1000, raising=False)
    argv = runtime_mod.DockerRuntime(_config()).run_argv("img:v1", {})
    assert ("--user", "1000:1000") in list(zip(argv, argv[1:]))


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------


def test_an_absent_container_is_not_running_rather_than_an_error():
    """The normal state on every tick where the machine was busy."""
    docker = FakeDocker(inspect=None)
    status = runtime_mod.DockerRuntime(_config(), runner=docker).status()
    assert status.running is False
    assert status.image is None


def test_a_running_container_reports_its_image():
    docker = FakeDocker(inspect=_inspect(True, "ganymede/worker-llm:v2"))
    status = runtime_mod.DockerRuntime(_config(), runner=docker).status()
    assert status.running
    assert status.image == "ganymede/worker-llm:v2"


def test_a_container_that_started_and_died_is_distinguishable_from_one_that_never_started():
    docker = FakeDocker(inspect=_inspect(False, status="exited"))
    status = runtime_mod.DockerRuntime(_config(), runner=docker).status()
    assert status.running is False
    assert status.exited is True


def test_inspect_output_that_is_not_json_does_not_raise():
    class Weird(FakeDocker):
        def __call__(self, argv, *, timeout, check, env=None):
            self.calls.append(list(argv))
            return runtime_mod._Completed(0, "not json at all", "")

    assert runtime_mod.DockerRuntime(_config(), runner=Weird()).status().running is False


# --------------------------------------------------------------------------
# start / stop / pull
# --------------------------------------------------------------------------


def test_start_clears_a_stale_container_name_first():
    """`docker run` refuses a name an exited container still holds, so every
    tick has to be able to assume nothing about the last one."""
    docker = FakeDocker(inspect=None)
    runtime_mod.DockerRuntime(_config(), runner=docker).start("img:v1", {})
    verbs = [c[1] for c in docker.calls]
    assert verbs.index("rm") < verbs.index("run")


def test_ensure_image_does_not_repull_a_tag_already_held():
    """4.1 pins by digest because a republished tag is a silent PyTorch bump
    across a running swarm."""
    docker = FakeDocker(inspect=_inspect(True))
    pulled = runtime_mod.DockerRuntime(_config(), runner=docker).ensure_image("img:v1")
    assert pulled is False
    assert docker.argv_for("pull") is None


def test_ensure_image_pulls_when_the_image_is_absent():
    docker = FakeDocker(inspect=None)
    pulled = runtime_mod.DockerRuntime(_config(), runner=docker).ensure_image("img:v1")
    assert pulled is True
    assert docker.argv_for("pull")[-1] == "img:v1"


def test_stop_passes_the_configured_grace_period():
    docker = FakeDocker(inspect=_inspect(True))
    runtime_mod.DockerRuntime(_config(stop_timeout_sec=90), runner=docker).stop()
    assert docker.argv_for("stop")[:4] == ["docker", "stop", "--time", "90"]


def test_a_failed_docker_call_carries_the_command_and_stderr():
    docker = FakeDocker(inspect=None, fail={"run"})
    with pytest.raises(runtime_mod.CommandFailed) as exc:
        runtime_mod.DockerRuntime(_config(), runner=docker).start("img:v1", {})
    assert "scripted failure" in str(exc.value)


def test_a_missing_docker_binary_is_a_typed_error_not_a_filenotfound():
    cfg = _config(docker_bin="definitely-not-a-real-binary")
    with pytest.raises(runtime_mod.CommandFailed) as exc:
        runtime_mod.DockerRuntime(cfg).ensure_image("img:v1")
    assert "not found on PATH" in str(exc.value)


# --------------------------------------------------------------------------
# NativeRuntime -- the macOS path (4.1, 6.8)
# --------------------------------------------------------------------------


class FakeProc:
    def __init__(self, pid):
        self.pid = pid


def _native(tmp_path, **kw):
    cfg = _config(runtime="native", state_dir=str(tmp_path), **kw)
    return cfg, runtime_mod.NativeRuntime(cfg, popen=lambda *a, **k: FakeProc(12345))


def test_no_recorded_worker_is_not_running(tmp_path):
    _, native = _native(tmp_path)
    assert native.status().running is False


def test_a_live_process_we_started_reads_as_running(tmp_path):
    import os

    cfg = _config(runtime="native", state_dir=str(tmp_path))
    native = runtime_mod.NativeRuntime(cfg, popen=lambda *a, **k: FakeProc(os.getpid()))
    native.start("native", {})
    assert native.status().running is True


def test_a_recycled_pid_does_not_read_as_our_worker(tmp_path):
    """The agent can be down for days -- a laptop that was closed -- and pids
    are recycled. Signalling a stranger's process is the worst thing this
    software could do to a volunteer, so the start time is compared too."""
    import os

    cfg = _config(runtime="native", state_dir=str(tmp_path))
    (tmp_path / "worker.json").write_text(
        json.dumps({"pid": os.getpid(), "started_at": 1.0, "image": "native"})
    )
    assert runtime_mod.NativeRuntime(cfg).status().running is False


def test_start_clears_a_stale_stop_sentinel(tmp_path):
    """Left behind by the last shutdown, it would make the new worker exit
    immediately and the agent restart it on every tick forever."""
    cfg, native = _native(tmp_path)
    (tmp_path / "stop").touch()
    native.start("native", {})
    assert not (tmp_path / "stop").exists()


def test_stop_writes_the_sentinel_before_it_ever_signals(tmp_path):
    """4.4's ordering: the sentinel is the mechanism that ports, so it has to
    be the one that is always exercised."""
    cfg, native = _native(tmp_path)
    native.stop()
    assert (tmp_path / "stop").exists()


def test_stop_with_no_recorded_worker_still_leaves_the_sentinel(tmp_path):
    cfg, native = _native(tmp_path)
    native.stop()
    assert (tmp_path / "stop").exists()


def test_a_corrupt_record_is_treated_as_no_worker(tmp_path):
    cfg = _config(runtime="native", state_dir=str(tmp_path))
    (tmp_path / "worker.json").write_text("{ this is not json")
    assert runtime_mod.NativeRuntime(cfg).status().running is False


def test_native_ensure_image_is_a_no_op(tmp_path):
    _, native = _native(tmp_path)
    assert native.ensure_image("anything") is False


# --------------------------------------------------------------------------
# for_config
# --------------------------------------------------------------------------


def test_for_config_picks_the_runtime_the_config_names():
    assert isinstance(runtime_mod.for_config(_config(runtime="docker")), runtime_mod.DockerRuntime)
    assert isinstance(runtime_mod.for_config(_config(runtime="native")), runtime_mod.NativeRuntime)


def test_an_unknown_runtime_name_is_rejected_loudly():
    with pytest.raises(ValueError):
        runtime_mod.for_config(_config(runtime="podman-maybe"))
