"""Runtime confinement and the kill path (docs/11 §2, §3).

Every subprocess in ``sandbox.py`` goes through an injectable runner, so all of
this runs on a machine with no container runtime installed -- which is the point
of the seam, not a limitation of the tests. What is asserted here is the flag
template, the digest check that gates ``docker load``, and the two kill modes;
what cannot be asserted here is that Docker honours those flags, which is a
statement about Docker.

The flags are asserted individually rather than against a golden argv. A golden
list would be rewritten by whoever next adds a flag, which is exactly the edit
that must not silently drop ``--read-only``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ganymede.worker import sandbox


class FakeRunner:
    """Records argv and answers from a scripted table."""

    def __init__(self, answers: dict[str, sandbox.Completed] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.answers = answers or {}

    def __call__(self, argv, *, timeout, env=None, check=None):  # noqa: ARG002
        self.calls.append(list(argv))
        for key, answer in self.answers.items():
            if key in argv:
                return answer
        return sandbox.Completed(returncode=0, stdout="", stderr="")

    def argv_for(self, verb: str) -> list[str] | None:
        for call in self.calls:
            if verb in call:
                return call
        return None


@pytest.fixture
def config(tmp_path) -> sandbox.SandboxConfig:
    return sandbox.SandboxConfig(scratch_root=tmp_path / "jobs")


@pytest.fixture
def job(config) -> sandbox.JobContainer:
    return sandbox.JobContainer(task_id="task-abcdef123456", config=config,
                                runner=FakeRunner())


def _flag_value(argv: list[str], flag: str) -> str | None:
    for i, item in enumerate(argv):
        if item == flag and i + 1 < len(argv):
            return argv[i + 1]
    return None


def _flag_values(argv: list[str], flag: str) -> list[str]:
    return [argv[i + 1] for i, item in enumerate(argv)
            if item == flag and i + 1 < len(argv)]


# ==========================================================================
# The flag template (docs/11 §2.2, §2.3, §2.4)
# ==========================================================================


def test_the_isolation_baseline_is_inherited_whole(job):
    argv = job.run_argv("sha256:" + "a" * 64)
    for flag in ("--cap-drop=ALL", "--security-opt=no-new-privileges", "--read-only"):
        assert flag in argv, f"{flag} missing -- §4.6 baseline"


def test_swap_is_off_because_memory_is_a_deadline_not_a_suggestion(job):
    """`--memory` alone lets a job over its cap grind against host disk for
    hours. Equal `--memory-swap` is how Docker spells 'no swap', and a task
    that dies is better for the fleet than one that takes twenty times its
    budget."""
    argv = job.run_argv("img")
    assert _flag_value(argv, "--memory") == _flag_value(argv, "--memory-swap")


def test_the_job_runs_as_a_non_root_uid_whatever_the_image_says(job):
    """The scan flags a root image (docs/11 §1.3), but a flag is advice. This
    is the enforcement, and it does not consult the image."""
    argv = job.run_argv("img")
    user = _flag_value(argv, "--user")
    assert user == f"{sandbox.CONTAINER_UID}:{sandbox.CONTAINER_UID}"
    assert not user.startswith("0:")


def test_the_network_is_off(job):
    """docs/11 §2.4's default-deny. The common job reads /scratch/in, computes
    and writes /scratch/out -- the worker does every transfer, so the job needs
    no route anywhere."""
    assert _flag_value(job.run_argv("img"), "--network") == "none"


def test_ipc_and_core_dumps_are_closed(job):
    argv = job.run_argv("img")
    assert "--ipc=private" in argv
    assert "core=0" in _flag_values(argv, "--ulimit")


def test_scratch_is_the_only_writable_mount(job):
    """§2.3: scratch plus two tmpfs, and nothing else. In particular not the
    worker's state dir -- a process that can write there can forge the
    contributor's kill switch, and the job is exactly that process."""
    argv = job.run_argv("img")
    mounts = _flag_values(argv, "-v")
    assert mounts == [f"{job.scratch}:{sandbox.CONTAINER_SCRATCH}"]
    assert set(_flag_values(argv, "--tmpfs")) == {"/tmp", "/run"}
    assert not any("ganymede" in m and "state" in m for m in mounts)


def test_env_is_passed_by_name_never_by_value(job):
    """A NAME=value here lands in the host process table and in `inspect`
    output for the life of the container. Same reasoning as the worker's own
    run_argv."""
    argv = job.run_argv("img", env={"HF_TOKEN": "secret-value"})
    assert "HF_TOKEN" in argv
    assert not any("secret-value" in item for item in argv)


def test_the_image_is_run_by_id_and_it_is_the_last_argument(job):
    image_id = "sha256:" + "b" * 64
    argv = job.run_argv(image_id)
    assert argv[-1] == image_id


def test_the_runtime_ceiling_clamps_a_generous_spec(config):
    """§2.2: `spec.max_runtime_sec` is the job's ask, `job_max_runtime_sec` is
    the operator's ceiling, and the ceiling wins."""
    tight = sandbox.SandboxConfig(scratch_root=config.scratch_root,
                                  job_max_runtime_sec=600)
    job = sandbox.JobContainer("t", tight, runner=FakeRunner())
    assert job.runtime_ceiling(99_999) == 600
    assert job.runtime_ceiling(60) == 60
    assert job.runtime_ceiling(None) == 600


def test_the_ceiling_does_not_leak_between_starts(config):
    """It is a function of its argument, not of whichever call ran last: a
    caller that builds argv once and starts twice must not inherit the previous
    task's clamp."""
    runner = FakeRunner()
    job = sandbox.JobContainer("t", config, runner=runner)
    job.start("img", max_runtime_sec=60)
    job.start("img", max_runtime_sec=99_999)
    assert job.runtime_ceiling(60) == 60


def test_the_storage_quota_is_off_by_default(job, config):
    """`--storage-opt size=` is a hard error on overlay2, the common driver. A
    flag that refuses to start the job is worse than a quota that is a
    directory the worker wipes."""
    assert "--storage-opt" not in job.run_argv("img")
    opted_in = sandbox.JobContainer(
        "t", sandbox.SandboxConfig(scratch_root=config.scratch_root,
                                   storage_opt_size=True, scratch_gb=20),
        runner=FakeRunner())
    assert "size=20G" in _flag_values(opted_in.run_argv("img"), "--storage-opt")


def test_gpus_can_be_withheld(config):
    job = sandbox.JobContainer(
        "t", sandbox.SandboxConfig(scratch_root=config.scratch_root, gpus=None),
        runner=FakeRunner())
    assert "--gpus" not in job.run_argv("img")


# ==========================================================================
# Pull, verify, load (docs/11 §2.3)
# ==========================================================================


def test_a_matching_digest_lands_the_archive(job):
    payload = b"a plausible docker save archive"
    digest = hashlib.sha256(payload).hexdigest()
    path = job.fetch_archive(lambda url: payload, "http://store/img.tar", digest)
    assert path.read_bytes() == payload


def test_a_sha256_prefixed_digest_is_accepted(job):
    payload = b"x"
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    assert job.fetch_archive(lambda url: payload, "u", digest).exists()


def test_a_mismatched_digest_raises_and_leaves_nothing_behind(job):
    """§2.3's last gate. `docker load` parses an archive the author controls,
    so running it on bytes that failed their check would be trusting the thing
    being checked."""
    with pytest.raises(sandbox.DigestMismatch):
        job.fetch_archive(lambda url: b"wrong bytes", "u", "0" * 64)
    assert not (job.scratch / "image.tar").exists()


def test_an_absent_digest_is_a_mismatch_not_a_pass(job):
    """Fail closed: a task payload with no digest is a bug upstream, and the
    reading that costs a re-queue is better than the one that loads unverified
    bytes."""
    with pytest.raises(sandbox.DigestMismatch):
        job.fetch_archive(lambda url: b"anything", "u", "")


def test_load_returns_the_content_addressed_id(config):
    image_id = "sha256:" + "c" * 64
    runner = FakeRunner({"load": sandbox.Completed(
        0, stdout=f"Loaded image ID: {image_id}\n")})
    job = sandbox.JobContainer("t", config, runner=runner)
    assert job.load_image(Path("x.tar")) == image_id


def test_load_failure_is_a_sandbox_error(config):
    runner = FakeRunner({"load": sandbox.Completed(1, stderr="no such file")})
    job = sandbox.JobContainer("t", config, runner=runner)
    with pytest.raises(sandbox.SandboxError):
        job.load_image(Path("x.tar"))


def test_a_load_that_names_no_image_is_a_sandbox_error(config):
    """An archive that loads but reports nothing leaves the worker with no id
    to run. Guessing the tag it carries is exactly what §2.3 forbids."""
    runner = FakeRunner({"load": sandbox.Completed(0, stdout="Loaded image: job:latest")})
    job = sandbox.JobContainer("t", config, runner=runner)
    with pytest.raises(sandbox.SandboxError):
        job.load_image(Path("x.tar"))


# ==========================================================================
# Scratch
# ==========================================================================


def test_prepare_scratch_wipes_the_previous_attempt(job):
    """A retry of the same task id must not find the last attempt's
    half-written output and take it for its own."""
    job.prepare_scratch()
    stale = job.scratch / "out" / "partial.json"
    stale.write_text("half a result")
    job.prepare_scratch()
    assert not stale.exists()
    assert (job.scratch / "in").is_dir() and (job.scratch / "out").is_dir()


def test_cleanup_removes_the_container_and_the_directory(job):
    job.prepare_scratch()
    job.cleanup()
    assert not job.scratch.exists()
    assert job.runner.argv_for("rm") is not None


# ==========================================================================
# The kill path (docs/11 §3 step 3)
# ==========================================================================


def test_soft_is_a_stop_with_the_grace_period(config):
    runner = FakeRunner()
    job = sandbox.JobContainer("t", config, runner=runner)
    job.cancel("soft")
    argv = runner.argv_for("stop")
    assert argv is not None
    assert _flag_value(argv, "--time") == str(config.cancel_grace_sec)
    assert runner.argv_for("kill") is None


def test_hard_is_a_kill_now_with_no_grace(config):
    runner = FakeRunner()
    job = sandbox.JobContainer("t", config, runner=runner)
    job.cancel("hard")
    assert runner.argv_for("kill") is not None
    assert runner.argv_for("stop") is None


def test_a_shorter_grace_can_be_forced(config):
    """§3: the drain never runs past the lease, so the caller clamps."""
    runner = FakeRunner()
    job = sandbox.JobContainer("t", config, runner=runner)
    job.cancel("soft", grace_sec=5)
    assert _flag_value(runner.argv_for("stop"), "--time") == "5"


# ==========================================================================
# The capability report (docs/11 §4)
# ==========================================================================


def test_detect_runtime_names_the_binary_when_it_answers():
    runner = FakeRunner({"version": sandbox.Completed(0, stdout="27.0.3")})
    assert sandbox.detect_runtime(runner=runner, runtime_bin="docker") == "docker"


def test_detect_runtime_is_none_when_the_daemon_is_unreachable():
    """The binary being present is not the capability. Inside the worker
    container `docker` can exist while the socket it needs does not, and a
    profile that overclaims costs a real task a real lease."""
    runner = FakeRunner({"version": sandbox.Completed(1, stderr="cannot connect")})
    assert sandbox.detect_runtime(runner=runner, runtime_bin="docker") is None


def test_detect_runtime_survives_a_missing_binary():
    def explode(argv, *, timeout, env=None, check=None):  # noqa: ARG001
        raise OSError("no such file")

    assert sandbox.detect_runtime(runner=explode) is None


# ==========================================================================
# The lease crumb (docs/11 §3, wedged-worker path)
# ==========================================================================


def test_the_crumb_round_trips(tmp_path):
    sandbox.write_lease_crumb(tmp_path, "task1", "2026-09-07T00:00:00+00:00", "c1")
    crumb = sandbox.read_lease_crumb(tmp_path)
    assert crumb["task_id"] == "task1"
    assert crumb["container"] == "c1"


def test_the_crumb_is_written_atomically(tmp_path):
    """The host agent reads this file on a timer with no locking. A partial
    write would parse as a missing crumb and reap a live job."""
    sandbox.write_lease_crumb(tmp_path, "task1", "2026-09-07T00:00:00+00:00", "c1")
    sandbox.write_lease_crumb(tmp_path, "task2", "2026-09-07T00:01:00+00:00", "c2")
    assert sandbox.read_lease_crumb(tmp_path)["task_id"] == "task2"
    assert not list(tmp_path.glob("*.tmp"))


def test_reading_an_absent_crumb_is_none_not_an_error(tmp_path):
    assert sandbox.read_lease_crumb(tmp_path) is None
    sandbox.clear_lease_crumb(tmp_path)  # must not raise either


def test_a_corrupt_crumb_reads_as_absent(tmp_path):
    (tmp_path / "lease.json").write_text("{not json")
    assert sandbox.read_lease_crumb(tmp_path) is None


# ==========================================================================
# The wedged-worker backstop (docs/11 §3, second half)
# ==========================================================================


def _host_config(tmp_path, **kw):
    from ganymede.host.config import HostConfig

    return HostConfig(coordinator_url="http://c", key="k",
                      job_scratch_dir=str(tmp_path), **kw)


def _fresh(seconds_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()


def test_a_live_lease_is_left_alone(tmp_path):
    from ganymede.host import agent

    sandbox.write_lease_crumb(tmp_path, "t1", _fresh(30), "ganymede-job-t1")
    runner = FakeRunner({"inspect": sandbox.Completed(0, stdout="true")})
    assert agent.reap_orphaned_jobs(_host_config(tmp_path), runner=runner) == []
    assert runner.argv_for("kill") is None


def test_a_stale_crumb_over_a_running_container_is_reaped(tmp_path):
    """The container outlived the lease behind it, so whatever it is computing
    is work nobody will accept. Soft/hard collapses to hard here: there is no
    worker left to drain gracefully."""
    from ganymede.host import agent

    sandbox.write_lease_crumb(tmp_path, "t1", _fresh(5000), "ganymede-job-t1")
    runner = FakeRunner({"inspect": sandbox.Completed(0, stdout="true")})
    killed = agent.reap_orphaned_jobs(_host_config(tmp_path), runner=runner)

    assert killed == ["ganymede-job-t1"]
    assert runner.argv_for("kill") is not None
    assert runner.argv_for("rm") is not None
    assert sandbox.read_lease_crumb(tmp_path) is None


def test_a_stale_crumb_with_no_container_just_clears(tmp_path):
    """The ordinary end of a task that finished while the agent slept. Not an
    anomaly, and not worth a log line about killing something."""
    from ganymede.host import agent

    sandbox.write_lease_crumb(tmp_path, "t1", _fresh(5000), "ganymede-job-t1")
    runner = FakeRunner({"inspect": sandbox.Completed(1, stderr="No such object")})
    assert agent.reap_orphaned_jobs(_host_config(tmp_path), runner=runner) == []
    assert runner.argv_for("kill") is None
    assert sandbox.read_lease_crumb(tmp_path) is None


def test_an_unparseable_crumb_is_treated_as_stale(tmp_path):
    """Fail-safe direction: the cost is killing a container that might have
    been fine, and the alternative is never killing one."""
    from ganymede.host import agent

    (tmp_path / "lease.json").write_text(json.dumps(
        {"task_id": "t1", "renewed_at": "not a timestamp",
         "container": "ganymede-job-t1"}))
    runner = FakeRunner({"inspect": sandbox.Completed(0, stdout="true")})
    assert agent.reap_orphaned_jobs(_host_config(tmp_path), runner=runner) == \
        ["ganymede-job-t1"]


def test_the_reaper_survives_its_real_runner(tmp_path):
    """The tests above inject a runner; production resolves
    ``runtime._run``. The tick swallows exceptions from the reaper and logs
    them, so a signature drift here would silently stop the backstop with the
    whole suite still green -- this is the path that would drift."""
    from ganymede.host import agent

    sandbox.write_lease_crumb(tmp_path, "t1", _fresh(5000), "ganymede-job-t1")
    config = _host_config(tmp_path)
    config.docker_bin = "definitely-not-a-real-binary-xyz"
    assert agent.reap_orphaned_jobs(config) == []


def test_a_host_that_never_opted_in_reaps_nothing(tmp_path):
    """No job scratch means this machine does not run submitter code at all,
    so there is nothing for the reaper to have an opinion about."""
    from ganymede.host import agent
    from ganymede.host.config import HostConfig

    config = HostConfig(coordinator_url="http://c", key="k")
    runner = FakeRunner()
    assert agent.reap_orphaned_jobs(config, runner=runner) == []
    assert runner.calls == []


def test_the_worker_container_gets_the_scratch_mount_only_when_opted_in(tmp_path):
    from ganymede.host import runtime as runtime_mod
    from ganymede.host.config import CONTAINER_JOB_SCRATCH

    opted_in = runtime_mod.DockerRuntime(_host_config(tmp_path)).run_argv("img", {})
    assert any(m.endswith(CONTAINER_JOB_SCRATCH) for m in _flag_values(opted_in, "-v"))

    from ganymede.host.config import HostConfig
    plain = runtime_mod.DockerRuntime(
        HostConfig(coordinator_url="http://c", key="k")).run_argv("img", {})
    assert not any(CONTAINER_JOB_SCRATCH in m for m in _flag_values(plain, "-v"))


def test_the_state_dir_stays_read_only_next_to_the_new_mount(tmp_path):
    """The writable scratch is an addition, not a relaxation: the argument for
    a read-only state dir -- a worker that can write there can forge the kill
    switch -- does not weaken because the worker now needs somewhere else to
    write."""
    from ganymede.host import runtime as runtime_mod

    argv = runtime_mod.DockerRuntime(_host_config(tmp_path)).run_argv("img", {})
    state_mounts = [m for m in _flag_values(argv, "-v") if m.endswith(":ro")]
    assert state_mounts, "the state dir mount should still be there, read-only"
