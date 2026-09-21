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
# The container name (shared by JobContainer, the heartbeat thread's cancel
# handler, and the lease crumb -- three places that must agree on a name
# without sharing an instance)
# ==========================================================================


def test_container_name_for_is_deterministic_from_the_task_id_alone():
    assert sandbox.container_name_for("task-abcdef123456") == \
        sandbox.container_name_for("task-abcdef123456")
    assert sandbox.container_name_for("task-abcdef123456").startswith("ganymede-job-")


def test_a_job_container_names_itself_the_same_way(config):
    """The name ``loop.Worker._run_contained`` computes for a cancel, and the
    name a lease crumb records, must be the exact name a freshly constructed
    ``JobContainer`` (never given one explicitly) picks for itself -- they are
    never the same instance."""
    job = sandbox.JobContainer(task_id="task-abcdef123456", config=config,
                               runner=FakeRunner())
    assert job.container_name == sandbox.container_name_for("task-abcdef123456")


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


def test_an_explicit_empty_override_withholds_gpus_even_with_devices(config):
    """``gpus=None`` (above) means "no override, defer to the lease". An
    explicit empty string is a different thing -- an operator who set
    ``GANYMEDE_JOB_GPUS=`` on purpose -- and it wins over a real lease
    allocation exactly like a non-empty override would, just withholding
    rather than granting."""
    job = sandbox.JobContainer(
        "t", sandbox.SandboxConfig(scratch_root=config.scratch_root, gpus=""),
        runner=FakeRunner())
    assert "--gpus" not in job.run_argv("img", backend="cuda", devices=[0])


# ==========================================================================
# Backend-aware container device pinning (docs/14 §2)
# ==========================================================================


@pytest.mark.parametrize("backend,indices,expected", [
    ("cuda", [1], ["--gpus", '"device=1"']),
    ("cuda", [1, 2, 3], ["--gpus", '"device=1,2,3"']),
    ("cuda", [3, 1, 2], ["--gpus", '"device=1,2,3"']),  # sorted regardless of input order
    ("rocm", [1], ["--device=/dev/kfd", "--device=/dev/dri/renderD129",
                   "--group-add", "video"]),
    ("rocm", [0, 1], ["--device=/dev/kfd", "--device=/dev/dri/renderD128",
                      "--device=/dev/dri/renderD129", "--group-add", "video"]),
    ("xpu", [0], ["--device=/dev/dri/renderD128"]),
    ("xpu", [0, 2], ["--device=/dev/dri/renderD128", "--device=/dev/dri/renderD130"]),
    ("cpu", [0], []),
    ("cpu", [0, 1], []),  # GANYMEDE_CPU_SLOTS > 1: no real core mapping to pin to.
])
def test_device_argv_matches_docs_14s_table(backend, indices, expected):
    # ``env={}``: these assert the *table*, and a developer with an ambient
    # CUDA_VISIBLE_DEVICES set should not see them fail. The composition that
    # variable triggers is tested separately below.
    assert sandbox.device_argv(backend, indices, env={}) == expected


def test_device_argv_is_a_noop_with_nothing_to_pin():
    """Unreachable for a real lease (``devices.allocate`` raises on
    ``count <= 0``), but reachable from a payload built before docs/14 landed
    -- must not invent a pin for a device that was never named."""
    assert sandbox.device_argv("cuda", [], env={}) == []


def test_device_argv_refuses_mps_with_a_device_to_pin():
    """mps is always exactly one device and its in-process pin is a correct
    no-op (``loop._pin_env`` returns ``{}``), but a container has no way to
    reach that device at all -- Docker Desktop for Mac has no Metal
    passthrough. Unlike the in-process case, there is no safe unpinned
    fallback here, so this refuses rather than silently start a GPU-less
    container."""
    assert sandbox.device_argv("mps", [0]) is None


def test_device_argv_refuses_an_unrecognised_backend_with_devices_to_pin():
    """docs/14 §2: a backend with no known pinning form refuses to launch
    rather than falling back to 'all devices'."""
    assert sandbox.device_argv("some_future_backend", [0, 1]) is None


def test_a_single_device_host_pins_to_that_one_device(config):
    """Backward compatibility: the overwhelmingly common case (one card, one
    lease) must end up with exactly that device, not 'all'."""
    job = sandbox.JobContainer("t", config, runner=FakeRunner())
    argv = job.run_argv("img", backend="cuda", devices=[0])
    assert _flag_value(argv, "--gpus") == '"device=0"'


def test_a_host_that_never_reports_devices_still_starts(config):
    """A payload with no ``devices`` field at all -- an old coordinator, or a
    job type that never named one -- must not crash and must not refuse.

    On a discrete backend it also must not quietly hand over *nothing*: an
    absent field means the coordinator predates docs/14, that coordinator is
    still enforcing one lease per machine, and ``all`` is both safe and what
    this host used to get. ``None`` and ``[]`` are the same case and take the
    same path.
    """
    job = sandbox.JobContainer("t", config, runner=FakeRunner())
    argv = job.run_argv("img", backend="cuda", devices=None)
    assert _flag_value(argv, "--gpus") == "all"


def test_an_unpinnable_backend_refuses_to_start_rather_than_run_unconfined(config):
    """The container-launch fail-closed rule, exercised through ``start`` --
    not just ``device_argv`` in isolation. Landing here as ``SandboxError``
    matters: ``worker.loop._run_contained`` catches exactly that type and
    abandons-and-backs-off, which is the correct handling for "this host
    cannot run this safely" (a new exception type would fall through to the
    abandon-and-*reraise* handler and take the worker down)."""
    job = sandbox.JobContainer("t", config, runner=FakeRunner())
    with pytest.raises(sandbox.SandboxError):
        job.run_argv("img", backend="mps", devices=[0])


def test_an_operator_override_wins_over_the_leases_own_devices(config, caplog):
    """docs/14 §2 keeps ``GANYMEDE_JOB_GPUS`` as an operator escape hatch --
    the same relationship every other ``SandboxConfig`` field has to the task
    (``job_max_runtime_sec`` clamps, never the reverse). Trusting it is a
    deliberate, logged act: setting it on a multi-lease host can re-open the
    double-booking docs/14 exists to prevent, so it is not silent."""
    override = sandbox.JobContainer(
        "t", sandbox.SandboxConfig(scratch_root=config.scratch_root, gpus="all"),
        runner=FakeRunner())
    with caplog.at_level("WARNING"):
        argv = override.run_argv("img", backend="cuda", devices=[2, 3])
    assert _flag_value(argv, "--gpus") == "all"
    assert any("GANYMEDE_JOB_GPUS" in r.message for r in caplog.records)


def test_pinned_argv_keeps_every_confinement_flag(config):
    """``device_argv`` adds flags; it must never come at the cost of the
    §4.6 baseline this file's other tests check one at a time. A golden argv
    would be rewritten by whoever next adds a flag -- exactly the edit that
    must not silently drop one of these -- so they are asserted individually,
    against the *pinned* path specifically, which is easy to exercise only in
    isolation and never as the integrated argv a real multi-GPU host sends."""
    job = sandbox.JobContainer("t", config, runner=FakeRunner())
    argv = job.run_argv("img", backend="cuda", devices=[0, 1])
    for flag in ("--cap-drop=ALL", "--security-opt=no-new-privileges", "--read-only"):
        assert flag in argv, f"{flag} missing from the pinned path"
    assert _flag_value(argv, "--user") == f"{sandbox.CONTAINER_UID}:{sandbox.CONTAINER_UID}"
    assert _flag_value(argv, "--network") == "none"
    assert _flag_value(argv, "--gpus") == '"device=0,1"'


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


def test_cancelling_a_container_that_does_not_exist_yet_does_not_raise(config):
    """A cancel latched on the heartbeat thread can land before ``jt.run`` has
    called ``start`` -- during the archive pull, which has no should_stop
    check of its own. ``docker kill``/``stop`` on an unknown name is a nonzero
    exit, not a subprocess error, and ``cancel`` never inspects the return
    code -- so acting early must be a harmless no-op, never a raise out of the
    caller (the worker's heartbeat thread, for ``loop.Worker._run_contained``'s
    ``on_cancel``)."""
    runner = FakeRunner({
        "kill": sandbox.Completed(returncode=1, stderr="Error: No such container"),
        "stop": sandbox.Completed(returncode=1, stderr="Error: No such container"),
    })
    job = sandbox.JobContainer("never-started", config, runner=runner)
    job.cancel("hard")  # must not raise
    job.cancel("soft")  # must not raise


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
    crumb = sandbox.read_lease_crumb(tmp_path, "task1")
    assert crumb["task_id"] == "task1"
    assert crumb["container"] == "c1"


def test_the_crumb_is_written_atomically(tmp_path):
    """The host agent reads this file on a timer with no locking. A partial
    write would parse as a missing crumb and reap a live job."""
    sandbox.write_lease_crumb(tmp_path, "task1", "2026-09-07T00:00:00+00:00", "c1")
    sandbox.write_lease_crumb(tmp_path, "task1", "2026-09-07T00:01:00+00:00", "c2")
    assert sandbox.read_lease_crumb(tmp_path, "task1")["container"] == "c2"
    assert not list((tmp_path / "leases").glob("*.tmp"))


def test_two_tasks_crumbs_do_not_clobber_each_other(tmp_path):
    """The point of splitting the crumb per task (Step 0): a host running more
    than one contained job at once must not have the second task's heartbeat
    stamp over the first's record, the way one shared ``lease.json`` would."""
    sandbox.write_lease_crumb(tmp_path, "task1", "2026-09-07T00:00:00+00:00", "c1")
    sandbox.write_lease_crumb(tmp_path, "task2", "2026-09-07T00:01:00+00:00", "c2")

    crumb1 = sandbox.read_lease_crumb(tmp_path, "task1")
    crumb2 = sandbox.read_lease_crumb(tmp_path, "task2")
    assert crumb1["task_id"] == "task1" and crumb1["container"] == "c1"
    assert crumb2["task_id"] == "task2" and crumb2["container"] == "c2"


def test_reading_an_absent_crumb_is_none_not_an_error(tmp_path):
    assert sandbox.read_lease_crumb(tmp_path, "task1") is None
    sandbox.clear_lease_crumb(tmp_path, "task1")  # must not raise either
    sandbox.clear_lease_crumb(tmp_path)  # nor the legacy (task_id=None) path


def test_a_corrupt_crumb_reads_as_absent(tmp_path):
    (tmp_path / "leases").mkdir(parents=True)
    (tmp_path / "leases" / "task1.json").write_text("{not json")
    assert sandbox.read_lease_crumb(tmp_path, "task1") is None
    assert sandbox.read_all_lease_crumbs(tmp_path) == []


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
    assert sandbox.read_lease_crumb(tmp_path, "t1") is None


def test_a_stale_crumb_with_no_container_just_clears(tmp_path):
    """The ordinary end of a task that finished while the agent slept. Not an
    anomaly, and not worth a log line about killing something."""
    from ganymede.host import agent

    sandbox.write_lease_crumb(tmp_path, "t1", _fresh(5000), "ganymede-job-t1")
    runner = FakeRunner({"inspect": sandbox.Completed(1, stderr="No such object")})
    assert agent.reap_orphaned_jobs(_host_config(tmp_path), runner=runner) == []
    assert runner.argv_for("kill") is None
    assert sandbox.read_lease_crumb(tmp_path, "t1") is None


def test_a_legacy_single_file_crumb_is_tolerated_and_treated_as_stale(tmp_path):
    """A worker built before Step 0 split the crumb per task may still leave a
    bare ``lease.json`` behind. The reaper must not crash on it, and an
    unparseable *timestamp* inside it (distinct from unparseable JSON) is the
    fail-safe direction: the cost is killing a container that might have been
    fine, and the alternative is never killing one."""
    from ganymede.host import agent

    (tmp_path / "lease.json").write_text(json.dumps(
        {"task_id": "t1", "renewed_at": "not a timestamp",
         "container": "ganymede-job-t1"}))
    runner = FakeRunner({"inspect": sandbox.Completed(0, stdout="true")})
    assert agent.reap_orphaned_jobs(_host_config(tmp_path), runner=runner) == \
        ["ganymede-job-t1"]
    assert not (tmp_path / "lease.json").exists()


def test_the_reaper_considers_every_live_crumb(tmp_path):
    """Concurrent tasks mean concurrent orphans; the sweep must not stop at
    the first crumb it finds a container name in."""
    from ganymede.host import agent

    sandbox.write_lease_crumb(tmp_path, "t1", _fresh(5000), "ganymede-job-t1")
    sandbox.write_lease_crumb(tmp_path, "t2", _fresh(5000), "ganymede-job-t2")
    runner = FakeRunner({"inspect": sandbox.Completed(0, stdout="true")})
    killed = agent.reap_orphaned_jobs(_host_config(tmp_path), runner=runner)

    assert sorted(killed) == ["ganymede-job-t1", "ganymede-job-t2"]
    assert sandbox.read_lease_crumb(tmp_path, "t1") is None
    assert sandbox.read_lease_crumb(tmp_path, "t2") is None


def test_the_reaper_leaves_a_fresh_crumb_beside_a_stale_one(tmp_path):
    """One task going quiet must not implicate another that is still
    renewing -- exactly the failure a single shared crumb file could not have
    avoided."""
    from ganymede.host import agent

    sandbox.write_lease_crumb(tmp_path, "t1", _fresh(30), "ganymede-job-t1")
    sandbox.write_lease_crumb(tmp_path, "t2", _fresh(5000), "ganymede-job-t2")
    runner = FakeRunner({"inspect": sandbox.Completed(0, stdout="true")})
    killed = agent.reap_orphaned_jobs(_host_config(tmp_path), runner=runner)

    assert killed == ["ganymede-job-t2"]
    assert sandbox.read_lease_crumb(tmp_path, "t1") is not None
    assert sandbox.read_lease_crumb(tmp_path, "t2") is None


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


def test_a_gpu_host_whose_coordinator_named_no_devices_still_gets_every_device(
    job, caplog
):
    """Version skew, not a ledger violation -- and the two want opposite
    answers. A coordinator that allocates always names at least one device, so
    an empty list means the coordinator predates docs/14; that coordinator is
    still enforcing one lease per machine, which makes ``all`` both safe and
    what this host used to get. Emitting nothing would silently run a
    submitter's GPU job on CPU: it "works", far slower, and nobody finds out.
    """
    with caplog.at_level("WARNING"):
        argv = job.run_argv("img", backend="cuda", devices=[])

    assert _flag_value(argv, "--gpus") == "all"
    assert any("named no devices" in r.message for r in caplog.records), (
        "the fallback has to be visible, or a real coordinator bug hides in it"
    )


def test_a_cpu_host_with_no_devices_named_gets_no_device_flags(job):
    """The same empty list means something different here: there is no
    accelerator to hand over in the first place, so nothing is the right answer
    and the discrete-backend fallback must not fire."""
    argv = job.run_argv("img", backend="cpu", devices=[])

    assert "--gpus" not in argv


def test_a_named_device_on_an_unpinnable_backend_still_refuses(job):
    """The fallback above must not soften the refusal it sits next to. Devices
    named + no way to honour them is the ledger case: obeying loosely would
    double-book a card already promised to another lease."""
    with pytest.raises(sandbox.SandboxError):
        job.run_argv("img", backend="mps", devices=[0])


# --------------------------------------------------------------------------
# A visibility restriction already on the worker (docs/14 §2, review-added)
# --------------------------------------------------------------------------


def test_device_argv_composes_through_an_ambient_cuda_restriction():
    """The NVIDIA container runtime addresses cards by absolute physical index
    and does not inherit the worker's own ``CUDA_VISIBLE_DEVICES``.

    A worker launched restricted to ``4,5,6,7`` reports its cards to the
    coordinator as local indices 0-3 (``probe.run_probe`` enumerates whatever
    torch can see). Writing a lease's local ``[2]`` straight into
    ``--gpus device=`` would hand the container physical card 2 -- one
    deliberately withheld from Ganymede -- instead of card 6.
    """
    env = {"CUDA_VISIBLE_DEVICES": "4,5,6,7"}
    assert sandbox.device_argv("cuda", [2], env=env) == ["--gpus", '"device=6"']
    assert sandbox.device_argv("cuda", [0, 3], env=env) == ["--gpus", '"device=4,7"']


def test_device_argv_composes_a_uuid_valued_ambient_list():
    """``--gpus device=`` accepts a UUID wherever it accepts an index, so a
    UUID-valued restriction composes through unchanged."""
    env = {"CUDA_VISIBLE_DEVICES": "GPU-aaa,GPU-bbb"}
    assert sandbox.device_argv("cuda", [1], env=env) == ["--gpus", '"device=GPU-bbb"']


def test_device_argv_refuses_a_local_index_the_ambient_list_cannot_resolve():
    """Refuse, rather than hand the container a card this lease does not hold
    -- the same answer this function already gives an unpinnable backend."""
    assert sandbox.device_argv("cuda", [4], env={"CUDA_VISIBLE_DEVICES": "4,5"}) is None


def test_device_argv_is_unchanged_when_nothing_is_ambient():
    """The ordinary deployment, byte for byte."""
    assert sandbox.device_argv("cuda", [0, 2], env={"CUDA_VISIBLE_DEVICES": ""}) ==         ["--gpus", '"device=0,2"']


def test_the_in_process_and_container_pins_share_one_translation():
    """``loop._pin_env`` and ``device_argv`` must not drift: both resolve a
    lease's local indices through the same ``compose_visible``."""
    from ganymede.worker import loop as loop_mod

    env = {"CUDA_VISIBLE_DEVICES": "4,5,6,7"}
    assert loop_mod._pin_env("cuda", [2], env=env)["CUDA_VISIBLE_DEVICES"] == "6"
    assert sandbox.device_argv("cuda", [2], env=env) == ["--gpus", '"device=6"']
