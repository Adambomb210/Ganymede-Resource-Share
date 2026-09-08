"""Runtime confinement for submitter job code (docs/11 §2, §3).

The worker owns the claim / heartbeat / submit loop and now also supervises a
**sibling** job container: it pulls the archive the coordinator named, verifies
the digest before anything is loaded, stages inputs into a scratch directory,
runs the image under §2.2's flags, and signals it on a cancel.

Threat model is Decision 3 -- a trusted author's honest mistake, not an attacker
against the runtime. That buys the flags below and does not buy gVisor / Kata
(docs/11 §5).

**Who launches it.** §2.1 forbids handing the worker the host's Docker socket:
that is root-equivalent on the machine and defeats the §4.6 hardening the worker
itself runs under. What this module does instead is invoke a runtime *binary*
and never a socket path -- so whether that binary talks to a scoped socket proxy
(``DOCKER_HOST`` pointing at it) or is a rootless Podman is a deployment
decision, made where the host agent is configured, not a code path here. The
deviation to record honestly: **Ganymede does not yet ship the socket proxy**, so
an operator who points ``GANYMEDE_JOB_RUNTIME`` at a plain ``docker`` on the
host socket has given the worker more than §2.1 wants it to have. The interface
is the seam that makes fixing that a config change.

Every subprocess goes through an injectable ``runner``, the pattern
``host/runtime.py`` already uses -- so the flag template, the digest check and
the kill path are all unit-testable on a machine with no container runtime at
all, which is the machine this was written on.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("ganymede.worker.sandbox")

# A pull of a multi-GB archive; everything else is a control-plane call that
# should answer immediately or is wedged.
PULL_TIMEOUT_SEC = 3600
RUNTIME_TIMEOUT_SEC = 30

# docs/11 §3: how long a soft cancel waits for the job to checkpoint and exit
# before it becomes a hard one. Never past the lease -- the caller clamps.
DEFAULT_CANCEL_GRACE_SEC = 120

# Where the job sees its own scratch. The host side is a per-task directory.
CONTAINER_SCRATCH = "/scratch"

# The uid the job runs as. Matches the worker image's own unprivileged user
# (host/config.CONTAINER_UID) so a bind-mounted scratch dir written by one is
# readable by the other.
CONTAINER_UID = 1000


class SandboxError(RuntimeError):
    """The job could not be run. The task is abandoned, not failed."""


class DigestMismatch(SandboxError):
    """The archive did not hash to what the coordinator said (docs/11 §2.3).

    Its own type because its handling is specific: abandon with reason
    ``image_digest_mismatch`` and let the task be re-queued. It is not a
    verdict on the job -- the bytes in flight were wrong, and the next worker
    may pull them intact.
    """


@dataclass(frozen=True)
class SandboxConfig:
    """The half of §2.2 that is a deployment's to choose.

    Defaults are the §4.6 baseline the worker container itself runs under, so a
    job never gets more of the machine than the worker was given.
    """

    scratch_root: Path
    runtime_bin: str = "docker"
    memory: str = "16g"
    cpus: str = "0.9"
    gpus: str | None = "all"
    pids_limit: int = 512
    scratch_gb: int = 50
    # §2.2: the ceiling on spec.max_runtime_sec, not the value itself.
    job_max_runtime_sec: int = 86_400
    cancel_grace_sec: int = DEFAULT_CANCEL_GRACE_SEC
    # §2.2's storage quota is driver-dependent (`--storage-opt size=` needs
    # devicemapper/btrfs/zfs or xfs with pquota). Off by default because on
    # overlay2 -- the common case -- passing it is a hard error at `run`, and a
    # flag that refuses to start the job is worse than a quota that is a
    # directory the worker wipes.
    storage_opt_size: bool = False

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "SandboxConfig":
        env = os.environ if environ is None else environ
        root = env.get("GANYMEDE_JOB_SCRATCH")
        if not root:
            raise SandboxError(
                "GANYMEDE_JOB_SCRATCH is not set: a contained job needs a "
                "host-visible scratch directory (docs/11 §2.3)"
            )
        return cls(
            scratch_root=Path(root),
            runtime_bin=env.get("GANYMEDE_JOB_RUNTIME", "docker"),
            memory=env.get("GANYMEDE_JOB_MEMORY", "16g"),
            cpus=env.get("GANYMEDE_JOB_CPUS", "0.9"),
            gpus=env.get("GANYMEDE_JOB_GPUS", "all") or None,
            pids_limit=int(env.get("GANYMEDE_JOB_PIDS_LIMIT", "512")),
            scratch_gb=int(env.get("GANYMEDE_JOB_SCRATCH_GB", "50")),
            job_max_runtime_sec=int(env.get("GANYMEDE_JOB_MAX_RUNTIME_SEC", "86400")),
            cancel_grace_sec=int(
                env.get("GANYMEDE_JOB_CANCEL_GRACE_SEC", str(DEFAULT_CANCEL_GRACE_SEC))
            ),
            storage_opt_size=env.get("GANYMEDE_JOB_STORAGE_QUOTA", "").lower()
            in {"1", "true", "yes", "on"},
        )


def detect_runtime(runner=None, runtime_bin: str | None = None) -> str | None:
    """The runtime this machine can launch a job container with, or ``None``.

    Reported in the compute profile, where the coordinator's claim gate reads
    it (docs/11 §4): a machine that answers ``None`` is refused submitter jobs
    with ``no_container_runtime`` rather than handed an image it cannot run.

    Deliberately an actual call, not a ``shutil.which``: inside the worker
    container the binary can be present while the socket it needs is not, and a
    profile that claims a capability the machine does not have costs a real
    task a real lease.
    """
    binary = runtime_bin or os.environ.get("GANYMEDE_JOB_RUNTIME", "docker")
    run = runner or _run
    try:
        result = run([binary, "version", "--format", "{{.Server.Version}}"],
                     timeout=RUNTIME_TIMEOUT_SEC)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return Path(binary).stem


_LOADED_ID = re.compile(r"sha256:[0-9a-f]{64}")


@dataclass
class JobContainer:
    """One contained job: pull, verify, load, run, and the kill path."""

    task_id: str
    config: SandboxConfig
    runner: object = None
    container_name: str = ""
    image_id: str | None = None

    def __post_init__(self) -> None:
        self._run = self.runner or _run
        if not self.container_name:
            self.container_name = f"ganymede-job-{self.task_id[:12]}"

    # -- scratch (§2.3) ---------------------------------------------------

    @property
    def scratch(self) -> Path:
        return self.config.scratch_root / self.task_id

    def prepare_scratch(self) -> Path:
        """``/scratch/in`` for staged inputs, ``/scratch/out`` for results.

        Wiped first, not merged: a retry of the same task id must not see the
        previous attempt's half-written output and mistake it for its own.
        """
        self.wipe_scratch()
        (self.scratch / "in").mkdir(parents=True, exist_ok=True)
        (self.scratch / "out").mkdir(parents=True, exist_ok=True)
        return self.scratch

    def wipe_scratch(self) -> None:
        shutil.rmtree(self.scratch, ignore_errors=True)

    # -- pull and verify (§2.3) -------------------------------------------

    def fetch_archive(self, download, url: str, expected_digest: str) -> Path:
        """Pull the archive and hash it *before* anything loads it.

        The order is the point. ``docker load`` parses an archive an author
        controls; running it on bytes that failed their check would be trusting
        the thing being checked. The digest is the coordinator's row (docs/11
        §1.1), which is the only value a worker can independently recompute.
        """
        self.scratch.mkdir(parents=True, exist_ok=True)
        path = self.scratch / "image.tar"
        payload = download(url)
        path.write_bytes(payload)
        actual = hashlib.sha256(payload).hexdigest()
        want = (expected_digest or "").split(":")[-1].lower()
        if not want or actual != want:
            path.unlink(missing_ok=True)
            raise DigestMismatch(
                f"image archive hashed {actual}, coordinator said {want or '(none)'}"
            )
        return path

    def load_image(self, archive: Path) -> str:
        """``docker load`` and return the image **ID**, never the tag.

        The archive carries its own ``RepoTags``, which the author chose and
        which can collide with an image already on the machine -- running by tag
        would let a submitter's archive decide which bytes execute. The id is
        content-addressed and cannot.
        """
        result = self._run(
            [self.config.runtime_bin, "load", "-i", str(archive)],
            timeout=PULL_TIMEOUT_SEC,
        )
        if result.returncode != 0:
            raise SandboxError(f"image load failed: {result.stderr.strip()}")
        match = _LOADED_ID.search(result.stdout or "")
        if match is None:
            raise SandboxError(
                f"image load reported no image id: {(result.stdout or '').strip()}"
            )
        self.image_id = match.group(0)
        return self.image_id

    # -- run (§2.2, §2.3, §2.4) -------------------------------------------

    def runtime_ceiling(self, max_runtime_sec: int | None) -> int:
        """The job's ask, clamped by the operator's ceiling (§2.2).

        Its own method rather than a value stashed on the instance by whichever
        of ``run_argv`` / ``start`` ran last: a caller that builds argv once and
        starts twice would otherwise inherit the previous task's clamp, and
        nothing would say so.
        """
        return min(max_runtime_sec or self.config.job_max_runtime_sec,
                   self.config.job_max_runtime_sec)

    def run_argv(self, image_id: str, *, env: dict[str, str] | None = None,
                 max_runtime_sec: int | None = None) -> list[str]:
        """The flag template. §4.6's baseline plus §2.2's additions.

        Notes on the ones that are not obvious:

        - ``--memory`` and ``--memory-swap`` are set **equal**, which is how
          Docker spells "no swap". Without it a job over its memory cap slows to
          a crawl against the host's disk instead of dying, and a task that
          takes twenty times its budget is worse for the fleet than one that
          fails.
        - ``--network none`` is unconditional here. §2.4's per-job allowlist and
          its CONNECT proxy are a separate piece; the common job reads
          ``/scratch/in``, computes, writes ``/scratch/out`` and needs no network
          at all, because the worker does every transfer.
        - The worker's own state dir is **not** mounted. §4.6's reasoning is
          that a process which can write there can forge the kill switch, and
          the job is exactly the process that must not.
        - ``--user`` is forced to a non-root uid regardless of what the image's
          own ``USER`` says. The scan flags a root image (docs/11 §1.3) but a
          flag is advice; this is the enforcement.
        """
        cfg = self.config
        argv = [
            cfg.runtime_bin, "run",
            "--detach",
            "--name", self.container_name,
            # §4.6 baseline, inherited whole.
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--read-only",
            "--user", f"{CONTAINER_UID}:{CONTAINER_UID}",
            # §2.2 additions.
            "--memory", cfg.memory,
            "--memory-swap", cfg.memory,
            "--cpus", cfg.cpus,
            "--pids-limit", str(cfg.pids_limit),
            "--ipc=private",
            "--ulimit", "core=0",
            "--network", "none",
            "--stop-timeout", str(cfg.cancel_grace_sec),
            # §2.3: scratch, /tmp and /run. Nothing else is writable.
            "-v", f"{self.scratch}:{CONTAINER_SCRATCH}",
            "--tmpfs", "/tmp",
            "--tmpfs", "/run",
        ]
        if cfg.storage_opt_size:
            argv += ["--storage-opt", f"size={cfg.scratch_gb}G"]
        if cfg.gpus:
            argv += ["--gpus", cfg.gpus]
        for name in sorted(env or {}):
            # Name only, never NAME=value: a value here lands in the host
            # process table and in `inspect` output for the life of the
            # container. Same reasoning as host/runtime.run_argv.
            argv += ["-e", name]
        argv += [
            "-e", "GANYMEDE_SCRATCH",
            "-e", "GANYMEDE_MAX_RUNTIME_SEC",
            image_id,
        ]
        return argv

    def start(self, image_id: str, *, env: dict[str, str] | None = None,
              max_runtime_sec: int | None = None) -> str:
        argv = self.run_argv(image_id, env=env, max_runtime_sec=max_runtime_sec)
        child_env = dict(os.environ)
        child_env.update(env or {})
        child_env["GANYMEDE_SCRATCH"] = CONTAINER_SCRATCH
        child_env["GANYMEDE_MAX_RUNTIME_SEC"] = str(
            self.runtime_ceiling(max_runtime_sec)
        )
        result = self._run(argv, timeout=RUNTIME_TIMEOUT_SEC, env=child_env)
        if result.returncode != 0:
            raise SandboxError(f"could not start job container: {result.stderr.strip()}")
        return self.container_name

    def is_running(self) -> bool:
        result = self._run(
            [self.config.runtime_bin, "inspect", "-f", "{{.State.Running}}",
             self.container_name],
            timeout=RUNTIME_TIMEOUT_SEC,
        )
        return result.returncode == 0 and (result.stdout or "").strip() == "true"

    def exit_code(self) -> int | None:
        result = self._run(
            [self.config.runtime_bin, "inspect", "-f", "{{.State.ExitCode}}",
             self.container_name],
            timeout=RUNTIME_TIMEOUT_SEC,
        )
        if result.returncode != 0:
            return None
        try:
            return int((result.stdout or "").strip())
        except ValueError:
            return None

    # -- the kill path (§3) -----------------------------------------------

    def cancel(self, mode: str, *, grace_sec: int | None = None) -> None:
        """Act on a cancel that arrived on a heartbeat (docs/11 §3 step 3).

        ``soft`` is SIGTERM with a deadline: the job is expected to trap it,
        checkpoint the unit it is on, and exit. One that ignores it is SIGKILLed
        at the timeout -- the grace is a promise about how long the swarm waits,
        not about whether the container stops.

        ``hard`` is SIGKILL now, and takes no view on what the job was doing.
        """
        grace = grace_sec if grace_sec is not None else self.config.cancel_grace_sec
        if mode == "hard":
            self._run([self.config.runtime_bin, "kill", self.container_name],
                      timeout=RUNTIME_TIMEOUT_SEC)
            return
        self._run(
            [self.config.runtime_bin, "stop", "--time", str(grace),
             self.container_name],
            timeout=grace + RUNTIME_TIMEOUT_SEC,
        )

    def remove(self) -> None:
        self._run([self.config.runtime_bin, "rm", "-f", self.container_name],
                  timeout=RUNTIME_TIMEOUT_SEC)

    def cleanup(self) -> None:
        """Everything this task left on the machine. Called on every exit path,
        including the failures -- a scratch dir that survives a crash is a disk
        that fills up over a week of them."""
        self.remove()
        self.wipe_scratch()


# --------------------------------------------------------------------------
# The lease crumb (docs/11 §3, wedged-worker path)
# --------------------------------------------------------------------------


def write_lease_crumb(scratch_root: Path, task_id: str, renewed_at: str,
                      container: str | None = None) -> Path:
    """Record that the lease was renewed, where the *host agent* can read it.

    Not in the state dir, which is mounted read-only into the worker precisely
    so the worker cannot forge the contributor's kill switch (host/runtime
    ``run_argv``). The job scratch root is the one directory both the worker and
    the host agent can see and the worker may write -- and it exists already,
    because §2.3's bind mount needs it.
    """
    scratch_root.mkdir(parents=True, exist_ok=True)
    path = scratch_root / "lease.json"
    payload = {"task_id": task_id, "renewed_at": renewed_at, "container": container}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)
    return path


def read_lease_crumb(scratch_root: Path) -> dict | None:
    try:
        return json.loads((scratch_root / "lease.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def clear_lease_crumb(scratch_root: Path) -> None:
    try:
        (scratch_root / "lease.json").unlink()
    except OSError:
        pass


# --------------------------------------------------------------------------
# subprocess
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Completed:
    returncode: int
    stdout: str = ""
    stderr: str = ""


def _run(argv: list[str], *, timeout: int, env: dict[str, str] | None = None) -> Completed:
    try:
        proc = subprocess.run(  # noqa: S603 - argv is built here, never a shell string
            argv, capture_output=True, text=True, timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired:
        # A control-plane call that does not answer means a wedged daemon. The
        # caller's next tick tries again; blocking on it would pile up.
        return Completed(returncode=124, stderr=f"timeout after {timeout}s")
    except (OSError, subprocess.SubprocessError) as exc:
        return Completed(returncode=127, stderr=str(exc))
    return Completed(proc.returncode, proc.stdout or "", proc.stderr or "")
