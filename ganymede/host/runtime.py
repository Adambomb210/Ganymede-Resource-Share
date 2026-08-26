"""Starting and stopping the worker (docs/02-architecture-v2.md 7, step 4).

Two implementations behind one small protocol, because 4.1's delivery table
says the container is *a* packaging option rather than *the* architecture:

    Linux + NVIDIA        container, full 4.6 hardening
    macOS + Apple Silicon pip + launchd -- no container, the runtime cannot
                          reach the GPU at all (6.8)
    Windows + NVIDIA      either; Docker Desktop reaches the GPU through WSL2

The host agent must not care which. It decides *whether* to run and *what* to
run, calls `start`, and the runtime knows how.

Standard library only, like the rest of ``ganymede/host``. That rules out the
Docker SDK, which is a real cost -- argv construction by hand is more code than
`client.containers.run(...)` -- and buys a host agent that installs by copying
a directory onto a machine with nothing but a Python interpreter on it.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ganymede.host.config import HostConfig

log = logging.getLogger("ganymede.host.runtime")

# Long enough for `docker pull` of a ~2 GB image on a slow connection. Every
# other docker call gets the short timeout: a `docker inspect` that has not
# answered in 30 s means the daemon is wedged, and blocking the agent's tick on
# it just means the timer fires again into a pile-up.
PULL_TIMEOUT_SEC = 3600
DOCKER_TIMEOUT_SEC = 30

# How long to wait for a native worker to notice its stop sentinel before
# escalating to a signal. It polls on the heartbeat interval (4.4), so this has
# to exceed one of those with room to spare.
NATIVE_STOP_GRACE_SEC = 90


class RuntimeError_(RuntimeError):
    """Base for this module, so a caller can catch the whole surface."""


class CommandFailed(RuntimeError_):
    """A subprocess exited non-zero, or did not exit at all.

    Modelled on ``CoordinatorError`` in ``ganymede/worker/client.py``: the
    command and its stderr travel with the exception, because "docker failed"
    in a log a contributor pastes into an issue is not a diagnosis.
    """

    def __init__(self, argv: list[str], returncode: int, stderr: str):
        super().__init__(f"{' '.join(argv)} exited {returncode}: {stderr.strip()[:400]}")
        self.argv = argv
        self.returncode = returncode
        self.stderr = stderr


@dataclass(frozen=True)
class RuntimeStatus:
    running: bool
    image: str | None = None
    # Present for a container that exists but has stopped. The agent uses it to
    # tell "never started" from "started and died", which are different
    # problems: the second one wants the exit code surfaced, not another start.
    exited: bool = False
    exit_code: int | None = None
    detail: str = ""


class WorkerRuntime(Protocol):
    def status(self) -> RuntimeStatus: ...
    def ensure_image(self, image: str) -> bool: ...
    def start(self, image: str, env: dict[str, str]) -> None: ...
    def stop(self) -> None: ...


# --------------------------------------------------------------------------
# Docker
# --------------------------------------------------------------------------


class DockerRuntime:
    """The 7 step 4 invocation, built as argv.

    The flags are 4.6's isolation baseline and they are not decoration: this
    runs somebody else's machine, and every one of them is the difference
    between borrowing a GPU and being handed the box.
    """

    def __init__(self, config: HostConfig, *, runner=None):
        self.config = config
        # Injected so tests can assert on composed argv without a daemon. The
        # argv is the thing worth testing -- a dropped `--read-only` is silent
        # and permanent, and nothing about the worker's behaviour reveals it.
        self._run = runner or _run

    # -- queries ---------------------------------------------------------

    def status(self) -> RuntimeStatus:
        """`docker inspect` on the fixed container name.

        An absent container is *not running*, not an error. It is the normal
        state on every tick where the machine was busy, and raising here would
        turn the most common path into an exception.
        """
        argv = [self.config.docker_bin, "inspect", self.config.container_name]
        result = self._run(argv, timeout=DOCKER_TIMEOUT_SEC, check=False)
        if result.returncode != 0:
            return RuntimeStatus(running=False, detail="no such container")

        try:
            payload = json.loads(result.stdout)
        except ValueError:
            return RuntimeStatus(running=False, detail="docker inspect returned no JSON")
        if not payload:
            return RuntimeStatus(running=False, detail="no such container")

        info = payload[0]
        state = info.get("State") or {}
        image = (info.get("Config") or {}).get("Image")
        running = bool(state.get("Running"))
        return RuntimeStatus(
            running=running,
            image=image,
            exited=not running and state.get("Status") == "exited",
            exit_code=state.get("ExitCode"),
            detail=str(state.get("Status") or ""),
        )

    def local_image_exists(self, image: str) -> bool:
        argv = [self.config.docker_bin, "image", "inspect", image]
        return self._run(argv, timeout=DOCKER_TIMEOUT_SEC, check=False).returncode == 0

    # -- actions ---------------------------------------------------------

    def ensure_image(self, image: str) -> bool:
        """Pull if absent. True when a pull actually happened.

        Only when *absent*, deliberately: 4.1 pins `torch-base` by digest
        because a silent PyTorch bump across a running swarm is a nasty class of
        bug, and re-pulling a tag we already hold is precisely how a republished
        tag would slip in. Moving the fleet to a new build means naming a new
        tag on the run, which is visible in the manifest, rather than
        republishing one and hoping.
        """
        if self.local_image_exists(image):
            return False
        log.info("pulling %s", image)
        self._run([self.config.docker_bin, "pull", image], timeout=PULL_TIMEOUT_SEC, check=True)
        return True

    def start(self, image: str, env: dict[str, str]) -> None:
        # A container from a previous tick that exited leaves its name taken,
        # and `docker run` refuses the name rather than reusing it. Clearing it
        # first makes start idempotent, which is what a timer needs: every tick
        # has to be able to assume nothing about the last one.
        self._run(
            [self.config.docker_bin, "rm", "-f", self.config.container_name],
            timeout=DOCKER_TIMEOUT_SEC,
            check=False,
        )
        argv = self.run_argv(image, env)
        log.info("starting %s as %s", image, self.config.container_name)
        self._run(argv, timeout=DOCKER_TIMEOUT_SEC, check=True, env=_child_env(env))

    def stop(self) -> None:
        """`docker stop`, which sends SIGTERM and waits out the stop timeout.

        The worker catches SIGTERM as an optimization (4.4) and abandons its
        lease in milliseconds; if the signal is missed the sentinel in the state
        volume is the correctness path. Either way the shard is released for
        re-lease rather than sat on until the lease expires.
        """
        self._run(
            [
                self.config.docker_bin, "stop",
                "--time", str(self.config.stop_timeout_sec),
                self.config.container_name,
            ],
            timeout=self.config.stop_timeout_sec + DOCKER_TIMEOUT_SEC,
            check=False,
        )

    # -- the argv itself -------------------------------------------------

    def run_argv(self, image: str, env: dict[str, str]) -> list[str]:
        """7 step 4, reconciled with the working invocation in docker/README.md.

        Two differences from the spec block in 7, both from the README having
        been run for real in M2:

        - ``-v ganymede-state:/var/lib/ganymede``. 7 lists only the HF cache
          volume. Without the state volume the contributor's `pause` and `stop`
          sentinels cannot reach inside a ``--read-only`` container at all, so
          4.4's portable stop mechanism -- the one that is not a signal -- would
          have nowhere to land. The kill switch is what makes the ask
          reasonable (7.1); it does not get to depend on signals arriving.
        - ``--user`` is passed only when the agent is *not* root, where 7 and
          the README both pass it unconditionally. Under systemd the agent
          needs the Docker socket and usually has it by being root -- and
          `--user 0:0` would then run the worker as root inside the container,
          quietly undoing the `USER ganymede` the image already set for itself.
          Passing nothing lets the image's own unprivileged user stand, which
          is the stronger of the two. A rootless-Docker contributor still gets
          their own uid, which is what keeps the cache volume readable for
          them across an image rebuild.
        """
        cfg = self.config
        argv = [
            cfg.docker_bin, "run",
            "--detach",
            "--name", cfg.container_name,
            "--stop-timeout", str(cfg.stop_timeout_sec),
            # 4.6 isolation baseline.
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--read-only",
            # --read-only is the flag most likely to surprise: transformers,
            # Triton and torch.compile all write caches, and these two tmpfs
            # mounts are where they land. Not optional.
            "--tmpfs", "/tmp",
            "--tmpfs", "/run/ganymede",
            "-v", f"{cfg.hf_volume}:/cache/hf",
            "-v", f"{cfg.state_volume}:/var/lib/ganymede",
            "--memory", cfg.memory,
            "--cpus", cfg.cpus,
            "--pids-limit", str(cfg.pids_limit),
        ]
        user = _user_flag()
        if user:
            argv += ["--user", user]
        if cfg.gpus:
            argv += ["--gpus", cfg.gpus]

        for name in sorted(env):
            # Name only, never `NAME=value`. A value here lands in the host's
            # process table and in `docker inspect` output for the life of the
            # container, and GANYMEDE_KEY is a bearer token (6.3) -- a secret
            # that survives in `docker inspect` is a secret in every support
            # log a contributor ever pastes. Docker reads the value from the
            # agent's own environment, which `start` populates for the child.
            argv += ["-e", name]

        argv.append(image)
        return argv


def _user_flag() -> str | None:
    """The uid:gid to run as, or None to leave the image's own USER in place.

    None in two cases, and for the same reason both times -- the image already
    specifies a correct unprivileged user, so the only thing overriding it can
    do here is make things worse. Root would override it with root; Windows has
    no uid to override it with, and Docker Desktop ignores the flag for Linux
    containers anyway.
    """
    getuid = getattr(os, "getuid", None)
    getgid = getattr(os, "getgid", None)
    if getuid is None or getgid is None:
        return None
    uid = getuid()
    if uid == 0:
        return None
    return f"{uid}:{getgid()}"


# --------------------------------------------------------------------------
# Native
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _NativeRecord:
    pid: int
    started_at: float
    image: str


class NativeRuntime:
    """The pip path: macOS under launchd, and Linux on your own hardware (4.1).

    macOS is not an option here but a requirement -- 6.8 records that the
    container runtime on macOS cannot reach the GPU, so a Mac contributes
    natively or not at all.
    """

    def __init__(self, config: HostConfig, *, popen=None, now=None):
        self.config = config
        self._popen = popen or subprocess.Popen
        self._now = now or time.time

    @property
    def _record_path(self) -> Path:
        return self.config.resolved_state_dir() / "worker.json"

    # -- queries ---------------------------------------------------------

    def status(self) -> RuntimeStatus:
        record = self._read_record()
        if record is None:
            return RuntimeStatus(running=False, detail="no recorded worker")
        if not self._alive(record):
            return RuntimeStatus(running=False, image=record.image, detail="recorded pid is gone")
        return RuntimeStatus(running=True, image=record.image, detail=f"pid {record.pid}")

    def _alive(self, record: _NativeRecord) -> bool:
        """Is that pid still *our* process?

        A bare `kill(pid, 0)` is not enough. The agent can be down for days --
        a laptop that was closed -- and pids are recycled, so the pid in the
        record may now belong to a browser tab. Signalling it on the next
        `stop` would kill the contributor's own process, which is the single
        worst thing this software could do to a volunteer.

        So the record carries a start time and this compares it against the
        process's actual start time where the platform will tell us. Where it
        will not, the fallback is deliberately the *cautious* direction: report
        not-running, which risks a second worker rather than risking killing
        something that is not ours.
        """
        try:
            os.kill(record.pid, 0)
        except (ProcessLookupError, PermissionError, OSError):
            return False

        actual = _process_start_time(record.pid)
        if actual is None:
            return False
        # Clock granularity differs between the boot-time arithmetic below and
        # time.time(); a couple of seconds of slack is noise, days are not.
        return abs(actual - record.started_at) < 5.0

    # -- actions ---------------------------------------------------------

    def ensure_image(self, image: str) -> bool:  # noqa: ARG002
        """No images on this path -- but the seam stays.

        A native install's version is whatever `pip install` resolved, and the
        host agent cannot change it. Returning False keeps `WorkerRuntime` one
        protocol, so the agent's tick has no branch on runtime type; 4.1 asks
        native installs to report their resolved versions in `compute_profile`
        instead, which puts the mismatch in front of the coordinator rather
        than in front of the contributor.
        """
        return False

    def start(self, image: str, env: dict[str, str]) -> None:
        state_dir = self.config.resolved_state_dir()
        state_dir.mkdir(parents=True, exist_ok=True)
        # A stale stop sentinel from the last shutdown would make the new worker
        # exit immediately, and the agent would start it again on every tick
        # forever. Clearing it is part of starting.
        try:
            (state_dir / "stop").unlink()
        except FileNotFoundError:
            pass

        argv = list(self.config.native_command)
        log.info("starting native worker: %s", " ".join(argv))
        proc = self._popen(
            argv,
            env=_child_env(env),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            # Detach from the agent's process group. The agent is a `--once`
            # tick that is about to exit, and a worker in its group would take
            # the same terminal signals it does.
            start_new_session=True,
        )
        self._write_record(_NativeRecord(pid=proc.pid, started_at=_process_start_time(proc.pid) or self._now(), image=image))

    def stop(self) -> None:
        """Sentinel first, signal second -- 4.4's ordering, not politeness.

        The sentinel is the portable mechanism and the one Windows has at all.
        Signalling first would mean the correctness path is the one that does
        not port, and the sentinel would only ever be exercised on the platform
        where it is hardest to notice it had broken.
        """
        record = self._read_record()
        state_dir = self.config.resolved_state_dir()
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "stop").touch()

        if record is None:
            return
        deadline = self._now() + NATIVE_STOP_GRACE_SEC
        while self._now() < deadline:
            if not self._alive(record):
                self._clear_record()
                return
            time.sleep(1.0)

        log.warning("native worker %s ignored the stop sentinel; signalling", record.pid)
        if self._alive(record):
            try:
                os.kill(record.pid, getattr(signal, "SIGTERM", signal.SIGINT))
            except OSError:
                pass
        self._clear_record()

    # -- the record ------------------------------------------------------

    def _read_record(self) -> _NativeRecord | None:
        try:
            raw = json.loads(self._record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        try:
            return _NativeRecord(
                pid=int(raw["pid"]),
                started_at=float(raw["started_at"]),
                image=str(raw.get("image") or ""),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _write_record(self, record: _NativeRecord) -> None:
        self._record_path.write_text(
            json.dumps({"pid": record.pid, "started_at": record.started_at, "image": record.image}),
            encoding="utf-8",
        )

    def _clear_record(self) -> None:
        try:
            self._record_path.unlink()
        except FileNotFoundError:
            pass


def _process_start_time(pid: int) -> float | None:
    """Epoch seconds at which `pid` started, or None where we cannot tell.

    Linux reads field 22 of ``/proc/<pid>/stat`` (start time in clock ticks
    since boot) and adds the boot time from ``/proc/stat``. macOS has no
    ``/proc``, so it shells out to ``ps -o lstart=``. Windows has neither and
    returns None, which `_alive` treats as "assume not ours".
    """
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            data = fh.read().decode("utf-8", errors="replace")
        # The comm field is parenthesised and may itself contain spaces and
        # parentheses, so split after the last ')' rather than on whitespace.
        after = data[data.rindex(")") + 2:].split()
        ticks = float(after[19])
        clock = os.sysconf("SC_CLK_TCK")
        with open("/proc/stat", "rb") as fh:
            for line in fh.read().decode("utf-8", errors="replace").splitlines():
                if line.startswith("btime "):
                    return float(line.split()[1]) + ticks / clock
        return None
    except (OSError, ValueError, IndexError, AttributeError):
        pass

    try:
        out = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        import email.utils

        parsed = email.utils.parsedate_to_datetime(out.stdout.strip())
        return parsed.timestamp()
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        return None


# --------------------------------------------------------------------------
# shared
# --------------------------------------------------------------------------


def _child_env(env: dict[str, str]) -> dict[str, str]:
    """The agent's environment plus the worker's settings.

    Inherit rather than replace: the child needs PATH, HOME and the proxy
    variables a contributor behind a corporate proxy has set, none of which the
    agent knows to pass through by name.
    """
    merged = dict(os.environ)
    merged.update(env)
    return merged


@dataclass(frozen=True)
class _Completed:
    returncode: int
    stdout: str
    stderr: str


def _run(argv: list[str], *, timeout: int, check: bool, env: dict[str, str] | None = None):
    """Every subprocess in this module goes through here, and every one has a
    timeout. A `docker` call against a wedged daemon otherwise hangs the agent
    forever, and a hung agent is indistinguishable from a machine that simply
    never became idle -- silent, and the contributor's GPU quietly stops
    contributing."""
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False, env=env,
        )
    except FileNotFoundError as exc:
        raise CommandFailed(argv, 127, f"{argv[0]} not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise CommandFailed(argv, -1, f"timed out after {timeout}s") from exc

    if check and proc.returncode != 0:
        raise CommandFailed(argv, proc.returncode, proc.stderr or "")
    return _Completed(proc.returncode, proc.stdout or "", proc.stderr or "")


def for_config(config: HostConfig) -> WorkerRuntime:
    """The runtime this host's config asks for."""
    if config.runtime == "native":
        return NativeRuntime(config)
    if config.runtime == "docker":
        return DockerRuntime(config)
    raise ValueError(f"unknown runtime {config.runtime!r}: expected 'docker' or 'native'")
