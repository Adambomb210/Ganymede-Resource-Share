"""Host agent configuration (docs/02-architecture-v2.md 7).

Two differences from the coordinator's config, both because of who is holding
the machine.

**A file, not just environment variables.** The coordinator is started by an
operator who controls its environment. The host agent is started by a *timer* --
systemd, launchd, or the Windows scheduler -- and a timer's environment is
whatever the init system decided it was, which is usually close to nothing.
Making the environment the only channel would mean every contributor edits a
unit file to change a setting. So the file is primary, the environment overrides
it, and both are documented in ``INSTALL.md``.

**Every value has a default that works.** A contributor should be able to
install this, set two things -- where the coordinator is and what their key is
-- and have a correct, conservatively-limited worker. Anything they must decide
before the software runs at all is a step at which they give up, so the only
required values are the two that cannot be guessed.
"""

from __future__ import annotations

import json
import os
import platform
import shlex
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

# The container is named, not discovered. "Is a Ganymede worker already
# running?" is step 1 of the tick (7) and it runs every timer interval, so it
# has to be one cheap, unambiguous question -- a fixed name makes it
# `docker inspect`, where a label search makes it a filter over every container
# on the box and a judgement call about what counts as ours.
DEFAULT_CONTAINER_NAME = "ganymede-worker"

# Named volumes rather than bind mounts. A bind mount into a read-only container
# needs a host path that exists with the right ownership before the first run,
# which is a support conversation; a named volume Docker creates itself.
DEFAULT_HF_VOLUME = "ganymede-hf"
DEFAULT_STATE_VOLUME = "ganymede-state"

# 4.1: the image the `llm_finetune` runs name. The *tag* comes from the
# manifest (7 step 3) -- this is only the repository half.
DEFAULT_IMAGE_REPO = "ganymede/worker-llm"

# 6.7 wants "conservative". 100 GB holds about six ~16 GB base models, which is
# several runs' worth of history, and it is a number a contributor with a 500 GB
# laptop disk will not blink at. The eviction is LRU, so the cost of the cap
# being slightly too small is a re-download, not a failure.
DEFAULT_CACHE_CAP_GB = 100.0

# 4.4: Docker's default stop grace is 10 s, which is not enough for a worker to
# notice its sentinel, abandon a lease and exit. 120 s is what 7 specifies.
DEFAULT_STOP_TIMEOUT_SEC = 120

# 4.6 isolation defaults. Deliberately restrictive: a contributor lending a
# machine has not agreed to let it be saturated, and a worker that leaves the
# desktop usable is one that stays installed.
DEFAULT_MEMORY = "16g"
DEFAULT_CPUS = "4"
DEFAULT_PIDS_LIMIT = 256

# How long the machine must have been untouched before it counts as idle.
# Fifteen minutes is longer than a coffee and shorter than a lunch: it will not
# grab the GPU while someone is reading, and it will not sit out an afternoon.
DEFAULT_USER_IDLE_SEC = 900


def default_config_path() -> Path:
    """Where the agent looks when nobody told it where to look."""
    env = os.environ.get("GANYMEDE_HOST_CONFIG")
    if env:
        return Path(env)
    if platform.system() == "Windows":
        return Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / "Ganymede" / "host.json"
    return Path("/etc/ganymede/host.json")


def default_state_dir() -> Path:
    """The sentinel directory, shared with the worker's ``control.py``.

    Deliberately the same computation as ``ganymede.worker.control`` rather than
    an import of it: the host agent must run on a machine where the worker is a
    *container image* and the worker package is not installed on the host at
    all. Two lines of duplication buys the host agent an empty dependency list.
    """
    env = os.environ.get("GANYMEDE_STATE")
    if env:
        return Path(env)
    if platform.system() == "Windows":
        return Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / "Ganymede"
    return Path("/var/lib/ganymede")


def default_cache_dir() -> Path:
    env = os.environ.get("GANYMEDE_CACHE_DIR") or os.environ.get("HF_HOME")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "huggingface"


@dataclass
class HostConfig:
    """What the agent needs to decide whether and what to run.

    Field names are the config-file keys and, upper-cased and prefixed with
    ``GANYMEDE_``, the environment variables -- one name per setting, so a
    contributor reading a unit file and a contributor reading ``host.json`` are
    reading the same vocabulary.
    """

    # --- the two that cannot be guessed ---------------------------------
    coordinator_url: str = ""
    key: str = ""

    # --- what to run ----------------------------------------------------
    # "docker" everywhere a container can reach the GPU; "native" on macOS,
    # where it cannot (6.8), and for anyone on their own hardware who would
    # rather have a pip install (4.1).
    runtime: str = "docker"
    image_repo: str = DEFAULT_IMAGE_REPO
    # Pins the tag and disables manifest reconciliation. For an operator who
    # wants to hold a fleet on a known image while something is investigated.
    image_tag: str | None = None
    container_name: str = DEFAULT_CONTAINER_NAME
    docker_bin: str = "docker"
    # The native runtime's command. A list, not a string, so nothing is ever
    # handed to a shell.
    native_command: list[str] = field(
        default_factory=lambda: ["python", "-m", "ganymede.worker.loop"]
    )

    # --- how much of the machine it may have ----------------------------
    gpus: str = "all"
    memory: str = DEFAULT_MEMORY
    cpus: str = DEFAULT_CPUS
    pids_limit: int = DEFAULT_PIDS_LIMIT
    stop_timeout_sec: int = DEFAULT_STOP_TIMEOUT_SEC

    # --- where things live ----------------------------------------------
    state_dir: str = ""
    cache_dir: str = ""
    hf_volume: str = DEFAULT_HF_VOLUME
    state_volume: str = DEFAULT_STATE_VOLUME
    cache_cap_gb: float = DEFAULT_CACHE_CAP_GB

    # --- when it may run -------------------------------------------------
    # Seconds of no keyboard or mouse before the machine counts as idle. Zero
    # disables the check, which is the right setting for a headless box that
    # nobody is sitting at -- there, "user idle" is always true and asking the
    # question just costs a subprocess.
    user_idle_sec: int = DEFAULT_USER_IDLE_SEC
    # A worker of somebody else's on the GPU means this machine is busy. Off
    # only for the deliberate case of two workers sharing a large card.
    require_gpu_free: bool = True
    # Optional local-time window, "23:00-07:00". Empty means any hour.
    active_window: str = ""

    verify_tls: bool = True

    # --- derived ---------------------------------------------------------

    def resolved_state_dir(self) -> Path:
        return Path(self.state_dir) if self.state_dir else default_state_dir()

    def resolved_cache_dir(self) -> Path:
        return Path(self.cache_dir) if self.cache_dir else default_cache_dir()

    @property
    def pause_path(self) -> Path:
        """The contributor's kill switch (7.1). Same file the worker reads."""
        return self.resolved_state_dir() / "pause"

    def missing(self) -> list[str]:
        """The settings with no sensible default, for one good error message.

        Returned rather than raised because the agent has something useful to
        say about *all* of them at once, and because ``--check`` in the CLI
        wants to report them without failing.
        """
        gaps = []
        if not self.coordinator_url:
            gaps.append("coordinator_url")
        if not self.key:
            gaps.append("key")
        return gaps

    # --- loading ---------------------------------------------------------

    @classmethod
    def load(cls, path: Path | None = None, environ: dict[str, str] | None = None) -> "HostConfig":
        """File first, environment second, because a timer has no environment.

        A missing config file is not an error: the container path can be
        configured entirely through ``-e`` flags, and a contributor who has done
        that should not also have to create an empty JSON file.
        """
        environ = os.environ if environ is None else environ
        path = path if path is not None else default_config_path()

        data: dict[str, Any] = {}
        try:
            raw = path.read_text(encoding="utf-8")
        except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
            raw = ""
        if raw.strip():
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError(f"{path}: expected a JSON object, found {type(data).__name__}")

        known = {f.name: f for f in fields(cls)}
        unknown = sorted(set(data) - set(known))
        if unknown:
            # Loud, because the failure this prevents is silent: a typo'd key in
            # host.json means the contributor's setting is simply not applied,
            # and the agent behaves reasonably while ignoring them.
            raise ValueError(f"{path}: unknown settings {', '.join(unknown)}")

        cfg = cls()
        for name, value in data.items():
            setattr(cfg, name, _coerce(known[name].type, value, f"{path}:{name}"))

        for name, f in known.items():
            env_name = f"GANYMEDE_{name.upper()}"
            if env_name in environ:
                setattr(cfg, name, _coerce(f.type, environ[env_name], env_name))

        # Aliases. These names predate the host agent -- the worker and the
        # container README already use them -- and a contributor who has
        # exported GANYMEDE_COORDINATOR_URL for a manual `docker run` should not
        # discover that the agent spells it differently.
        for env_name, attr in (
            ("GANYMEDE_COORDINATOR_URL", "coordinator_url"),
            ("GANYMEDE_CACHE_DIR", "cache_dir"),
            ("GANYMEDE_STATE", "state_dir"),
        ):
            if env_name in environ and environ[env_name]:
                setattr(cfg, attr, environ[env_name])

        return cfg


def _coerce(kind: Any, value: Any, where: str) -> Any:
    """Environment variables are strings; JSON values mostly are not.

    One function for both so that ``pids_limit`` means the same thing whether it
    arrived as ``256`` from a file or ``"256"`` from a unit file.
    """
    if isinstance(kind, str):  # `from __future__ import annotations` stringifies these
        name = kind
    else:
        name = getattr(kind, "__name__", str(kind))

    try:
        if name.startswith("list"):
            if isinstance(value, list):
                return [str(v) for v in value]
            return shlex.split(str(value))
        if name == "bool":
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"1", "true", "yes", "on"}
        if name == "int":
            return int(value)
        if name == "float":
            return float(value)
        if name.startswith("str"):  # str, and `str | None`
            return None if value is None else str(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: {exc}") from exc
    return value
