"""Whether the contributor's machine is available right now (docs/02-architecture-v2.md 7, 7.1).

The host agent's tick (7) already knows no Ganymede container is running for
this GPU before it ever calls this module -- that is step 1. Everything here
answers the next question: is it okay to start one?

Checks run cheapest-and-most-authoritative first, because the agent runs this
every timer interval forever and most ticks are going to say "no" for the same
reason as the last one:

1. **pause sentinel** -- the contributor's kill switch (7.1). Must work with no
   network and no coordinator, so it is a file check and nothing else.
2. **active window** -- an optional local-time-of-day restriction.
3. **GPU free** -- ``nvidia-smi``, when the config asks for it.
4. **user idle** -- keyboard/mouse inactivity, per platform.

Nothing here may raise. Same rule as ``worker/probe.py``: a host agent that
dies because ``ioreg`` hung leaves the machine contributing nothing until
someone notices, which is a worse outcome than any wrong answer this module
could give. Every subprocess call has a timeout and a missing binary is an
answer, not a crash.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ganymede.host.config import HostConfig

# Every subprocess this module runs is a local, synchronous status query
# (nvidia-smi, ioreg, xprintidle). None of them should ever take this long;
# if one does, the machine has bigger problems than one missed tick, and the
# agent should not hang waiting to find out.
SUBPROCESS_TIMEOUT_SEC = 5.0


class IdleBackend(Protocol):
    """7.1, kept exactly as specified -- one predicate, whatever the platform.

    ``vast`` and ``tensordock`` (7.1, later) will query a rental API instead of
    the local machine; nothing about the tick loop that calls ``is_idle()``
    needs to know which kind it is talking to.
    """

    def is_idle(self) -> bool: ...


@dataclass(frozen=True)
class IdleReport:
    """The richer answer behind ``is_idle()``.

    "Why do I never get work" is a real support question, and a bare bool has
    no answer to it. This is what ``LocalIdleBackend.report()`` and the CLI at
    the bottom of this module hand back instead.
    """

    idle: bool
    reason: str


# --------------------------------------------------------------------------
# 1. Pause sentinel
# --------------------------------------------------------------------------


def _pause_check(config: HostConfig) -> IdleReport | None:
    """None means "no objection"; the caller moves on to the next check."""
    if config.pause_path.exists():
        return IdleReport(False, f"paused: {config.pause_path} exists")
    return None


# --------------------------------------------------------------------------
# 2. Active window
# --------------------------------------------------------------------------


def _parse_hhmm(text: str) -> dt.time:
    hour, _, minute = text.strip().partition(":")
    return dt.time(int(hour), int(minute))


def _within_active_window(window: str, now: dt.datetime | None = None) -> tuple[bool, str]:
    """``"23:00-07:00"`` in local time, wrapping across midnight.

    Empty ``window`` is handled by the caller (it means "no restriction" and
    never reaches here). A malformed one fails *open* -- treated as no
    restriction, with a reason that says so -- rather than closed. Failing
    closed would mean a typo in ``host.json`` silently stops the machine from
    ever contributing, which looks exactly like the "why do I never get work"
    problem this module exists to make answerable; failing open means the
    contributor sees the agent running outside the window they meant to set,
    which is a mistake they will actually notice and fix.
    """
    now = now or dt.datetime.now()
    try:
        start_s, _, end_s = window.partition("-")
        start = _parse_hhmm(start_s)
        end = _parse_hhmm(end_s)
    except (ValueError, IndexError):
        return True, f"active_window {window!r} is unparseable; ignoring it"

    current = now.time()
    if start <= end:
        within = start <= current < end
    else:  # wraps past midnight, e.g. "23:00-07:00"
        within = current >= start or current < end

    if within:
        return True, f"within active window {window}"
    return False, f"outside active window {window} (local time {current.strftime('%H:%M')})"


def _active_window_check(config: HostConfig, now: dt.datetime | None) -> IdleReport | None:
    if not config.active_window:
        return None
    within, detail = _within_active_window(config.active_window, now)
    return None if within else IdleReport(False, detail)


# --------------------------------------------------------------------------
# 3. GPU free
# --------------------------------------------------------------------------


def _looks_like_ganymede(compute_app_line: str, config: HostConfig) -> bool:
    """Best-effort attribution of one ``nvidia-smi`` compute-app row to us.

    By the time this runs, the tick has already confirmed no Ganymede
    container exists for this GPU (step 1 of 7's loop), so in the container
    case this filter is largely redundant -- any row left really is someone
    else. It earns its keep in the native-runtime case (4.1, macOS/no
    container), where the worker process shares the host's process table
    directly and ``nvidia-smi`` may report it by name. It is deliberately not
    load-bearing: a false negative here just means one extra "busy" tick, not
    a correctness problem.
    """
    name = compute_app_line.rsplit(",", 1)[-1].strip().strip('"').lower()
    return "ganymede" in name or (config.container_name and config.container_name.lower() in name)


def _gpu_busy(config: HostConfig) -> tuple[bool, str]:
    """No non-Ganymede compute process on the GPU (7.1).

    Absence of ``nvidia-smi`` -- or any failure running it -- is answered as
    "free", never "busy". A Mac, an AMD box, or a CPU-only host has no NVIDIA
    tooling at all, and treating "I can't check" as "assume busy" would quietly
    exclude every one of those from ever contributing, which is exactly
    backwards for a project whose whole point is broad hardware compatibility
    (6.9). The check can only ever prove the GPU busy, never prove it free.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False, "nvidia-smi not found; assuming gpu free"

    try:
        proc = subprocess.run(
            [nvidia_smi, "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SEC,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"nvidia-smi failed ({exc}); assuming gpu free"

    if proc.returncode != 0:
        return False, f"nvidia-smi exited {proc.returncode}; assuming gpu free"

    apps = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    others = [line for line in apps if not _looks_like_ganymede(line, config)]
    if not others:
        return False, "gpu free"

    extra = f" (+{len(others) - 1} more)" if len(others) > 1 else ""
    return True, f"gpu in use: {others[0]}{extra}"


def _gpu_check(config: HostConfig) -> IdleReport | None:
    if not config.require_gpu_free:
        return None
    busy, detail = _gpu_busy(config)
    return IdleReport(False, detail) if busy else None


# --------------------------------------------------------------------------
# 4. User idle, per platform
# --------------------------------------------------------------------------


def _idle_seconds_macos() -> float | None:
    ioreg = shutil.which("ioreg")
    if not ioreg:
        return None
    try:
        proc = subprocess.run(
            [ioreg, "-c", "IOHIDSystem"], capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SEC
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    match = re.search(r'"HIDIdleTime"\s*=\s*(\d+)', proc.stdout)
    if not match:
        return None
    return int(match.group(1)) / 1_000_000_000  # ns -> s


def _idle_seconds_windows() -> float | None:
    # Guarded import: `ctypes` itself is stdlib and safe to import anywhere,
    # but `ctypes.windll` only exists on Windows, so nothing here may be
    # touched at module import time or this module stops importing on Linux
    # and macOS, where it is imported unconditionally by the config/agent code.
    try:
        import ctypes

        class _LastInputInfo(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

        info = _LastInputInfo()
        info.cbSize = ctypes.sizeof(_LastInputInfo)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):  # type: ignore[attr-defined]
            return None
        tick_count = ctypes.windll.kernel32.GetTickCount()  # type: ignore[attr-defined]
        idle_ms = tick_count - info.dwTime
        if idle_ms < 0:
            # GetTickCount wraps every ~49.7 days; a negative delta means it
            # wrapped between the two reads. Wrong answer is worse than no
            # answer here, so this is "unknown", not "just active".
            return None
        return idle_ms / 1000.0
    except Exception:  # noqa: BLE001 - any ctypes/WinAPI failure is "unknown"
        return None


def _idle_seconds_linux() -> float | None:
    xprintidle = shutil.which("xprintidle")
    if not xprintidle:
        # No X11 idle tool is the *expected* state for a headless Linux box --
        # the single most likely donated machine, since it is the cheapest
        # kind to leave running unattended. It has no input device to be idle
        # on. Returning None here and treating unknown as idle (see the
        # docstring below) is what lets that machine contribute at all;
        # treating "can't tell" as "busy" would exclude it permanently.
        return None
    try:
        proc = subprocess.run([xprintidle], capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SEC)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip()) / 1000.0  # ms -> s
    except ValueError:
        return None


def idle_seconds() -> float | None:
    """Seconds since the last keyboard/mouse input, or ``None`` if unknown.

    ``None`` is a real, explicit answer -- not a magic 0 or a huge number --
    because both of those would be indistinguishable from a genuine
    measurement and would silently pick a side. The caller decides what
    "unknown" means (§7.1: treated as idle); this function's job is only to
    say when it does not know.
    """
    system = platform.system()
    if system == "Darwin":
        return _idle_seconds_macos()
    if system == "Windows":
        return _idle_seconds_windows()
    if system == "Linux":
        return _idle_seconds_linux()
    return None


def _user_idle_check(config: HostConfig) -> IdleReport | None:
    if config.user_idle_sec <= 0:
        return None
    secs = idle_seconds()
    if secs is None:
        # Unknown is treated as idle (7.1's judgement call, made explicit
        # here): a headless box with no input device would otherwise fail
        # this check forever, for a reason no contributor could fix.
        return None
    if secs < config.user_idle_sec:
        return IdleReport(False, f"user active {secs:.0f}s ago (< {config.user_idle_sec}s threshold)")
    return None


# --------------------------------------------------------------------------
# Local backend
# --------------------------------------------------------------------------


def evaluate(config: HostConfig, *, now: dt.datetime | None = None) -> IdleReport:
    """Run every check in order, cheapest first, and stop at the first "no"."""
    for check in (
        lambda: _pause_check(config),
        lambda: _active_window_check(config, now),
        lambda: _gpu_check(config),
        lambda: _user_idle_check(config),
    ):
        verdict = check()
        if verdict is not None:
            return verdict
    return IdleReport(True, "idle: no pause, in window, gpu free, user idle")


class LocalIdleBackend:
    """7.1's ``local`` backend -- own hardware, no rental API involved."""

    def __init__(self, config: HostConfig):
        self.config = config

    def is_idle(self) -> bool:
        return self.report().idle

    def report(self, *, now: dt.datetime | None = None) -> IdleReport:
        return evaluate(self.config, now=now)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python3 -m ganymede.host.idle",
        description="Report whether this machine currently counts as idle (7.1) -- "
        "the answer to 'why do I never get work'.",
    )
    p.add_argument("--config", default=None, help="path to host.json; default is the agent's own search path")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    config = HostConfig.load(Path(args.config) if args.config else None)
    report = LocalIdleBackend(config).report()

    if args.json:
        print(json.dumps({"idle": report.idle, "reason": report.reason}))
    else:
        print(f"{'idle' if report.idle else 'busy'}: {report.reason}")
    return 0 if report.idle else 1


if __name__ == "__main__":
    sys.exit(main())
