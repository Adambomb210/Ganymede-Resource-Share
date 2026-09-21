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


@dataclass(frozen=True)
class GpuDeviceStatus:
    """Per-device answer to "is somebody else on this card" (docs/14 §9).

    ``total`` is however many devices ``nvidia-smi --query-gpu`` enumerated;
    ``busy`` maps the index of every device carrying a non-Ganymede compute
    process to one description of it (the first such process on that card --
    enough for the reason string ``_gpu_busy`` builds from this, and picking
    only one is deliberate rather than an accident of iteration order: this
    type answers "which cards, if any", never "how many processes per card").
    A free index is simply absent from ``busy``.
    """

    total: int
    busy: dict[int, str]


def _compute_memory_mb(field: str) -> int | None:
    """The MiB figure nvidia-smi attributed to a compute process, or ``None``
    when it attributed none.

    ``used_memory`` comes back as ``"1234 MiB"`` where the driver knows, and as
    ``"[N/A]"`` or ``"[Insufficient Permissions]"`` where it does not -- the
    latter two being the only thing Windows/WDDM ever reports, for every
    process, including real CUDA ones. Returning ``None`` for anything that is
    not a number is what lets the caller treat "no memory attributed" as "not
    evidence of a compute client" rather than as "a compute client using zero".
    """
    field = field.strip()
    if not field or field.startswith("["):
        return None
    number = field.split()[0]
    try:
        return int(float(number))
    except ValueError:
        return None


def _gpu_device_status(config: HostConfig) -> tuple[GpuDeviceStatus | None, str]:
    """The per-device busy map, plus a reason.

    Returns ``(None, reason)`` when nvidia-smi cannot be asked at all -- the
    binary is missing, either call times out or raises, or either call exits
    non-zero -- and ``(status, "")`` otherwise. The reason on every failure
    path ends "; assuming gpu free", the same convention every other check in
    this module already uses (see ``_gpu_busy``'s own docstring for why
    "cannot tell" must never read as "busy"), and it is what lets ``_gpu_busy``
    hand a contributor a specific, debuggable line ("nvidia-smi not found" vs.
    "nvidia-smi exited 1" vs. "nvidia-smi failed (timed out)") instead of one
    generic "unknown" for every way this can fail.

    Two ``nvidia-smi`` calls, not one: ``--query-compute-apps`` can report a
    process's ``gpu_uuid`` but not its physical index, so the only way to say
    *which card* a process is on is to resolve the uuid through the separate,
    authoritative ``--query-gpu`` listing this function asks for first.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None, "nvidia-smi not found; assuming gpu free"

    try:
        gpu_proc = subprocess.run(
            [nvidia_smi, "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SEC,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"nvidia-smi failed ({exc}); assuming gpu free"
    if gpu_proc.returncode != 0:
        return None, f"nvidia-smi exited {gpu_proc.returncode}; assuming gpu free"

    index_map: dict[str, int] = {}
    for line in gpu_proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        idx_s, _, uuid = line.partition(",")
        try:
            index_map[uuid.strip()] = int(idx_s.strip())
        except ValueError:
            continue

    try:
        apps_proc = subprocess.run(
            [nvidia_smi,
             "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SEC,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"nvidia-smi failed ({exc}); assuming gpu free"
    if apps_proc.returncode != 0:
        return None, f"nvidia-smi exited {apps_proc.returncode}; assuming gpu free"

    busy: dict[int, str] = {}
    for line in apps_proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # Four CSV fields requested above, and a process name can itself
        # contain a comma on some platforms -- so the *last* field is split
        # off the right first, and the name is then whatever remains after
        # uuid and pid on the left.
        head, _, mem = line.rpartition(",")
        parts = [p.strip() for p in head.split(",", 2)]
        if len(parts) != 3:
            continue
        uuid, pid, name = parts
        if _compute_memory_mb(mem) is None:
            # No memory attributed to this process, so nvidia-smi is not
            # telling us it is a *compute* client -- and the spec's rule is
            # about CUDA processes specifically (`01` §"No non-Ganymede CUDA
            # process holds the GPU", `02` 6.9). On Windows/WDDM this is the
            # difference between a check that works and one that can never
            # pass: `--query-compute-apps` there enumerates every process
            # holding any graphics context -- explorer.exe, the shell, a
            # browser, a chat app, forty of them on an idle desktop -- and
            # attributes `[N/A]` memory to all of them, including a genuine
            # torch CUDA process (measured on a real RTX 3060 box). With no
            # memory figure there is nothing that distinguishes the compositor
            # from a training run, so the query cannot *prove* the card busy,
            # and this module's rule for that is unambiguous and stated three
            # times: "the check can only ever prove the GPU busy, never prove
            # it free". Counting these rows is why `require_gpu_free` -- which
            # defaults to True -- made the host agent refuse to start the
            # worker forever on every Windows contributor's machine, which is
            # precisely backwards for the platform most likely to have an idle
            # gaming GPU to donate.
            #
            # On Linux the driver reports real per-process memory, so every
            # genuine CUDA client keeps its row and behaviour is unchanged.
            # A row skipped here is never a row that *would* have been
            # actionable: it is one nvidia-smi declined to describe.
            continue
        idx = index_map.get(uuid)
        if idx is None:
            continue
        # ``_looks_like_ganymede`` wants the "pid, name" shape the old
        # single-call reason string handed it -- reconstructed explicitly
        # here, since that function's own rsplit(",", 1) assumes the name is
        # the line's last field, and the raw line here carries the uuid too.
        if _looks_like_ganymede(f"{pid}, {name}", config):
            continue
        # First process wins the reason string for that card (class
        # docstring); a card can host more than one, but which one is named
        # here is cosmetic, not load-bearing.
        busy.setdefault(idx, name)
    return GpuDeviceStatus(total=len(index_map), busy=busy), ""


def _gpu_busy(config: HostConfig) -> tuple[bool, str]:
    """The machine-wide answer this module's other callers still want: is
    there a non-Ganymede compute process on the GPU (7.1)?

    Absence of ``nvidia-smi`` -- or any failure running it -- is answered as
    "free", never "busy". A Mac, an AMD box, or a CPU-only host has no NVIDIA
    tooling at all, and treating "I can't check" as "assume busy" would quietly
    exclude every one of those from ever contributing, which is exactly
    backwards for a project whose whole point is broad hardware compatibility
    (6.9). The check can only ever prove the GPU busy, never prove it free.

    **Multi-GPU semantics, docs/14 §9.** "Busy" here now means *every* card
    ``nvidia-smi`` reports carries a non-Ganymede process, not merely one of
    several. On a single-GPU host the two readings are identical -- one busy
    card is the only card, so this is a byte-for-byte-unchanged answer for
    the fleet's overwhelming majority. On a multi-GPU host, the old
    "any busy card marks the machine busy" reading meant one contributor
    running a game on card 3 of a four-card box silently stopped the worker
    from ever starting on cards 0-2 -- the whole donated machine going idle
    over one occupied slot, forever, since the worker never gets the chance
    to register a device inventory that would let the coordinator route
    around the busy one. ``require_gpu_free``'s own comment in ``config.py``
    is updated to say this too, since that setting is what a contributor
    reads before deciding whether they still want the check on.

    This is a *start/stop* gate, not a routing decision -- it answers "should
    the worker container run at all", never "which card should it use". A
    card this function finds busy is not communicated to the worker's own
    probe once it starts (``worker/probe.py`` enumerates every device the
    backend reports, unconditionally), so the coordinator can still allocate
    the occupied card to a job. See the step's own report for why that gap is
    recorded rather than closed here.
    """
    status, reason = _gpu_device_status(config)
    if status is None:
        return False, reason
    if status.total == 0 or not status.busy:
        return False, "gpu free"
    if len(status.busy) >= status.total:
        names = list(status.busy.values())
        extra = f" (+{len(names) - 1} more)" if len(names) > 1 else ""
        return True, f"gpu in use: {names[0]}{extra}"

    # At least one card is free even though not every card is (the fix this
    # rewrite exists for) -- named explicitly, since "why did it start with a
    # busy card in the fleet" has to have an answer in the log.
    detail = ", ".join(f"gpu{i}: {name}" for i, name in sorted(status.busy.items()))
    return False, f"gpu free ({len(status.busy)}/{status.total} device(s) busy: {detail})"


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
        # ``restype`` is set explicitly, and it is load-bearing. ctypes
        # defaults an unspecified return type to ``c_int`` -- *signed* -- but
        # ``GetTickCount`` returns a ``DWORD``, and ``dwTime`` above is a
        # ``c_uint``. Left at the default, the two stop agreeing the moment
        # the tick count passes 2^31 ms, which is **24.9 days of uptime**, not
        # the 49.7-day wrap the guard below is about: ``tick_count`` reads as a
        # large negative number while ``dwTime`` is still a large positive one,
        # so ``idle_ms`` is about -4294967296 on every call and this function
        # returns ``None`` forever after.
        #
        # ``None`` means "unknown", and ``_user_idle_check`` treats unknown as
        # *idle* -- deliberately, so a headless box can contribute. So the
        # failure is silent and it fails open: past 24.9 days of uptime a
        # Windows contributor's machine would read as idle while they were
        # actively typing on it, and the worker would run anyway. That is now
        # the only user-facing protection left on Windows, since `_gpu_busy`
        # there cannot prove a card busy either (see its docstring), which is
        # what makes this worth a line of ctypes rather than a comment.
        get_tick_count = ctypes.windll.kernel32.GetTickCount  # type: ignore[attr-defined]
        get_tick_count.restype = ctypes.c_uint
        tick_count = get_tick_count()
        idle_ms = tick_count - info.dwTime
        if idle_ms < 0:
            # Both are now unsigned 32-bit, so this is the genuine article:
            # GetTickCount wraps every ~49.7 days, and a negative delta means
            # it wrapped between the two reads. Wrong answer is worse than no
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
