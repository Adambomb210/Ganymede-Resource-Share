"""Stop and pause, as files rather than signals (docs/02-architecture-v2.md 4.4, 7.1).

Signals don't port. ``SIGTERM`` is delivered reliably to a Linux container and to
a native macOS process, but native Windows has no real equivalent -- console
control handlers and service control codes are different mechanisms with
different semantics. Building the correctness path on them means three
implementations of the one thing that must not be flaky.

So the primary mechanism is a **sentinel file the worker polls**. It is already
polling (heartbeats run every 60 s), so this costs one ``os.path.exists`` per
cycle and behaves identically on all three platforms. Signal handlers stay as an
*optimization* where they work: catching ``SIGTERM`` lets a container abandon its
lease within milliseconds rather than within a poll interval. That is worth
having, and it is not what correctness rests on.

Two files, two meanings
-----------------------
``stop`` means *exit* -- the host is shutting the worker down. ``pause`` means
*stay running but take no new work*, which is the contributor's own control
(7.1): someone about to play a game wants their GPU back for an hour without
uninstalling anything.

The difference matters mid-round. Both abandon the current lease, because
finishing under time pressure risks a half-uploaded artifact and abandoning
releases the shard for immediate re-lease -- strictly better for the swarm. But
``pause`` then waits and resumes, while ``stop`` exits 0.
"""

from __future__ import annotations

import os
import platform
import signal
import threading
from pathlib import Path

STOP_FILE = "stop"
PAUSE_FILE = "pause"


def default_state_dir() -> Path:
    """Where the sentinels live, per platform.

    A fixed, documented location rather than something derived: a contributor
    has to be able to stop the worker from a shell or a shortcut without knowing
    how it was installed, and a support answer that starts "find the directory
    where..." is not an answer.
    """
    env = os.environ.get("GANYMEDE_STATE")
    if env:
        return Path(env)
    if platform.system() == "Windows":
        return Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / "Ganymede"
    return Path("/var/lib/ganymede")


class ControlFiles:
    """Polls the sentinels, and catches signals where the platform has them."""

    def __init__(self, state_dir: Path | None = None, install_signal_handlers: bool = True):
        self.state_dir = Path(state_dir) if state_dir else default_state_dir()
        # Set by a signal handler as well as by the file, so a container gets
        # millisecond response and a Windows service still gets correct
        # behaviour from polling alone.
        self._signalled = threading.Event()
        if install_signal_handlers:
            self.install_signal_handlers()

    @property
    def stop_path(self) -> Path:
        return self.state_dir / STOP_FILE

    @property
    def pause_path(self) -> Path:
        return self.state_dir / PAUSE_FILE

    def install_signal_handlers(self) -> None:
        """Best-effort. Absence of a signal is not an error on any platform.

        Only installable from the main thread, and `SIGTERM` does not exist on
        every platform -- both are normal, and neither should stop a worker from
        starting, because the sentinel file already covers correctness.
        """
        for name in ("SIGTERM", "SIGINT"):
            handler = getattr(signal, name, None)
            if handler is None:
                continue
            try:
                signal.signal(handler, self._on_signal)
            except (ValueError, OSError, RuntimeError):
                pass  # not the main thread, or unsupported here

    def _on_signal(self, signum, frame) -> None:  # noqa: ARG002
        self._signalled.set()

    def should_stop(self) -> bool:
        return self._signalled.is_set() or self.stop_path.exists()

    def should_pause(self) -> bool:
        return self.pause_path.exists()

    def reason(self) -> str | None:
        """Why the worker is stopping, for the log line that explains an exit."""
        if self._signalled.is_set():
            return "signal"
        if self.stop_path.exists():
            return f"stop file at {self.stop_path}"
        if self.pause_path.exists():
            return f"pause file at {self.pause_path}"
        return None

    # -- helpers for the CLI and for tests -------------------------------

    def request_stop(self) -> Path:
        return self._touch(self.stop_path)

    def request_pause(self) -> Path:
        return self._touch(self.pause_path)

    def clear(self) -> None:
        self._signalled.clear()
        for path in (self.stop_path, self.pause_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def _touch(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path
