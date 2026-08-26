"""Stop and pause as files (§4.4, §7.1).

Signals don't port. The sentinel file is what correctness rests on, and these
tests are the reason: they run identically on all three platforms, whereas a
suite built on SIGTERM would be testing a mechanism that native Windows does not
have.
"""

from __future__ import annotations

import os
import platform

import pytest

from ganymede.worker.control import ControlFiles, default_state_dir


@pytest.fixture
def control(tmp_path):
    return ControlFiles(tmp_path, install_signal_handlers=False)


def test_a_clean_worker_neither_stops_nor_pauses(control):
    assert not control.should_stop()
    assert not control.should_pause()
    assert control.reason() is None


def test_the_stop_file_stops(control):
    control.request_stop()
    assert control.should_stop()
    assert "stop" in control.reason()


def test_the_pause_file_pauses_without_stopping(control):
    """Different meanings: pause is the contributor's own control (7.1) -- stay
    installed, take no new work. Someone about to play a game wants their GPU
    back for an hour, not an uninstall."""
    control.request_pause()
    assert control.should_pause()
    assert not control.should_stop()


def test_clear_removes_both_and_is_safe_when_neither_exists(control):
    control.request_stop()
    control.request_pause()
    control.clear()
    assert not control.should_stop() and not control.should_pause()
    control.clear()  # idempotent


def test_a_signal_stops_without_any_file(control):
    """SIGTERM stays an optimization: a container abandons its lease in
    milliseconds rather than within a poll interval. Correctness does not rest
    on it, but where it exists it should work."""
    control._on_signal(15, None)
    assert control.should_stop()
    assert control.reason() == "signal"
    assert not control.stop_path.exists()


def test_the_state_dir_is_created_on_demand(tmp_path):
    nested = tmp_path / "does" / "not" / "exist"
    control = ControlFiles(nested, install_signal_handlers=False)
    path = control.request_stop()
    assert path.exists()
    assert control.should_stop()


def test_the_default_location_is_documented_per_platform(monkeypatch):
    """A contributor has to be able to stop the worker from a shell without
    knowing how it was installed. A support answer that begins "find the
    directory where..." is not an answer."""
    monkeypatch.delenv("GANYMEDE_STATE", raising=False)
    expected = r"C:\ProgramData" if platform.system() == "Windows" else "/var/lib/ganymede"
    assert str(default_state_dir()).startswith(expected.split("\\")[0])


def test_the_environment_overrides_the_default(monkeypatch, tmp_path):
    monkeypatch.setenv("GANYMEDE_STATE", str(tmp_path))
    assert default_state_dir() == tmp_path


def test_installing_signal_handlers_off_the_main_thread_is_not_fatal(tmp_path):
    """Only the main thread may install them, and SIGTERM does not exist
    everywhere. Both are normal; neither should stop a worker from starting,
    because the sentinel file already covers correctness."""
    import threading

    failures = []

    def build():
        try:
            ControlFiles(tmp_path, install_signal_handlers=True)
        except Exception as exc:  # noqa: BLE001
            failures.append(exc)

    thread = threading.Thread(target=build)
    thread.start()
    thread.join()
    assert failures == []
