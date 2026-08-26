"""The idle predicate the host agent's tick calls every interval (docs/02-architecture-v2.md 7, 7.1).

Fast and hermetic: every subprocess boundary (``nvidia-smi``, ``ioreg``,
``xprintidle``) is monkeypatched, never invoked for real, so these tests run
identically on whatever CI happens to be, with or without a GPU.
"""

from __future__ import annotations

import datetime as dt
import subprocess

import pytest

from ganymede.host import idle
from ganymede.host.config import HostConfig


def _config(**overrides) -> HostConfig:
    cfg = HostConfig(coordinator_url="https://coordinator.test", key="k")
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return cfg


# --------------------------------------------------------------------------
# Pause sentinel: the contributor's kill switch, and it wins over everything
# --------------------------------------------------------------------------


def test_a_clean_machine_with_no_restrictions_is_idle(tmp_path, monkeypatch):
    monkeypatch.setattr(idle, "_gpu_busy", lambda config: (False, "gpu free"))
    monkeypatch.setattr(idle, "idle_seconds", lambda: 10_000.0)
    cfg = _config(state_dir=str(tmp_path))
    report = idle.evaluate(cfg)
    assert report.idle is True


def test_the_pause_file_beats_every_other_check(tmp_path, monkeypatch):
    """Even a machine that is otherwise idle in every other respect must stop
    the instant `pause` exists -- it's the no-network, no-coordinator kill
    switch (7.1), and it must not be shadowed by anything checked later."""
    monkeypatch.setattr(idle, "_gpu_busy", lambda config: (False, "gpu free"))
    monkeypatch.setattr(idle, "idle_seconds", lambda: 10_000.0)
    cfg = _config(state_dir=str(tmp_path), active_window="", require_gpu_free=False, user_idle_sec=0)
    (tmp_path / "pause").touch()

    report = idle.evaluate(cfg)
    assert report.idle is False
    assert "paused" in report.reason


def test_is_idle_is_a_thin_wrapper_over_report(tmp_path, monkeypatch):
    monkeypatch.setattr(idle, "_gpu_busy", lambda config: (False, "gpu free"))
    monkeypatch.setattr(idle, "idle_seconds", lambda: 10_000.0)
    cfg = _config(state_dir=str(tmp_path))
    backend = idle.LocalIdleBackend(cfg)
    assert backend.is_idle() == backend.report().idle


# --------------------------------------------------------------------------
# Active window: local time, wrapping across midnight
# --------------------------------------------------------------------------


def test_empty_active_window_disables_the_check(tmp_path):
    cfg = _config(state_dir=str(tmp_path), active_window="")
    assert idle._active_window_check(cfg, dt.datetime(2026, 1, 1, 13, 0)) is None


def test_a_daytime_window_excludes_the_evening():
    within, _ = idle._within_active_window("09:00-17:00", dt.datetime(2026, 1, 1, 20, 0))
    assert within is False


def test_a_daytime_window_includes_midday():
    within, _ = idle._within_active_window("09:00-17:00", dt.datetime(2026, 1, 1, 12, 0))
    assert within is True


def test_an_overnight_window_wraps_across_midnight():
    """"23:00-07:00" -- the interesting case, because start > end and a naive
    `start <= now < end` comparison would reject every hour of the night."""
    late_night = dt.datetime(2026, 1, 1, 23, 30)
    early_morning = dt.datetime(2026, 1, 2, 5, 0)
    midday = dt.datetime(2026, 1, 2, 13, 0)

    assert idle._within_active_window("23:00-07:00", late_night)[0] is True
    assert idle._within_active_window("23:00-07:00", early_morning)[0] is True
    assert idle._within_active_window("23:00-07:00", midday)[0] is False


def test_the_window_boundaries_are_half_open():
    assert idle._within_active_window("23:00-07:00", dt.datetime(2026, 1, 1, 23, 0))[0] is True
    assert idle._within_active_window("23:00-07:00", dt.datetime(2026, 1, 1, 7, 0))[0] is False


def test_a_malformed_window_fails_open_rather_than_closed():
    """A typo in host.json must not silently strand the machine forever -- see
    the reasoning in idle.py's docstring. Failing open (still runs) is a
    mistake a contributor can notice; failing closed is not."""
    within, reason = idle._within_active_window("garbage", dt.datetime(2026, 1, 1, 3, 0))
    assert within is True
    assert "unparseable" in reason


# --------------------------------------------------------------------------
# GPU free: absence of tooling must never read as "busy"
# --------------------------------------------------------------------------


def test_missing_nvidia_smi_means_free_not_busy(monkeypatch):
    """A Mac, an AMD box, or a CPU-only host has no nvidia-smi at all. If that
    read as "busy" it would permanently exclude exactly the machines the
    project most wants to accept."""
    monkeypatch.setattr(idle.shutil, "which", lambda name: None)
    busy, reason = idle._gpu_busy(_config())
    assert busy is False
    assert "not found" in reason


def test_nvidia_smi_timing_out_means_free_not_busy(monkeypatch):
    monkeypatch.setattr(idle.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def _raise(*a, **k):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)

    monkeypatch.setattr(idle.subprocess, "run", _raise)
    busy, reason = idle._gpu_busy(_config())
    assert busy is False
    assert "failed" in reason


def test_an_other_process_on_the_gpu_means_busy(monkeypatch):
    monkeypatch.setattr(idle.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        idle.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="1234, steam.exe\n", stderr=""),
    )
    busy, reason = idle._gpu_busy(_config())
    assert busy is True
    assert "steam.exe" in reason


def test_an_empty_compute_apps_list_means_free(monkeypatch):
    monkeypatch.setattr(idle.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        idle.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="", stderr=""),
    )
    busy, _ = idle._gpu_busy(_config())
    assert busy is False


def test_require_gpu_free_false_skips_the_check_entirely(monkeypatch):
    def explode(*a, **k):
        raise AssertionError("should not be called")

    monkeypatch.setattr(idle, "_gpu_busy", explode)
    cfg = _config(require_gpu_free=False)
    assert idle._gpu_check(cfg) is None


# --------------------------------------------------------------------------
# User idle: unknown means idle, per platform, and 0 disables the check
# --------------------------------------------------------------------------


def test_unknown_idle_time_is_treated_as_idle(monkeypatch):
    """A headless Linux box -- the single most likely donated machine -- has
    no input device and no xprintidle. Refusing to run there would exclude
    exactly the contributors the project most wants."""
    monkeypatch.setattr(idle, "idle_seconds", lambda: None)
    cfg = _config(user_idle_sec=900)
    assert idle._user_idle_check(cfg) is None


def test_recent_user_activity_is_not_idle(monkeypatch):
    monkeypatch.setattr(idle, "idle_seconds", lambda: 30.0)
    cfg = _config(user_idle_sec=900)
    verdict = idle._user_idle_check(cfg)
    assert verdict is not None
    assert verdict.idle is False


def test_sufficient_user_idle_time_passes(monkeypatch):
    monkeypatch.setattr(idle, "idle_seconds", lambda: 1000.0)
    cfg = _config(user_idle_sec=900)
    assert idle._user_idle_check(cfg) is None


def test_zero_user_idle_sec_disables_the_check(monkeypatch):
    monkeypatch.setattr(idle, "idle_seconds", lambda: 0.0)
    cfg = _config(user_idle_sec=0)
    assert idle._user_idle_check(cfg) is None


def test_linux_with_no_xprintidle_reports_unknown(monkeypatch):
    monkeypatch.setattr(idle.platform, "system", lambda: "Linux")
    monkeypatch.setattr(idle.shutil, "which", lambda name: None)
    assert idle.idle_seconds() is None


def test_macos_idle_seconds_parses_hid_idle_time(monkeypatch):
    monkeypatch.setattr(idle.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(idle.shutil, "which", lambda name: "/usr/sbin/ioreg")
    fake_output = '"HIDIdleTime" = 5000000000\n'
    monkeypatch.setattr(
        idle.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=fake_output, stderr=""),
    )
    assert idle.idle_seconds() == pytest.approx(5.0)


def test_linux_with_xprintidle_parses_milliseconds(monkeypatch):
    monkeypatch.setattr(idle.platform, "system", lambda: "Linux")
    monkeypatch.setattr(idle.shutil, "which", lambda name: "/usr/bin/xprintidle")
    monkeypatch.setattr(
        idle.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="12345\n", stderr=""),
    )
    assert idle.idle_seconds() == pytest.approx(12.345)


def test_an_unrecognized_platform_reports_unknown(monkeypatch):
    monkeypatch.setattr(idle.platform, "system", lambda: "Plan9")
    assert idle.idle_seconds() is None


# --------------------------------------------------------------------------
# Nothing here may raise
# --------------------------------------------------------------------------


def test_a_broken_subprocess_call_never_raises_out_of_gpu_busy(monkeypatch):
    monkeypatch.setattr(idle.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def _raise(*a, **k):
        raise OSError("no such device")

    monkeypatch.setattr(idle.subprocess, "run", _raise)
    busy, _ = idle._gpu_busy(_config())
    assert busy is False


def test_a_broken_subprocess_call_never_raises_out_of_idle_seconds(monkeypatch):
    monkeypatch.setattr(idle.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(idle.shutil, "which", lambda name: "/usr/sbin/ioreg")

    def _raise(*a, **k):
        raise OSError("ioreg vanished")

    monkeypatch.setattr(idle.subprocess, "run", _raise)
    assert idle.idle_seconds() is None
