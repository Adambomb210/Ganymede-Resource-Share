"""The idle predicate the host agent's tick calls every interval (docs/02-architecture-v2.md 7, 7.1).

Fast and hermetic: every subprocess boundary (``nvidia-smi``, ``ioreg``,
``xprintidle``) is monkeypatched, never invoked for real, so these tests run
identically on whatever CI happens to be, with or without a GPU.
"""

from __future__ import annotations

import ctypes
import datetime as dt
import platform
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


def _fake_nvidia_smi(gpu_csv: str, apps_csv: str):
    """Build a ``subprocess.run`` stand-in that answers the two distinct
    ``nvidia-smi`` calls ``_gpu_device_status`` now makes: ``--query-gpu``
    (the index/uuid map) and ``--query-compute-apps`` (who is running where).
    A single fixed return value, as the pre-rewrite tests used, silently
    breaks once there are two different queries in flight -- the index map
    would be parsed out of compute-app rows and vice versa.

    ``apps_csv`` rows carry four fields, including ``used_memory``,
    because that is what ``--query-compute-apps`` really returns and
    because the memory column is now load-bearing: a row with no memory
    attributed to it is not counted as a compute client
    (``_compute_memory_mb``). Fixtures written to the three-field shape
    would be describing output nvidia-smi does not produce."""
    def _run(argv, **kwargs):
        cmd = " ".join(argv)
        if "--query-gpu=" in cmd:
            return subprocess.CompletedProcess(argv, 0, stdout=gpu_csv, stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout=apps_csv, stderr="")
    return _run


# One GPU, uuid "GPU-0", the shape every single-device test below uses.
_ONE_GPU_CSV = "0, GPU-0\n"


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
        _fake_nvidia_smi(_ONE_GPU_CSV, "GPU-0, 1234, steam.exe, 512 MiB\n"),
    )
    busy, reason = idle._gpu_busy(_config())
    assert busy is True
    assert "steam.exe" in reason


def test_an_empty_compute_apps_list_means_free(monkeypatch):
    monkeypatch.setattr(idle.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(idle.subprocess, "run", _fake_nvidia_smi(_ONE_GPU_CSV, ""))
    busy, _ = idle._gpu_busy(_config())
    assert busy is False


def test_require_gpu_free_false_skips_the_check_entirely(monkeypatch):
    def explode(*a, **k):
        raise AssertionError("should not be called")

    monkeypatch.setattr(idle, "_gpu_busy", explode)
    cfg = _config(require_gpu_free=False)
    assert idle._gpu_check(cfg) is None


# --------------------------------------------------------------------------
# GPU free, per device (docs/14 §9): one busy card no longer parks the whole
# multi-GPU box idle.
# --------------------------------------------------------------------------


_FOUR_GPU_CSV = "0, GPU-0\n1, GPU-1\n2, GPU-2\n3, GPU-3\n"


def test_one_busy_card_on_a_four_gpu_box_still_reads_free(monkeypatch):
    """The bug this rewrite exists to fix: the pre-rewrite aggregate marked
    the whole machine busy the moment *any* card was, which on a donated
    4-GPU box meant one contributor game on card 2 silently idled cards 0, 1
    and 3 forever -- the worker never got the chance to start and offer the
    coordinator the free cards at all."""
    monkeypatch.setattr(idle.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        idle.subprocess, "run",
        _fake_nvidia_smi(_FOUR_GPU_CSV, "GPU-2, 555, steam.exe, 512 MiB\n"),
    )
    busy, reason = idle._gpu_busy(_config())
    assert busy is False
    assert "1/4" in reason
    assert "gpu2: steam.exe" in reason


def test_every_card_busy_on_a_four_gpu_box_reads_busy(monkeypatch):
    monkeypatch.setattr(idle.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    apps = "\n".join(
        f"GPU-{i}, {100 + i}, other-job-{i}.exe, {512 + i} MiB" for i in range(4)
    ) + "\n"
    monkeypatch.setattr(idle.subprocess, "run", _fake_nvidia_smi(_FOUR_GPU_CSV, apps))
    busy, reason = idle._gpu_busy(_config())
    assert busy is True
    assert "other-job-0.exe" in reason


# --------------------------------------------------------------------------
# Windows/WDDM: --query-compute-apps enumerates graphics contexts, not just
# CUDA clients, and attributes memory to none of them (docs/14 §9)
# --------------------------------------------------------------------------

# Verbatim from a real RTX 3060 / Windows 11 / driver 610.74 desktop session,
# trimmed to a representative handful of the forty rows it actually returned.
# Two things in here are the whole point, and neither was guessable from a
# hand-written Linux fixture: ordinary GUI processes are listed as "compute
# apps" at all, and *every* row -- including a genuine torch CUDA process
# measured on the same box -- carries "[N/A]" where Linux carries a number.
_WINDOWS_APPS_CSV = (
    "GPU-a728, 1984, [Insufficient Permissions], [N/A]\n"
    "GPU-a728, 10028, C:\\Windows\\explorer.exe, [N/A]\n"
    "GPU-a728, 28472, C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe, [N/A]\n"
    "GPU-a728, 6088, C:\\Users\\x\\AppData\\Local\\Discord\\Discord.exe, [N/A]\n"
    "GPU-a728, 44144, C:\\Users\\x\\python.exe, [N/A]\n"
)
_WINDOWS_GPU_CSV = "0, GPU-a728\n"


def test_a_windows_desktop_is_not_reported_busy_by_its_own_gui_processes(monkeypatch):
    """The bug this fixture exists for: ``require_gpu_free`` defaults to True,
    and on Windows every one of these rows counted as "somebody else is on the
    GPU" -- so the host agent refused to start the worker, forever, on the
    platform most likely to have an idle gaming GPU to donate.

    nvidia-smi attributes no memory to any of them, so it is not telling us any
    of them is a *compute* client (`01` §"No non-Ganymede CUDA process holds the
    GPU"). It therefore cannot prove the card busy, and this module's rule for
    that case is stated three times over: never read "cannot tell" as "busy".
    """
    monkeypatch.setattr(idle.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        idle.subprocess, "run",
        _fake_nvidia_smi(_WINDOWS_GPU_CSV, _WINDOWS_APPS_CSV),
    )
    busy, reason = idle._gpu_busy(_config())
    assert busy is False, reason
    # And the gate raises no objection, which is the part a contributor feels.
    assert idle._gpu_check(_config()) is None


def test_a_real_cuda_process_is_still_busy_when_memory_is_attributed(monkeypatch):
    """The other half: the filter keys on whether nvidia-smi described the
    process, not on the platform. Give one row a real memory figure -- what
    Linux reports for every genuine CUDA client -- and it counts again."""
    monkeypatch.setattr(idle.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    apps = _WINDOWS_APPS_CSV + "GPU-a728, 77, /usr/bin/python3, 4096 MiB\n"
    monkeypatch.setattr(
        idle.subprocess, "run", _fake_nvidia_smi(_WINDOWS_GPU_CSV, apps),
    )
    busy, reason = idle._gpu_busy(_config())
    assert busy is True
    assert "python3" in reason


def test_a_ganymede_process_with_real_memory_is_still_excluded(monkeypatch):
    """Attribution still wins over the memory filter: our own worker holding
    4 GB is not somebody else's work."""
    monkeypatch.setattr(idle.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    apps = "GPU-a728, 77, python-ganymede-worker, 4096 MiB\n"
    monkeypatch.setattr(
        idle.subprocess, "run", _fake_nvidia_smi(_WINDOWS_GPU_CSV, apps),
    )
    busy, _ = idle._gpu_busy(_config())
    assert busy is False


def test_compute_memory_mb_reads_only_a_real_figure():
    assert idle._compute_memory_mb("512 MiB") == 512
    assert idle._compute_memory_mb(" 4096 MiB ") == 4096
    assert idle._compute_memory_mb("[N/A]") is None
    assert idle._compute_memory_mb("[Insufficient Permissions]") is None
    assert idle._compute_memory_mb("") is None
    assert idle._compute_memory_mb("not-a-number") is None
    # Zero is a real attribution, not an absent one.
    assert idle._compute_memory_mb("0 MiB") == 0


def test_a_process_name_containing_a_comma_still_parses(monkeypatch):
    """The name is no longer the line's last field, so it is split off from
    the right -- a name with a comma in it must survive that."""
    monkeypatch.setattr(idle.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    apps = "GPU-a728, 77, weird, name.exe, 2048 MiB\n"
    monkeypatch.setattr(
        idle.subprocess, "run", _fake_nvidia_smi(_WINDOWS_GPU_CSV, apps),
    )
    status, _ = idle._gpu_device_status(_config())
    assert status.busy == {0: "weird, name.exe"}


def test_a_ganymede_process_is_excluded_per_card_not_just_globally(monkeypatch):
    """The native-runtime attribution filter (``_looks_like_ganymede``) still
    has to work once a process line carries a uuid ahead of the pid/name
    pair -- a card running only our own worker must read as free, not busy,
    on a multi-GPU box exactly as it does on a single-GPU one."""
    monkeypatch.setattr(idle.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    apps = ("GPU-0, 42, python-ganymede-worker, 900 MiB\n"
            "GPU-1, 43, steam.exe, 512 MiB\n")
    monkeypatch.setattr(
        idle.subprocess, "run", _fake_nvidia_smi(_FOUR_GPU_CSV, apps),
    )
    busy, reason = idle._gpu_busy(_config())
    assert busy is False
    assert "1/4" in reason
    assert "gpu1: steam.exe" in reason


def test_gpu_device_status_reports_the_free_and_busy_split(monkeypatch):
    """The per-device primitive directly, not just the aggregate it feeds."""
    monkeypatch.setattr(idle.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        idle.subprocess, "run",
        _fake_nvidia_smi(_FOUR_GPU_CSV, "GPU-2, 555, steam.exe, 512 MiB\n"),
    )
    status, reason = idle._gpu_device_status(_config())
    assert reason == ""
    assert status.total == 4
    assert status.busy == {2: "steam.exe"}


def test_gpu_device_status_none_when_the_index_map_call_fails(monkeypatch):
    """A failure on the *first* of the two calls (the index/uuid map) must
    read as unknown, not silently proceed with an empty map that would then
    misattribute every compute-app row to no device at all."""
    monkeypatch.setattr(idle.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        idle.subprocess, "run",
        lambda argv, **k: subprocess.CompletedProcess(argv, 1, stdout="", stderr="boom"),
    )
    status, reason = idle._gpu_device_status(_config())
    assert status is None
    assert "exited 1" in reason


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


# --------------------------------------------------------------------------
# The Windows probe. No coverage existed for it before, because every other
# platform probe is a mockable subprocess and this one is ctypes -- which is
# exactly how the signed/unsigned bug below survived.
# --------------------------------------------------------------------------


def test_the_tick_count_and_last_input_must_be_compared_as_unsigned():
    """``GetTickCount`` returns a ``DWORD`` and ``dwTime`` is a ``c_uint``, but
    ctypes defaults an unspecified ``restype`` to *signed* ``c_int``.

    Left at that default the two disagree once the tick count passes 2^31 ms --
    **24.9 days of uptime**, not the 49.7-day wrap the function's own guard is
    about. ``tick_count`` reads large-negative while ``dwTime`` is still
    large-positive, ``idle_ms`` is about -4294967296 on every call, and the
    probe returns ``None`` forever after.

    That matters because ``None`` means "unknown" and ``_user_idle_check``
    treats unknown as *idle* on purpose, so the failure is silent and fails
    open: the machine reads as idle while somebody is typing on it.

    This test is the arithmetic, so it runs on every platform rather than only
    where the bug can be reproduced.
    """
    uptime_ms = int(25.0 * 86400000)       # past 2^31, the interesting side
    last_input = uptime_ms - 5000          # five seconds ago

    as_signed = ctypes.c_int(uptime_ms & 0xFFFFFFFF).value
    as_unsigned = ctypes.c_uint(uptime_ms & 0xFFFFFFFF).value
    dw_time = ctypes.c_uint(last_input & 0xFFFFFFFF).value

    assert as_signed < 0, "precondition: past 2^31 a signed read goes negative"
    assert as_signed - dw_time < 0, "the bug: a negative delta reads as unknown"
    assert as_unsigned - dw_time == 5000, "the fix: unsigned gives the real gap"


@pytest.mark.skipif(platform.system() != "Windows", reason="Windows-only probe")
def test_the_windows_probe_returns_a_sane_measurement_on_a_real_box():
    """Runs for real against the live WinAPI on CI's windows runner -- the one
    place the ctypes path is actually exercised rather than described."""
    secs = idle._idle_seconds_windows()
    assert secs is not None, "GetLastInputInfo should succeed on a real desktop"
    assert secs >= 0.0
    # Sanity: the answer must not exceed the machine's own uptime.
    get_tick_64 = ctypes.windll.kernel32.GetTickCount64
    get_tick_64.restype = ctypes.c_ulonglong
    assert secs <= get_tick_64() / 1000.0 + 1.0


@pytest.mark.skipif(platform.system() != "Windows", reason="Windows-only probe")
def test_the_windows_probe_sets_an_unsigned_restype():
    """The fix itself: whatever else happens, the tick count must not be read
    back through ctypes' signed default."""
    idle._idle_seconds_windows()
    restype = ctypes.windll.kernel32.GetTickCount.restype
    assert restype in (ctypes.c_uint, ctypes.c_ulong), restype
    assert ctypes.c_uint(0xFFFFFFFF).value > 0  # i.e. the type really is unsigned
