"""The scheduler files parse, and say what the agent expects them to say.

These exist because of how the failures look. A plist with a doubled hyphen
inside a comment is not well-formed XML, and launchd's response to that is to
silently decline to load it; a Task Scheduler document with its elements in the
wrong order is rejected at import. In both cases the contributor's install
"succeeds", nothing ticks, and there is no error anywhere to find. None of this
can be caught by running the agent, because the agent is not what reads them.

What these cannot check is behaviour: no systemd, launchd or Task Scheduler
exists on CI, so "the timer actually fires" is a claim only a real install on a
real machine can make. See packaging/README.md for the per-platform commands.
"""

from __future__ import annotations

import configparser
import plistlib
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

PACKAGING = Path(__file__).resolve().parent.parent / "packaging"
TASK_NS = "{http://schemas.microsoft.com/windows/2004/02/mit/task}"


# --------------------------------------------------------------------------
# launchd
# --------------------------------------------------------------------------


def test_the_plist_is_well_formed():
    """XML forbids `--` inside a comment, and a flag like `--once` written into
    a rationale comment is the natural way to trip over that."""
    plistlib.loads((PACKAGING / "com.ganymede.host.plist").read_bytes())


def test_the_plist_runs_a_single_tick_not_a_daemon():
    d = plistlib.loads((PACKAGING / "com.ganymede.host.plist").read_bytes())
    assert d["ProgramArguments"][-1] == "--once"
    assert d["KeepAlive"] is False
    assert d["StartInterval"] > 0


def test_the_plist_carries_no_key():
    """~/Library/LaunchAgents is mode 0644. A bearer token (6.3) in an
    EnvironmentVariables block would be readable by every local user.

    Asserted against the parsed plist, not the raw bytes: the file *mentions*
    both names in the comment explaining why they are absent, and a text search
    cannot tell an explanation from the thing it warns about."""
    d = plistlib.loads((PACKAGING / "com.ganymede.host.plist").read_bytes())
    assert "EnvironmentVariables" not in d
    assert "GANYMEDE_KEY" not in str(d)


# --------------------------------------------------------------------------
# Windows Task Scheduler
# --------------------------------------------------------------------------


def test_the_task_xml_is_well_formed_and_its_declared_encoding_is_true():
    """It previously declared UTF-16 while being UTF-8 on disk, which every
    parser rejects before reading a single element."""
    ET.parse(PACKAGING / "ganymede-host-task.xml")


def test_the_task_elements_are_in_the_order_the_importer_demands():
    root = ET.parse(PACKAGING / "ganymede-host-task.xml").getroot()
    order = [c.tag.replace(TASK_NS, "") for c in root]
    assert order == ["RegistrationInfo", "Triggers", "Principals", "Settings", "Actions"]


def test_the_task_runs_a_single_tick():
    root = ET.parse(PACKAGING / "ganymede-host-task.xml").getroot()
    assert root.find(f"{TASK_NS}Actions/{TASK_NS}Exec/{TASK_NS}Arguments").text == "--once"


def test_the_task_leaves_the_idle_decision_to_the_agent():
    """Task Scheduler has its own idle gate. Using it would put a second,
    Windows-only opinion next to 7.1's `is_idle`, and they would disagree."""
    root = ET.parse(PACKAGING / "ganymede-host-task.xml").getroot()
    assert root.find(f"{TASK_NS}Settings/{TASK_NS}RunOnlyIfIdle").text == "false"


def test_the_installer_placeholders_are_still_the_ones_the_installer_replaces():
    """A renamed token here is a task registered to run a literal
    `__GANYMEDE_HOST_EXE__`, which fails on every fire."""
    raw = (PACKAGING / "ganymede-host-task.xml").read_text()
    script = (PACKAGING / "install-windows.ps1").read_text()
    for token in ("__GANYMEDE_HOST_EXE__", "__GANYMEDE_USER__"):
        assert token in raw
        assert token in script


# --------------------------------------------------------------------------
# systemd
# --------------------------------------------------------------------------


def _unit(name: str) -> configparser.ConfigParser:
    # systemd allows repeated keys; ConfigParser does not, so read them as
    # lists rather than rejecting a legal unit file.
    parser = configparser.ConfigParser(strict=False, allow_no_value=True)
    parser.optionxform = str
    parser.read_string((PACKAGING / name).read_text())
    return parser


def test_the_service_is_a_oneshot_tick():
    unit = _unit("ganymede-host.service")
    assert unit["Service"]["Type"] == "oneshot"
    assert unit["Service"]["ExecStart"].endswith("--once")


def test_the_service_reads_the_key_from_a_file_rather_than_the_unit():
    """`systemctl cat` and `systemctl show` are readable by any local user."""
    unit = _unit("ganymede-host.service")
    assert "EnvironmentFile" in unit["Service"]
    # Directives only. The unit explains at length why the key is not here, and
    # a raw text search cannot tell that explanation from an actual assignment.
    directives = "\n".join(
        line for line in (PACKAGING / "ganymede-host.service").read_text().splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "GANYMEDE_KEY" not in directives


def test_the_service_can_write_the_state_directory_under_protectsystem_strict():
    """ProtectSystem=strict makes everything read-only except ReadWritePaths.
    The agent writes sentinels into the state directory; leaving it out turns
    `--pause` into a permission error."""
    unit = _unit("ganymede-host.service")
    assert "/var/lib/ganymede" in unit["Service"]["ReadWritePaths"]


def test_the_timer_jitters_so_a_fleet_does_not_poll_in_lockstep():
    unit = _unit("ganymede-host.timer")
    assert unit["Timer"]["RandomizedDelaySec"]
    assert unit["Timer"]["Persistent"] == "true"


def test_the_three_platforms_agree_on_a_cadence():
    """The exit criterion is "idles -> starts within one timer interval", so a
    platform that quietly ticks at a different rate has a different criterion."""
    timer = _unit("ganymede-host.timer")["Timer"]["OnUnitActiveSec"]
    plist = plistlib.loads((PACKAGING / "com.ganymede.host.plist").read_bytes())
    task = ET.parse(PACKAGING / "ganymede-host-task.xml").getroot().find(
        f"{TASK_NS}Triggers/{TASK_NS}TimeTrigger/{TASK_NS}Repetition/{TASK_NS}Interval").text

    assert timer == "5min"
    assert plist["StartInterval"] == 300
    assert task == "PT5M"


# --------------------------------------------------------------------------
# install scripts
# --------------------------------------------------------------------------


@pytest.mark.parametrize("script", ["install-linux.sh", "install-macos.sh"])
def test_the_shell_installers_are_syntactically_valid(script):
    import subprocess

    proc = subprocess.run(["bash", "-n", str(PACKAGING / script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize(
    "script", ["install-linux.sh", "install-macos.sh", "install-windows.ps1"]
)
def test_every_installer_offers_a_way_back_out(script):
    """A volunteer who cannot cleanly remove your software will not install it
    in the first place."""
    raw = (PACKAGING / script).read_text().lower()
    assert "uninstall" in raw


@pytest.mark.parametrize(
    "script", ["install-linux.sh", "install-macos.sh", "install-windows.ps1"]
)
def test_every_installer_ends_by_verifying_rather_than_by_going_quiet(script):
    raw = (PACKAGING / script).read_text()
    assert "--check" in raw
