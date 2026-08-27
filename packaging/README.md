# Scheduler packaging

Three operating systems, three schedulers, one job: run `ganymede-host --once`
every five minutes and let the agent decide whether anything should happen.

This is the operator-facing file. `INSTALL.md` at the repository root is the
contributor-facing one, and it is the document that matters more.

```
ganymede-host.service      Linux, systemd  -- the tick
ganymede-host.timer        Linux, systemd  -- the cadence
com.ganymede.host.plist    macOS, launchd
ganymede-host-task.xml     Windows, Task Scheduler
install-linux.sh           }
install-macos.sh           } and each of them --uninstall
install-windows.ps1        }
```

## Why the scheduler decides nothing

Every one of these files fires unconditionally, on a fixed interval, and asks
no questions. All three have some form of built-in idle gate — systemd has
none, launchd has `ThrottleInterval`, Task Scheduler has a whole
`RunOnlyIfIdle` machinery — and none of them is used.

The reason is that §7.1's `is_idle()` is meant to be *one* predicate shared by
every platform. Delegating half of it to Task Scheduler would put a
Windows-only second opinion beside the Python one, and the two would disagree
in ways nobody could reproduce: a contributor reports "it never runs on my
desktop", and the answer is buried in a checkbox in a Microsoft UI rather than
in a function with tests. One source of truth, in Python, on all three.

## Which delivery path each platform takes

From §4.1's table:

| Platform | Path | Why |
|---|---|---|
| Linux + NVIDIA | container (default) or native | Full §4.6 hardening available |
| macOS + Apple Silicon | **native only** | §6.8: the container runtime cannot reach the GPU here |
| Windows + NVIDIA | native (default) or container | Docker Desktop reaches the GPU through WSL2, but requiring WSL2 is a real hurdle for a volunteer |

`runtime` in `host.json` selects it. The macOS installer is the only one that
installs the trainer stack (`transformers`, `peft`, `torch`), because it is the
only platform with no image to carry them.

## LaunchAgent, not LaunchDaemon

The real decision on macOS, and worth stating outside the plist where a stray
hyphen cannot break the file.

A LaunchDaemon runs system-wide with no user session, which looks like strictly
more idle time to harvest. But the worker on this platform is a native process
doing GPU compute through Metal, and Metal is far more reliably available
inside a logged-in user's session than from a session-less root daemon.
Headless Metal is a known soft spot, and it is not the thing to build the one
mechanism that must work on.

So: a LaunchAgent, which ticks while the contributor is logged in. Locking the
screen is fine — that is the normal case, and `ioreg`'s `HIDIdleTime` is
exactly what reports it. A full log out pauses contribution until next login,
which for a laptop volunteer is rare and is documented in `INSTALL.md` rather
than hidden.

The idle predicate did not decide this. Metal availability did.

## Where the key lives, per platform

Never in the scheduler file. All three of those are readable by every local
user in their default state.

| Platform | Key in | Protected by |
|---|---|---|
| Linux | `/etc/ganymede/host.env` | root-owned, mode 0600, referenced by `EnvironmentFile=` |
| macOS | `/etc/ganymede/host.json` | owned by the contributor, mode 0600 (launchd has no `EnvironmentFile` equivalent) |
| Windows | `%PROGRAMDATA%\Ganymede\host.json` | ACL inheritance disabled, Administrators and SYSTEM only |

`systemctl cat` and `systemctl show` will happily print a unit file to any
local user, and a plist in `~/Library/LaunchAgents` is mode 0644. A bearer
token (§6.3) in either is a bearer token in every support log a contributor
ever pastes into an issue.

## The Docker socket, honestly

The Linux unit runs as root. Putting the agent in the `docker` group instead
would not sandbox anything: anything that can reach `/var/run/docker.sock` can
bind-mount the host root into a container and read or write it as root. It is
root-equivalent by design. Spelling it as a group membership would be the same
privilege wearing a smaller word.

`NoNewPrivileges`, `ProtectSystem=strict` and `PrivateTmp` are still in the
unit, and they still narrow this process's own footprint. They do not constrain
what a container launched through that socket can do, and the unit says so.

## Verifying it actually fires

The single most useful thing in this file. "Is it running?" is the first
question every contributor asks, and none of these three answer it the same way.

**Linux**
```sh
systemctl list-timers ganymede-host.timer     # NEXT and LEFT columns
systemctl status ganymede-host.service        # last run's exit status
journalctl -u ganymede-host.service -n 50     # what the last few ticks decided
```

**macOS**
```sh
launchctl list | grep ganymede                # exit code of the last run
tail -f /var/log/ganymede/host-agent.log
```

**Windows**
```powershell
Get-ScheduledTask -TaskPath '\Ganymede\' | Get-ScheduledTaskInfo   # LastRunTime, LastTaskResult
Get-WinEvent -LogName Microsoft-Windows-TaskScheduler/Operational -MaxEvents 20
```

On every platform, `ganymede-host --check` is the one command that reports what
the agent resolved, whether the machine reads as idle right now, and what image
the coordinator wants — without starting anything.

## What CI can and cannot check

`tests_host/test_packaging.py` parses all four scheduler files and asserts the
things that fail *silently*: a doubled hyphen inside an XML comment (which
makes a plist launchd will not load), a Task Scheduler document whose elements
are in the wrong order (rejected at import), an encoding claim of any kind in
the task XML, a cadence that has drifted apart between platforms.

The `host` job in `.github/workflows/ci.yml` goes further, on real operating
systems: Task Scheduler *imports* the task on windows-latest, `plutil` lints
the plist on macos-latest, PowerShell parses the installer, `systemd-analyze`
verifies the unit and then systemd *runs* it, and each platform's real idle
probe is executed rather than faked.

That distinction earned its keep immediately. **A parser is not an importer**:
the task XML parsed cleanly in Python for two commits while Task Scheduler
refused it outright, and only the Windows runner could say so. Every one of
these files shipped broken in some way on its first draft.

What still cannot be checked here is whether a timer *fires on a contributor's
machine over hours*, and whether the idle probes agree with a real desktop.
Those come from a real install, using the commands above.

## Editing notes

- **XML comments cannot contain `--`.** This bites immediately, because the flag
  being documented is `--once`. Keep prose in this file.
- **`ganymede-host-task.xml` carries no XML declaration, deliberately**, and
  it took two failures to land there. It first declared UTF-16 while being
  ASCII on disk, which every parser rejects. Declaring UTF-8 instead fixed the
  parsers and broke the real consumer: the installer reads the file into a
  .NET string, which is already UTF-16 in memory, and Task Scheduler's importer
  refuses a contrary encoding claim inside one — `unable to switch the
  encoding`, at column 40. With no declaration both are satisfied, since XML
  defaults to UTF-8 absent a declaration and a BOM. Only the Windows CI job
  could have caught the second one; a parser is not an importer.

  If you ever switch to `schtasks /create /xml`, that reads a *file* rather
  than a string and historically wants UTF-16 — convert at that point and add
  a matching declaration then.
- The two `__GANYMEDE_*__` placeholders in the task XML are substituted on a
  copy. The template on disk never changes, which is what makes re-running the
  installer a repair rather than a rewrite.
