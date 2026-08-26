# Lending Ganymede your GPU

Ganymede trains language-model adapters across borrowed hardware. When your
machine is idle, it does a few minutes of training and uploads a small file.
When you want your machine back, it stops.

This document is for the person lending the machine. It is not long, and the
most important thing in it is [how to make it stop](#the-off-switch).

---

## The off switch

Read this first, before installing anything. This is the promise the rest of
the document rests on.

Creating one empty file stops Ganymede from using your GPU:

| Your machine | Create this file |
|---|---|
| Linux | `/var/lib/ganymede/pause` |
| macOS | `~/Library/Application Support/Ganymede/pause` |
| Windows | `C:\ProgramData\Ganymede\pause` |

```sh
sudo touch /var/lib/ganymede/pause          # Linux
touch ~/Library/Application\ Support/Ganymede/pause   # macOS
```
```powershell
New-Item -ItemType File "$env:PROGRAMDATA\Ganymede\pause"   # Windows
```

Any running work stops within a couple of minutes and nothing new starts. There
is nothing to log into and nothing to configure. It works with **no network
connection**, with the coordinator down, and with no explanation to anyone.

Delete the file to resume, or run `ganymede-host --resume`.

`ganymede-host --pause` does the same thing if you would rather type a command.
The file is the real mechanism; the command just creates it.

---

## What it actually does to your machine

- **Only when you are not using it.** Fifteen minutes with no keyboard or mouse
  activity, *and* no other program using the GPU. Touch the machine and the
  current round is abandoned within a couple of minutes.
- **Bounded.** By default: 16 GB of RAM, 4 CPU cores, 256 processes. On Linux
  and Windows the worker runs in a container with no capabilities, no ability to
  gain privileges, and a read-only filesystem.
- **It never reads your files.** The worker sees exactly two directories: a
  cache of downloaded models, and a directory holding your pause file. Nothing
  else on the machine is visible to it.
- **Uploads are small.** A few tens of MB per round — an adapter, not a model.
- **Laptops:** on Windows it will not start on battery and stops if you unplug.
  On any platform, the resource limits above are the ones that keep the machine
  usable while it works.

You can lower any of the limits — see [Settings](#settings).

---

## Before you start

**Disk.** The numbers, so you can check before committing:

| | Docker path (Linux, Windows) | Native path (macOS) |
|---|---|---|
| Worker image | ~12 GB | — |
| Python + PyTorch | — | ~5 GB |
| One base model | 3–16 GB depending on the run | same |
| **Minimum free** | **35 GB** | **25 GB** |

The default cache ceiling is **100 GB**, so over time, across several runs with
different base models, the model cache will grow toward that. It never exceeds
it: the oldest unused model is deleted when a new one would push the cache over
the cap. If 100 GB is more than you want to lend, set `cache_cap_gb` lower — 40
is fine, it just means occasionally re-downloading a model.

`ganymede-host --check` prints your actual free space and the cap it is
enforcing. `ganymede-cache --cap-gb 40` reclaims down to a cap immediately.

**Everything else.**

| | Requirement |
|---|---|
| Python | 3.11 or newer |
| Linux | systemd, and Docker (or `--runtime native`) |
| macOS | Apple Silicon. No Docker — the container cannot reach the GPU on macOS |
| Windows | Python; Docker Desktop only if you want container isolation |
| GPU | Optional. A machine with no GPU registers fine and simply never matches work |

**A key.** Ask whoever runs the coordinator you are contributing to. It looks
like `ganymede_...`, and it identifies your machine's contributions.

---

## Install

### Linux

```sh
git clone https://github.com/Adambomb210/Ganymede-Resource-Share
cd Ganymede-Resource-Share
sudo ./packaging/install-linux.sh \
    --coordinator https://coordinator.example \
    --key ganymede_your_key_here
```

### macOS

```sh
git clone https://github.com/Adambomb210/Ganymede-Resource-Share
cd Ganymede-Resource-Share
./packaging/install-macos.sh \
    --coordinator https://coordinator.example \
    --key ganymede_your_key_here
```

Not with `sudo` — the agent runs as you, and the script asks for root only for
the two steps that need it. It installs PyTorch, so allow a few minutes.

macOS runs the agent while you are **logged in**. Locking the screen is the
normal case and works fine; a full log out pauses contribution until you log
back in.

### Windows

From an elevated PowerShell:

```powershell
git clone https://github.com/Adambomb210/Ganymede-Resource-Share
cd Ganymede-Resource-Share
.\packaging\install-windows.ps1 `
    -Coordinator https://coordinator.example `
    -Key ganymede_your_key_here
```

Add `-Runtime docker` if you have Docker Desktop and want the container path.

---

## Checking it works

Every installer ends by running this, but you can run it any time:

```sh
ganymede-host --check
```

```
coordinator : https://coordinator.example
key         : set
runtime     : docker
state dir   : /var/lib/ganymede
cache dir   : /var/cache/ganymede/hf
cache cap   : 100 GB
free disk   : 412.7 GB
idle now    : no (user active 41s ago, needs 900s)
active runs : 2
image       : ganymede/worker-llm:v3 -- required by 2 active run(s)
```

`idle now: no` while you are sitting at the machine is correct — that is the
software working. Leave it alone for fifteen minutes and it will read `yes`.

**Is the scheduler firing?**

```sh
systemctl list-timers ganymede-host.timer          # Linux
launchctl list | grep ganymede                     # macOS
```
```powershell
Get-ScheduledTask -TaskPath '\Ganymede\' | Get-ScheduledTaskInfo   # Windows
```

**What has it been doing?**

```sh
journalctl -u ganymede-host.service -n 50          # Linux
tail -f /var/log/ganymede/host-agent.log           # macOS
```

---

## Settings

`/etc/ganymede/host.json` on Linux and macOS,
`C:\ProgramData\Ganymede\host.json` on Windows. Edit and save; the next tick
picks it up. Everything is optional except the coordinator URL and the key.

```json
{
  "coordinator_url": "https://coordinator.example",
  "cache_cap_gb": 40,
  "memory": "8g",
  "cpus": "2",
  "user_idle_sec": 1800,
  "active_window": "23:00-07:00"
}
```

| Setting | Default | What it does |
|---|---|---|
| `cache_cap_gb` | `100` | Ceiling on downloaded base models. Oldest-unused deleted first |
| `memory`, `cpus` | `16g`, `4` | What the worker may use |
| `pids_limit` | `256` | Process ceiling |
| `user_idle_sec` | `900` | Seconds untouched before your machine counts as idle. `0` disables the check — right for a headless server nobody sits at |
| `require_gpu_free` | `true` | Never share the GPU with your own work |
| `active_window` | *(any hour)* | Restrict to a local-time window, e.g. `"23:00-07:00"` |
| `runtime` | `docker` | `docker` or `native` |
| `image_tag` | *(from coordinator)* | Pin an image instead of following the run |
| `gpus` | `all` | e.g. `"device=1"` to lend one card of several |

A typo in a setting name is a hard error rather than a shrug — the agent will
refuse to start and name the key it did not recognise. A setting that is
silently ignored is worse than one that fails loudly.

---

## Uninstalling

```sh
sudo ./packaging/install-linux.sh --uninstall      # Linux
./packaging/install-macos.sh --uninstall           # macOS
```
```powershell
.\packaging\install-windows.ps1 -Uninstall         # Windows
```

This stops any running worker and removes the scheduler entry. It deliberately
does **not** delete your config or your downloaded models — it tells you where
they are so you can remove them yourself. Then:

```sh
pip uninstall ganymede
```

---

## If something is wrong

**"It never gets any work."** Run `ganymede-host --check`. The `idle now` line
gives the reason, and `active runs` tells you whether the coordinator has
anything to hand out at all. A GPU that does not meet a run's requirements is a
normal outcome, not a fault — your machine stays registered and picks up work
from a run it does suit.

**"It filled my disk."** It should not: the cap is enforced before each start.
`ganymede-cache --cap-gb 40` reclaims immediately, and
`ganymede-cache --dry-run` shows what would go first. If the cache is genuinely
over its cap, that is a bug — please report it.

**"I want it off right now."** [The off switch](#the-off-switch). No network
needed.

**Reporting a problem.** `ganymede-host --check` output plus the last 50 lines
of the log is almost always enough. Neither contains your key.

---

## What Ganymede sees, precisely

Because it matters, and because you should not have to take it on trust:

- Your machine reports what kind of GPU it has, how much memory, and roughly how
  fast it is. That is how the coordinator decides what work fits.
- It downloads a public base model and a small adapter file, trains for a few
  minutes on data the coordinator supplies, and uploads the resulting adapter.
- It never reads, uploads, or has access to anything else on your machine. On
  Linux and Windows this is enforced by the container: the only two paths it can
  see are the model cache and the directory holding your pause file, and the
  second of those is mounted read-only.

The details are in `docs/02-architecture-v2.md` §4.6 and §7.
