<#
Install the Ganymede host agent as a Scheduled Task (docs/02-architecture-v2.md 7, 4.1).

    .\packaging\install-windows.ps1 -Coordinator https://... -Key ganymede_...
    .\packaging\install-windows.ps1 -Uninstall

Windows is where most consumer GPUs actually live, and 4.1 is explicit that
requiring WSL2 is a real hurdle for a volunteer -- so the default here is the
native pip path, with `-Runtime docker` available for anyone who has Docker
Desktop and wants container isolation.

Run from an elevated PowerShell: registering a scheduled task and writing under
PROGRAMDATA both need it. The task itself runs at LeastPrivilege (see the XML).
#>

[CmdletBinding()]
param(
    [string]$Coordinator,
    [string]$Key,
    [ValidateSet('native', 'docker')]
    [string]$Runtime = 'native',
    [double]$CacheCapGb = 100,
    [switch]$Uninstall
)

$ErrorActionPreference = 'Stop'

$TaskPath  = '\Ganymede\'
$TaskName  = 'ganymede-host'
$FullTask  = "$TaskPath$TaskName"
$DataDir   = Join-Path $env:PROGRAMDATA 'Ganymede'
$ConfPath  = Join-Path $DataDir 'host.json'
$CacheDir  = Join-Path $DataDir 'hf-cache'
$LogDir    = Join-Path $DataDir 'logs'

function Die($msg) { Write-Error $msg; exit 1 }
function Note($msg) { Write-Host "  $msg" }

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    (New-Object Security.Principal.WindowsPrincipal $id).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-Admin)) {
    Die "run this from an elevated PowerShell (Run as Administrator)"
}

# ------------------------------------------------------------------ uninstall
# First, not last. A volunteer who cannot cleanly remove your software will not
# install it, and an uninstall path written as an afterthought is one nobody has
# ever run.
if ($Uninstall) {
    Write-Host "Removing the Ganymede host agent."
    try { & ganymede-host --stop 2>$null } catch { }
    try { docker rm -f ganymede-worker 2>$null | Out-Null } catch { }
    if (Get-ScheduledTask -TaskPath $TaskPath -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskPath $TaskPath -TaskName $TaskName -Confirm:$false
        Note "task removed"
    } else {
        Note "no task registered"
    }
    Write-Host ""
    Write-Host "Removed: the scheduled task and any running worker."
    Write-Host "Left alone, delete by hand if you want them gone:"
    Write-Host "  $DataDir  (config, key, sentinels, and downloaded base models)"
    Write-Host "The package is untouched: pip uninstall ganymede"
    exit 0
}

# -------------------------------------------------------------- prerequisites
# Loudly, and before touching anything: a half-install that fails at step six
# leaves the contributor working out what state their machine is in.
Write-Host "Checking prerequisites."

$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { Die "python not found on PATH" }
$ver = & python -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ([version]$ver -lt [version]'3.11') { Die "python 3.11 or newer is required (found $ver)" }

if ($Runtime -eq 'docker') {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Die "docker not found -- install Docker Desktop, or use -Runtime native"
    }
    try { docker info 2>$null | Out-Null } catch { Die "the docker daemon is not reachable -- is Docker Desktop running?" }
}

if (-not $Coordinator) { Die "-Coordinator is required" }
if (-not $Key) { Die "-Key is required" }
Note "ok"

# ------------------------------------------------------------------- install
Write-Host "Installing the ganymede package."
# The trainer extra only on the native path: there is no image to carry
# transformers and peft, so the host has to. The docker path gets them from
# the worker image and installing them here would be several GB for nothing.
$spec = if ($Runtime -eq 'native') { '.[trainer]' } else { '.' }
& python -m pip install --quiet --upgrade $spec
if ($LASTEXITCODE -ne 0) { Die "pip install failed" }

# pip's Scripts directory is not reliably on PATH for a task running outside an
# interactive shell, so the task gets an absolute path rather than a bare name.
$agent = (Get-Command ganymede-host -ErrorAction SilentlyContinue)
if (-not $agent) {
    $scripts = & python -c "import sysconfig; print(sysconfig.get_path('scripts'))"
    $candidate = Join-Path $scripts 'ganymede-host.exe'
    if (Test-Path $candidate) { $agentPath = $candidate } else { Die "ganymede-host.exe not found after install" }
} else {
    $agentPath = $agent.Source
}
Note "agent at $agentPath"

Write-Host "Creating directories."
foreach ($d in @($DataDir, $CacheDir, $LogDir)) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

Write-Host "Writing configuration."
# The key lives in host.json, and host.json is locked to Administrators and
# SYSTEM. PROGRAMDATA is world-readable by default, so inheritance is disabled
# and the ACL rebuilt -- otherwise a bearer token (6.3) sits in a directory
# every local user can read.
$conf = [ordered]@{
    coordinator_url = $Coordinator
    key             = $Key
    runtime         = $Runtime
    state_dir       = $DataDir
    cache_dir       = $CacheDir
    cache_cap_gb    = $CacheCapGb
}
$conf | ConvertTo-Json | Set-Content -Path $ConfPath -Encoding UTF8

$acl = Get-Acl $ConfPath
$acl.SetAccessRuleProtection($true, $false)   # stop inheriting PROGRAMDATA's ACL
$acl.Access | ForEach-Object { $acl.RemoveAccessRule($_) | Out-Null }
foreach ($who in @('BUILTIN\Administrators', 'NT AUTHORITY\SYSTEM')) {
    $acl.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule(
        $who, 'FullControl', 'Allow')))
}
Set-Acl -Path $ConfPath -AclObject $acl
Note "config at $ConfPath (Administrators and SYSTEM only)"

Write-Host "Registering the scheduled task."
$template = Join-Path $PSScriptRoot 'ganymede-host-task.xml'
if (-not (Test-Path $template)) { Die "missing $template" }

# Substituted on a copy: the template on disk never changes, so re-running the
# installer is a repair rather than a rewrite.
$user = "$env:USERDOMAIN\$env:USERNAME"
$xml = (Get-Content $template -Raw).
    Replace('__GANYMEDE_HOST_EXE__', $agentPath).
    Replace('__GANYMEDE_USER__', $user)

# Unregister first so this is idempotent rather than "task already exists".
if (Get-ScheduledTask -TaskPath $TaskPath -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskPath $TaskPath -TaskName $TaskName -Confirm:$false
}
Register-ScheduledTask -Xml $xml -TaskName $TaskName -TaskPath $TaskPath | Out-Null
Note "task registered as $FullTask, running as $user"

# -------------------------------------------------------------------- verify
# The last step, always: a contributor should finish an install looking at a
# pass or a fail, never at silence.
Write-Host ""
Write-Host "Verifying."
& $agentPath --check
$status = if ($LASTEXITCODE -eq 0) { 'ok' } else { 'failed' }

@"

------------------------------------------------------------------
Ganymede host agent installed. Check: $status

  Pause (take your GPU back, no network needed):
      New-Item -ItemType File "$DataDir\pause"
  Resume:
      ganymede-host --resume

  Is it running?      Get-ScheduledTask -TaskPath '$TaskPath' | Get-ScheduledTaskInfo
  What did it do?     Get-WinEvent -LogName Microsoft-Windows-TaskScheduler/Operational
  Uninstall:          .\packaging\install-windows.ps1 -Uninstall

  Note: the task will not start while on battery, and stops if you unplug.
------------------------------------------------------------------
"@ | Write-Host

if ($status -ne 'ok') { exit 1 }
