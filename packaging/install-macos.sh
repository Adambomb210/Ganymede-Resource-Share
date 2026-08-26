#!/usr/bin/env bash
# Install the Ganymede host agent under launchd (docs/02-architecture-v2.md 7, 4.1).
#
#   ./packaging/install-macos.sh --coordinator https://... --key ganymede_...
#   ./packaging/install-macos.sh --uninstall
#
# macOS is the one platform with no container option at all: 6.8 records that
# the container runtime here cannot reach the GPU, so a Mac contributes through
# the native pip path or not at all. That is why this script installs the
# trainer stack -- the one place any installer here pulls real dependencies.
#
# Run as your normal user, not with sudo. The LaunchAgent runs as you (see the
# plist for why), so the install has to leave things you own. The two steps that
# genuinely need root ask for it individually.
set -euo pipefail

PLIST_LABEL=com.ganymede.host
PLIST_DIR="$HOME/Library/LaunchAgents"
PLIST_PATH="$PLIST_DIR/$PLIST_LABEL.plist"
PREFIX_ETC=/etc/ganymede
STATE_DIR="$HOME/Library/Application Support/Ganymede"
CACHE_DIR="$HOME/.cache/huggingface"
LOG_DIR=/var/log/ganymede

COORDINATOR=""
KEY=""
CACHE_CAP_GB=100
UNINSTALL=0

die() { echo "error: $*" >&2; exit 1; }
note() { echo "  $*"; }

usage() {
    cat <<EOF
usage: $0 --coordinator URL --key KEY [--cache-cap-gb N]
       $0 --uninstall
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --coordinator) COORDINATOR="${2:-}"; shift 2 ;;
        --key)         KEY="${2:-}"; shift 2 ;;
        --cache-cap-gb) CACHE_CAP_GB="${2:-}"; shift 2 ;;
        --uninstall)   UNINSTALL=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *) usage >&2; die "unknown argument $1" ;;
    esac
done

[ "$(id -u)" -ne 0 ] || die "run this as your normal user, not with sudo (the agent runs as you)"
[ "$(uname -s)" = Darwin ] || die "this is the macOS installer; use install-linux.sh"

# ---------------------------------------------------------------- uninstall --
if [ "$UNINSTALL" -eq 1 ]; then
    echo "Removing the Ganymede host agent."
    launchctl unload "$PLIST_PATH" 2>/dev/null || true
    ganymede-host --stop 2>/dev/null || true
    rm -f "$PLIST_PATH"
    echo
    echo "Removed: the LaunchAgent."
    echo "Left alone, delete by hand if you want them gone:"
    echo "  $PREFIX_ETC       (config and key -- needs sudo)"
    echo "  $CACHE_DIR        (downloaded base models)"
    echo "  $STATE_DIR        (sentinels)"
    echo "The package is untouched: pip3 uninstall ganymede"
    exit 0
fi

# ------------------------------------------------------------ prerequisites --
echo "Checking prerequisites."
command -v python3 >/dev/null 2>&1 || die "python3 not found (try: brew install python@3.12)"
python3 - <<'PY' || die "python 3.11 or newer is required"
import sys
sys.exit(0 if sys.version_info >= (3, 11) else 1)
PY
[ -n "$COORDINATOR" ] || { usage >&2; die "--coordinator is required"; }
[ -n "$KEY" ] || { usage >&2; die "--key is required"; }
note "ok"

# ---------------------------------------------------------------- install ---
# The trainer extra, unlike every other installer here. There is no image to
# carry transformers and peft on this platform, so the host has to.
echo "Installing the ganymede package and the trainer stack (this pulls torch -- a few minutes)."
python3 -m pip install --quiet --upgrade ".[trainer]" || die "pip install failed"

AGENT_BIN="$(command -v ganymede-host || true)"
[ -n "$AGENT_BIN" ] || die "ganymede-host is not on PATH after install -- check your pip --user bin dir"
note "agent at $AGENT_BIN"

echo "Creating directories."
mkdir -p "$STATE_DIR" "$CACHE_DIR" "$PLIST_DIR"
# /var/log is root-owned and a LaunchAgent runs as you, so this is one of the
# two steps that needs sudo. Asked for explicitly rather than by running the
# whole script as root.
sudo mkdir -p "$LOG_DIR"
sudo chown "$(id -u):$(id -g)" "$LOG_DIR"
sudo mkdir -p "$PREFIX_ETC"

echo "Writing configuration."
# The key lives in host.json here rather than a separate env file: launchd has
# no EnvironmentFile equivalent, and a plist in ~/Library/LaunchAgents is 0644
# by default -- so the key must not be in the plist. A 0600 file owned by the
# contributor is the equivalent protection.
TMP_CONF="$(mktemp)"
cat > "$TMP_CONF" <<EOF
{
  "coordinator_url": "$COORDINATOR",
  "key": "$KEY",
  "runtime": "native",
  "state_dir": "$STATE_DIR",
  "cache_dir": "$CACHE_DIR",
  "cache_cap_gb": $CACHE_CAP_GB
}
EOF
sudo install -m 0600 -o "$(id -u)" -g "$(id -g)" "$TMP_CONF" "$PREFIX_ETC/host.json"
rm -f "$TMP_CONF"
note "config at $PREFIX_ETC/host.json (mode 0600, owned by you)"

echo "Installing the LaunchAgent."
sed "s|<string>/usr/local/bin/ganymede-host</string>|<string>$AGENT_BIN</string>|" \
    "$(cd "$(dirname "$0")" && pwd)/$PLIST_LABEL.plist" > "$PLIST_PATH"
chmod 0644 "$PLIST_PATH"

# Unload first so re-running is a repair rather than a "service already loaded".
launchctl unload "$PLIST_PATH" 2>/dev/null || true
launchctl load "$PLIST_PATH"
note "agent loaded"

# ------------------------------------------------------------------- verify --
echo
echo "Verifying."
if "$AGENT_BIN" --check; then
    STATUS=ok
else
    STATUS=failed
fi

cat <<EOF

------------------------------------------------------------------
Ganymede host agent installed. Check: $STATUS

  Pause (take your GPU back, no network needed):
      touch "$STATE_DIR/pause"
  Resume:
      ganymede-host --resume

  Is it running?      launchctl list | grep ganymede
  What did it do?     tail -f $LOG_DIR/host-agent.log
  Uninstall:          $0 --uninstall

  Note: this is a LaunchAgent, so it ticks while you are logged in.
  Locking the screen is fine; a full log out pauses it until next login.
------------------------------------------------------------------
EOF

[ "$STATUS" = ok ]
