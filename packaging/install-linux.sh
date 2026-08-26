#!/usr/bin/env bash
# Install the Ganymede host agent under systemd (docs/02-architecture-v2.md 7).
#
#   sudo ./packaging/install-linux.sh --coordinator https://... --key ganymede_...
#   sudo ./packaging/install-linux.sh --uninstall
#
# Idempotent: re-running is a repair, not a mess. Every step either does nothing
# or fixes the thing it owns, so "run it again" is a real answer to a broken
# install rather than advice to start over.
set -euo pipefail

PREFIX_ETC=/etc/ganymede
STATE_DIR=/var/lib/ganymede
CACHE_DIR=/var/cache/ganymede/hf
UNIT_DIR=/etc/systemd/system
# docker/worker-core.Dockerfile creates uid 1000 and runs as it. The cache bind
# mount has to be writable by that uid or the very first model download fails
# inside a --read-only container, which is a confusing place to discover a
# permissions problem.
CONTAINER_UID=1000

COORDINATOR=""
KEY=""
RUNTIME=docker
CACHE_CAP_GB=100
UNINSTALL=0

die() { echo "error: $*" >&2; exit 1; }
note() { echo "  $*"; }

usage() {
    cat <<EOF
usage: $0 --coordinator URL --key KEY [options]
       $0 --uninstall

  --coordinator URL   the coordinator to contribute to
  --key KEY           your contributor key (issued by the coordinator operator)
  --runtime RUNTIME   docker (default) or native
  --cache-cap-gb N    base-model cache ceiling, default ${CACHE_CAP_GB}
  --uninstall         remove everything this script installs
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --coordinator) COORDINATOR="${2:-}"; shift 2 ;;
        --key)         KEY="${2:-}"; shift 2 ;;
        --runtime)     RUNTIME="${2:-}"; shift 2 ;;
        --cache-cap-gb) CACHE_CAP_GB="${2:-}"; shift 2 ;;
        --uninstall)   UNINSTALL=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *) usage >&2; die "unknown argument $1" ;;
    esac
done

[ "$(id -u)" -eq 0 ] || die "run this with sudo -- it writes to /etc and /var"

# ---------------------------------------------------------------- uninstall --
# First, not last. A volunteer who cannot cleanly remove your software will not
# install it in the first place, and an uninstall path that is an afterthought
# is one that has never been run.
if [ "$UNINSTALL" -eq 1 ]; then
    echo "Removing the Ganymede host agent."
    systemctl disable --now ganymede-host.timer 2>/dev/null || true
    systemctl stop ganymede-host.service 2>/dev/null || true
    ganymede-host --stop 2>/dev/null || true
    docker rm -f ganymede-worker 2>/dev/null || true
    rm -f "$UNIT_DIR/ganymede-host.service" "$UNIT_DIR/ganymede-host.timer"
    systemctl daemon-reload
    echo
    echo "Removed: the timer, the unit, and any running worker."
    echo "Left alone, delete by hand if you want them gone:"
    echo "  $PREFIX_ETC     (your config and key)"
    echo "  $CACHE_DIR      (downloaded base models -- $(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1 || echo 0))"
    echo "  $STATE_DIR      (sentinels)"
    echo "The 'ganymede' pip package is untouched: pip uninstall ganymede"
    exit 0
fi

# ------------------------------------------------------------ prerequisites --
# Loudly, and before touching anything. A half-install that fails at step six is
# worse than a refusal at step one, because the contributor now has to work out
# what state their machine is in.
echo "Checking prerequisites."

command -v systemctl >/dev/null 2>&1 || die "no systemctl -- this script is for systemd hosts"
command -v python3 >/dev/null 2>&1 || die "python3 not found"

python3 - <<'PY' || die "python 3.11 or newer is required"
import sys
sys.exit(0 if sys.version_info >= (3, 11) else 1)
PY

if [ "$RUNTIME" = docker ]; then
    command -v docker >/dev/null 2>&1 || die "docker not found (install it, or use --runtime native)"
    docker info >/dev/null 2>&1 || die "the docker daemon is not reachable -- is it running?"
fi

[ -n "$COORDINATOR" ] || { usage >&2; die "--coordinator is required"; }
[ -n "$KEY" ] || { usage >&2; die "--key is required"; }
note "ok"

# ------------------------------------------------------------------ install --
echo "Installing the ganymede package."
# --no-deps on the docker path, deliberately. `ganymede/host` imports nothing
# outside the standard library (that is why it can be installed by copying),
# and the package's declared dependencies include torch -- roughly 2 GB that
# this machine already has inside the worker image and will never load on the
# host. A native install does need them, and asks for them.
if [ "$RUNTIME" = docker ]; then
    python3 -m pip install --quiet --upgrade --no-deps . || die "pip install failed"
else
    python3 -m pip install --quiet --upgrade ".[trainer]" || die "pip install failed"
fi

# The unit's ExecStart has to name a path that exists. pip puts console scripts
# in /usr/local/bin on a system Python and /usr/bin on a distro-packaged one,
# and hardcoding either is how this breaks on somebody else's distribution.
AGENT_BIN="$(command -v ganymede-host || true)"
[ -n "$AGENT_BIN" ] || die "ganymede-host is not on PATH after install"
note "agent at $AGENT_BIN"

echo "Creating directories."
install -d -m 0755 "$PREFIX_ETC"
install -d -m 0755 "$STATE_DIR"
install -d -m 0755 "$(dirname "$CACHE_DIR")"
install -d -m 0755 -o "$CONTAINER_UID" -g "$CONTAINER_UID" "$CACHE_DIR"
install -d -m 0755 /var/log/ganymede

echo "Writing configuration."
# The key goes in the environment file, never in host.json and never in the unit.
# `systemctl cat` and `systemctl show` are readable by any local user; 0600 and
# root-owned is what keeps a bearer token (6.3) out of them.
umask 077
cat > "$PREFIX_ETC/host.env" <<EOF
GANYMEDE_KEY=$KEY
EOF
chmod 0600 "$PREFIX_ETC/host.env"
chown root:root "$PREFIX_ETC/host.env"
umask 022

cat > "$PREFIX_ETC/host.json" <<EOF
{
  "coordinator_url": "$COORDINATOR",
  "runtime": "$RUNTIME",
  "state_dir": "$STATE_DIR",
  "cache_dir": "$CACHE_DIR",
  "cache_cap_gb": $CACHE_CAP_GB
}
EOF
chmod 0644 "$PREFIX_ETC/host.json"
note "config at $PREFIX_ETC/host.json (key is in host.env, mode 0600)"

echo "Installing the systemd units."
SRC="$(cd "$(dirname "$0")" && pwd)"
sed "s|^ExecStart=.*|ExecStart=$AGENT_BIN --once|" \
    "$SRC/ganymede-host.service" > "$UNIT_DIR/ganymede-host.service"
cp "$SRC/ganymede-host.timer" "$UNIT_DIR/ganymede-host.timer"
chmod 0644 "$UNIT_DIR/ganymede-host.service" "$UNIT_DIR/ganymede-host.timer"

systemctl daemon-reload
systemctl enable --now ganymede-host.timer
note "timer enabled"

# ------------------------------------------------------------------- verify --
# The last step, always, and it is the point of the whole script: a contributor
# should finish an install looking at a pass or a fail, never at silence.
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
      sudo touch $STATE_DIR/pause
  Resume:
      sudo ganymede-host --resume

  Is it running?      systemctl list-timers ganymede-host.timer
  What did it do?     journalctl -u ganymede-host.service -n 50
  Uninstall:          sudo $0 --uninstall
------------------------------------------------------------------
EOF

[ "$STATUS" = ok ]
