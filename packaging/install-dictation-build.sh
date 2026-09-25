#!/bin/bash
# Install a push-to-talk dictation build over Homebrew tink-agent 0.1.0 without git.
#
# Usage:
#   curl -L -o /tmp/tink-agent-dictation.tar.gz '<tarball-url>'
#   bash packaging/install-dictation-build.sh /tmp/tink-agent-dictation.tar.gz
#
# Rollback:
#   bash packaging/install-dictation-build.sh --rollback
#
# Paths match Homebrew Cellar layout on Apple Silicon.
set -euo pipefail

CELLAR="${TINK_AGENT_CELLAR:-/opt/homebrew/Cellar/tink-agent/0.1.0}"
APP="$CELLAR/libexec/app"
VENV_PY="$CELLAR/libexec/venv/bin/python"
PLIST="$HOME/Library/LaunchAgents/io.github.tajchert.tinkagent.plist"
CONFIG="$HOME/.tink-agent/config.json"
BACKUP_ROOT="$HOME/.tink-agent/backups"
STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="$BACKUP_ROOT/$STAMP"

rollback() {
  latest="$(ls -1dt "$BACKUP_ROOT"/*/ 2>/dev/null | head -1 || true)"
  if [ -z "$latest" ]; then
    echo "error: no backup under $BACKUP_ROOT" >&2
    exit 1
  fi
  echo "Restoring from $latest"
  rm -rf "$APP"
  cp -a "$latest/app" "$APP"
  if [ -f "$latest/config.json" ]; then
    cp -a "$latest/config.json" "$CONFIG"
  fi
  uid="$(id -u)"
  launchctl bootout "gui/$uid" "$PLIST" 2>/dev/null || true
  launchctl bootstrap "gui/$uid" "$PLIST"
  echo "Rollback complete. LaunchAgent restarted."
  exit 0
}

if [ "${1:-}" = "--rollback" ]; then
  rollback
fi

TARBALL="${1:-}"
if [ -z "$TARBALL" ] || [ ! -f "$TARBALL" ]; then
  echo "usage: $0 <branch-tarball.tar.gz>|--rollback" >&2
  exit 1
fi

if [ ! -x "$VENV_PY" ]; then
  echo "error: expected Homebrew venv at $VENV_PY" >&2
  exit 1
fi

mkdir -p "$BACKUP_ROOT"
mkdir -p "$BACKUP"
echo "Backing up app -> $BACKUP/app"
cp -a "$APP" "$BACKUP/app"
if [ -f "$CONFIG" ]; then
  cp -a "$CONFIG" "$BACKUP/config.json"
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT
tar -xzf "$TARBALL" -C "$WORKDIR"
SRC="$(find "$WORKDIR" -mindepth 1 -maxdepth 2 -type d -name 'tink_agent' -print -quit | xargs dirname)"
if [ -z "$SRC" ] || [ ! -d "$SRC/tink_agent" ]; then
  echo "error: tarball must contain tink_agent/ at top level or one folder down" >&2
  exit 1
fi

echo "Installing source into $APP"
rm -rf "$APP"
mkdir -p "$(dirname "$APP")"
cp -a "$SRC/." "$APP/"

VENV_DIR="$(dirname "$VENV_PY")"
echo "Installing Python dependencies into $VENV_DIR"
"$VENV_PY" -m pip install -q -r "$APP/requirements.txt"

echo "Merging new config keys (existing values preserved)"
PYTHONPATH="$APP" "$VENV_PY" - "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path

from tink_agent.config import Config

path = Path(sys.argv[1])
if path.exists():
    existing = json.loads(path.read_text())
else:
    existing = {}
defaults = Config().to_dict()
merged = {**defaults, **existing}
# Preserve user slot maps but backfill missing keys from defaults.
for key in ("tones", "slot_actions"):
    if key in defaults:
        merged[key] = {**defaults[key], **(existing.get(key) or {})}
if not existing.get("dictation_profiles"):
    merged["dictation_profiles"] = defaults["dictation_profiles"]
if not existing.get("fx_button_templates"):
    merged["fx_button_templates"] = defaults.get("fx_button_templates") or {}
if "button_detector" not in existing:
    merged["button_detector"] = defaults.get("button_detector", "fxmic")
for key in (
    "knob_serial_enabled", "knob_poll_ms", "knob_start_debounce_ms",
    "dictation_release_action", "dictation_idle_cap_ms",
    "dictation_min_toggle_gap_ms",
):
    if key not in existing:
        merged[key] = defaults.get(key)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(merged, indent=2))
print(f"Wrote merged config: {path}")
PY

uid="$(id -u)"
echo "Restarting LaunchAgent"
launchctl bootout "gui/$uid" "$PLIST" 2>/dev/null || true
launchctl bootstrap "gui/$uid" "$PLIST"

echo ""
echo "Installed. Backup saved at: $BACKUP"
echo "Rollback: bash $(dirname "$0")/install-dictation-build.sh --rollback"
echo "Or manually: cp -a $BACKUP/app $APP && relaunch LaunchAgent."
