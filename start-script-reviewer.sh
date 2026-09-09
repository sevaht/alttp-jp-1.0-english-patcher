#!/usr/bin/env bash
# Launches the dialogue script reviewer (tools/script_reviewer.py). Progress
# (edits + completed flags) is saved to tools/reviewer_state.json, tracked
# in-repo alongside this tool.
#
# Requires GBA_ROM (no default -- it's your own ROM, never assumed/committed).
# SNES_ROM/PORT are overridable too:
#   GBA_ROM=/path/to/rom.gba ./start-script-reviewer.sh
#
# For your own convenience, put a gitignored start-script-reviewer.local.sh
# next to this one that hardcodes GBA_ROM and execs this script.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ -z "${GBA_ROM:-}" ]; then
  echo "error: set GBA_ROM to your GBA ROM's path, e.g.:" >&2
  echo "  GBA_ROM=/path/to/rom.gba ./start-script-reviewer.sh" >&2
  exit 1
fi

SNES_ROM="${SNES_ROM:-alttp-us.sfc}"
PORT="${PORT:-8123}"

echo "Script Reviewer: http://127.0.0.1:${PORT}/"

exec python3 tools/script_reviewer.py \
  --gba-rom "$GBA_ROM" \
  --snes-rom "$SNES_ROM" \
  --port "$PORT"
