#!/usr/bin/env bash
# Print the chip-window end GW for `fplai predict --through-gw`, derived from
# the newest FPL bootstrap snapshot on disk. Prints nothing (exit 0) when no
# upcoming GW exists (between seasons) — callers then fall back to a plain
# `fplai predict`.
#
# Usage: scripts/through_gw.sh [snapshot_root]
#   snapshot_root  directory of daily YYYY-MM-DD snapshot dirs
#                  (default: data/raw/fpl_api/snapshots)
#
# Mirrors fplai.models.sampler.chip_window_end for 2025+ seasons: set-1 chips
# (WC1/FH1/BB1/TC1) expire at the GW19 deadline, set 2 covers GW20-38 — so the
# prediction window a pre-GW20 `fplai simulate` needs ends at GW19, after that
# at GW38.
set -euo pipefail

SNAP_ROOT=${1:-data/raw/fpl_api/snapshots}

if [ ! -d "$SNAP_ROOT" ]; then
  exit 0
fi

latest_day=$(find "$SNAP_ROOT" -mindepth 1 -maxdepth 1 -type d \
               -name '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]' \
               -exec basename {} \; | LC_ALL=C sort | tail -n 1)
bootstrap="$SNAP_ROOT/$latest_day/bootstrap.json"
if [ -z "$latest_day" ] || [ ! -f "$bootstrap" ]; then
  exit 0
fi

# The next GW: the event flagged is_next, else the first unfinished event.
next_gw=$(jq -r '[.events[] | select(.is_next == true)][0].id // empty' "$bootstrap")
if [ -z "$next_gw" ]; then
  next_gw=$(jq -r '[.events[] | select(.finished | not)][0].id // empty' "$bootstrap")
fi
if [ -z "$next_gw" ]; then
  exit 0
fi

if [ "$next_gw" -le 19 ]; then
  echo 19
else
  echo 38
fi
