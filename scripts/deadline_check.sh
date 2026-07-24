#!/usr/bin/env bash
# Decide whether the hourly deadline-watch should dispatch an extra model-run.
#
# Emits `dispatch=true|false` to $GITHUB_OUTPUT (stdout when unset) and always
# exits 0 — the watch must stay quiet and green whatever happens upstream.
#
# Policy: dispatch iff BOTH
#   1. the next FPL deadline is 2.5-4.5h away (a 2h window checked hourly is
#      guaranteed to hit at least once, ~3h before the deadline), AND
#   2. no model-run (any status, including queued/in-progress) was created in
#      the last 3h — so the 2h window cannot double-fire and a recent daily
#      05:30 run suppresses redundant work.
# If the run-history check itself errors while the deadline is in the window,
# we dispatch anyway: one redundant queued run (model-run's concurrency group
# serialises) is cheaper than missing the pre-deadline refresh.
#
# Env (all optional):
#   FPL_BOOTSTRAP_URL   override the bootstrap endpoint (tests use file fixtures)
#   GH_REPO / GITHUB_REPOSITORY, GH_TOKEN   repo + token for the gh API check
#   WATCH_WINDOW_MIN_H / WATCH_WINDOW_MAX_H / WATCH_RECENT_RUN_H   tuning knobs
set -euo pipefail

BOOTSTRAP_URL=${FPL_BOOTSTRAP_URL:-https://fantasy.premierleague.com/api/bootstrap-static/}
WINDOW_MIN=${WATCH_WINDOW_MIN_H:-2.5}
WINDOW_MAX=${WATCH_WINDOW_MAX_H:-4.5}
RECENT_H=${WATCH_RECENT_RUN_H:-3}
# The FPL API rejects default curl UAs — send a browser UA like fplai.data.fpl_api.
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"

emit() { # emit <true|false> <reason>
  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    echo "dispatch=$1" >> "$GITHUB_OUTPUT"
  fi
  echo "deadline-watch: $2 (dispatch=$1)"
  exit 0
}

# Portable UTC date maths: GNU date (runners) first, BSD date (local dev) fallback.
iso_to_epoch() {
  date -u -d "$1" +%s 2>/dev/null \
    || date -u -j -f "%Y-%m-%dT%H:%M:%SZ" "$1" +%s
}
epoch_hours_ago() {
  date -u -d "$1 hours ago" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null \
    || date -u -v "-${1}H" +%Y-%m-%dT%H:%M:%SZ
}

if ! payload=$(curl -fsSL --max-time 30 --retry 2 -A "$UA" "$BOOTSTRAP_URL"); then
  emit false "bootstrap-static fetch failed — skipping quietly"
fi
if ! deadline=$(jq -r '[.events[] | select(.is_next == true)][0].deadline_time // empty' \
                  <<<"$payload" 2>/dev/null); then
  emit false "unparseable bootstrap payload — skipping quietly"
fi
if [ -z "$deadline" ]; then
  emit false "no upcoming deadline (between seasons?)"
fi

now_s=$(date -u +%s)
if ! deadline_s=$(iso_to_epoch "$deadline"); then
  emit false "unparseable deadline_time '$deadline' — skipping quietly"
fi
hours=$(awk -v a="$deadline_s" -v b="$now_s" 'BEGIN { printf "%.2f", (a - b) / 3600 }')
in_window=$(awk -v h="$hours" -v lo="$WINDOW_MIN" -v hi="$WINDOW_MAX" \
              'BEGIN { print (h >= lo && h <= hi) ? 1 : 0 }')
if [ "$in_window" != 1 ]; then
  emit false "next deadline $deadline is ${hours}h away — outside the ${WINDOW_MIN}-${WINDOW_MAX}h window"
fi

repo=${GH_REPO:-${GITHUB_REPOSITORY:-}}
if [ -z "$repo" ]; then
  emit true "deadline ${hours}h away; no repo context for the run-history check — dispatching"
fi
since=$(epoch_hours_ago "$RECENT_H")
# Any model-run created in the last RECENT_H hours, whatever its status.
if ! count=$(gh api -X GET "repos/$repo/actions/workflows/model-run.yml/runs" \
               -f "created=>=$since" -F per_page=1 --jq '.total_count' 2>/dev/null); then
  emit true "deadline ${hours}h away; run-history check failed — dispatching anyway"
fi
if [ "${count:-0}" -gt 0 ]; then
  emit false "deadline ${hours}h away but a model-run was created since $since"
fi
emit true "next deadline $deadline is ${hours}h away and no model-run since $since"
