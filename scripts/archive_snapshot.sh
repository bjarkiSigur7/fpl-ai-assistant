#!/usr/bin/env bash
# Append the newest daily FPL API snapshot, gzipped, to an orphan archive branch.
#
# Usage: scripts/archive_snapshot.sh [snapshot_root] [branch]
#   snapshot_root  directory of daily YYYY-MM-DD snapshot dirs
#                  (default: data/raw/fpl_api/snapshots)
#   branch         archive branch name (default: data-archive)
#
# Behaviour:
#   * picks the NEWEST YYYY-MM-DD dir under snapshot_root and stores it as
#     snapshots/<day>.tar.gz on the archive branch (append-only history of the
#     raw bootstrap/fixtures payloads — survives cache eviction);
#   * creates the orphan branch gracefully on first run;
#   * idempotent: exits 0 without a commit when <day>.tar.gz already exists;
#   * exits 0 with a notice when there is nothing to archive (no snapshot yet).
#
# Must run from the repo root of a clone with push credentials for the remote
# (in GitHub Actions: actions/checkout's persisted GITHUB_TOKEN and
# `permissions: contents: write` on the workflow). Uses a temporary git
# worktree so the main checkout is never disturbed.
set -euo pipefail

SNAP_ROOT=${1:-data/raw/fpl_api/snapshots}
BRANCH=${2:-data-archive}
REMOTE=${ARCHIVE_REMOTE:-origin}
GIT_NAME=${ARCHIVE_GIT_NAME:-"github-actions[bot]"}
GIT_EMAIL=${ARCHIVE_GIT_EMAIL:-"41898282+github-actions[bot]@users.noreply.github.com"}

log() { echo "archive_snapshot: $*"; }

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  log "not inside a git repository — run from the repo root"
  exit 1
fi

if [ ! -d "$SNAP_ROOT" ]; then
  log "$SNAP_ROOT does not exist — nothing to archive"
  exit 0
fi

# Newest daily snapshot dir (lexicographic sort == chronological for ISO dates).
DAY=$(find "$SNAP_ROOT" -mindepth 1 -maxdepth 1 -type d \
        -name '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]' \
        -exec basename {} \; | LC_ALL=C sort | tail -n 1)
if [ -z "$DAY" ]; then
  log "no dated snapshot dirs under $SNAP_ROOT — nothing to archive"
  exit 0
fi
TARBALL="snapshots/$DAY.tar.gz"
log "archiving $SNAP_ROOT/$DAY -> $BRANCH:$TARBALL"

# Fetch the archive branch if it exists on the remote (first run: it doesn't).
git fetch --depth 1 "$REMOTE" "+refs/heads/$BRANCH:refs/remotes/$REMOTE/$BRANCH" \
  2>/dev/null || log "remote branch $BRANCH not found — will create it"

WT="$(mktemp -d)/wt"
cleanup() {
  git worktree remove --force "$WT" >/dev/null 2>&1 || true
  rm -rf "$(dirname "$WT")"
  git worktree prune >/dev/null 2>&1 || true
}
trap cleanup EXIT

if git rev-parse -q --verify "refs/remotes/$REMOTE/$BRANCH" >/dev/null; then
  # Existing archive: check it out into the worktree.
  git worktree add --no-track -B "$BRANCH" "$WT" "refs/remotes/$REMOTE/$BRANCH" >/dev/null
else
  # First run: create the orphan branch (no shared history with main).
  # `git worktree add --orphan` needs git >= 2.42 (GitHub runners have it);
  # fall back to a detached worktree + `checkout --orphan` for older gits.
  if ! git worktree add --orphan -b "$BRANCH" "$WT" >/dev/null 2>&1; then
    git worktree add --detach "$WT" >/dev/null
    git -C "$WT" checkout --orphan "$BRANCH" >/dev/null 2>&1
    git -C "$WT" rm -rf --quiet . >/dev/null 2>&1 || true
    # Drop any leftover working-tree files inherited from HEAD.
    find "$WT" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
  fi
  {
    echo "# data-archive"
    echo
    echo "Append-only archive of daily raw FPL API snapshots"
    # shellcheck disable=SC2016  # literal markdown backticks, not expansions
    echo '(`bootstrap.json` + `fixtures.json`, gzipped per day) written by the'
    # shellcheck disable=SC2016
    echo '`model-run` GitHub Actions workflow via `scripts/archive_snapshot.sh`.'
    echo "Orphan branch — no shared history with \`main\`."
  } > "$WT/README.md"
fi

if [ -f "$WT/$TARBALL" ]; then
  log "$TARBALL already archived — nothing to do"
  exit 0
fi

mkdir -p "$WT/snapshots"
# gzip -n: no filename/timestamp in the gzip header (reproducible bytes).
tar -C "$SNAP_ROOT" -cf - "$DAY" | gzip -n -9 > "$WT/$TARBALL"

git -C "$WT" add -A
if git -C "$WT" diff --cached --quiet 2>/dev/null; then
  log "nothing changed — no commit"
  exit 0
fi
git -C "$WT" -c user.name="$GIT_NAME" -c user.email="$GIT_EMAIL" \
  commit -q -m "archive: FPL snapshot $DAY"
git -C "$WT" push -u "$REMOTE" "$BRANCH"
log "pushed $TARBALL to $REMOTE/$BRANCH"
