#!/usr/bin/env bash
# Sync the non-git data (_results/) with Google Drive.
#
# Mirrors the convention used for the harmonica / neural-forward-emg projects: the local
# ~/projects/<name> maps to Drive work/<name>, and rclone syncs the gitignored data there.
# Git holds the code (branch feat/learned-vc on origin); Drive holds the heavy artifacts
# (FEM meshes, datasets, checkpoints, benchmarks, plots) that _results/ is gitignored to keep
# out of the repo.
#
# Requires: rclone with a 'gdrive' remote (`rclone config` -> type=drive). The token
# auto-refreshes. Verify with:  rclone lsd gdrive:work
#
# Usage:
#   scripts/sync_drive.sh push        # local _results/ -> Drive   (default)
#   scripts/sync_drive.sh pull        # Drive -> local _results/   (on a fresh checkout)
#   scripts/sync_drive.sh push --dry-run
set -euo pipefail

REMOTE="gdrive:work/emgforge/_results"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

cmd="${1:-push}"; shift || true
# --copy-links is REQUIRED: _results/sanity/fem_cache is a symlink into the main repo (the
# worktree shares the FEM cache), and without -L rclone skips it — silently dropping the WR
# mesh + fibre config, which are the essential regeneration inputs. -L uploads the target's
# content as real files, which is also what a fresh checkout / the cluster wants (no symlink).
COMMON=(--copy-links --transfers 8 --stats 5s --exclude '_cluster_state/**')

case "$cmd" in
  push)
    echo "push  _results/ -> $REMOTE"
    rclone sync _results/ "$REMOTE/" --progress "${COMMON[@]}" "$@"
    ;;
  pull)
    # NOTE: `sync` makes the destination match the source and DELETES local extras. Use
    # `copy` instead if you have local-only files you want to keep.
    echo "pull  $REMOTE -> _results/"
    rclone sync "$REMOTE/" _results/ --progress "${COMMON[@]}" "$@"
    ;;
  *)
    echo "usage: $0 [push|pull] [rclone flags e.g. --dry-run]" >&2
    exit 1
    ;;
esac
