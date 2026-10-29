#!/usr/bin/env bash
# Start one long job detached, and only that job: run every check (git pull, config greps) as separate commands
# BEFORE calling this, never chained into a background '&' list.
# A chain like 'cd X && git pull && check && (job) &' backgrounds the whole chain, so the checks' output is lost
# (2026-10-27, twice). This script detaches only the job, prints its PID, and verifies it is alive after a few seconds.
# The job runs from the repo root (this script's parent folder), whatever the caller's directory: τ² records the
# git commit of its working directory, and logged "Failed to get git hash" as an ERROR in every run log launched
# from <workspace> (2026-10-29). Pass absolute paths in the command; the log path may be relative to the caller.
# Usage: scripts/launch.sh LOGFILE -- command [args...]
set -euo pipefail
log=$1; shift
[ "${1:-}" = "--" ] && shift
[ "$#" -gt 0 ] || { echo "usage: launch.sh LOGFILE -- command [args...]" >&2; exit 2; }
mkdir -p "$(dirname "$log")"
log="$(cd "$(dirname "$log")" && pwd)/$(basename "$log")"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
( cd "$repo" && nohup "$@" > "$log" 2>&1; echo "exit $?" >> "$log" ) > /dev/null 2>&1 &
pid=$!
echo "launched pid $pid -> $log"
sleep "${LAUNCH_CHECK_S:-8}"
if kill -0 "$pid" 2>/dev/null; then
  echo "alive after ${LAUNCH_CHECK_S:-8} s; first lines of the log:"
  head -n 5 "$log" | cut -c1-200
else
  echo "FAILED: the job exited within ${LAUNCH_CHECK_S:-8} s; log:" >&2
  tail -n 20 "$log" >&2
  exit 1
fi
