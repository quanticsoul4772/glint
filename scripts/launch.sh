#!/usr/bin/env bash
# Start one long job detached, and only that job: run every check (git pull, config greps) as separate commands
# BEFORE calling this, never chained into a background '&' list.
# A chain like 'cd X && git pull && check && (job) &' backgrounds the whole chain, so the checks' output is lost
# (2026-10-27, twice). This script detaches only the job, prints its PID, and verifies it is alive after a few seconds.
# Usage: scripts/launch.sh LOGFILE -- command [args...]
set -euo pipefail
log=$1; shift
[ "${1:-}" = "--" ] && shift
[ "$#" -gt 0 ] || { echo "usage: launch.sh LOGFILE -- command [args...]" >&2; exit 2; }
mkdir -p "$(dirname "$log")"
( nohup "$@" > "$log" 2>&1; echo "exit $?" >> "$log" ) > /dev/null 2>&1 &
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
