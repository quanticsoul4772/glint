#!/usr/bin/env bash
# Block until a launch.sh job's log shows the job ended, then print the ending lines and exit. Run it in the
# background so the caller is woken once, at the end, instead of re-arming a 30-minute watch (2026-11-01: seven
# re-arms in one run). Failures that end the job (a Traceback, a cap, a nonzero exit) end the wait too.
# Usage: scripts/wait_run.sh LOG
set -uo pipefail
log=${1:?usage: wait_run.sh LOG}
end='^exit |Traceback|GPU CAP|BudgetExceeded|SystemExit|glint-serve stopped'
until grep -Eq "$end" "$log" 2>/dev/null; do sleep 30; done
sleep 5  # the final lines are written together
grep -E "$end|\] done:|this arm" "$log" | tail -n 12
