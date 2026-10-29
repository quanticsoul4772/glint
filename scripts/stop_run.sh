#!/usr/bin/env bash
# Stop a running tools_ab set safely: kill its job's whole process tree, stop the Glimmer app on Modal (with --yes:
# without a TTY the stop waits for a confirmation that never comes, 2026-10-29), and wait until no container is left.
# A stopped arm writes no arm file, so a later resume starts that arm again (finished runs stay in τ²'s checkpoint).
# Usage: scripts/stop_run.sh SET
# MODAL names the Modal CLI (default: modal on PATH); MODAL_CONFIG_PATH is passed through. STOP_WAIT_S bounds the wait.
set -uo pipefail
set_name=${1:?usage: stop_run.sh SET}
MODAL=${MODAL:-modal}
wait_s=${STOP_WAIT_S:-300}
app=glint-serve
# the job's processes: every command line naming "tools_ab.py SET" (nohup, python, and their children)
if command -v powershell >/dev/null 2>&1; then
  pids=$(powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'tools_ab\.py $set_name( |\$)' -and \$_.CommandLine -notmatch 'stop_run|Get-CimInstance' } | ForEach-Object { \$_.ProcessId }" | tr -d '\r')
else
  pids=$(pgrep -f "tools_ab\.py $set_name( |\$)" | grep -v "^$$\$" || true)
fi
if [ -z "$pids" ]; then
  echo "no job running for $set_name"
else
  for pid in $pids; do
    if command -v taskkill >/dev/null 2>&1; then taskkill //PID "$pid" //T //F >/dev/null 2>&1; else kill -9 "$pid" 2>/dev/null; fi
  done
  echo "killed the job's process tree for $set_name: $(echo $pids | tr '\n' ' ')"
fi
# the job's own exit path stops the app; a killed job does not, so stop it here and wait
"$MODAL" app stop "$app" --yes >/dev/null 2>&1 || echo "WARNING: modal app stop $app exited non-zero (it may already be stopped)"
deadline=$((SECONDS + wait_s))
while :; do
  # Modal's table truncates the name ("glint-se"): matching the full name found no row and read as stopped
  row=$("$MODAL" app list 2>/dev/null | grep -E "(^|[[:space:]])${app:0:8}" | head -n 1)
  # a row's task count is the container count; a stopped app shows 0 (or is gone)
  running=$(echo "$row" | awk '{for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+$/) {print $i; exit}}')
  if [ -z "$row" ] || [ "${running:-0}" = "0" ]; then
    echo "$app: 0 containers running"
    exit 0
  fi
  if [ $SECONDS -ge $deadline ]; then
    echo "FAILED: $app still has $running container(s) after ${wait_s}s: stop it from the Modal dashboard" >&2
    exit 1
  fi
  sleep 10
done
