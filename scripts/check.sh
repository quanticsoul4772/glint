#!/usr/bin/env bash
# The three CI checks (pytest with coverage, the coverage gate, ruff) and the two gold replays (gate_replay: τ²'s gold
# actions through the gates, 0 false refusals; bank_replay: the same through glint-bank's MCP path, 0 false refusals,
# 0 database hash mismatches, the recorded wrong writes stopped) for each git ref given, each in a throwaway worktree
# of its own, so the checkout you work in and the run folder are never touched. Prints one line per ref with each
# step's exit code; exits 1 when any step of any ref fails.
# Usage: scripts/check.sh REF [REF ...]      (a branch, a tag or a commit)
# PY and RUFF name the tools; by default the checkout's own .venv (glint-dev's has pytest-cov), else PATH.
set -uo pipefail
[ "$#" -gt 0 ] || { echo "usage: check.sh REF [REF ...]" >&2; exit 2; }
repo=$(git rev-parse --show-toplevel)
venv="$repo/.venv/Scripts"
[ -d "$venv" ] || venv="$repo/.venv/bin"
if [ -x "$venv/python.exe" ]; then py_default="$venv/python.exe"; elif [ -x "$venv/python" ]; then py_default="$venv/python"; else py_default=python; fi
if [ -x "$venv/ruff.exe" ]; then ruff_default="$venv/ruff.exe"; elif [ -x "$venv/ruff" ]; then ruff_default="$venv/ruff"; else ruff_default=ruff; fi
PY=${PY:-$py_default}
RUFF=${RUFF:-$ruff_default}
# the coverage step needs pytest-cov; without it every ref would fail on --cov (the run folder's .venv lacks it)
"$PY" -c "import pytest_cov" 2>/dev/null || { echo "check.sh: $PY has no pytest-cov; set PY to a python that has it (glint-dev's .venv)" >&2; exit 2; }
status=0
for ref in "$@"; do
  # next to the repo, not in the system temp folder: tests find the vendored τ² checkout relative to the repo's
  # parent (../vendor/tau2-bench), which a worktree in /tmp does not have (2026-10-29)
  wt=$(mktemp -d "$(dirname "$repo")/glint-check-XXXXXX")
  if ! git -C "$repo" worktree add -q --detach "$wt" "$ref" 2>/dev/null; then
    echo "$ref: FAILED to check out"; status=1; rmdir "$wt" 2>/dev/null; continue
  fi
  (cd "$wt" && PYTHONPATH=src $PY -X utf8 -m pytest -q -p no:cacheprovider --cov --cov-report=json:coverage.json \
    > pytest.log 2>&1); p=$?
  (cd "$wt" && $PY scripts/coverage_gate.py > cov.log 2>&1); c=$?
  (cd "$wt" && $RUFF check src tests scripts --line-length 120 > ruff.log 2>&1); r=$?
  (cd "$wt" && PYTHONPATH=src $PY -X utf8 scripts/gate_replay.py > gate_replay.log 2>&1); g=$?
  (cd "$wt" && PYTHONPATH=src $PY -X utf8 scripts/bank_replay.py > bank_replay.log 2>&1); b=$?
  echo "$ref: pytest exit $p ($(tail -n 1 "$wt/pytest.log")); coverage exit $c ($(tail -n 1 "$wt/cov.log")); ruff exit $r"
  echo "$ref: gate_replay exit $g ($(grep -E "^TOTAL" "$wt/gate_replay.log")); bank_replay exit $b"
  grep -E "^(gold|wrong writes):" "$wt/bank_replay.log" | sed "s/^/  /"
  if [ $p -ne 0 ]; then grep -E "^FAILED" "$wt/pytest.log" | head -n 10; fi
  if [ $r -ne 0 ]; then head -n 20 "$wt/ruff.log"; fi
  if [ $g -ne 0 ]; then grep -E "^  (FALSE|false)|Traceback|Error" "$wt/gate_replay.log" | head -n 10; tail -n 5 "$wt/gate_replay.log"; fi
  if [ $b -ne 0 ]; then grep -E "BAD|refusal|mismatch|Traceback|Error" "$wt/bank_replay.log" | head -n 10; fi
  [ $((p + c + r + g + b)) -eq 0 ] || status=1
  git -C "$repo" worktree remove --force "$wt"
done
exit $status
