#!/usr/bin/env bash
# The three CI checks (pytest with coverage, the coverage gate, ruff) for each git ref given, each in a throwaway
# worktree of its own, so the checkout you work in and the run folder are never touched. Prints one line per ref with
# each step's exit code; exits 1 when any step of any ref fails.
# Usage: scripts/check.sh REF [REF ...]      (a branch, a tag or a commit)
# PY and RUFF name the tools (default: python and ruff on PATH; on this machine, glint-dev's .venv has pytest-cov).
set -uo pipefail
[ "$#" -gt 0 ] || { echo "usage: check.sh REF [REF ...]" >&2; exit 2; }
PY=${PY:-python}
RUFF=${RUFF:-ruff}
repo=$(git rev-parse --show-toplevel)
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
  echo "$ref: pytest exit $p ($(tail -n 1 "$wt/pytest.log")); coverage exit $c ($(tail -n 1 "$wt/cov.log")); ruff exit $r"
  if [ $p -ne 0 ]; then grep -E "^FAILED" "$wt/pytest.log" | head -n 10; fi
  if [ $r -ne 0 ]; then head -n 20 "$wt/ruff.log"; fi
  [ $((p + c + r)) -eq 0 ] || status=1
  git -C "$repo" worktree remove --force "$wt"
done
exit $status
