#!/usr/bin/env bash
# Run a Python command against this checkout's own src with the shared venv, from any worktree.
# Usage: scripts/wt.sh pytest -q tests/test_desk.py      (runs python -X utf8 -m pytest ...)
#        scripts/wt.sh scripts/export_results.py           (runs a script)
# The venv is the run folder's (PY overrides it); PYTHONPATH is this checkout's src, so the worktree's code is what
# runs. GLINT_HOME defaults to the checkouts' parent folder, as in glint.paths.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-/d/Projects/meta/glint/.venv/Scripts/python.exe}
[ -x "$PY" ] || PY=python
src=$(cd "$here/src" && pwd -W 2>/dev/null || pwd)
export PYTHONPATH="$src"
cd "$here"
case "${1:-}" in
  "") echo "usage: wt.sh pytest ARGS | wt.sh SCRIPT ARGS" >&2; exit 2 ;;
  pytest) shift; exec "$PY" -X utf8 -m pytest "$@" ;;
  *) exec "$PY" -X utf8 "$@" ;;
esac
