"""scripts/check.sh and scripts/stop_run.sh with stub tools: no real test run, no Modal, no running job touched."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash is not on PATH")


def _stub(path: Path, body: str) -> str:
    path.write_text("#!/usr/bin/env bash\n" + body + "\n", encoding="utf-8", newline="\n")
    path.chmod(0o755)
    return path.as_posix()


def _run(script: str, args: list[str], env: dict, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run([BASH, str(ROOT / "scripts" / script), *args], cwd=cwd, capture_output=True, text=True,
                          timeout=120, env={**os.environ, **env}, check=False)


def test_check_reports_each_step_and_removes_its_worktree(tmp_path):
    ok = _stub(tmp_path / "py", 'echo "3 passed"; exit 0')
    lint = _stub(tmp_path / "ruff", 'echo "All checks passed!"; exit 0')
    before = subprocess.run(["git", "worktree", "list"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    r = _run("check.sh", ["HEAD"], {"PY": ok, "RUFF": lint})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "HEAD: pytest exit 0 (3 passed); coverage exit 0 (3 passed); ruff exit 0" in r.stdout
    after = subprocess.run(["git", "worktree", "list"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert after == before  # the throwaway worktree is gone


def test_check_fails_when_a_step_fails_and_on_an_unknown_ref(tmp_path):
    bad = _stub(tmp_path / "py", 'echo "FAILED tests/test_x.py::test_y"; echo "1 failed"; exit 1')
    lint = _stub(tmp_path / "ruff", "exit 0")
    r = _run("check.sh", ["HEAD"], {"PY": bad, "RUFF": lint})
    assert r.returncode == 1 and "pytest exit 1 (1 failed)" in r.stdout and "FAILED tests/test_x.py::test_y" in r.stdout
    r = _run("check.sh", ["no-such-ref-xyz"], {"PY": bad, "RUFF": lint})
    assert r.returncode == 1 and "no-such-ref-xyz: FAILED to check out" in r.stdout
    assert _run("check.sh", [], {}).returncode == 2


def test_stop_run_stops_the_app_with_yes_and_waits_for_zero_containers(tmp_path):
    calls = tmp_path / "calls.txt"
    modal = _stub(tmp_path / "modal", f'echo "$*" >> "{calls.as_posix()}"\n'
                  'if [ "$1 $2" = "app list" ]; then echo "  ap-1  glint-se  stopped  0  2026-10-  2026-10"; fi')
    r = _run("stop_run.sh", ["no-such-set-xyz"], {"MODAL": modal, "STOP_WAIT_S": "5"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "no job running for no-such-set-xyz" in r.stdout and "glint-serve: 0 containers running" in r.stdout
    assert "app stop glint-serve --yes" in calls.read_text(encoding="utf-8")


def test_stop_run_fails_loud_when_containers_stay(tmp_path):
    modal = _stub(tmp_path / "modal", 'if [ "$1 $2" = "app list" ]; then echo "  ap-1  glint-se  stopping  1  x  y"; fi')
    r = _run("stop_run.sh", ["no-such-set-xyz"], {"MODAL": modal, "STOP_WAIT_S": "0"})
    assert r.returncode == 1 and "FAILED: glint-serve still has 1 container(s)" in r.stderr
