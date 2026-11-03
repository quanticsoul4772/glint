"""Importing glint.declare before glint.gates works (regression: glint.declare imports glint.gates.rules, whose
package loads glint.gates.reminders, which read declare.WRAPPER at import time and raised a circular-import
AttributeError). Each check runs in a fresh interpreter, so the import order of this test session cannot hide it."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str]) -> subprocess.CompletedProcess:
    env = os.environ | {"PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    return subprocess.run([sys.executable, "-X", "utf8", *args], check=False, capture_output=True, text=True,
                          env=env, cwd=ROOT, timeout=300)


def test_glint_declare_imports_first_in_a_fresh_interpreter():
    done = _run(["-c", "import glint.declare, glint.gates.reminders as r; print(glint.declare.WRAPPER)"])
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == "call_discoverable_agent_tool"


def test_declare_grade_script_starts():
    done = _run([str(ROOT / "scripts" / "declare_grade.py"), "--help"])
    assert "Traceback" not in done.stderr, done.stderr[-2000:]
    assert done.returncode == 2 and "Usage: python scripts/declare_grade.py" in done.stdout


def test_llm_harness_tests_pass_when_run_alone(tmp_path):
    # its own --basetemp: the default (.pytest_tmp, pyproject addopts) is this session's, and pytest clears it at start
    done = _run(["-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-cov", f"--basetemp={tmp_path / 'child'}",
                 str(ROOT / "tests" / "test_llm_harness.py")])
    assert done.returncode == 0, (done.stdout + done.stderr)[-3000:]
