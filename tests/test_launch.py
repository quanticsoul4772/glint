"""scripts/launch.sh: the job runs from the repo root whatever the caller's directory, and the log lands where asked."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")


@pytest.mark.skipif(BASH is None, reason="bash is not on PATH")
def test_the_job_runs_from_the_repo_root_and_logs_to_a_path_relative_to_the_caller(tmp_path):
    caller = tmp_path / "elsewhere"
    caller.mkdir()
    r = subprocess.run([BASH, str(ROOT / "scripts" / "launch.sh"), "logs/job.log", "--", "bash", "-c",
                        "pwd; git rev-parse --is-inside-work-tree"],
                       # this job ends at once, so launch.sh's alive check reports it and exits 1: judged by the log
                       check=False, cwd=caller, capture_output=True, text=True, timeout=60,
                       env={"LAUNCH_CHECK_S": "3", "PATH": os.environ["PATH"]})
    log = caller / "logs" / "job.log"
    assert log.exists(), r.stdout + r.stderr
    lines = log.read_text(encoding="utf-8").splitlines()
    # the job's working directory is the repo root (bash prints it in its own path form), inside the git work tree
    assert lines[-1] == "exit 0" and "true" in lines
    assert lines[0].rstrip("/").rsplit("/", 1)[-1] == ROOT.name
