"""The archived dead code stays out of the package and the scripts, and no shell script carries a local path (judge
panel 2026-11-03, technical report: conformal, verifiers and the M0-M2 scripts; scripts/wt.sh:9's /d/Projects path)."""
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARCHIVED = {"glint": ("conformal", "verifiers"), "scripts": ("dryrun_m0", "m1_c2_rerun", "m1_deepinfra_checks",
                                                           "m2_judge")}


def test_the_archived_modules_are_in_archive_and_nothing_imports_them():
    for folder, names in ARCHIVED.items():
        for name in names:
            assert (ROOT / "archive" / folder / f"{name}.py").exists()
            assert not (ROOT / ("src/glint" if folder == "glint" else "scripts") / f"{name}.py").exists()
    assert importlib.util.find_spec("glint.conformal") is None and importlib.util.find_spec("glint.verifiers") is None
    names = "|".join(n for ns in ARCHIVED.values() for n in ns)
    pattern = re.compile(rf"^\s*(from|import)\s+(glint\.)?({names})\b|^\s*from glint import .*\b({names})\b",
                         re.MULTILINE)
    users = [str(p.relative_to(ROOT)) for d in ("src", "scripts", "tests") for p in (ROOT / d).rglob("*.py")
             if pattern.search(p.read_text(encoding="utf-8"))]
    assert users == []


def test_no_shell_script_names_a_local_drive_path():
    for script in (ROOT / "scripts").glob("*.sh"):
        assert not re.search(r"/[a-z]/Projects/", script.read_text(encoding="utf-8")), script.name


# on Windows, "bash" from Python can resolve to WSL's, which cannot read the checkout's CRLF scripts; CI runs it on Linux
@pytest.mark.skipif(shutil.which("bash") is None or sys.platform == "win32", reason="needs a POSIX bash")
def test_wt_sh_finds_the_venv_of_the_main_checkout_from_the_repo_location():
    """With PY unset, wt.sh uses the .venv of the checkout that holds the shared git directory, else python."""
    common = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            check=True, capture_output=True, text=True).stdout.strip()
    main = Path(common).parent
    venv = [p for p in (main / ".venv" / "Scripts" / "python.exe", main / ".venv" / "bin" / "python") if p.exists()]
    env = {k: v for k, v in os.environ.items() if k != "PY"}
    r = subprocess.run(["bash", "scripts/wt.sh", "-c", "import sys; print(sys.executable)"], cwd=ROOT,
                       check=False, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    used = Path(r.stdout.strip().splitlines()[-1]).resolve()
    if venv:
        assert used.parent.parent.resolve() == venv[0].parent.parent.resolve()
    else:
        assert used.exists()  # PATH's python
