"""glint.paths finds its workspace in the repo layout and in a shallow deployed layout."""
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_repo_layout_defaults_to_the_folder_that_holds_the_repo(monkeypatch):
    monkeypatch.delenv("GLINT_HOME", raising=False)
    from glint import paths

    importlib.reload(paths)
    here = Path(paths.__file__).resolve()
    assert paths.HOME == here.parents[3]


def test_shallow_layout_does_not_fail(tmp_path):
    # Modal puts the package at /root/glint/paths.py: there is no fourth parent (2026-10-27 deploy crash)
    pkg = tmp_path / "glint"
    pkg.mkdir()
    shutil.copy(Path(__file__).resolve().parents[1] / "src" / "glint" / "paths.py", pkg / "paths.py")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "GLINT_HOME"} | {"PYTHONPATH": str(tmp_path)}
    out = subprocess.run([sys.executable, "-c", "import glint.paths as p; print(p.HOME)"], cwd=tmp_path,
                         capture_output=True, text=True, env=env, check=False)
    assert out.returncode == 0, out.stderr
    assert Path(out.stdout.strip()) == tmp_path
