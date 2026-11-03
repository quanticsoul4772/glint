"""An installed (non-editable) Glint finds its data inside the package (judge panel 2026-11-03, technical B15).

Before the fix the reminders read data-public/clauses/reconciled.json three folders above the package, so an installed
copy crashed with FileNotFoundError at the first Enforcer. CI also builds the wheel, installs it into a fresh venv and
runs the bank server's selftest and an Enforcer from it (.github/workflows/tests.yml, job installed-wheel)."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from glint.gates import reminders

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "glint"


def test_the_clause_lists_are_package_data_and_the_source_tree_reads_them():
    assert reminders.declare.CLAUSE_DIR == PACKAGE / "clauses"
    assert reminders.CLAUSES == PACKAGE / "clauses" / "reconciled.json" and reminders.CLAUSES.exists()
    assert (PACKAGE / "clauses" / "gold_violations.json").exists()
    assert not (ROOT / "data-public" / "clauses").exists()  # one copy only, inside the package
    assert reminders.declare_reminder_clauses()  # the source tree (PYTHONPATH=src) still loads them


def test_a_copy_of_the_package_with_no_source_tree_around_it_builds_an_enforcer(tmp_path):
    """The judge's reproduction: src/glint copied to a folder with no data-public above it, then an Enforcer."""
    site = tmp_path / "site"
    shutil.copytree(PACKAGE, site / "glint", ignore=shutil.ignore_patterns("__pycache__"))
    code = ("import json, glint; from glint.gates import Enforcer, EventLog; "
            "e = Enforcer('banking_knowledge', EventLog(':memory:')); "
            "print(json.dumps({'file': glint.__file__, 'clauses': len(e.declare_clauses)}))")
    env = os.environ | {"PYTHONPATH": str(site), "GLINT_HOME": str(tmp_path)}
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", code], check=False, cwd=tmp_path, env=env,
                       capture_output=True, text=True, encoding="utf-8", timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert Path(out["file"]).resolve().is_relative_to(site.resolve())
    assert out["clauses"] == len(reminders.declare_reminder_clauses()) > 0
