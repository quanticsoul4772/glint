"""The gates never read gold: nothing under src/glint/gates imports glint.inspect (which reads each task's expected
actions), directly or through another module."""
import ast
import os
import subprocess
import sys
from pathlib import Path

GATES = Path(__file__).resolve().parents[1] / "src" / "glint" / "gates"


def _imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


def test_no_gate_module_imports_glint_inspect():
    files = sorted(GATES.glob("*.py"))
    assert files
    offenders = {f.name: sorted(n for n in _imports(f) if n == "glint.inspect" or n.startswith("glint.inspect."))
                 for f in files}
    assert {k: v for k, v in offenders.items() if v} == {}


def test_importing_every_gate_module_does_not_load_glint_inspect():
    modules = ", ".join(f"glint.gates.{f.stem}" for f in sorted(GATES.glob("*.py")) if f.stem != "__init__")
    code = f"import sys, glint.gates, {modules}; print('glint.inspect' in sys.modules)"
    src = str(GATES.parents[1])
    env = os.environ | {"PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    done = subprocess.run([sys.executable, "-X", "utf8", "-c", code], check=False, capture_output=True, text=True,
                          env=env, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == "False"
