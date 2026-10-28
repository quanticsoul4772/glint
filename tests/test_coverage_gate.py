"""scripts/coverage_gate.py: fails, naming the module, when coverage drops below a floor or a goal."""
import json

import coverage_gate


def _report(tmp_path, files: dict) -> str:
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps({"files": {f: {"summary": {"percent_covered": p}} for f, p in files.items()}}),
                    encoding="utf-8")
    return str(path)


def test_the_gate_passes_at_the_goals_and_floors(tmp_path, monkeypatch):
    monkeypatch.setattr(coverage_gate, "FLOORS", {"scripts/tools_ab.py": 82.0})
    assert coverage_gate.main(_report(tmp_path, {"src\\glint\\meter.py": 95.0, "scripts/tools_ab.py": 82.5,
                                                 "src/glint/labels.py": 71.0})) == 0


def test_the_gate_fails_loudly_below_a_floor_or_a_goal(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(coverage_gate, "FLOORS", {"scripts/tools_ab.py": 82.0})
    assert coverage_gate.main(_report(tmp_path, {"src/glint/meter.py": 89.0, "scripts/tools_ab.py": 81.0})) == 1
    err = capsys.readouterr().err
    assert "src/glint/meter.py: 89.0% < floor 90%" in err and "scripts/tools_ab.py: 81.0% < floor 82%" in err


def test_a_floor_for_a_module_that_is_gone_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(coverage_gate, "FLOORS", {"scripts/removed.py": 50.0})
    assert coverage_gate.main(_report(tmp_path, {"src/glint/labels.py": 100.0})) == 1
    assert "scripts/removed.py" in capsys.readouterr().err
