"""Per-module coverage gate, run after `pytest --cov --cov-report=json:coverage.json`.

The goal (2026-10-27):
- Total line coverage of the library and the measurement pipeline: at least 85% (pyproject `fail_under`).
- The code that spends money or produces the published numbers (CRITICAL): 90%.
- Every other module: at least 70%.

FLOORS hold a module's measured coverage, rounded down, while it is below its goal. A floor may only be raised, never
lowered: raise it when tests add coverage, and delete the entry when the module reaches its goal. The gate fails,
naming each module and its number, when any module falls below its floor, or a module at its goal falls below it.

Usage: python scripts/coverage_gate.py [coverage.json]
"""
import json
import sys
from pathlib import Path

CRITICAL_GOAL, OTHER_GOAL = 90.0, 70.0
CRITICAL = {"src/glint/meter.py", "src/glint/runlog.py", "src/glint/calllog.py", "src/glint/glimmer_serve.py",
            "src/glint/desk/live.py", "scripts/tools_ab.py", "scripts/export_results.py",
            "scripts/export_scenarios.py", "scripts/gate_labels.py"}
FLOORS: dict[str, float] = {}  # every module met its goal on 2026-10-27; a module that drops below its goal
# fails the gate (add a floor here only for new code that starts below its goal)


def main(path: str = "coverage.json") -> int:
    files = json.loads(Path(path).read_text(encoding="utf-8"))["files"]
    failed, below_goal = [], []
    for name, data in sorted(files.items()):
        module = name.replace("\\", "/")
        pct = data["summary"]["percent_covered"]
        goal = CRITICAL_GOAL if module in CRITICAL else OTHER_GOAL
        floor = FLOORS.get(module, goal)
        if pct < floor:
            failed.append(f"  {module}: {pct:.1f}% < floor {floor:.0f}%")
        elif pct < goal:
            below_goal.append(f"  {module}: {pct:.1f}% (floor {floor:.0f}%, goal {goal:.0f}%)")
    stale = [m for m in FLOORS if m not in {n.replace("\\", "/") for n in files}]
    if below_goal:
        print("below the goal, held at their floors:\n" + "\n".join(below_goal))
    if stale:
        failed.append(f"  FLOORS names modules the report does not have: {stale}")
    if failed:
        print("COVERAGE GATE FAILED:\n" + "\n".join(failed), file=sys.stderr)
        return 1
    print(f"coverage gate passed ({len(files)} modules)")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
