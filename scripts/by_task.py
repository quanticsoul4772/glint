"""Passes by task for one or more finished tools_ab sets, side by side, with tuned and untuned tasks kept apart.

Runs of one task are not independent, so results are read by task, and tasks whose failures shaped a tool or a gate
(design data) are never pooled with the rest. Each column is one arm of one set (`<set folder>:<arm>`); a cell is
passes / result runs, with runs that are not results (infrastructure errors, evaluation errors, cap stops) counted
apart. Reads the arm files the run wrote (`<arm>-<domain>.json`); no model, no network.

Usage: python scripts/by_task.py DATA_DIR [DATA_DIR ...] [--domain banking_knowledge] [--tuned task_037,task_038,...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from glint.runlog import NOT_RESULTS, outcome

# banking12 tasks that are design data (research/00-decisions.md): 038, 054 (the banking-failures set), 039-041
# (dispute planning), 037 and 081 (the filing order)
TUNED = ("task_037", "task_038", "task_039", "task_040", "task_041", "task_054", "task_081")


def column(folder: Path, arm_file: Path) -> str:
    return f"{folder.name.removeprefix('tools_ab_').removesuffix('_modal')}:{arm_file.name.split('-', 1)[0]}"


def table(folders: list[Path], domain: str) -> dict[str, dict[str, list[int]]]:
    """task -> column -> [passes, result runs, runs that are not results]."""
    rows: dict[str, dict[str, list[int]]] = {}
    for folder in folders:
        for arm_file in sorted(folder.glob(f"*-{domain}.json")):
            col = column(folder, arm_file)
            for sim in json.loads(arm_file.read_text(encoding="utf-8"))["simulations"]:
                kind = outcome(sim)
                cell = rows.setdefault(str(sim["task_id"]), {}).setdefault(col, [0, 0, 0])
                if kind in NOT_RESULTS:
                    cell[2] += 1
                else:
                    cell[0] += kind == "pass"
                    cell[1] += 1
    return rows


def render(rows: dict[str, dict[str, list[int]]], tuned: tuple[str, ...]) -> str:
    cols = sorted({c for r in rows.values() for c in r})
    lines = ["task       tuned  " + "  ".join(f"{c:>26}" for c in cols)]
    totals = {g: {c: [0, 0] for c in cols} for g in ("untuned", "tuned")}
    for task in sorted(rows):
        group = "tuned" if task in tuned else "untuned"
        cells = []
        for c in cols:
            p, n, x = rows[task].get(c, [0, 0, 0])
            totals[group][c][0] += p
            totals[group][c][1] += n
            cells.append(f"{p}/{n}" + (f" ({x} not results)" if x else ""))
        lines.append(f"{task:10} {'yes' if group == 'tuned' else '':5}  " + "  ".join(f"{c:>26}" for c in cells))
    for group in ("untuned", "tuned"):
        lines.append(f"{group.upper():17}" + "  ".join(f"{f'{p}/{n}':>26}" for p, n in totals[group].values()))
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    domain = argv[argv.index("--domain") + 1] if "--domain" in argv else "banking_knowledge"
    tuned = tuple(argv[argv.index("--tuned") + 1].split(",")) if "--tuned" in argv else TUNED
    skip = {argv.index(f) + 1 for f in ("--domain", "--tuned") if f in argv}
    folders = [Path(a) for i, a in enumerate(argv) if not a.startswith("--") and i not in skip]
    if not folders:
        print(__doc__)
        return 2
    rows = table(folders, domain)
    if not rows:
        print(f"no {domain} arm files in {[str(f) for f in folders]}")
        return 1
    print(render(rows, tuned))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
