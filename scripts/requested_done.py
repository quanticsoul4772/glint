"""The offline requested-vs-done grader (glint.grade) over a finished set's arm files: one line per run with a
finding, then the arm's totals. Read-only; reads no gold. Each run is graded with its task's initial database
(request_recount.task_db: τ²'s environment for the task, set to its initial state).

Usage: python scripts/requested_done.py DATA_FOLDER [DATA_FOLDER ...] [--domain banking_knowledge] [--quiet]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from request_recount import task_db

from glint import grade


def arm_files(folder: Path, domain: str) -> list[Path]:
    return sorted(folder.glob(f"*-{domain}.json"))


def run_line(g: dict) -> str | None:
    parts = []
    for key in ("requested_not_done", "done_not_requested", "unrecorded", "required_not_done"):
        if g[key]:
            items = [f"{x.get('tool') or x.get('rule')} on {x.get('target_id') or x.get('args')}" for x in g[key]]
            parts.append(f"{key} {len(items)}: " + "; ".join(str(i)[:160] for i in items))
    return f"  {g['task_id']} t{g['trial']}: " + " | ".join(parts) if parts else None


def report(path: Path, quiet: bool = False, db_of=task_db) -> list[str]:
    sims = json.loads(path.read_text(encoding="utf-8")).get("simulations") or []
    grades = [grade.grade(s, db_of(str(s.get("task_id")))) for s in sorted(sims, key=lambda s: (str(s.get("task_id")), int(s.get("trial") or 0)))]
    lines = [f"{path.parent.name}/{path.name}: {grade.totals(grades)}"]
    if not quiet:
        lines += [x for x in map(run_line, grades) if x]
    return lines


def main(argv: list[str], db_of=task_db) -> int:
    domain = argv[argv.index("--domain") + 1] if "--domain" in argv else "banking_knowledge"
    skip = {argv.index("--domain") + 1} if "--domain" in argv else set()
    folders = [Path(a) for i, a in enumerate(argv) if not a.startswith("--") and i not in skip]
    if not folders:
        print(__doc__)
        return 2
    files = [f for folder in folders for f in arm_files(folder, domain)]
    if not files:
        print(f"no {domain} arm files in {[str(f) for f in folders]}")
        return 1
    for f in files:
        print("\n".join(report(f, "--quiet" in argv, db_of)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
