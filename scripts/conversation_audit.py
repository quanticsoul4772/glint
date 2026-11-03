"""The conversation-text checks (glint.convcheck) over a finished set's arm files: one line per run with a finding,
then the arm's counts. Read-only; reads no gold; calls no model. Counts on the test split are tuned.

Usage: python scripts/conversation_audit.py DATA_FOLDER [DATA_FOLDER ...] [--domain banking_knowledge]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from glint import convcheck

CHECKS = ("possession_conflicts", "unsupplied_values", "uncited_records", "dispute_without_filing")


def run_findings(sim: dict) -> dict:
    return convcheck.audit(sim)


def report(path: Path) -> list[str]:
    sims = json.loads(path.read_text(encoding="utf-8")).get("simulations") or []
    counts = dict.fromkeys(CHECKS, 0)
    lines = []
    for s in sorted(sims, key=lambda s: (str(s.get("task_id")), int(s.get("trial") or 0))):
        f = run_findings(s)
        hit = [k for k in CHECKS if f[k]]
        for k in hit:
            counts[k] += 1
        if hit:
            lines.append(f"  {s.get('task_id')} t{s.get('trial')}: " + "; ".join(
                f"{k} {len(f[k]) if isinstance(f[k], list) else 1}" for k in hit))
    return [f"{path.parent.name}/{path.name}: {len(sims)} runs, runs with a finding per check: {counts}", *lines]


def main(argv: list[str]) -> int:
    domain = argv[argv.index("--domain") + 1] if "--domain" in argv else "banking_knowledge"
    skip = {argv.index("--domain") + 1} if "--domain" in argv else set()
    folders = [Path(a) for i, a in enumerate(argv) if not a.startswith("--") and i not in skip]
    if not folders:
        print(__doc__)
        return 2
    files = sorted(f for folder in folders for f in folder.glob(f"*-{domain}.json"))
    if not files:
        print(f"no {domain} arm files in {[str(f) for f in folders]}")
        return 1
    for f in files:
        print("\n".join(report(f)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
