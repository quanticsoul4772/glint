"""Print each run's diff against τ²'s expected writes for a data folder (read-only; nothing is written).

Usage: python -m glint.inspect diff DATA_FOLDER [--task T] [--arm tools]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

# τ² logs its whole registry at DEBUG when imported (through glint.gates); this command prints only its report
logger.remove()
logger.add(sys.stderr, level="WARNING")

from glint import inspect  # after the log setup above
from glint.runlog import FAILED_RUNS, NOT_RESULTS, outcome


def _value(v) -> str:
    return json.dumps(v, default=str)


def _context(w: dict) -> str:
    return f" [{', '.join(f'{k} {v}' for k, v in w['context'].items())}]" if w.get("context") else ""


def _reward(r: dict | None) -> str:
    if r is None:
        return "τ²: no reward"
    parts = [f"τ² reward {r['reward']}", f"db_match {r['db_match']}"]
    for key in ("actions", "env_assertions", "nl_assertions", "communicate_checks"):
        if key in r:
            parts.append(f"{key} {r[key]['met']}/{r[key]['total']}")
    if r.get("note"):
        parts.append(r["note"])
    return ", ".join(parts)


def run_lines(diff: dict) -> list[str]:
    """One run: its summary line, then one indented line per differing, missing, extra and rejected write."""
    c = inspect.compact(diff)
    lines = [f"{diff['task_id']} trial {diff['trial']} {diff['outcome'].upper()}: {c['text']}; {_reward(diff['reward'])}"]
    for w in diff["changed"]:
        target = ", ".join(f"{k} {v}" for k, v in w["target"].items())
        fields = "; ".join(f"{f['field']} gold {_value(f['gold'])} run {_value(f['run'])}" for f in w["fields"])
        lines.append(f"  changed {w['tool']} ({target}){_context(w)}: {fields}")
    for kind in ("missing", "extra"):
        for w in diff[kind]:
            lines.append(f"  {kind} {w['tool']} {_value(w['args'])}{_context(w)}")
    for w in diff["rejected"]:
        lines.append(f"  rejected by {w['by']} {w['tool']}" + (f" (rule {w['rule_id']})" if w["by"] == "gate" else "")
                     + f": {w['reason'][:160]}")
    if diff["reward"] and diff["reward"].get("actions", {}).get("unmet"):
        lines.append(f"  τ² unmet actions: {', '.join(diff['reward']['actions']['unmet'])}")
    return lines


def folder_lines(data: Path, arm: str = "tools", task: str | None = None) -> list[str]:
    """Every run of `arm` in the folder (or of one task): its diff, then the arm's DIFFS line over the failed runs."""
    files = sorted(data.glob(f"{arm}-*.json"))
    if not files:
        raise FileNotFoundError(f"no {arm} arm file ({arm}-<domain>.json) in {data}")
    lines = []
    for f in files:
        domain = f.stem.removeprefix(f"{arm}-")
        sims = [s for s in json.loads(f.read_text(encoding="utf-8"))["simulations"]
                if task is None or str(s["task_id"]) == task]
        failed = []
        lines.append(f"== {arm} {domain}: {len(sims)} runs")
        for s in sorted(sims, key=lambda s: (str(s["task_id"]), int(s.get("trial") or 0))):
            kind = outcome(s)
            if kind in NOT_RESULTS:
                lines.append(f"{s['task_id']} trial {s.get('trial')}: not a result ({kind})")
                continue
            diff = inspect.diff_run(domain, s)
            lines += run_lines(diff)
            if kind in FAILED_RUNS:
                failed.append(inspect.compact(diff))
        lines.append(inspect.diffs_line(failed))
    return lines


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m glint.inspect", description=__doc__.strip().splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    diff = sub.add_parser("diff", help="per-run diffs against τ²'s expected writes")
    diff.add_argument("data", type=Path)
    diff.add_argument("--task")
    diff.add_argument("--arm", default="tools")
    args = parser.parse_args(argv)
    for line in folder_lines(args.data, args.arm, args.task):
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
