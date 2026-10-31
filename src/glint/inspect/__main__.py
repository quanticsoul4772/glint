"""Print each run's diff against τ²'s expected writes for a data folder (read-only; nothing is written).

Usage: python -m glint.inspect diff DATA_FOLDER [--task T] [--arm tools]
       python -m glint.inspect sequence DATA_FOLDER --task T [--trial N] [--arm tools]
       python -m glint.inspect messages DATA_FOLDER --task T --grep REGEX [--trial N] [--arm tools]
       python -m glint.inspect refusals DATA_FOLDER [--arm tools]
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
                lines.append(inspect.facts_line(inspect.run_facts(s)))
                continue
            diff = inspect.diff_run(domain, s)
            lines += run_lines(diff)
            lines.append(inspect.facts_line(inspect.run_facts(s)))
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
    seq = sub.add_parser("sequence", help="each run's dispute, replacement and increase steps in order")
    msgs = sub.add_parser("messages", help="the messages of runs matching a pattern")
    for p in (seq, msgs):
        p.add_argument("data", type=Path)
        p.add_argument("--task", required=True)
        p.add_argument("--trial", type=int)
        p.add_argument("--arm", default="tools")
    msgs.add_argument("--grep", required=True, help="a regex, case-insensitive")
    ref = sub.add_parser("refusals", help="each gate refusal against gold: false, held for a precondition, or differs")
    ref.add_argument("data", type=Path)
    ref.add_argument("--arm", default="tools")
    args = parser.parse_args(argv)
    if args.command == "refusals":
        lines = refusal_lines(args.data, args.arm)
    elif args.command == "diff":
        lines = folder_lines(args.data, args.arm, args.task)
    else:
        lines = run_lines_of(args.data, args.arm, args.task, args.trial,
                             inspect.run_sequence if args.command == "sequence"
                             else (lambda s: inspect.run_messages(s, args.grep)))
    for line in lines:
        print(line, flush=True)
    return 0


def refusal_lines(data: Path, arm: str = "tools") -> list[str]:
    """Every gate refusal of the folder's arm, classified against gold (inspect.live_refusals), per rule, with each
    live false refusal and plan-caused refusal named; the live counterpart of the gold replays' false-refusal
    count."""
    files = sorted(data.glob(f"{arm}-*.json"))
    if not files:
        raise FileNotFoundError(f"no {arm} arm file ({arm}-<domain>.json) in {data}")
    lines = []
    for f in files:
        domain = f.stem.removeprefix(f"{arm}-")
        per_rule: dict[str, dict[str, int]] = {}
        named, total, runs = [], dict.fromkeys(("refused", *inspect.KINDS), 0), 0
        for s in json.loads(f.read_text(encoding="utf-8"))["simulations"]:
            if outcome(s) in NOT_RESULTS:
                continue
            runs += 1
            r = inspect.live_refusals(domain, s)
            total["refused"] += r["refused"]
            for kind in inspect.KINDS:
                total[kind] += len(r[kind])
                for row in r[kind]:
                    per_rule.setdefault(row["rule_id"], {}).setdefault(kind, 0)
                    per_rule[row["rule_id"]][kind] += 1
                    if kind in ("live_false_refusals", "plan_differs_from_gold"):
                        named.append(f"{kind.replace('_', ' ').upper()}: {s['task_id']} trial {s.get('trial')}: "
                                     f"{row['tool']} refused by {row['rule_id']} (message {row['index']})")
        # only a write gold also makes can be judged false; a gated tool that is not a write is never judged
        judged = total["refused"] - total["differs_from_gold"] - total["gated_tool"]
        lines.append(f"== {arm} {domain}: {runs} runs; {total['refused']} gate refusals, {judged} judged (the refused "
                     "write matches gold): " + ", ".join(f"{total[k]} {k.replace('_', ' ')}" for k in inspect.KINDS))
        for rule, kinds in sorted(per_rule.items(), key=lambda kv: -sum(kv[1].values())):
            lines.append(f"  {rule}: " + ", ".join(f"{k.replace('_', ' ')} {n}" for k, n in sorted(kinds.items())))
        lines += [f"  {n}" for n in named]
    return lines


def run_lines_of(data: Path, arm: str, task: str, trial: int | None, per_run) -> list[str]:
    """`per_run(sim)`'s lines for each run of one task (or one trial) in the folder's arm files, under a header."""
    files = sorted(data.glob(f"{arm}-*.json"))
    if not files:
        raise FileNotFoundError(f"no {arm} arm file ({arm}-<domain>.json) in {data}")
    lines = []
    for f in files:
        sims = [s for s in json.loads(f.read_text(encoding="utf-8"))["simulations"]
                if str(s["task_id"]) == task and (trial is None or int(s.get("trial") or 0) == trial)]
        for s in sorted(sims, key=lambda s: int(s.get("trial") or 0)):
            lines.append(f"== {s['task_id']} trial {s.get('trial')} ({outcome(s)})")
            lines += per_run(s)
    return lines


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
