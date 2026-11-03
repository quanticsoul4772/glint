"""The independent Declare grader (agenda T10, appendix #41) and the rule miner (appendix #61, #62), offline.

Subcommands (all read-only, deterministic, no model, no network):
- `gold [--clauses PATH] [--all] [--out PATH]`: every clause over τ²'s banking TRAINING gold traces (the 77 tasks
  gate_replay reads; the test split stays unread). A clause gold violates is either a wrong encoding or a place where
  gold departs from the written policy; each is listed with its tasks. In-sample on training gold. `--out` writes
  {clause id: tasks} (src/glint/clauses/gold_violations.json: the "declare" reminder skips those clauses).
- `runs DATA_FOLDER [...] [--clauses PATH] [--all]`: every clause over each saved run of the banking arm files; one
  line per run with a violation, then per-clause totals. Results on the test split are tuned (M10).
- `mine [--min-support N] [--require-negative] [--out PATH]`: response/precedence rules mined from training gold
  ("fitted to τ² gold"), with the negative-trace filter over data-public/wrong-write-runs.json and recall against the
  clause list; never a gate.
By default only the clauses both blind passes agreed on (status "agreed" in reconciled.json) are judged; `--all` adds
the single-pass ones.

Usage: python scripts/declare_grade.py gold|runs|mine [...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from glint import declare, mine

ROOT = Path(__file__).resolve().parents[1]
CLAUSES = declare.CLAUSE_DIR / "reconciled.json"
NEGATIVES = ROOT / "data-public" / "wrong-write-runs.json"
DOMAIN = "banking_knowledge"


def clauses_of(path: Path, include_all: bool = False) -> list[dict]:
    """The judged clauses of a reconciled list (agreed, plus a_only and b_only with include_all) or of a plain list."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "agreed" not in data:
        return declare.load(path)
    picked = data["agreed"] + (data["a_only"] + data["b_only"] if include_all else [])
    return [declare.validate(c) for c in picked]


def gold_traces() -> dict[str, list[dict]]:
    """Task id -> gold trace, for the banking training tasks (gate_replay.tasks_of)."""
    import gate_replay
    return {t.id: declare.gold_trace(t.evaluation_criteria.actions or []) for t in gate_replay.tasks_of(DOMAIN)
            if t.evaluation_criteria is not None}


def gold_violations(clauses: list[dict], traces: dict[str, list[dict]]) -> dict[str, list[str]]:
    """Clause id -> the training tasks whose gold trace violates it."""
    bad: dict[str, list[str]] = {}
    for tid, trace in sorted(traces.items()):
        for cid in declare.grade(clauses, trace)["violated"]:
            bad.setdefault(cid, []).append(tid)
    return dict(sorted(bad.items()))


def grade_gold(clauses: list[dict], traces: dict[str, list[dict]]) -> list[str]:
    bad = gold_violations(clauses, traces)
    checkable = sum(1 for c in clauses if c["checkable"])
    head = (f"gold (training, in-sample): {len(traces)} tasks, {len(clauses)} clauses ({checkable} checkable), "
            f"{len(bad)} clauses violated by gold")
    lines = [head]
    return lines + [f"  {cid}: {len(t)} tasks: {', '.join(t[:8])}" for cid, t in bad.items()]


def grade_runs(clauses: list[dict], folders: list[Path]) -> list[str]:
    lines = []
    for f in sorted(p for folder in folders for p in folder.glob(f"*-{DOMAIN}.json")):
        sims = json.loads(f.read_text(encoding="utf-8")).get("simulations") or []
        per_clause: dict[str, int] = {}
        lines.append(f"{f.parent.name}/{f.name}: {len(sims)} runs")
        for s in sorted(sims, key=lambda s: (str(s.get("task_id")), int(s.get("trial") or 0))):
            violated = declare.grade(clauses, declare.trace_of(s))["violated"]
            for cid in violated:
                per_clause[cid] = per_clause.get(cid, 0) + 1
            if violated:
                lines.append(f"  {s.get('task_id')} t{s.get('trial')}: {', '.join(violated)}")
        lines.append("  per clause: " + (", ".join(f"{k} {v}" for k, v in sorted(per_clause.items())) or "none"))
    return lines


def run_mine(traces: dict[str, list[dict]], clauses: list[dict], min_support: int, require_negative: bool,
             negatives_path: Path | None = None) -> dict:
    gold = list(traces.values())
    rows = json.loads((negatives_path or NEGATIVES).read_text(encoding="utf-8"))
    negatives, vocabulary = mine.negative_traces(rows)
    rules = mine.negative_filter(mine.mine(gold, min_support=min_support), negatives, require_negative, vocabulary)
    auto = mine.Automaton(gold, vocabulary)
    return {"source": "fitted to τ² gold (training tasks)", "rules": rules,
            "negative_traces": len(negatives),
            "negatives_with_unseen_steps": sum(1 for t in negatives if auto.unseen_steps(t)),
            "recall_of_clause_list": declare.recall(clauses, rules)}


def _opt(argv: list[str], name: str, default: str | None = None) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else default


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("gold", "runs", "mine"):
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    clauses = clauses_of(Path(_opt(rest, "--clauses", str(CLAUSES))), "--all" in rest)
    if cmd == "runs":
        values = {_opt(rest, o) for o in ("--clauses", "--min-support", "--out")}
        folders = [Path(a) for a in rest if not a.startswith("--") and a not in values]
        if not folders:
            print(__doc__)
            return 2
        print("\n".join(grade_runs(clauses, folders)))
        return 0
    traces = gold_traces()
    if cmd == "gold":
        print("\n".join(grade_gold(clauses, traces)))
        if _opt(rest, "--out"):
            Path(_opt(rest, "--out")).write_text(json.dumps(gold_violations(clauses, traces), indent=1) + "\n",
                                                 encoding="utf-8")
        return 0
    out = run_mine(traces, clauses, int(_opt(rest, "--min-support", "2")), "--require-negative" in rest)
    text = json.dumps(out, indent=1)
    if _opt(rest, "--out"):
        Path(_opt(rest, "--out")).write_text(text, encoding="utf-8")
    r = out["recall_of_clause_list"]
    broken = sum(1 for x in out["rules"] if x["negative_violations"])
    print(f"mined {len(out['rules'])} rules from {len(traces)} training gold traces (fitted to τ² gold); "
          f"{broken} broken by some of {out['negative_traces']} "
          f"negative traces; clause-list recall {r['exact']}/{r['reference']} exact, {r['activities_only']} "
          "activities only")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
