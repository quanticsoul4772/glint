"""The LLM-stage harnesses over saved banking runs, every model call pluggable and off by default (glint.modelcall).
No paid service is called unless `--model module:function` names a function that calls one; this repo ships none.

Subcommands:
- `deny-states DATA_FOLDER [...]`: every gate refusal with τ²'s gold call on the same tool and the gold fault
  (glint.denystate, T11a); with `--model` (an agent call), also the next call after an oracle refusal and whether it
  recovered. Without --model it is offline and free.
- `extract DATA_FOLDER [...] --model M [--facts f1,f2] [--labels LABELS.jsonl]`: single-fact extraction per run
  (glint.extract, T7), scored against hand labels when given.
- `second-sim DATA_FOLDER [...] --model M`: the second customer model's turns against the recorded ones
  (glint.second_sim, #80).
- `draft --model M [--docs DIR] [--clauses RECONCILED] [--only id1,id2]`: model-drafted rules from the policy
  documents and their recall against the blind clause list (glint.draft_rules, T8).
Results on the banking test split are tuned (M10).

Usage: python scripts/llm_harness.py SUBCOMMAND [...] [--model module:function] [--out OUT.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from glint import declare, denystate, draft_rules, extract, paths, second_sim
from glint.modelcall import ModelOff, resolve

DOMAIN = "banking_knowledge"
BANKING = paths.tau2_data() / "tau2" / "domains" / DOMAIN
CLAUSES = declare.CLAUSE_DIR / "reconciled.json"
OPTIONS = ("--model", "--facts", "--labels", "--docs", "--clauses", "--only", "--out", "--tasks")


def _opt(argv: list[str], name: str, default: str | None = None) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else default


def _folders(argv: list[str]) -> list[Path]:
    values = {i + 1 for i, a in enumerate(argv) if a in OPTIONS}
    return [Path(a) for i, a in enumerate(argv) if not a.startswith("--") and i not in values]


def _sims(folders: list[Path]) -> list[dict]:
    out = []
    for f in sorted(p for folder in folders for p in folder.glob(f"*-{DOMAIN}.json")):
        out += [s | {"arm": f.stem} for s in json.loads(f.read_text(encoding="utf-8")).get("simulations") or []]
    return sorted(out, key=lambda s: (str(s.get("task_id")), int(s.get("trial") or 0)))


def _tasks(path: Path) -> dict[str, dict]:
    return {t["id"]: t for t in json.loads(path.read_text(encoding="utf-8"))}


def _tools() -> list[str]:
    from tau2.domains.banking_knowledge.tools import KnowledgeTools
    return sorted(n for n in dir(KnowledgeTools) if not n.startswith("_") and callable(getattr(KnowledgeTools, n)))


def deny_states(sims: list[dict], tasks: dict[str, dict], model) -> dict:
    rows = []
    for s in sims:
        gold = ((tasks.get(str(s.get("task_id"))) or {}).get("evaluation_criteria") or {}).get("actions") or []
        for st in denystate.states(s, gold):
            row = {"arm": s.get("arm"), "task_id": s.get("task_id"), "trial": s.get("trial")} | st
            if model is not None and st["oracle"] is not None:
                row["recovered"] = denystate.recovered(denystate.resume(s, st, model), st)
            rows.append(row)
    judged = [r for r in rows if "recovered" in r]
    return {"refusals": len(rows), "with_gold": sum(1 for r in rows if not r["no_gold"]),
            "argument_faults": sum(1 for r in rows if r["fault"]),
            "recovered": sum(1 for r in judged if r["recovered"]) if judged else None, "rows": rows}


def run_extract(sims: list[dict], facts: list[str], model, labels: list[dict] | None) -> dict:
    preds = [{"task_id": s.get("task_id"), "trial": s.get("trial")} | extract.extract(s, f, model)
             for s in sims for f in facts]
    return {"predictions": preds, "score": extract.score(preds, labels) if labels is not None else None}


def run_second_sim(sims: list[dict], tasks: dict[str, dict], model) -> dict:
    rows = {f"{s.get('arm')} {s.get('task_id')} t{s.get('trial')}": second_sim.cross_check(
        s, second_sim.instructions_of(tasks[str(s.get("task_id"))]), model) for s in sims
        if str(s.get("task_id")) in tasks}
    return {"summary": second_sim.summary(rows), "rows": rows}


def run_draft(docs: Path, clauses: Path, model, only: list[str] | None) -> dict:
    drafted = draft_rules.draft_all(docs, _tools(), model, only)
    data = json.loads(clauses.read_text(encoding="utf-8"))
    reference = [declare.validate(c) for c in data["agreed"]]
    return drafted | {"recall_of_clause_list": declare.recall(reference, drafted["clauses"])}


def main(argv: list[str]) -> int:
    cmds = ("deny-states", "extract", "second-sim", "draft")
    if not argv or argv[0] not in cmds:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    model = resolve(_opt(rest, "--model"))
    tasks_path = Path(_opt(rest, "--tasks", str(BANKING / "tasks.json")))
    try:
        if cmd == "deny-states":
            out = deny_states(_sims(_folders(rest)), _tasks(tasks_path), model)
            line = (f"{out['refusals']} refusals, {out['with_gold']} with a gold call on the same tool, "
                    f"{out['argument_faults']} with an argument fault; recovered: {out['recovered']}")
        elif cmd == "extract":
            facts = (_opt(rest, "--facts") or "card_in_possession,pin_compromised").split(",")
            labels = extract.load_jsonl(_opt(rest, "--labels")) if _opt(rest, "--labels") else None
            out = run_extract(_sims(_folders(rest)), facts, model, labels)
            line = f"{len(out['predictions'])} extractions; score: {out['score']}"
        elif cmd == "second-sim":
            out = run_second_sim(_sims(_folders(rest)), _tasks(tasks_path), model)
            line = f"second simulator: {out['summary']}"
        else:
            only = _opt(rest, "--only")
            out = run_draft(Path(_opt(rest, "--docs", str(BANKING / "documents"))),
                            Path(_opt(rest, "--clauses", str(CLAUSES))), model, only.split(",") if only else None)
            r = out["recall_of_clause_list"]
            line = (f"drafted {len(out['clauses'])} clauses from {out['documents']} documents ({out['invalid']} "
                    f"invalid); recall {r['exact']}/{r['reference']} exact")
    except ModelOff as e:
        print(f"off: {e}")
        return 2
    if _opt(rest, "--out"):
        Path(_opt(rest, "--out")).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
