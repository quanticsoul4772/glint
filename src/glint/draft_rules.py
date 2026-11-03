"""Model-drafted rules and their recall against the blind clause list (agenda T8, appendix #44 and #86; design §4-§5):
the harness for "recall of Muse-drafted rules".

A drafting model reads one policy document at a time and returns clauses in the clause-list schema (glint.declare);
each is validated (an invalid clause is counted and dropped, never repaired) and tagged `source` "drafted". The
drafted list is then measured against the reconciled blind list with glint.declare.recall. The model call is
pluggable (glint.modelcall) and off by default: with none, `draft` raises ModelOff and calls nothing. Drafting with
Muse Spark needs Meta API credit, which is nearly spent: a run needs the owner's yes. The T8 lead found drafted
rules reach no failure class of the test split, so this measures coverage only.
"""
from __future__ import annotations

import json
from pathlib import Path

from glint import declare
from glint.modelcall import ModelCall, parse_object, require

SCHEMA = ("Return one JSON object {\"clauses\": [...]}. Each clause has: template (one of "
          f"{list(declare.TEMPLATES)}), a and b (tool names; b null for unary templates), a_where and b_where "
          f"(lists of {{field, op, value}} with op one of {list(declare.OPS)}), correlate (pairs of argument names "
          "equal between the a and b calls), n (integer for at_most, else null), condition_text (what the clause "
          "depends on that tool arguments cannot show, or \"\"), quote (the sentence it encodes).")


def prompt(doc_id: str, title: str, content: str, tools: list[str]) -> str:
    return (f"Policy document {doc_id}: {title}\n\n{content}\n\nThe agent's tools: {', '.join(tools)}.\n"
            "List every rule in this document that constrains when or how the agent may call one of these tools. "
            + SCHEMA)


def draft(doc: dict, tools: list[str], model: ModelCall | None) -> dict:
    """Clauses drafted from one document ({id, title, content}): {doc, clauses, invalid}."""
    call = require(model, "rule drafting")
    reply = parse_object(call(prompt(doc["id"], doc.get("title", ""), doc.get("content", ""), tools))) or {}
    good, invalid = [], 0
    for k, c in enumerate(reply.get("clauses") or [], 1):
        try:
            v = declare.validate({"id": f"D-{doc['id']}-{k:02d}", "doc": doc["id"], **(c if isinstance(c, dict)
                                                                                        else {})})
        except (ValueError, TypeError):
            invalid += 1
            continue
        if v["a"] not in tools or (v["b"] is not None and v["b"] not in tools):
            invalid += 1
            continue
        good.append(v | {"source": "drafted"})
    return {"doc": doc["id"], "clauses": good, "invalid": invalid}


def draft_all(docs_dir: str | Path, tools: list[str], model: ModelCall | None, only: list[str] | None = None) -> dict:
    """Drafts over every document of a folder (or the ids in `only`), in file-name order."""
    require(model, "rule drafting")
    out = {"clauses": [], "invalid": 0, "documents": 0}
    for path in sorted(Path(docs_dir).glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc.setdefault("id", path.stem)
        if only and doc["id"] not in only and path.stem not in only:
            continue
        r = draft(doc, tools, model)
        out["clauses"] += r["clauses"]
        out["invalid"] += r["invalid"]
        out["documents"] += 1
    return out
