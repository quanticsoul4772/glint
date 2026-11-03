"""Declare constraints over tool-call traces: the independent grader (agenda T10, appendix #41), the runtime monitor
(appendix #60, reader 1 1b) and the clause-list reconciler (agenda T8, appendix #86).

Source: fix-research 2026-11-02, appendix "Omissions, loops, extras, grading, simulator". Every rule this module
judges comes from a clause list (src/glint/clauses/, CLAUSE_DIR: package data), written from the banking policy documents alone by two
separate blind passes (pass_a.json, pass_b.json) and reconciled (reconciled.json), or from the miner (glint.mine,
"fitted to τ² gold"). Nothing here reads τ²'s gold or a run's outcome; it never decides a write.

A trace is a list of events {"tool": name, "args": dict}: the calls that ran, in order (trace_of builds it from a
saved simulation, gold_trace from τ²'s gold actions). A clause is one Declare template over two activities a and b
(tool names), with argument predicates (a_where, b_where) and correlations (pairs of argument names that must be
equal between the a call and the b call). Templates (Declare's standard semantics, open world: a constraint never
activated is satisfied, reader 1 counter I10):
- existence(a): some a occurs;           absence(a): no a occurs;         at_most(a, n): a occurs at most n times;
- response(a, b): every a is followed by a b;   precedence(a, b): every a is preceded by a b;
- not_succession(a, b): no b after an a;        not_coexistence(a, b): not both an a and a b;
- responded_existence(a, b): every a has a b anywhere;  value(a): every call to tool a satisfies a_where.

A clause with `checkable` false depends on database state or on what the customer said (`condition_text`); the
grader reports it `not_checkable` and never judges it (M12: no conversation text reaches Glint).

Runtime states on a partial trace (Maggi et al., four-valued LTL semantics, as reader 1 1b quotes):
permanently_satisfied, permanently_violated, possibly_satisfied, possibly_violated. `pending` lists the
possibly_violated response and existence obligations, the ones a later call can still satisfy: the monitor's
reminder (glint.gates.enforce, GLINT_DECLARE_REMINDER, default off).
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from importlib import resources
from pathlib import Path

from glint.gates.rules import parse_hidden_arguments
from glint.gates.summary import norm, run_calls

# the blind clause lists and their reconciliation: package data, so an installed (non-editable) Glint has them
CLAUSE_DIR = Path(str(resources.files("glint") / "clauses"))

TEMPLATES = ("existence", "absence", "at_most", "response", "precedence", "not_succession", "not_coexistence",
             "responded_existence", "value")
BINARY = ("response", "precedence", "not_succession", "not_coexistence", "responded_existence")
OPS = ("eq", "ne", "in", "not_in", "lt", "le", "gt", "ge", "present")
KEYS = ("id", "doc", "quote", "template", "a", "b", "a_where", "b_where", "correlate", "n", "condition_text",
        "checkable", "ambiguity", "confidence")
WRAPPER = "call_discoverable_agent_tool"
STATES = ("permanently_satisfied", "permanently_violated", "possibly_satisfied", "possibly_violated")


def validate(clause: dict) -> dict:
    """The clause with defaults filled in; raises ValueError naming the clause and the fault."""
    c = {"a_where": [], "b_where": [], "correlate": [], "n": None, "condition_text": "", "checkable": True,
         "ambiguity": "", "confidence": "medium", "b": None, "doc": "", "quote": ""} | dict(clause)
    cid = c.get("id", "?")
    if c.get("template") not in TEMPLATES:
        raise ValueError(f"clause {cid}: unknown template {c.get('template')!r}")
    if not c.get("a"):
        raise ValueError(f"clause {cid}: no activity a")
    if c["template"] in BINARY and not c.get("b"):
        raise ValueError(f"clause {cid}: template {c['template']} needs activity b")
    if c["template"] == "at_most" and not isinstance(c.get("n"), int):
        raise ValueError(f"clause {cid}: at_most needs an integer n")
    for side in ("a_where", "b_where"):
        for p in c[side] or []:
            if p.get("op") not in OPS or not p.get("field"):
                raise ValueError(f"clause {cid}: bad predicate {p!r} in {side}")
    for pair in c["correlate"] or []:
        if len(pair) != 2:
            raise ValueError(f"clause {cid}: a correlation is a pair of argument names, got {pair!r}")
    if str(c.get("condition_text") or "").strip():
        c["checkable"] = False
    return c


def load(path: str | Path) -> list[dict]:
    """The clauses of a clause-list file ({"clauses": [...]}) or a bare list, each validated."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [validate(c) for c in (data["clauses"] if isinstance(data, dict) else data)]


def _num(v):
    v = norm(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    return v


def holds(pred: dict, args: dict) -> bool:
    """One argument predicate on a call's arguments. Values compare as τ² compares them (numbers as floats, bool
    strings as bools); an ordering against a missing or non-numeric value is false."""
    field, op, want = pred["field"], pred["op"], pred.get("value")
    if op == "present":
        return field in args and args[field] not in (None, "")
    if field not in args:
        return op in ("ne", "not_in")
    have = _num(args[field])
    if op in ("in", "not_in"):
        member = have in [_num(w) for w in (want if isinstance(want, list) else [want])]
        return member if op == "in" else not member
    want = _num(want)
    if op == "eq":
        return have == want
    if op == "ne":
        return have != want
    if not isinstance(have, float) or not isinstance(want, float):
        return False
    return {"lt": have < want, "le": have <= want, "gt": have > want, "ge": have >= want}[op]


def _matches(event: dict, tool: str, where: list[dict]) -> bool:
    return event["tool"] == tool and all(holds(p, event["args"] or {}) for p in where or [])


def _correlated(a_args: dict, b_args: dict, pairs: list) -> bool:
    return all(norm((a_args or {}).get(x)) == norm((b_args or {}).get(y)) for x, y in pairs or [])


def _a_hits(c: dict, trace: list[dict]) -> list[int]:
    where = [] if c["template"] == "value" else c["a_where"]
    return [i for i, e in enumerate(trace) if _matches(e, c["a"], where)]


def _b_hit(c: dict, trace: list[dict], i: int, positions: Iterable[int]) -> bool:
    return any(_matches(trace[j], c["b"], c["b_where"]) and _correlated(trace[i]["args"], trace[j]["args"],
                                                                        c["correlate"]) for j in positions)


def violations(c: dict, trace: list[dict]) -> list[int]:
    """Indexes of the a events that violate the clause on a COMPLETE trace ([-1] for a whole-trace violation:
    existence missing, at_most exceeded). Empty when satisfied or not activated."""
    hits, n = _a_hits(c, trace), len(trace)
    t = c["template"]
    if t == "existence":
        return [] if hits else [-1]
    if t == "absence":
        return hits
    if t == "at_most":
        return [-1] if len(hits) > c["n"] else []
    if t == "value":
        return [i for i in hits if not all(holds(p, trace[i]["args"] or {}) for p in c["a_where"])]
    if t == "response":
        return [i for i in hits if not _b_hit(c, trace, i, range(i + 1, n))]
    if t == "precedence":
        return [i for i in hits if not _b_hit(c, trace, i, range(i))]
    if t == "not_succession":
        return [i for i in hits if _b_hit(c, trace, i, range(i + 1, n))]
    if t == "not_coexistence":
        return [i for i in hits if _b_hit(c, trace, i, (j for j in range(n) if j != i))]
    return [i for i in hits if not _b_hit(c, trace, i, (j for j in range(n) if j != i))]  # responded_existence


def check(c: dict, trace: list[dict]) -> dict:
    """The grader's verdict on one clause over a complete trace: {id, status, activations, violations}. status is
    not_checkable, vacuous (open world: never activated), satisfied or violated."""
    if not c["checkable"]:
        return {"id": c["id"], "status": "not_checkable", "activations": 0, "violations": []}
    acts = len(_a_hits(c, trace)) if c["template"] not in ("existence", "at_most") else 1
    bad = violations(c, trace)
    status = "violated" if bad else ("vacuous" if acts == 0 else "satisfied")
    return {"id": c["id"], "status": status, "activations": acts, "violations": bad}


def state(c: dict, prefix: list[dict]) -> str:
    """The clause's runtime state on a partial trace (one of STATES). Permanent states cannot change with more
    calls; possible ones can."""
    bad = violations(c, prefix)
    t = c["template"]
    if t in ("absence", "at_most", "value", "precedence", "not_succession", "not_coexistence"):
        return "permanently_violated" if bad else "possibly_satisfied"
    if t == "existence":
        return "possibly_violated" if bad else "permanently_satisfied"
    return "possibly_violated" if bad else "possibly_satisfied"  # response, responded_existence


def pending(clauses: list[dict], prefix: list[dict]) -> list[dict]:
    """The checkable clauses in possibly_violated state on a partial trace, each with the a events still owed a b:
    the obligations a later call can still meet."""
    out = []
    for c in clauses:
        if c["checkable"] and state(c, prefix) == "possibly_violated":
            out.append({"id": c["id"], "template": c["template"], "a": c["a"], "b": c["b"], "doc": c["doc"],
                        "owed": [prefix[i]["args"] for i in violations(c, prefix) if i >= 0]})
    return out


def reminder(item: dict) -> str:
    """One open-items line for a pending obligation (Glint's [open] footer form)."""
    if item["template"] == "existence":
        return f"[open] policy {item['doc']} ({item['id']}): {item['a']} is required and not called yet"
    ids = sorted({str(v) for args in item["owed"] for k, v in (args or {}).items() if k.endswith("_id")})
    on = f" for {', '.join(ids)}" if ids else ""
    return f"[open] policy {item['doc']} ({item['id']}): after {item['a']}{on}, {item['b']} is required and not called yet"


def grade(clauses: list[dict], trace: list[dict]) -> dict:
    """Every clause's verdict over one complete trace, plus counts by status."""
    verdicts = [check(c, trace) for c in clauses]
    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    return {"verdicts": verdicts, "counts": counts, "violated": [v["id"] for v in verdicts if v["status"] == "violated"]}


def trace_of(sim: dict) -> list[dict]:
    """The calls of a saved simulation that ran (no tool error, no gate refusal, readable arguments), hidden banking
    tools unwrapped (glint.gates.summary.run_calls)."""
    return [{"tool": c["tool"], "args": c["args"]} for c in run_calls(sim)
            if c["tool"] and c["args"] is not None and not c["error"] and c["deny"] is None]


def gold_trace(actions: list) -> list[dict]:
    """τ²'s gold actions (objects or dicts with name and arguments) as a trace, hidden tools unwrapped."""
    out = []
    for a in actions:
        name = a["name"] if isinstance(a, dict) else a.name
        args = (a.get("arguments") if isinstance(a, dict) else a.arguments) or {}
        if name == WRAPPER:
            name, args = str(args.get("agent_tool_name", "")), parse_hidden_arguments(args.get("arguments", "{}"))
        if args is not None:
            out.append({"tool": name, "args": args})
    return out


# --- reconciling two blind passes (T8, appendix #86) ---

def _where_key(where: list[dict]) -> tuple:
    return tuple(sorted((p["field"], p["op"], json.dumps(p.get("value"), sort_keys=True)) for p in where or []))


def key(c: dict) -> tuple:
    """What two encodings must share to count as the same rule: template, activities, predicates, correlations and n.
    Quotes, ids, documents and notes may differ."""
    return (c["template"], c["a"], c["b"] or "", _where_key(c["a_where"]), _where_key(c["b_where"]),
            tuple(sorted(tuple(p) for p in c["correlate"] or [])), c["n"], bool(c["checkable"]))


def _pair(c: dict) -> tuple:
    return tuple(sorted((c["a"], c["b"] or "")))


def reconcile(pass_a: list[dict], pass_b: list[dict]) -> dict:
    """Two blind clause lists reconciled. `agreed`: rules both passes encoded identically (key). `conflicts`: the same
    activities (in either order) encoded differently by the two passes: an ambiguity finding each (D§5). `a_only`,
    `b_only`: rules only one pass found, whose activities the other pass never paired. Each output clause keeps its
    pass's id; agreed clauses carry both ids."""
    keys_b = {}
    for c in pass_b:
        keys_b.setdefault(key(c), []).append(c)
    agreed, used_b, rest_a = [], set(), []
    for c in pass_a:
        match = next((m for m in keys_b.get(key(c), []) if m["id"] not in used_b), None)
        if match is None:
            rest_a.append(c)
            continue
        used_b.add(match["id"])
        # the key compares structure only: for an uncheckable clause the two condition texts are kept side by side
        agreed.append(c | {"id": f"{c['id']}+{match['id']}", "docs": sorted({c["doc"], match["doc"]}),
                           "status": "agreed", "other": {k: match[k] for k in ("id", "doc", "quote", "condition_text",
                                                                               "ambiguity")}})
    rest_b = [c for c in pass_b if c["id"] not in used_b]
    pairs_a, pairs_b = {}, {}
    for c in rest_a:
        pairs_a.setdefault(_pair(c), []).append(c)
    for c in rest_b:
        pairs_b.setdefault(_pair(c), []).append(c)
    conflicts = [{"activities": list(p), "a": pairs_a[p], "b": pairs_b[p]} for p in sorted(pairs_a.keys() & pairs_b)]
    return {"agreed": agreed, "conflicts": conflicts,
            "a_only": [c | {"status": "a_only"} for p in sorted(pairs_a.keys() - pairs_b) for c in pairs_a[p]],
            "b_only": [c | {"status": "b_only"} for p in sorted(pairs_b.keys() - pairs_a) for c in pairs_b[p]]}


def recall(reference: list[dict], candidate: list[dict]) -> dict:
    """How many reference rules a candidate list encodes identically (key) and how many share only the activities:
    the T8 measure of drafted or mined rules against the blind clause list."""
    keys, pairs = {key(c) for c in candidate}, {_pair(c) for c in candidate}
    exact = [c["id"] for c in reference if key(c) in keys]
    near = [c["id"] for c in reference if key(c) not in keys and _pair(c) in pairs]
    n = len(reference)
    return {"reference": n, "exact": len(exact), "activities_only": len(near), "exact_ids": exact,
            "recall": round(len(exact) / n, 4) if n else None}
