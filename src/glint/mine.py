"""Mined follow-up rules from gold traces, with the negative-trace filter (appendix #61; reader 1 1c, 5b; design §4),
and the safe set as an automaton over call sequences (appendix #62; reader 2 L).

Source: "fitted to τ² gold". Every rule here is mined from τ²'s training gold actions, so it cannot fail on that gold
(reader 1 counter R7) and overfits uniform gold (#42). The output is a review list, never a gate: mined rules are
written in the clause-list schema (glint.declare) with `source` "mined", for comparison against the blind policy
clause list (glint.declare.recall: rules the miner found that the policy lacks go to review; policy rules never mined
show trace-coverage gaps).

Mining (Declare discovery, response and precedence templates): for every ordered pair of tools (x, y) that both occur
in at least `min_support` gold traces, a candidate response(x, y) holds when every x in every gold trace is followed
by a y, and precedence(x, y) when every x is preceded by a y. A correlation on an `*_id` argument is added when every
x and its matching y share that argument's value. Confidence is the share of x events satisfied; only rules at or
above `min_confidence` are kept.

Negative-trace filter (reader 1 5b: "keep a candidate rule if all gold writes satisfy it and extra writes break it"):
given negative traces (runs with labeled off-policy writes, data-public/wrong-write-runs.json), each rule records how
many negative traces violate it; `require_negative` keeps only rules some negative trace breaks.

Automaton (#62): the prefix-closed set of tool-name sequences gold shows, as a deterministic automaton over tool
names, generalized to a "follows" relation: a call is accepted when gold ever shows its tool directly after the
previous call's tool (a first-order Markov acceptor). `unseen_steps` lists the steps of a trace gold never shows: a
flag for review, never a refusal (an automaton from 77 tasks refuses any order gold did not happen to use).
"""
from __future__ import annotations

from collections import Counter

from glint.declare import violations
from glint.gates.summary import norm

START = "<start>"


def _clause(template: str, x: str, y: str, correlate: list, support: int, confidence: float, n: int) -> dict:
    return {"id": f"M-{template}-{x}-{y}", "doc": "", "quote": "", "template": template, "a": x, "b": y,
            "a_where": [], "b_where": [], "correlate": correlate, "n": None, "condition_text": "", "checkable": True,
            "ambiguity": "", "confidence": "high" if confidence == 1.0 else "low", "source": "mined",
            "support": support, "rule_confidence": round(confidence, 4), "activations": n}


def _shared_ids(traces: list[list[dict]], x: str, y: str, after: bool) -> list:
    """The `*_id` arguments every x shares with its nearest y (after it for response, before it for precedence)."""
    common = None
    for t in traces:
        for i, e in enumerate(t):
            if e["tool"] != x:
                continue
            rng = range(i + 1, len(t)) if after else range(i - 1, -1, -1)
            j = next((j for j in rng if t[j]["tool"] == y), None)
            if j is None:
                continue
            same = {k for k, v in (e["args"] or {}).items() if k.endswith("_id")
                    and norm((t[j]["args"] or {}).get(k)) == norm(v)}
            common = same if common is None else common & same
    return [[k, k] for k in sorted(common or ())]


def mine(gold: list[list[dict]], min_support: int = 2, min_confidence: float = 1.0) -> list[dict]:
    """Response and precedence rules over tool pairs from gold traces (module doc)."""
    support = Counter(tool for t in gold for tool in {e["tool"] for e in t})
    tools = sorted(t for t, s in support.items() if s >= min_support)
    rules = []
    for x in tools:
        for y in tools:
            if x == y:
                continue
            together = sum(1 for t in gold if any(e["tool"] == x for e in t) and any(e["tool"] == y for e in t))
            if together < min_support:
                continue
            for template, after in (("response", True), ("precedence", False)):
                n = ok = 0
                for t in gold:
                    for i, e in enumerate(t):
                        if e["tool"] != x:
                            continue
                        n += 1
                        rng = range(i + 1, len(t)) if after else range(i)
                        ok += any(t[j]["tool"] == y for j in rng)
                conf = ok / n if n else 0.0
                if n and conf >= min_confidence:
                    rules.append(_clause(template, x, y, _shared_ids(gold, x, y, after), together, conf, n))
    return rules


def negative_filter(rules: list[dict], negatives: list[list[dict]], require_negative: bool = False,
                    vocabulary: set[str] | None = None) -> list[dict]:
    """Each rule with `negative_violations`: how many negative traces break it. A rule naming a tool outside
    `vocabulary` (the tools the negative traces record at all) cannot be judged on them and gets None: a negative
    trace that records only writes would break every "after a read" rule vacuously. With require_negative, only
    rules some negative trace breaks are kept (reader 1 5b)."""
    out = []
    for r in rules:
        judged = vocabulary is None or {r["a"], r["b"]} <= vocabulary
        hits = sum(1 for t in negatives if violations(r, t)) if judged else None
        if hits or not require_negative:
            out.append(r | {"negative_violations": hits})
    return out


def negative_traces(rows: list[dict]) -> tuple[list[list[dict]], set[str]]:
    """Traces from data-public/wrong-write-runs.json rows, and the tools they record: every assistant write that ran,
    the labeled off-policy ones (`st` wrongarg, dup, extra) included, in order. These rows record writes and hidden
    calls only, not unlocks or plain reads."""
    out = []
    for row in rows:
        out.append([{"tool": w["name"], "args": w.get("args") or {}} for w in row.get("writes") or []
                    if w.get("role") == "assistant"])
    return out, {e["tool"] for t in out for e in t}


class Automaton:
    """The tool-to-tool transitions gold shows (module doc, #62)."""

    def __init__(self, gold: list[list[dict]], vocabulary: set[str] | None = None):
        """`vocabulary`, when given, projects every trace onto those tools first (a trace source that records only
        writes is compared with gold's writes only)."""
        self.vocabulary = vocabulary
        self.edges: set[tuple[str, str]] = set()
        for t in gold:
            prev = START
            for _, e in self._steps(t):
                self.edges.add((prev, e["tool"]))
                prev = e["tool"]

    def _steps(self, trace: list[dict]) -> list[tuple[int, dict]]:
        return [(i, e) for i, e in enumerate(trace) if self.vocabulary is None or e["tool"] in self.vocabulary]

    def unseen_steps(self, trace: list[dict]) -> list[dict]:
        """The steps of a trace whose transition gold never shows: {index, from, to}."""
        out, prev = [], START
        for i, e in self._steps(trace):
            if (prev, e["tool"]) not in self.edges:
                out.append({"index": i, "from": prev, "to": e["tool"]})
            prev = e["tool"]
        return out

    def accepts(self, trace: list[dict]) -> bool:
        return not self.unseen_steps(trace)
