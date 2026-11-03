"""Deny-state replay with a deny naming the gold fault (agenda T11a, appendix #88): the harness for asking "would the
agent have recovered if the refusal had named exactly what was wrong?".

The offline part needs no model: `states` finds every gate refusal in a saved simulation and pairs the refused call
with τ²'s gold call on the same tool (the gold call sharing the most `*_id` values; the earliest of equals). The gold
fault is the argument names whose values differ (compared as τ² compares them); `oracle_message` writes a refusal
naming those fields and gold's values. That message is an oracle: it hands over the answer key, so a recovery rate
measured with it is an upper bound on what any refusal wording could reach, never a gate design (M10).

The continuation is the paid part: `resume` sends the conversation up to the refusal, with the oracle refusal in place
of the recorded one, to a pluggable agent call (glint.modelcall: prompt -> the agent's next reply, expected as a JSON
tool call {"name", "arguments"}), and `recovered` compares the next call with gold. With no agent call it raises
ModelOff and calls nothing. Results on the test split are tuned (M10).
"""
from __future__ import annotations

import json

from glint.declare import gold_trace
from glint.gates.summary import norm, run_calls
from glint.modelcall import ModelCall, parse_object, require


def _shared_ids(a: dict, b: dict) -> int:
    return sum(1 for k in a.keys() & b.keys() if k.endswith("_id") and norm(a[k]) == norm(b[k]))


def gold_match(tool: str, args: dict, gold: list[dict]) -> dict | None:
    """The gold call on `tool` sharing the most `*_id` values with `args` (the earliest of equals), or None."""
    best, score = None, -1
    for g in gold:
        if g["tool"] == tool and _shared_ids(args, g["args"]) > score:
            best, score = g, _shared_ids(args, g["args"])
    return best


def fault(args: dict, gold_args: dict) -> dict:
    """Argument name -> [sent, gold] for every argument whose value differs (missing counts as None)."""
    return {k: [args.get(k), gold_args.get(k)] for k in sorted(args.keys() | gold_args.keys())
            if norm(args.get(k)) != norm(gold_args.get(k))}


def oracle_message(tool: str, diff: dict) -> str:
    if not diff:
        return (f"{tool} was NOT executed. The call matches the expected one; the refusal concerned the conversation "
                "state, not the arguments.")
    fields = "; ".join(f"{k} should be {json.dumps(g)} (sent {json.dumps(s)})" for k, (s, g) in diff.items())
    return f"{tool} was NOT executed. Fix these arguments and call it again: {fields}."


def states(sim: dict, gold_actions: list) -> list[dict]:
    """Every gate refusal in a saved simulation: {index, tool, args, rule_id, gold, fault, oracle, no_gold}. `index`
    is the refusal's message index; the conversation before it is sim["messages"][:index]."""
    gold = gold_trace(gold_actions)
    out = []
    for c in run_calls(sim):
        if c["deny"] is None or not c["tool"] or c["args"] is None:
            continue
        g = gold_match(c["tool"], c["args"], gold)
        diff = fault(c["args"], g["args"]) if g else {}
        out.append({"index": c["index"], "tool": c["tool"], "args": c["args"],
                    "rule_id": (c["deny"] or {}).get("rule_id"), "gold": g["args"] if g else None, "fault": diff,
                    "oracle": oracle_message(c["tool"], diff) if g else None, "no_gold": g is None})
    return out


def prompt(messages: list[dict], refusal: str) -> str:
    lines = []
    for m in messages:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            lines.append(f"{m['role']}: {m['content']}")
        for tc in m.get("tool_calls") or []:
            lines.append(f"assistant tool call: {tc.get('name')} {json.dumps(tc.get('arguments'))}")
        if m.get("role") == "tool":
            lines.append(f"tool: {str(m.get('content'))[:2000]}")
    return ("\n".join(lines) + f"\ntool: {refusal}\n\nReply with your next tool call as one JSON object "
            "{\"name\": tool name, \"arguments\": {...}}.")


def resume(sim: dict, state: dict, agent: ModelCall | None) -> dict | None:
    """The agent's next call after the oracle refusal ({tool, args}), or None when the reply holds no call. Hidden
    tools may be named directly or through call_discoverable_agent_tool."""
    call = require(agent, "deny-state replay")
    if state["oracle"] is None:
        raise ValueError("this refusal has no gold call to name; there is no oracle refusal to send")
    reply = parse_object(call(prompt((sim.get("messages") or [])[:state["index"]], state["oracle"])))
    if not reply or not reply.get("name"):
        return None
    trace = gold_trace([{"name": reply["name"], "arguments": reply.get("arguments") or {}}])
    return trace[0] if trace else None


def recovered(next_call: dict | None, state: dict) -> bool:
    """The next call is gold's call: same tool and every gold argument equal."""
    return bool(next_call and state["gold"] is not None and next_call["tool"] == state["tool"]
                and not fault(next_call["args"], state["gold"]))
