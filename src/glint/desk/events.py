"""Bank Desk events: a τ² conversation as a timeline the page shows (shared by replay export and live mode)."""
from __future__ import annotations

import json

from glint.decisions import GLINT_TOOLS

WRITES = {"file_credit_card_transaction_dispute_4829", "order_replacement_credit_card_7291",
          "submit_credit_limit_increase_request_7392", "approve_credit_limit_increase_5847",
          "deny_credit_limit_increase_5848"}


def _short(value, n=400) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= n else text[:n] + " …"


def timeline(messages: list[dict], overrules: dict[int, dict] | None = None) -> list[dict]:
    """Events in conversation order from τ² messages (dicts). `overrules`: assistant message index -> decision
    (tool, reasons) for writes a Glint tool would have overruled."""
    overrules = overrules or {}
    results = {m.get("id"): m.get("content") for m in messages if m.get("role") == "tool"}
    events = []
    for i, m in enumerate(messages):
        if m.get("role") == "user" and m.get("content"):
            events.append({"kind": "customer", "text": m["content"]})
        if m.get("role") != "assistant":
            continue
        if m.get("content"):
            events.append({"kind": "agent", "text": m["content"]})
        for tc in m.get("tool_calls") or []:
            name, args = tc["name"], tc.get("arguments") or {}
            result = results.get(tc.get("id"))
            if name == "unlock_discoverable_agent_tool":
                continue
            if name in GLINT_TOOLS:
                events.append({"kind": "verdict", "tool": name, "args": args,
                               "result": None if result is None else _short(result, 1500)})
                continue
            if name == "call_discoverable_agent_tool":
                inner = args.get("arguments")
                try:
                    name, args = args.get("agent_tool_name", ""), json.loads(inner) if isinstance(inner, str) else inner
                except json.JSONDecodeError:
                    name, args = args.get("agent_tool_name", ""), {"arguments": inner}
            event = {"kind": "action" if name in WRITES else "lookup", "tool": name, "args": args,
                     "result": None if result is None else _short(result),
                     "error": str(result or "").startswith("Error")}
            if i in overrules and overrules[i]["tool"] == name:
                event["overruled"] = overrules[i]["reasons"]
            events.append(event)
    return events


def outcome(sim: dict) -> dict:
    r = sim.get("reward_info") or {}
    return {"passed": r.get("reward") == 1.0, "database_matches_gold": (r.get("db_check") or {}).get("db_match")}


def customer_summary(instructions: str) -> str:
    """The customer's situation and goal from the simulator brief, without markdown (the brief also holds persona
    details and hidden facts the agent must ask for, which the page should not show)."""
    import re

    parts = []
    for label in ("Your situation", "Your goal"):
        # the section runs to the next blank line or the next **Label:** heading
        m = re.search(rf"\*\*{label}:\*\*\s*(.+?)(?=\n\s*\n|\*\*[A-Z][^*]+:\*\*|$)", instructions, re.DOTALL)
        if m:
            parts.append(" ".join(m.group(1).replace("**", "").split()))
    text = " ".join(parts) or " ".join(instructions.replace("**", "").split())
    return _short(text, 500)
