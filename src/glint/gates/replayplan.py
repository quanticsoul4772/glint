"""The plan_disputes call an agent following the plan gate's Fix makes, for replays of conversations recorded without
it: τ²'s gold actions (gate_replay, bank_replay) and Glimmer's runs recorded before the tool existed (gate_catch).

`with_plans(calls)` inserts, before each user's first credit card dispute filing, one plan_disputes call carrying
that user's filings in the list, in their order (the first filing of each transaction; a retry is not listed twice).
The plan orders them by its own rule (glint.assist.banking.plan_disputes), so a replay checks whether the filings
follow the plan the conversation's own disputes give. Deterministic.
"""
from __future__ import annotations

from tau2.data_model.message import ToolCall

from glint.assist.banking import PLAN_TOOL
from glint.gates.rules import DISPUTE, parse_hidden_arguments

WRAPPER = "call_discoverable_agent_tool"
PLAN_FIELDS = ("transaction_id", "dispute_reason", "contacted_merchant")


def filing(call: ToolCall) -> dict | None:
    """The arguments of an assistant's credit card dispute filing (by name or through the hidden-tool wrapper)."""
    if call.requestor != "assistant":
        return None
    args = call.arguments or {}
    if call.name == WRAPPER and args.get("agent_tool_name") == DISPUTE:
        return parse_hidden_arguments(args.get("arguments", "{}"))
    return dict(args) if call.name == DISPUTE else None


def plan_arguments(filings: list[dict]) -> dict[str, dict]:
    """user id -> the plan_disputes arguments for that user's filings, in order (first filing of each transaction)."""
    plans: dict[str, dict] = {}
    for f in filings:
        user = str(f.get("user_id", ""))
        items = plans.setdefault(user, {"user_id": user, "transactions": []})["transactions"]
        if all(i["transaction_id"] != f.get("transaction_id") for i in items):
            items.append({k: f.get(k) for k in PLAN_FIELDS if f.get(k) is not None})
    return plans


def with_plans(calls: list[ToolCall]) -> list[ToolCall]:
    """`calls` with a plan_disputes call before each user's first dispute filing (ids `<filing id>-plan`)."""
    plans = plan_arguments([f for c in calls if (f := filing(c))])
    out, planned = [], set()
    for c in calls:
        f = filing(c)
        user = str(f.get("user_id", "")) if f else None
        if f is not None and user not in planned:
            planned.add(user)
            out.append(ToolCall(id=f"{c.id}-plan", name=PLAN_TOOL, arguments=plans[user], requestor="assistant"))
        out.append(c)
    return out
