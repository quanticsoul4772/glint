"""The plan_disputes call an agent following the plan gate's Fix makes, for replays of conversations recorded without
it: τ²'s gold actions (gate_replay, bank_replay) and Glimmer's runs recorded before the tool existed (gate_catch).

`with_plans(calls)` inserts, before each user's first credit card dispute filing, one plan_disputes call carrying
that user's filings in the list, in their order (the first filing of each transaction; a retry is not listed twice).
The plan orders them by its own rule (glint.assist.banking.plan_disputes), so a replay checks whether the filings
follow the plan the conversation's own disputes give. When the conversation has not read the dispute history before
that filing (τ²'s gold in 031, 037, 038), the read the filing requires (glint.gates.procedure.REQUIRED_LOOKUPS) goes
in first, unlocked and called as an agent does; τ² does not log a read its gold lacks, so the scored database is
unchanged. `with_address_asks` inserts the order an agent sends before confirming the shipping address, which the
address_ask gate refuses once, before gold's first order per card to the address on file (039, 054). A replay's 0
false refusals is on gold plus these inserted steps. Deterministic.
"""
from __future__ import annotations

import json
from collections.abc import Callable

from tau2.data_model.message import ToolCall

from glint.assist.banking import DISPUTE_HISTORY_TOOL, PLAN_TOOL
from glint.gates.rules import DISPUTE, REPLACE, parse_hidden_arguments, same_address

WRAPPER = "call_discoverable_agent_tool"
# a filing's partial_refund_amount is the overcharge the plan takes for an incorrect_amount dispute
PLAN_FIELDS = ("transaction_id", "dispute_reason", "contacted_merchant", "partial_refund_amount", "issue_noticed_date",
               "card_action")


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


def _history_read(call: ToolCall) -> bool:
    args = call.arguments or {}
    return call.requestor == "assistant" and call.name == WRAPPER and args.get("agent_tool_name") == DISPUTE_HISTORY_TOOL


def with_plans(calls: list[ToolCall]) -> list[ToolCall]:
    """`calls` with a plan_disputes call before each user's first dispute filing (ids `<filing id>-plan`), and the
    dispute-history read before it when none came earlier (ids `<filing id>-history-unlock`, `<filing id>-history`)."""
    plans = plan_arguments([f for c in calls if (f := filing(c))])
    out, planned, read = [], set(), False
    for c in calls:
        f = filing(c)
        user = str(f.get("user_id", "")) if f else None
        if f is not None and user not in planned:
            planned.add(user)
            if not read:
                out.append(ToolCall(id=f"{c.id}-history-unlock", name="unlock_discoverable_agent_tool",
                                    arguments={"agent_tool_name": DISPUTE_HISTORY_TOOL}, requestor="assistant"))
                out.append(ToolCall(id=f"{c.id}-history", name=WRAPPER, requestor="assistant", arguments={
                    "agent_tool_name": DISPUTE_HISTORY_TOOL, "arguments": json.dumps({"user_id": user})}))
                read = True
            out.append(ToolCall(id=f"{c.id}-plan", name=PLAN_TOOL, arguments=plans[user], requestor="assistant"))
        read = read or _history_read(c)
        out.append(c)
    return out


ADDRESS_ASK_SUFFIX = "-address-ask"


def with_address_asks(calls: list[ToolCall], address_of: Callable[[str], str | None]) -> list[ToolCall]:
    """`calls` with a copy of each card's first replacement order to the customer's address on file (`address_of(user
    id)`) just before it (id `<order id>-address-ask`): the order an agent sends before asking, which the address_ask
    gate refuses once (glint.gates.rules.ADDRESS_ASK). Replays expect that copy refused by that rule and send it to no
    ungated twin."""
    out, asked = [], set()
    for c in calls:
        args = c.arguments or {}
        order = (parse_hidden_arguments(args.get("arguments", "{}")) if c.name == WRAPPER
                 and args.get("agent_tool_name") == REPLACE else dict(args) if c.name == REPLACE else None)
        if c.requestor == "assistant" and order:
            card, on_file = order.get("credit_card_account_id"), address_of(str(order.get("user_id", "")))
            if card not in asked and on_file and same_address(order.get("shipping_address"), on_file):
                out.append(c.model_copy(update={"id": f"{c.id}{ADDRESS_ASK_SUFFIX}"}))
            asked.add(card)
        out.append(c)
    return out
