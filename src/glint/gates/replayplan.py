"""The plan_disputes call an agent following the plan gate's Fix makes, for replays of conversations recorded without
it: τ²'s gold actions (gate_replay, bank_replay) and Glimmer's runs recorded before the tool existed (gate_catch).

`with_plans(calls)` inserts, before each user's first credit card dispute filing, one plan_disputes call carrying
that user's filings in the list, in their order (the first filing of each transaction; a retry is not listed twice).
The plan orders them by its own rule (glint.assist.banking.plan_disputes), so a replay checks whether the filings
follow the plan the conversation's own disputes give. When the conversation has not read the dispute history before
that filing (τ²'s gold in 031, 037, 038), the read the filing requires (glint.gates.procedure.REQUIRED_LOOKUPS) goes
in first, unlocked and called as an agent does; τ² does not log a read its gold lacks, so the scored database is
unchanged. `with_address_asks` inserts the order an agent sends before confirming the shipping address, which the
address_ask gate refuses once, before gold's first order per card to the address on file (039, 054).
`with_address_confirmations` instead inserts the record_address_confirmation call an agent makes after asking, before
each card's first order; the gold replays (gate_replay, bank_replay) use it since 2026-11-03. `with_plans` also inserts
the record_dispute_answers call the debit dispute answers gate requires, carrying gold's own answers. A replay's 0
false refusals is on gold plus these inserted steps. Deterministic.
"""
from __future__ import annotations

import json
from collections.abc import Callable

from tau2.data_model.message import ToolCall

from glint.assist.banking import (
    ADDRESS_TOOL,
    DISPUTE_ANSWERS_TOOL,
    DISPUTE_HISTORY_TOOL,
    PLAN_TOOL,
    REQUEST_TOOLS,
    REQUESTS_TOOL,
    RETENTION_TOOL,
)
from glint.gates.debit import DISPUTE as DEBIT_DISPUTE
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
        items = plans.setdefault(user, {"user_id": user, "transactions": [], "other_requests": []})["transactions"]
        if all(i["transaction_id"] != f.get("transaction_id") for i in items):
            items.append({k: f.get(k) for k in PLAN_FIELDS if f.get(k) is not None})
    return plans


def _history_read(call: ToolCall) -> bool:
    args = call.arguments or {}
    return call.requestor == "assistant" and call.name == WRAPPER and args.get("agent_tool_name") == DISPUTE_HISTORY_TOOL


CLOSE_CARD, CLOSURE_REASON = "close_credit_card_account_7834", "log_credit_card_closure_reason_4521"


def _hidden(call: ToolCall, tool: str) -> dict | None:
    """The arguments of an assistant's call of `tool`, by name or through the hidden-tool wrapper."""
    if call.requestor != "assistant":
        return None
    args = call.arguments or {}
    if call.name == WRAPPER and args.get("agent_tool_name") == tool:
        return parse_hidden_arguments(args.get("arguments", "{}"))
    return dict(args) if call.name == tool else None


def request_arguments(calls: list[ToolCall]) -> dict | None:
    """The record_requests arguments a conversation's own writes imply: one request per (action, target) among the
    writes the request gate covers, for the conversation's customer. None when it makes none of those writes."""
    user, items = None, []
    for c in calls:
        args = c.arguments or {}
        inner = parse_hidden_arguments(args.get("arguments", "{}")) if c.name == WRAPPER else dict(args)
        user = user or (inner or {}).get("user_id")
        for tool, (action, target_arg) in REQUEST_TOOLS.items():
            if (w := _hidden(c, tool)) is not None:
                item = {"action": action, "target_id": str(w.get(target_arg) or ""),
                        "customer_words": "(replay: the conversation makes this write)"}
                if all((i["action"], i["target_id"]) != (item["action"], item["target_id"]) for i in items):
                    items.append(item)
    return {"user_id": user, "requests": items} if items and user else None


def with_plans(calls: list[ToolCall]) -> list[ToolCall]:
    """`calls` with a plan_disputes call before each user's first dispute filing (ids `<filing id>-plan`), and the
    dispute-history read before it when none came earlier (ids `<filing id>-history-unlock`, `<filing id>-history`).
    Before each closure of a credit card account whose closure reason was logged earlier in the conversation, the
    record_retention_answer call (accepted false) an agent following the retention gate makes (id `<close id>-retention`):
    the conversation closes the account, so the customer declined the offer. When the logged reason is annual_fee the
    call carries fee_waiver_offered true: a conversation that closes after that reason had the customer decline what
    credit_card_account_logistics_003 step 4 offers (the waiver when due), which record_retention_answer requires.
    Before each debit card dispute filing whose card_in_possession and pin_compromised differ from the latest answers recorded for its card (none yet, at
    the first), the record_dispute_answers call carrying the filing's own answers (id `<filing id>-answers`)."""
    plans = plan_arguments([f for c in calls if (f := filing(c))])
    requests = request_arguments(calls)
    out, planned, read, logged, answered, recorded = [], set(), False, set(), set(), False
    fee_reason = set()  # accounts whose logged closure reason is annual_fee
    answers: dict[str, tuple] = {}
    for c in calls:
        if (debit := _hidden(c, DEBIT_DISPUTE)) is not None:
            card, pair = debit.get("card_id"), (debit.get("card_in_possession"), debit.get("pin_compromised"))
            if answers.get(card) != pair:
                answers[card] = pair
                out.append(ToolCall(id=f"{c.id}-answers", name=DISPUTE_ANSWERS_TOOL, requestor="assistant", arguments={
                    "card_id": card, "user_id": debit.get("user_id"), "card_in_possession": pair[0],
                    "pin_compromised": pair[1], "customer_words": "(replay: the conversation files these answers)"}))
        if requests and not recorded and any(_hidden(c, tool) is not None for tool in REQUEST_TOOLS):
            recorded = True
            out.append(ToolCall(id=f"{c.id}-requests", name=REQUESTS_TOOL, arguments=requests, requestor="assistant"))
        if (reason := _hidden(c, CLOSURE_REASON)) is not None:
            logged.add(reason.get("credit_card_account_id"))
            if reason.get("closure_reason") == "annual_fee":
                fee_reason.add(reason.get("credit_card_account_id"))
        close = _hidden(c, CLOSE_CARD)
        if close is not None and close.get("credit_card_account_id") in logged - answered:
            card = close.get("credit_card_account_id")
            answered.add(card)
            out.append(ToolCall(id=f"{c.id}-retention", name=RETENTION_TOOL, requestor="assistant", arguments={
                "credit_card_account_id": card, "user_id": close.get("user_id"), "offer": "retention offer by card tier",
                "accepted": False, "customer_words": "(replay: the conversation closes the account)",
                "fee_waiver_offered": card in fee_reason}))
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


UNLOCK = "unlock_discoverable_agent_tool"
LOOKUP_SUFFIX = "-lookup"


def _lookup(db, value: str) -> tuple[str, dict, bool] | None:
    """The read an agent makes to see identifier `value`: (tool, arguments, hidden), from the table that holds it in
    `db`; None when no table holds it (an id an earlier write created shows in that write's own result)."""
    def table(name: str) -> dict:
        return getattr(db, name).data

    if value in table("users"):
        return "get_user_information_by_name", {"customer_name": table("users")[value].get("name")}, False
    if value in table("credit_card_accounts"):
        return "get_credit_card_accounts_by_user", {"user_id": table("credit_card_accounts")[value].get("user_id")}, False
    if value in table("credit_card_transaction_history"):
        user = table("credit_card_transaction_history")[value].get("user_id")
        return "get_credit_card_transactions_by_user", {"user_id": user}, False
    if value in table("accounts"):
        return "get_all_user_accounts_by_user_id_3847", {"user_id": table("accounts")[value].get("user_id")}, True
    if value in table("debit_cards"):
        return "get_debit_cards_by_account_id_7823", {"account_id": table("debit_cards")[value].get("account_id")}, True
    if value in table("bank_account_transaction_history"):
        account = table("bank_account_transaction_history")[value].get("account_id")
        return "get_bank_account_transactions_9173", {"account_id": account}, True
    return None


def with_lookups(calls: list[ToolCall], db) -> list[ToolCall]:
    """`calls` with, before each assistant call that names an identifier (an argument named `*_id`) for the first
    time, the read an agent makes to see it (`_lookup` on the initial database `db`; ids `<call id>-lookup-<n>`, a
    hidden read unlocked first): τ²'s gold actions carry almost none of the lookups a conversation makes, and the
    identifier-provenance deny (glint.gates.aids, N10) judges a write on the earlier tool results. τ² logs no read its
    gold lacks, so the scored database is unchanged."""
    out, shown, unlocked = [], set(), set()
    for c in calls:
        args = c.arguments or {}
        inner = (parse_hidden_arguments(args.get("arguments", "{}")) or {}) if c.name == WRAPPER else dict(args)
        if c.requestor == "assistant":
            for key, value in sorted(inner.items()):
                if not (key.endswith("_id") and isinstance(value, str)) or value in shown:
                    continue
                shown.add(value)
                read = _lookup(db, value)
                if read is None:
                    continue
                tool, read_args, hidden = read
                n = sum(o.id.startswith(f"{c.id}{LOOKUP_SUFFIX}") for o in out)
                if hidden and tool not in unlocked:
                    unlocked.add(tool)
                    out.append(ToolCall(id=f"{c.id}{LOOKUP_SUFFIX}-{n}", name=UNLOCK, requestor="assistant",
                                        arguments={"agent_tool_name": tool}))
                    n += 1
                out.append(ToolCall(id=f"{c.id}{LOOKUP_SUFFIX}-{n}", requestor="assistant", **(
                    {"name": WRAPPER, "arguments": {"agent_tool_name": tool, "arguments": json.dumps(read_args)}}
                    if hidden else {"name": tool, "arguments": read_args})))
        out.append(c)
    return out


def lookups(calls: list[ToolCall], db) -> list[ToolCall]:
    """Only the reads `with_lookups` would insert for `calls`: what an agent looks up first to see every identifier
    those calls name (for conversations built call by call, such as the tests)."""
    return [c for c in with_lookups(calls, db) if LOOKUP_SUFFIX in c.id]


def with_address_confirmations(calls: list[ToolCall]) -> list[ToolCall]:
    """`calls` with a record_address_confirmation call before each card's first replacement order (id `<order
    id>-address`), carrying that order's shipping address: the confirmation an agent following the address_ask gate's
    Fix records instead of sending the order once to be refused (fix research 2026-11-02, row 8). The conversation
    ships the card there, so that is the address the customer confirmed."""
    out, confirmed = [], set()
    for c in calls:
        args = c.arguments or {}
        order = (parse_hidden_arguments(args.get("arguments", "{}")) if c.name == WRAPPER
                 and args.get("agent_tool_name") == REPLACE else dict(args) if c.name == REPLACE else None)
        if c.requestor == "assistant" and order and order.get("credit_card_account_id") not in confirmed:
            confirmed.add(order.get("credit_card_account_id"))
            out.append(ToolCall(id=f"{c.id}-address", name=ADDRESS_TOOL, requestor="assistant", arguments={
                "credit_card_account_id": order.get("credit_card_account_id"), "user_id": order.get("user_id"),
                "shipping_address": order.get("shipping_address"),
                "customer_words": "(replay: the conversation ships the card to this address)"}))
        out.append(c)
    return out
