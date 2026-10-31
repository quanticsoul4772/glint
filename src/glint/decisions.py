"""Gate labels: for every write Glimmer made in a recorded τ² run, would the decision tool have overruled it?

Each run is replayed call by call in a fresh environment set to the
task's initial state; just before each write, the matching decision tool (glint.assist) gives its verdict on the
database at that moment. A write the verdict contradicts is a positive. Deterministic, no model.

Only decisions the tools encode are judged. A write with no matching tool (airline booking) is skipped; an airline
cancellation that is allowed only for a covered reason the replay cannot see (insurance + health/weather) is marked
`uncertain`, not positive.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from glint.assist import airline, banking, retail

RETAIL_WRITES = {
    "cancel_pending_order": "cancel", "modify_pending_order_address": "modify_address",
    "modify_pending_order_payment": "modify_payment", "modify_pending_order_items": "modify_items",
    "return_delivered_order_items": "return", "exchange_delivered_order_items": "exchange",
}
AIRLINE_WRITES = ("cancel_reservation", "update_reservation_flights", "update_reservation_baggages",
                  "update_reservation_passengers", "send_certificate")
BANKING_WRITES = ("file_credit_card_transaction_dispute_4829", "order_replacement_credit_card_7291",
                  "submit_credit_limit_increase_request_7392", "approve_credit_limit_increase_5847",
                  "deny_credit_limit_increase_5848")
CLI_STEPS = ("submit_credit_limit_increase_request_7392", "approve_credit_limit_increase_5847",
             "deny_credit_limit_increase_5848")
# glint's own READ tools, absent from the plain toolkit the replay uses
GLINT_TOOLS = {"check_order_action", "find_user_items", "find_variants", "check_reservation_action",
               "get_upcoming_trips_cost", "check_credit_limit_increase", "check_card_dispute"}
WRITES = {"retail": set(RETAIL_WRITES), "airline": set(AIRLINE_WRITES), "banking_knowledge": set(BANKING_WRITES)}


@dataclass
class Decision:
    task_id: str
    trial: int
    message_index: int     # index of the assistant message that made the write
    tool: str
    arguments: dict
    label: str             # "overruled" | "agreed" | "uncertain"
    reasons: list[str]
    executed: bool = True  # False when the environment itself rejected the write (an error reply, no change)


def _calls(sim: dict) -> list[tuple[int, str, dict]]:
    """(message index, tool name, arguments) of every assistant tool call, hidden banking tools unwrapped."""
    out = []
    for i, m in enumerate(sim["messages"]):
        if m["role"] != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            out.append((i, tc["name"], tc["arguments"]))
    return out


def _unwrap(name: str, args: dict) -> tuple[str, dict | None]:
    """The hidden banking tool and its arguments; None arguments when the model wrote malformed JSON (the
    environment rejects that call, so it is not judged)."""
    if name == "call_discoverable_agent_tool":
        inner = args.get("arguments") or {}
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except json.JSONDecodeError:
                return args.get("agent_tool_name", ""), None
        return args.get("agent_tool_name", ""), inner if isinstance(inner, dict) else None
    return name, args


def _flag(value) -> bool:
    """A boolean argument as the model wrote it: τ² shows every bool to the model as "True"/"False", so a model can
    send the string back, and bool("False") would read it as true."""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _retail(db, name: str, a: dict) -> tuple[str, list[str]] | None:
    action = RETAIL_WRITES.get(name)
    if action is None:
        return None
    v = retail.check_eligibility(db, a.get("order_id", ""), action, a.get("payment_method_id"),
                                 a.get("item_ids"), a.get("new_item_ids"))
    return ("agreed", []) if v["allowed"] else ("overruled", v["reasons"])


def _airline(db, name: str, a: dict) -> tuple[str, list[str]] | None:
    rid = a.get("reservation_id", "")
    res = db.reservations.get(rid)
    if name == "cancel_reservation":
        plain = airline.check_eligibility(db, rid, "cancel", reason="change of plan")
        if plain["allowed"]:
            return "agreed", []
        if airline.check_eligibility(db, rid, "cancel", reason="health")["allowed"]:
            return "uncertain", ["allowed only if the reason is covered by insurance; the replay cannot see it"]
        return "overruled", plain["reasons"]
    if name == "update_reservation_flights" and res is not None:
        same_flights = [(f["flight_number"], f["date"]) for f in a.get("flights") or []] == \
                       [(f.flight_number, f.date) for f in res.flights]
        if same_flights and a.get("cabin") == res.cabin:
            return None  # neither flights nor cabin change (a payment-only call): nothing the tools judge
        action = "change_cabin" if same_flights else "change_flights"
    elif name == "update_reservation_baggages" and res is not None:
        action = "remove_baggage" if (a.get("total_baggages") or 0) < (res.total_baggages or 0) else "add_baggage"
    elif name == "update_reservation_passengers" and res is not None:
        if len(a.get("passengers") or []) != len(res.passengers):
            return "overruled", ["the number of passengers cannot change"]
        action = "change_passengers"
    elif name == "send_certificate":
        # The call names no reservation, so eligibility is only certain when it does not depend on which one the
        # user complained about; the amount (per passenger, by delay or cancellation) is not judged
        user = db.users.get(a.get("user_id", ""))
        if user is None:
            return None
        if user.membership in ("silver", "gold"):
            return "agreed", []
        rows = [db.reservations[r] for r in user.reservations if r in db.reservations]
        qualifies = [r.insurance == "yes" or r.cabin == "business" for r in rows]
        if not any(qualifies):
            return "overruled", [airline.RULES["compensation"]]
        if all(qualifies):
            return "agreed", []
        return "uncertain", ["eligible only for some of the user's reservations; the call does not say which"]
    else:
        return None
    v = airline.check_eligibility(db, rid, action)
    return ("agreed", []) if v["allowed"] else ("overruled", v["reasons"])


def _banking(db, name: str, a: dict, later: list[tuple[str, dict]]) -> tuple[str, list[str]] | None:
    """`later`: the hidden banking writes (name, arguments) this run makes after this one, which stand for the
    run's remaining requests. A pending dispute blocks a credit limit increase for the whole user; a pending
    replacement blocks it only for that card (credit_card_account_logistics_007, checks 3 and 4)."""
    uid, cc = a.get("user_id", ""), a.get("credit_card_account_id", "")

    def later_cli(account: str | None) -> bool:
        return any(n in CLI_STEPS and (account is None or la.get("credit_card_account_id") == account)
                   for n, la in later)

    if name == "file_credit_card_transaction_dispute_4829":
        txn = db.credit_card_transaction_history.data.get(a.get("transaction_id", ""), {})
        cc = txn.get("credit_card_account_id") or next(
            (k for k, v in db.credit_card_accounts.data.items()
             if v.get("user_id") == uid and v.get("card_type") == txn.get("credit_card_type")), "")
        replaced_later = any(n == "order_replacement_credit_card_7291" and la.get("credit_card_account_id") == cc
                             for n, la in later)
        contacted = a.get("contacted_merchant")
        v = banking.check_dispute(db, uid, cc, a.get("transaction_id", ""), a.get("dispute_reason", ""),
                                  None if contacted is None else _flag(contacted),
                                  replacement_requested=replaced_later,
                                  other_requests=["credit limit increase"] if later_cli(None) else [])
        if "card_action" not in v:
            return None
        reasons = []
        if a.get("card_action") != v["card_action"]:
            reasons.append(f"card_action {a.get('card_action')} != {v['card_action']}")
        if _flag(a.get("eligible_for_provisional_credit")) != v["eligible_for_provisional_credit"]:
            reasons.append(f"eligible_for_provisional_credit {a.get('eligible_for_provisional_credit')} != "
                           f"{v['eligible_for_provisional_credit']}")
        if "sequencing" in v:
            reasons.append(v["sequencing"])
        return ("overruled", reasons) if reasons else ("agreed", [])
    if name == "order_replacement_credit_card_7291":
        return ("overruled", [banking.LIMIT_FIRST]) if later_cli(cc) else ("agreed", [])
    if name == "submit_credit_limit_increase_request_7392":
        v = banking.check_credit_limit_increase(db, uid, cc, banking._money(a.get("requested_increase_amount")))
        return ("overruled", v["reasons"]) if v["verdict"] in ("reduce_amount", "ask_amount") else ("agreed", [])
    if name in ("approve_credit_limit_increase_5847", "deny_credit_limit_increase_5848"):
        acc = db.credit_card_accounts.data.get(cc, {})
        submitted = [r for r in db.credit_limit_increase_requests.data.values()
                     if r.get("credit_card_account_id") == cc and str(r.get("status", "")).upper() == "PENDING"]
        if not submitted:  # the procedure submits the request before any decision (logistics_007, step 1)
            return "overruled", ["no credit limit increase request was submitted before this decision"]
        amount = banking._money(submitted[-1].get("requested_increase_amount"))
        v = banking.check_credit_limit_increase(db, uid, cc, amount)
        if v["verdict"] not in ("approve", "deny"):
            return None
        if name.startswith("approve"):
            if v["verdict"] != "approve":
                return "overruled", v["reasons"]
            wanted = round(banking._money(acc.get("credit_limit")) + amount)
            new = banking._money(a.get("new_credit_limit"))
            return ("agreed", []) if round(new) == wanted else (
                "overruled", [f"new_credit_limit {new:,.0f} != current limit + submitted increase {wanted:,}"])
        if v["verdict"] == "approve":
            return "overruled", ["every check passes; the policy approves"]
        return ("agreed", []) if a.get("denial_reason") == v["denial_reason"] else (
            "overruled", [f"denial_reason {a.get('denial_reason')} != {v['denial_reason']}"])
    return None


def label_run(sim: dict, task, env, unjudged: dict | None = None) -> list[Decision]:
    """Label every write of one recorded run. `env` is a fresh environment of the run's domain; `task` its Task.
    A write that cannot be judged (malformed arguments, or no rule covers it) gets no Decision; pass a dict as
    `unjudged` to have each one counted by tool name, so they never vanish from the totals unseen."""
    init = task.initial_state
    env.set_state(init.initialization_data if init else None, init.initialization_actions if init else None, [])
    calls = [(i, *_unwrap(n, a), n, a) for i, n, a in _calls(sim) if n not in GLINT_TOOLS]
    domain = env.domain_name
    out = []
    from tau2.data_model.message import ToolCall

    for k, (i, name, args, raw_name, raw_args) in enumerate(calls):
        db = env.tools.db
        if args is None:  # malformed arguments: the environment rejects the call; nothing to judge
            verdict = None
        elif domain == "retail":
            verdict = _retail(db, name, args)
        elif domain == "airline":
            verdict = _airline(db, name, args) if name in AIRLINE_WRITES else None
        else:
            later = [(n, la) for _, n, la, _, _ in calls[k + 1:] if n in BANKING_WRITES and la is not None]
            verdict = _banking(db, name, args, later) if name in BANKING_WRITES else None
        reply = env.get_response(ToolCall(id=str(k), name=raw_name, arguments=raw_args))
        if verdict is not None:
            executed = not (reply.error or str(reply.content).startswith("Error"))
            out.append(Decision(sim["task_id"], sim["trial"], i, name, args, *verdict, executed=executed))
        elif unjudged is not None and name in WRITES[domain]:
            unjudged[name] = unjudged.get(name, 0) + 1
    return out


def to_rows(decisions: list[Decision]) -> list[dict]:
    return [asdict(d) for d in decisions]
