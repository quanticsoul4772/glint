"""Gate labels: for every write Glimmer made in a recorded τ² run, would the decision tool have overruled it?

research/3-project/activation-gate-design.md §2. Each run is replayed call by call in a fresh environment set to the
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


def _unwrap(name: str, args: dict) -> tuple[str, dict]:
    if name == "call_discoverable_agent_tool":
        inner = args.get("arguments") or {}
        return args.get("agent_tool_name", ""), json.loads(inner) if isinstance(inner, str) else inner
    return name, args


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
        action = "change_cabin" if same_flights and a.get("cabin") != res.cabin else "change_flights"
    elif name == "update_reservation_baggages" and res is not None:
        action = "remove_baggage" if (a.get("total_baggages") or 0) < (res.total_baggages or 0) else "add_baggage"
    elif name == "update_reservation_passengers" and res is not None:
        if len(a.get("passengers") or []) != len(res.passengers):
            return "overruled", ["the number of passengers cannot change"]
        action = "change_passengers"
    elif name == "send_certificate":
        user = db.users.get(a.get("user_id", ""))
        if user is None:
            return None
        rows = [db.reservations[r] for r in user.reservations if r in db.reservations]
        eligible = user.membership in ("silver", "gold") or any(r.insurance == "yes" or r.cabin == "business"
                                                                for r in rows)
        return ("agreed", []) if eligible else ("overruled", [airline.RULES["compensation"]])
    else:
        return None
    v = airline.check_eligibility(db, rid, action)
    return ("agreed", []) if v["allowed"] else ("overruled", v["reasons"])


def _banking(db, name: str, a: dict, later: list[str]) -> tuple[str, list[str]] | None:
    """`later`: the hidden banking writes this run makes after this one (the run's own remaining requests)."""
    uid, cc = a.get("user_id", ""), a.get("credit_card_account_id", "")
    cli_pending = any(n in CLI_STEPS for n in later)
    if name == "file_credit_card_transaction_dispute_4829":
        txn = db.credit_card_transaction_history.data.get(a.get("transaction_id", ""), {})
        cc = txn.get("credit_card_account_id") or next(
            (k for k, v in db.credit_card_accounts.data.items()
             if v.get("user_id") == uid and v.get("card_type") == txn.get("credit_card_type")), "")
        v = banking.check_dispute(db, uid, cc, a.get("transaction_id", ""), a.get("dispute_reason", ""),
                                  a.get("contacted_merchant"),
                                  replacement_requested="order_replacement_credit_card_7291" in later,
                                  other_requests=["credit limit increase"] if cli_pending else [])
        if "card_action" not in v:
            return None
        reasons = []
        if a.get("card_action") != v["card_action"]:
            reasons.append(f"card_action {a.get('card_action')} != {v['card_action']}")
        if bool(a.get("eligible_for_provisional_credit")) != v["eligible_for_provisional_credit"]:
            reasons.append(f"eligible_for_provisional_credit {a.get('eligible_for_provisional_credit')} != "
                           f"{v['eligible_for_provisional_credit']}")
        if "sequencing" in v:
            reasons.append(v["sequencing"])
        return ("overruled", reasons) if reasons else ("agreed", [])
    if name == "order_replacement_credit_card_7291":
        return ("overruled", [banking.LIMIT_FIRST]) if cli_pending else ("agreed", [])
    if name == "submit_credit_limit_increase_request_7392":
        v = banking.check_credit_limit_increase(db, uid, cc, float(a.get("requested_increase_amount") or 0))
        return ("overruled", v["reasons"]) if v["verdict"] in ("reduce_amount", "ask_amount") else ("agreed", [])
    if name in ("approve_credit_limit_increase_5847", "deny_credit_limit_increase_5848"):
        acc = db.credit_card_accounts.data.get(cc, {})
        submitted = [r for r in db.credit_limit_increase_requests.data.values()
                     if r.get("credit_card_account_id") == cc and r.get("status") == "PENDING"]
        amount = (float(a["new_credit_limit"]) - banking._money(acc.get("credit_limit"))
                  if name.startswith("approve") and a.get("new_credit_limit") is not None
                  else float(submitted[-1]["requested_increase_amount"]) if submitted else 0.0)
        v = banking.check_credit_limit_increase(db, uid, cc, amount)
        if v["verdict"] not in ("approve", "deny"):
            return None
        if name.startswith("approve"):
            return ("agreed", []) if v["verdict"] == "approve" else ("overruled", v["reasons"])
        if v["verdict"] == "approve":
            return "overruled", ["every check passes; the policy approves"]
        return ("agreed", []) if a.get("denial_reason") == v["denial_reason"] else (
            "overruled", [f"denial_reason {a.get('denial_reason')} != {v['denial_reason']}"])
    return None


def label_run(sim: dict, task, env) -> list[Decision]:
    """Label every write of one recorded run. `env` is a fresh environment of the run's domain; `task` its Task."""
    init = task.initial_state
    env.set_state(init.initialization_data if init else None, init.initialization_actions if init else None, [])
    calls = [(i, *_unwrap(n, a), n, a) for i, n, a in _calls(sim) if n not in GLINT_TOOLS]
    banking_names = [name for _, name, _, _, _ in calls]
    domain = env.domain_name
    out = []
    from tau2.data_model.message import ToolCall

    for k, (i, name, args, raw_name, raw_args) in enumerate(calls):
        db = env.tools.db
        if domain == "retail":
            verdict = _retail(db, name, args)
        elif domain == "airline":
            verdict = _airline(db, name, args) if name in AIRLINE_WRITES else None
        else:
            later = [n for n in banking_names[k + 1:] if n in BANKING_WRITES]
            verdict = _banking(db, name, args, later) if name in BANKING_WRITES else None
        reply = env.get_response(ToolCall(id=str(k), name=raw_name, arguments=raw_args))
        if verdict is not None:
            executed = not (reply.error or str(reply.content).startswith("Error"))
            out.append(Decision(sim["task_id"], sim["trial"], i, name, args, *verdict, executed=executed))
    return out


def to_rows(decisions: list[Decision]) -> list[dict]:
    return [asdict(d) for d in decisions]
