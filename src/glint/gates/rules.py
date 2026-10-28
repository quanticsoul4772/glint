"""The gates on Glimmer's writes, built on the existing checks in glint.assist (which write maps to which check
follows glint.decisions).

Every gate reads only the live database and the write's own arguments, so the same call against the same database
gets the same decision (τ²'s evaluator replays the agent's writes). Where a check depends on something the database
cannot show (the user's stated cancellation reason, a replacement card ordered later in the conversation), the gate
allows and says so in its rule_id; it never guesses. Writes no rule covers yet are allowed and logged under
`no_rule_yet` (NO_RULE_YET below names each one).
"""
from __future__ import annotations

import json

from glint.assist import airline, as_flag, banking, retail
from glint.gates.core import NO_RULE, Gate, GateRegistry, GateResult, allow, deny

RETAIL_ACTIONS = {
    "cancel_pending_order": "cancel", "modify_pending_order_address": "modify_address",
    "modify_pending_order_payment": "modify_payment", "modify_pending_order_items": "modify_items",
    "return_delivered_order_items": "return", "exchange_delivered_order_items": "exchange",
}
AIRLINE_GATED = ("cancel_reservation", "update_reservation_flights", "update_reservation_baggages",
                 "update_reservation_passengers")
DISPUTE, REPLACE = "file_credit_card_transaction_dispute_4829", "order_replacement_credit_card_7291"
SUBMIT, APPROVE, DENY = ("submit_credit_limit_increase_request_7392", "approve_credit_limit_increase_5847",
                         "deny_credit_limit_increase_5848")
BANKING_GATED = (DISPUTE, REPLACE, SUBMIT, APPROVE, DENY)

# Writes with no rule yet (phase 2 adds them): allowed, and logged as such
NO_RULE_YET = {
    "retail": ("modify_user_address",),
    "airline": ("book_reservation", "send_certificate"),
    "banking_knowledge": (
        "change_user_email", "log_verification", "example_agent_tool_0000", "update_transaction_rewards_3847",
        "file_debit_card_transaction_dispute_6281", "set_debit_card_recurring_block_7382",
        "log_credit_card_closure_reason_4521", "apply_statement_credit_8472", "apply_credit_card_account_flag_6147",
        "close_credit_card_account_7834", "pay_credit_card_from_checking_9182", "open_bank_account_4821",
        "close_bank_account_7392", "transfer_funds_between_bank_accounts_7291", "apply_checking_account_credit_5829",
        "apply_savings_account_credit_6831", "submit_interest_discrepancy_report_7294", "order_debit_card_5739",
        "activate_debit_card_8291", "activate_debit_card_8292", "activate_debit_card_8293", "close_debit_card_4721",
        "freeze_debit_card_3892", "unfreeze_debit_card_3893", "clear_debit_card_fraud_alert_4892",
        "reset_debit_card_pin_6284", "change_debit_card_pin_6285", "request_temporary_debit_card_limit_increase_8374",
    ),
}
RULED = {"retail": tuple(RETAIL_ACTIONS), "airline": AIRLINE_GATED, "banking_knowledge": BANKING_GATED}
# Boot check: every one of these write tools must have a gate before the first call is served
REQUIRED_GATES = {d: set(RULED[d]) | set(NO_RULE_YET[d]) for d in RULED}


# ---- retail -------------------------------------------------------------------------------------------------------

def _retail_gate(action: str):
    source = "glint.assist.retail.check_eligibility"

    def gate(db, a: dict) -> GateResult:
        v = retail.check_eligibility(db, a.get("order_id", ""), action, a.get("payment_method_id"),
                                     a.get("item_ids"), a.get("new_item_ids"))
        if v["allowed"]:
            return allow(f"retail.{action}", source, v)
        return deny(f"retail.{action}", source, "; ".join(v["reasons"]),
                    "Do not repeat this write. Tell the user what the policy does not allow and why (rules in the "
                    "verdict); use only the payment methods the verdict allows.", v)
    return gate


# ---- airline ------------------------------------------------------------------------------------------------------

AIRLINE_SOURCE = "glint.assist.airline.check_eligibility"
AIRLINE_FIX = "Do not repeat this write. Tell the user the policy does not allow it (rules in the verdict)."


def _airline_cancel(db, a: dict) -> GateResult:
    rid = a.get("reservation_id", "")
    plain = airline.check_eligibility(db, rid, "cancel", reason="change of plan")
    if plain["allowed"]:
        return allow("airline.cancel", AIRLINE_SOURCE, plain)
    covered = airline.check_eligibility(db, rid, "cancel", reason="health")
    if covered["allowed"]:
        # Allowed only for a reason insurance covers; the call does not carry the user's reason (glint.decisions
        # marks this case "uncertain"): allow and log, never deny what may be allowed
        return allow("airline.cancel.insurance_reason_unseen", AIRLINE_SOURCE, covered,
                     reason="allowed only if the user's reason is covered by insurance; the call does not show it")
    return deny("airline.cancel", AIRLINE_SOURCE, "; ".join(plain["reasons"]), AIRLINE_FIX, plain)


def _airline_flights(db, a: dict) -> GateResult:
    rid = a.get("reservation_id", "")
    res = db.reservations.get(rid)
    action = "change_flights"
    if res is not None:
        same = [(f.get("flight_number"), f.get("date")) for f in a.get("flights") or []] == \
               [(f.flight_number, f.date) for f in res.flights]
        if same and a.get("cabin") == res.cabin:
            return allow("airline.update_flights.no_change", AIRLINE_SOURCE,
                         reason="neither the flights nor the cabin change (payment only): nothing to judge")
        action = "change_cabin" if same else "change_flights"
    v = airline.check_eligibility(db, rid, action)
    if v["allowed"]:
        return allow(f"airline.{action}", AIRLINE_SOURCE, v)
    return deny(f"airline.{action}", AIRLINE_SOURCE, "; ".join(v["reasons"]), AIRLINE_FIX, v)


def _airline_baggages(db, a: dict) -> GateResult:
    rid = a.get("reservation_id", "")
    res = db.reservations.get(rid)
    removing = res is not None and (a.get("total_baggages") or 0) < (res.total_baggages or 0)
    action = "remove_baggage" if removing else "add_baggage"
    v = airline.check_eligibility(db, rid, action)
    if v["allowed"]:
        return allow(f"airline.{action}", AIRLINE_SOURCE, v)
    return deny(f"airline.{action}", AIRLINE_SOURCE, "; ".join(v["reasons"]), AIRLINE_FIX, v)


def _airline_passengers(db, a: dict) -> GateResult:
    rid = a.get("reservation_id", "")
    res = db.reservations.get(rid)
    if res is not None and len(a.get("passengers") or []) != len(res.passengers):
        details = {"reservation_passengers": len(res.passengers), "requested": len(a.get("passengers") or [])}
        return deny("airline.change_passengers.count", AIRLINE_SOURCE, "the number of passengers cannot change",
                    "Keep the same number of passengers; only names and details may change.",
                    details | {"rules": [airline.RULES["passengers"]]})
    v = airline.check_eligibility(db, rid, "change_passengers")
    if v["allowed"]:
        return allow("airline.change_passengers", AIRLINE_SOURCE, v)
    return deny("airline.change_passengers", AIRLINE_SOURCE, "; ".join(v["reasons"]), AIRLINE_FIX, v)


# ---- banking ------------------------------------------------------------------------------------------------------

def _pending_increase(db, card: str) -> list[str]:
    """Credit limit increase requests on `card` submitted but not yet decided. τ²'s approve record reuses the submit
    request's id, so its insert is dropped and the request stays PENDING after a decision; a decision is visible only
    as the approve or deny call recorded in agent_discoverable_tools."""
    called = {str(r.get("tool_name")) for r in db.agent_discoverable_tools.data.values()}
    if APPROVE in called or DENY in called:
        return []
    return [rid for rid, r in db.credit_limit_increase_requests.data.items()
            if r.get("credit_card_account_id") == card and str(r.get("status", "")).upper() == "PENDING"]


def _card_of_transaction(db, user_id: str, txn: dict) -> str:
    return txn.get("credit_card_account_id") or next(
        (k for k, v in db.credit_card_accounts.data.items()
         if v.get("user_id") == user_id and v.get("card_type") == txn.get("credit_card_type")), "")


def _dispute(db, a: dict) -> GateResult:
    source = "glint.assist.banking.check_dispute"
    uid, tid = a.get("user_id", ""), a.get("transaction_id", "")
    txn = db.credit_card_transaction_history.data.get(tid, {})
    card = _card_of_transaction(db, uid, txn)
    contacted = a.get("contacted_merchant")
    v = banking.check_dispute(db, uid, card, tid, a.get("dispute_reason", ""),
                              None if contacted is None else as_flag(contacted))
    if "card_action" not in v:
        return deny("banking.dispute.record", source, "; ".join(v["reasons"]),
                    "Check the user id and the transaction id; dispute only a transaction of this user.", v)
    reasons = []
    if v["card_action"] == "cancel_and_reissue" and a.get("card_action") != "cancel_and_reissue":
        reasons.append(f"card_action {a.get('card_action')!r}: a replacement for this card was ordered today, so "
                       "every dispute on it must use 'cancel_and_reissue'")
    if as_flag(a.get("eligible_for_provisional_credit")) != v["eligible_for_provisional_credit"]:
        failed = [k for k, ok in v["provisional_credit_criteria"].items() if not ok]
        reasons.append(f"eligible_for_provisional_credit {a.get('eligible_for_provisional_credit')} != "
                       f"{v['eligible_for_provisional_credit']}" + (f" (failed: {failed})" if failed else ""))
    if reasons:
        return deny("banking.dispute", source, "; ".join(reasons),
                    "File the dispute with the card_action and eligible_for_provisional_credit values in the verdict.",
                    v)
    if a.get("card_action") == "cancel_and_reissue" and v["card_action"] == "keep_active":
        # a replacement may still be ordered later in the conversation; the database cannot show that yet
        return allow("banking.dispute.card_action_unseen", source, v,
                     reason="cancel_and_reissue with no replacement ordered yet: allowed, a replacement may follow")
    return allow("banking.dispute", source, v)


def _increase_verdict(db, a: dict, amount: float) -> dict:
    return banking.check_credit_limit_increase(db, a.get("user_id", ""), a.get("credit_card_account_id", ""), amount)


CLI_SOURCE = "glint.assist.banking.check_credit_limit_increase"


def _submit(db, a: dict) -> GateResult:
    v = _increase_verdict(db, a, banking._money(a.get("requested_increase_amount")))
    if v["verdict"] in ("reduce_amount", "ask_amount", "error"):
        return deny(f"banking.cli.submit.{v['verdict']}", CLI_SOURCE, "; ".join(v["reasons"]),
                    "Submit an increase no larger than max_increase in the verdict, for an account of this user.", v)
    return allow(f"banking.cli.submit.{v['verdict']}", CLI_SOURCE, v)


def _decision(db, a: dict, approving: bool) -> GateResult:
    card = a.get("credit_card_account_id", "")
    acc = db.credit_card_accounts.data.get(card, {})
    submitted = [r for r in db.credit_limit_increase_requests.data.values()
                 if r.get("credit_card_account_id") == card and str(r.get("status", "")).upper() == "PENDING"]
    rule = "banking.cli.approve" if approving else "banking.cli.deny"
    if not submitted:
        return deny(f"{rule}.not_submitted", CLI_SOURCE,
                    "no credit limit increase request was submitted for this card before this decision",
                    "Submit the request first (submit_credit_limit_increase_request_7392), then do the lookups, then "
                    "decide.", {"credit_card_account_id": card})
    amount = banking._money(submitted[-1].get("requested_increase_amount"))
    v = _increase_verdict(db, a, amount)
    if v["verdict"] not in ("approve", "deny"):
        return allow(f"{rule}.no_verdict", CLI_SOURCE, v, reason="the check gives no approve/deny verdict here")
    if approving:
        if v["verdict"] != "approve":
            return deny(rule, CLI_SOURCE, f"the policy denies this increase: {v['denial_reason']} "
                        f"({'; '.join(v['reasons'])})", f"Deny it with denial_reason {v['denial_reason']!r}.", v)
        wanted = round(banking._money(acc.get("credit_limit")) + amount)
        new = banking._money(a.get("new_credit_limit"))
        if round(new) != wanted:
            return deny(f"{rule}.new_limit", CLI_SOURCE,
                        f"new_credit_limit {new:,.0f} != current limit + submitted increase {wanted:,}",
                        f"Approve with new_credit_limit {wanted}.", v)
        return allow(rule, CLI_SOURCE, v)
    if v["verdict"] == "approve":
        return deny(rule, CLI_SOURCE, "every check passes; the policy approves this increase",
                    f"Approve it with new_credit_limit {v['new_credit_limit']}.", v)
    if a.get("denial_reason") != v["denial_reason"]:
        return deny(f"{rule}.reason", CLI_SOURCE, f"denial_reason {a.get('denial_reason')!r} != "
                    f"{v['denial_reason']!r} (the first failing check in the policy's order)",
                    f"Deny with denial_reason {v['denial_reason']!r}.", v)
    return allow(rule, CLI_SOURCE, v)


def _replacement(db, a: dict) -> GateResult:
    card = a.get("credit_card_account_id", "")
    pending = _pending_increase(db, card)
    details = {"credit_card_account_id": card, "undecided_increase_requests": pending}
    if pending:
        return deny("banking.replacement.increase_first", "glint.assist.banking.LIMIT_FIRST", banking.LIMIT_FIRST,
                    "Approve or deny the submitted credit limit increase first, then order the replacement.", details)
    return allow("banking.replacement", "glint.assist.banking.LIMIT_FIRST", details)


# ---- registry -----------------------------------------------------------------------------------------------------

def _no_rule(tool: str):
    def gate(db, a: dict) -> GateResult:
        return allow(NO_RULE, "glint.gates.rules.NO_RULE_YET", reason=f"no rule covers {tool} yet (phase 2)")
    return gate


def build_registry(domain: str) -> GateRegistry:
    reg = GateRegistry()
    if domain == "retail":
        for tool, action in RETAIL_ACTIONS.items():
            reg.register(Gate(f"retail_{action}_gate", (tool,), f"Retail policy for '{action}': order status, items "
                              "of the same product, and the payment methods allowed.", _retail_gate(action)))
    elif domain == "airline":
        purpose = "Airline policy: who may cancel, basic economy, flown segments, bags, passenger count."
        for tool, fn in zip(AIRLINE_GATED, (_airline_cancel, _airline_flights, _airline_baggages,
                                            _airline_passengers)):
            reg.register(Gate(f"airline_{tool}_gate", (tool,), purpose, fn))
    elif domain == "banking_knowledge":
        reg.register(Gate("banking_dispute_gate", (DISPUTE,), "Knowledge base credit_cards_(general)_014/_015: the "
                          "card_action and provisional credit a dispute must carry.", _dispute))
        reg.register(Gate("banking_cli_submit_gate", (SUBMIT,), "credit_card_account_logistics_005: the maximum "
                          "increase per tier.", _submit))
        reg.register(Gate("banking_cli_approve_gate", (APPROVE,), "credit_card_account_logistics_006/_007: approve "
                          "only when every check passes, with the submitted amount.",
                          lambda db, a: _decision(db, a, True)))
        reg.register(Gate("banking_cli_deny_gate", (DENY,), "credit_card_account_logistics_006/_007: deny only when "
                          "a check fails, with the first failing reason.", lambda db, a: _decision(db, a, False)))
        reg.register(Gate("banking_replacement_gate", (REPLACE,), "credit_card_account_logistics_007: a pending "
                          "replacement makes a credit limit increase fail, so the increase is decided first.",
                          _replacement))
    else:
        raise ValueError(f"no gates for domain {domain!r}")
    for tool in NO_RULE_YET[domain]:
        reg.register(Gate(f"no_rule_{tool}", (tool,), "Every write is logged; no policy rule covers this one yet.",
                          _no_rule(tool)))
    return reg


def manifest(domain: str, reg: GateRegistry) -> dict:
    return {"domain": domain, "required": sorted(REQUIRED_GATES[domain]), "no_rule_yet": list(NO_RULE_YET[domain]),
            "gates": [{"name": g.name, "tools": list(g.tools), "purpose": g.purpose, "timeout_s": g.timeout_s}
                      for g in reg.gates()]}


def parse_hidden_arguments(raw) -> dict | None:
    """The hidden banking tool's arguments as τ²'s call_discoverable_agent_tool parses them (integers as floats);
    None when they do not parse (the tool rejects that call itself)."""
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}", parse_int=float)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None
