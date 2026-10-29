"""The gates on Glimmer's writes, built on the existing checks in glint.assist (which write maps to which check
follows glint.decisions).

Every gate reads only the live database and the write's own arguments, so the same call against the same database
gets the same decision (τ²'s evaluator replays the agent's writes). Where a check depends on something the database
cannot show (the user's stated cancellation reason, a replacement card ordered later in the conversation), the gate
allows and says so in its rule_id; it never guesses. Writes no rule covers yet are allowed and logged under
`no_rule_yet` (NO_RULE_REASONS below names each one and why).

Phase 2 (2026-10-28) adds the debit card rules (glint.gates.debit), the other banking writes (glint.gates.banking),
the procedure gates (glint.gates.procedure: verification, recorded lookups, order, the transfer reason) and the
airline and retail writes (glint.gates.travel). Rules are per write tool, never per task. Dispute planning
(2026-10-28) adds the plan gate on credit card disputes (procedure.dispute_plan).
"""
from __future__ import annotations

import json

from glint.assist import airline, as_flag, banking, retail
from glint.gates import banking as bank_rules
from glint.gates import debit, procedure, travel
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

# Writes with no rule of their own yet: allowed, and logged under no_rule_yet (each with the reason it has none)
NO_RULE_REASONS = {
    "example_agent_tool_0000": "a placeholder tool in the toolkit; no procedure uses it",
    "file_debit_card_transaction_dispute_6281": ("its policy values (category, liability, provisional credit, "
                                                 "card action) come from the user's account of what happened; the "
                                                 "procedure gates still require verification and the lookups"),
    "set_debit_card_recurring_block_7382": "the merchant and the block are the user's request",
    "pay_credit_card_from_checking_9182": "the amount is the user's choice; the procedure gates require verification",
    "open_bank_account_4821": ("the account class follows the user's stated needs; the tool checks eligibility "
                               "(the 14-day checking tenure)"),
    "close_bank_account_7392": ("whether the user also plans to open an account that needs this one's tenure is "
                                "not in the database"),
    "transfer_funds_between_bank_accounts_7291": "the amount and accounts are the user's choice",
    "apply_savings_account_credit_6831": ("the interest correction amount needs the rate tables; the procedure gates "
                                          "require the lookups"),
    "submit_interest_discrepancy_report_7294": "the discrepancy is computed from rate tables; lookups are gated",
    "clear_debit_card_fraud_alert_4892": "the fraud score inputs are the user's answers",
    "reset_debit_card_pin_6284": "the PIN rules are checked by the tool; verification is gated",
    "change_debit_card_pin_6285": "the PIN rules are checked by the tool; verification is gated",
    "request_temporary_debit_card_limit_increase_8374": "the amount and travel facts are the user's statement",
}
NO_RULE_YET = {
    "retail": (),
    "airline": (),
    "banking_knowledge": tuple(NO_RULE_REASONS),
}
# GENERIC (not WRITE) tools gated like writes: they end or redirect the conversation
GATED_GENERIC = {"banking_knowledge": ("transfer_to_human_agents",)}
PHASE2 = {
    "retail": ("modify_user_address",),
    "airline": ("book_reservation", "send_certificate"),
    "banking_knowledge": (
        "log_verification", "change_user_email", "update_transaction_rewards_3847",
        "log_credit_card_closure_reason_4521", "apply_statement_credit_8472", "apply_credit_card_account_flag_6147",
        "close_credit_card_account_7834", "apply_checking_account_credit_5829", *debit.DEBIT_TOOLS,
        "transfer_to_human_agents"),
}
RULED = {"retail": tuple(RETAIL_ACTIONS) + PHASE2["retail"], "airline": AIRLINE_GATED + PHASE2["airline"],
         "banking_knowledge": BANKING_GATED + PHASE2["banking_knowledge"]}
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
        return deny(f"retail.{action}", source, f"the retail policy does not allow this {action.replace('_', ' ')}",
                    "Do not repeat this write. Tell the user what the policy does not allow and why (rules in the "
                    "verdict); use only the payment methods the verdict allows.", v,
                    consequence=POLICY_NO if any(map(retail_rules_out, v["reasons"])) else RETAIL_ARGS_NO,
                    facts=v["reasons"])
    return gate


# ---- airline ------------------------------------------------------------------------------------------------------

AIRLINE_SOURCE = "glint.assist.airline.check_eligibility"
AIRLINE_FIX = "Do not repeat this write. Tell the user the policy does not allow it (rules in the verdict)."
POLICY_NO = ("This write cannot succeed for this record however it is retried. Tell the customer why, then continue "
             "with any other part of their request.")
# a retail deny on the call's own arguments (an order id, items, a payment method outside a non-empty allowed list)
# can succeed with corrected ones
def retail_rules_out(reason: str) -> bool:
    """A glint.assist.retail.check_eligibility reason no corrected call can get past: the order's status, or no
    payment method that covers the change at all."""
    return reason.startswith("status is") or reason.endswith("(no payment method covers it)")


RETAIL_ARGS_NO = ("Nothing changed. A call with corrected arguments (the items or payment method the policy allows) "
                  "can succeed; tell the customer what must change, then continue with any other part of their "
                  "request.")


def _airline_deny(rule: str, v: dict) -> GateResult:
    what = rule.split(".", 1)[1].replace("_", " ")
    return deny(rule, AIRLINE_SOURCE, f"the airline policy does not allow this {what}", AIRLINE_FIX, v,
                consequence=POLICY_NO, facts=v["reasons"])


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
    return _airline_deny("airline.cancel", plain)


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
    return _airline_deny(f"airline.{action}", v)


def _airline_baggages(db, a: dict) -> GateResult:
    rid = a.get("reservation_id", "")
    res = db.reservations.get(rid)
    removing = res is not None and (a.get("total_baggages") or 0) < (res.total_baggages or 0)
    action = "remove_baggage" if removing else "add_baggage"
    v = airline.check_eligibility(db, rid, action)
    if v["allowed"]:
        return allow(f"airline.{action}", AIRLINE_SOURCE, v)
    return _airline_deny(f"airline.{action}", v)


def _airline_passengers(db, a: dict) -> GateResult:
    rid = a.get("reservation_id", "")
    res = db.reservations.get(rid)
    if res is not None and len(a.get("passengers") or []) != len(res.passengers):
        details = {"reservation_passengers": len(res.passengers), "requested": len(a.get("passengers") or [])}
        return deny("airline.change_passengers.count", AIRLINE_SOURCE, "the number of passengers on a reservation "
                    "cannot change", "Keep the same number of passengers; only names and details may change.",
                    details | {"rules": [airline.RULES["passengers"]]},
                    consequence="Passenger details can still be corrected; adding or removing a passenger is not "
                    "possible on this reservation.",
                    facts=[f"passengers on {rid}: {details['reservation_passengers']}",
                           f"passengers in the call: {details['requested']}"])
    v = airline.check_eligibility(db, rid, "change_passengers")
    if v["allowed"]:
        return allow("airline.change_passengers", AIRLINE_SOURCE, v)
    return _airline_deny("airline.change_passengers", v)


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


def _dispute(db, a: dict) -> GateResult:
    source = "glint.assist.banking.check_dispute"
    uid, tid = a.get("user_id", ""), a.get("transaction_id", "")
    txn = db.credit_card_transaction_history.data.get(tid, {})
    card = banking.card_of_transaction(db, uid, txn)
    contacted = a.get("contacted_merchant")
    v = banking.check_dispute(db, uid, card, tid, a.get("dispute_reason", ""),
                              None if contacted is None else as_flag(contacted))
    if "card_action" not in v:
        return deny("banking.dispute.record", source, "the dispute does not name a transaction of this user",
                    "Check the user id and the transaction id; dispute only a transaction of this user.", v,
                    consequence="Nothing was filed. Look the transaction up on the customer's own cards before "
                    "filing.", facts=v["reasons"])
    reasons = []
    if v["card_action"] == "cancel_and_reissue" and a.get("card_action") != "cancel_and_reissue":
        reasons.append(f"card_action {a.get('card_action')!r}: a replacement for this card was ordered today, so "
                       "every dispute on it must use 'cancel_and_reissue'")
    if as_flag(a.get("eligible_for_provisional_credit")) != v["eligible_for_provisional_credit"]:
        reasons.append(f"eligible_for_provisional_credit given {a.get('eligible_for_provisional_credit')}, policy "
                       f"{v['eligible_for_provisional_credit']}")
    if reasons:
        failed = [k for k, ok in v["provisional_credit_criteria"].items() if not ok]
        return deny("banking.dispute", source, "the dispute's values do not match the policy",
                    "File the dispute with the card_action and eligible_for_provisional_credit values in the verdict.",
                    v, consequence="Nothing was filed. The values come from the policy, not from the customer's "
                    "wishes; file with them and tell the customer what they mean.",
                    facts=[*reasons, f"provisional credit criteria failed: {', '.join(failed) or 'none'}"])
    if a.get("card_action") == "cancel_and_reissue" and v["card_action"] == "keep_active":
        # a replacement may still be ordered later in the conversation; the database cannot show that yet
        return allow("banking.dispute.card_action_unseen", source, v,
                     reason="cancel_and_reissue with no replacement ordered yet: allowed, a replacement may follow")
    return allow("banking.dispute", source, v)


def _increase_verdict(db, a: dict, amount: float) -> dict:
    return banking.check_credit_limit_increase(db, a.get("user_id", ""), a.get("credit_card_account_id", ""), amount)


CLI_SOURCE = "glint.assist.banking.check_credit_limit_increase"
SUBMIT_CONSEQUENCE = {
    "reduce_amount": ("Nothing was submitted. Tell the customer the largest increase their card allows and ask whether "
                      "to submit that."),
    "ask_amount": "Nothing was submitted. Ask the customer how large an increase they want, in dollars.",
    "error": ("Nothing was submitted. The request names a card that is not this customer's; look up their cards "
              "first."),
}
INCREASE_ORDER = "The increase goes submit, then the required lookups, then approve or deny; nothing was decided."


def _submit(db, a: dict) -> GateResult:
    v = _increase_verdict(db, a, banking._money(a.get("requested_increase_amount")))
    if v["verdict"] in ("reduce_amount", "ask_amount", "error"):
        return deny(f"banking.cli.submit.{v['verdict']}", CLI_SOURCE, "the requested increase cannot be submitted "
                    "as given", "Submit an increase no larger than max_increase in the verdict, for an account of "
                    "this user.", v, consequence=SUBMIT_CONSEQUENCE[v["verdict"]], facts=v["reasons"])
    return allow(f"banking.cli.submit.{v['verdict']}", CLI_SOURCE, v)


def _decision(db, a: dict, approving: bool) -> GateResult:
    card = a.get("credit_card_account_id", "")
    acc = db.credit_card_accounts.data.get(card, {})
    submitted = [r for r in db.credit_limit_increase_requests.data.values()
                 if r.get("credit_card_account_id") == card and str(r.get("status", "")).upper() == "PENDING"]
    rule = "banking.cli.approve" if approving else "banking.cli.deny"
    if not submitted:
        return deny(f"{rule}.not_submitted", CLI_SOURCE, "a decision needs a submitted request, and none is on "
                    "record for this card", "Submit the request first (submit_credit_limit_increase_request_7392), "
                    "then do the lookups, then decide.", {"credit_card_account_id": card},
                    consequence=INCREASE_ORDER, facts=[f"pending increase requests for {card}: none"])
    amount = banking._money(submitted[-1].get("requested_increase_amount"))
    v = _increase_verdict(db, a, amount)
    if v["verdict"] not in ("approve", "deny"):
        return allow(f"{rule}.no_verdict", CLI_SOURCE, v, reason="the check gives no approve/deny verdict here")
    if approving:
        if v["verdict"] != "approve":
            return deny(rule, CLI_SOURCE, "a policy check fails, so the increase is denied, not approved",
                        f"Deny it with denial_reason {v['denial_reason']!r}.", v,
                        consequence="The request is decided by the checks; tell the customer the reason for the "
                        "denial.", facts=[f"first failing check: {v['denial_reason']}", *v["reasons"]])
        wanted = round(banking._money(acc.get("credit_limit")) + amount)
        new = banking._money(a.get("new_credit_limit"))
        if round(new) != wanted:
            return deny(f"{rule}.new_limit", CLI_SOURCE, "the new limit must be the current limit plus the "
                        "submitted increase", f"Approve with new_credit_limit {wanted}.", v,
                        consequence="The approval is right; only the new limit is wrong.",
                        facts=[f"current limit: {banking._money(acc.get('credit_limit')):,.0f}",
                               f"submitted increase: {amount:,.0f}", f"new_credit_limit given: {new:,.0f}"])
        return allow(rule, CLI_SOURCE, v)
    if v["verdict"] == "approve":
        return deny(rule, CLI_SOURCE, "every policy check passes, so the increase is approved, not denied",
                    f"Approve it with new_credit_limit {v['new_credit_limit']}.", v,
                    consequence="The request is decided by the checks; tell the customer it is approved and the new "
                    "limit.", facts=v["reasons"] or ["every check passed"])
    if a.get("denial_reason") != v["denial_reason"]:
        return deny(f"{rule}.reason", CLI_SOURCE, "the denial reason must be the first failing check in the "
                    "policy's order", f"Deny with denial_reason {v['denial_reason']!r}.", v,
                    consequence="The denial is right; only its reason is wrong.",
                    facts=[f"denial_reason given: {a.get('denial_reason')!r}",
                           f"first failing check: {v['denial_reason']!r}", *v["reasons"]])
    return allow(rule, CLI_SOURCE, v)


def _replacement(db, a: dict) -> GateResult:
    card = a.get("credit_card_account_id", "")
    pending = _pending_increase(db, card)
    details = {"credit_card_account_id": card, "undecided_increase_requests": pending}
    if pending:
        return deny("banking.replacement.increase_first", "glint.assist.banking.LIMIT_FIRST", "a credit limit "
                    "increase was submitted and not decided yet", "Approve or deny the submitted credit limit "
                    "increase first, then order the replacement.", details, consequence=procedure.INCREASE_FIRST,
                    facts=[f"undecided increase requests for {card}: {', '.join(map(str, pending))}"])
    return allow("banking.replacement", "glint.assist.banking.LIMIT_FIRST", details)


# ---- registry -----------------------------------------------------------------------------------------------------

def _no_rule(tool: str):
    def gate(db, a: dict) -> GateResult:
        return allow(NO_RULE, "glint.gates.rules.NO_RULE_YET", reason=f"no rule covers {tool} yet: "
                     f"{NO_RULE_REASONS.get(tool, 'not written')}")
    return gate


def _register_banking(reg: GateRegistry) -> None:
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
    debit_gates = ((debit.FREEZE, debit.freeze, "_026: only an ACTIVE card is frozen."),
                   (debit.UNFREEZE, debit.unfreeze, "_026: only a FROZEN card on an OPEN account is unfrozen."),
                   (debit.CLOSE, debit.close, ("_025: no pending transactions, the 14-day card age, a "
                    "frozen card unfrozen first.")),
                   (debit.ORDER, debit.order, ("_023/_029: fees, the replacement limit and the entry-tier wait by "
                    "checking tier.")))
    for tool, fn, purpose in debit_gates:
        reg.register(Gate(f"debit_{tool}_gate", (tool,), f"Knowledge base bank_accounts_(general){purpose}", fn))
    for tool in debit.ACTIVATE:
        reg.register(Gate(f"debit_{tool}_gate", (tool,), "Knowledge base bank_accounts_(general)_024: the activation "
                          "variant follows the card's issue_reason; the card is PENDING and unexpired.",
                          debit.activate(tool)))
    others = (("log_verification", bank_rules.log_verification, ("Verification records the user's details as on "
               "file at the domain's current time.")),
              ("change_user_email", bank_rules.change_email, "An email change sets a different address."),
              ("update_transaction_rewards_3847", bank_rules.update_rewards, ("credit_cards_(general)_004: rewards "
               "are corrected only after an approved cash back dispute.")),
              ("apply_checking_account_credit_5829", bank_rules.checking_credit, ("bank_accounts_(general)_017: one "
               "credit per account per interaction, a 14-day cooldown, a fee refund within the fees charged.")),
              ("log_credit_card_closure_reason_4521", bank_rules.log_closure_reason, ("credit_card_account_logistics"
               "_003 step 1: closure eligibility before the reason and retention steps.")),
              ("close_credit_card_account_7834", bank_rules.close_card, ("credit_card_account_logistics_002/_003: "
               "closure eligibility and a logged reason.")),
              ("apply_credit_card_account_flag_6147", bank_rules.account_flag, ("credit_card_account_logistics_003 "
               "step 4: an annual fee waiver answers a logged annual_fee reason, once a year.")),
              ("apply_statement_credit_8472", bank_rules.statement_credit, ("credit_card_account_logistics_003 step 5: "
               "the retention credit by card tier, after a logged reason, once a year.")),
              ("transfer_to_human_agents", procedure.transfer, ("bank_accounts_(general)_042: the transfer reason "
               "tier; a submitted credit limit increase is decided before a transfer.")))
    for tool, fn, purpose in others:
        reg.register(Gate(f"banking_{tool}_gate", (tool,), purpose, fn))
    # procedure gates: every banking write acting for a user, and every write whose procedure names lookups
    reg.register(Gate("procedure_verified_gate", procedure.VERIFIED_WRITES, "Every procedure starts with a verified "
                      "customer: the user's log_verification record from this conversation.", procedure.verified))
    for tool in procedure.REQUIRED_LOOKUPS:
        reg.register(Gate(f"procedure_lookups_{tool}", (tool,), "The lookups the procedure names, recorded in the "
                          "database before this write.", procedure.lookups(tool), toolkit=True))
    reg.register(Gate("procedure_increase_first_gate", (DISPUTE,), "credit_card_account_logistics_007: a submitted "
                      "credit limit increase is decided before a dispute.", procedure.increase_first))
    reg.register(Gate("procedure_card_digits_gate", (DISPUTE,), "A dispute carries the card's last 4 digits as the "
                      "customer's get_card_last_4_digits tool reports them, after the customer was given it.",
                      procedure.card_digits))
    reg.register(Gate("procedure_dispute_plan_gate", (DISPUTE,), "credit_cards_(general)_015 criterion 4: the "
                      "disputes filed first take the provisional credit, so each filing follows the plan_disputes "
                      "plan (order from τ² train gold, tasks 037-041 and 081).", procedure.dispute_plan, toolkit=True))


def build_registry(domain: str) -> GateRegistry:
    reg = GateRegistry()
    if domain == "retail":
        for tool, action in RETAIL_ACTIONS.items():
            reg.register(Gate(f"retail_{action}_gate", (tool,), f"Retail policy for '{action}': order status, items "
                              "of the same product, and the payment methods allowed.", _retail_gate(action)))
        reg.register(Gate("retail_modify_user_address_gate", ("modify_user_address",), "Retail policy: a complete "
                          "default address.", travel.modify_user_address))
    elif domain == "airline":
        purpose = "Airline policy: who may cancel, basic economy, flown segments, bags, passenger count."
        for tool, fn in zip(AIRLINE_GATED, (_airline_cancel, _airline_flights, _airline_baggages,
                                            _airline_passengers)):
            reg.register(Gate(f"airline_{tool}_gate", (tool,), purpose, fn))
        reg.register(Gate("airline_book_reservation_gate", ("book_reservation",), "Airline policy, Book flight: "
                          "passengers, payment method counts, free bag allowance.", travel.book_reservation))
        reg.register(Gate("airline_send_certificate_gate", ("send_certificate",), "Airline policy, Refunds and "
                          "Compensation: who may get a certificate, and for what.", travel.send_certificate))
    elif domain == "banking_knowledge":
        _register_banking(reg)
    else:
        raise ValueError(f"no gates for domain {domain!r}")
    for tool in NO_RULE_YET[domain]:
        reg.register(Gate(f"no_rule_{tool}", (tool,), "Every write is logged; no policy rule covers this one yet.",
                          _no_rule(tool)))
    return reg


def manifest(domain: str, reg: GateRegistry) -> dict:
    return {"domain": domain, "required": sorted(REQUIRED_GATES[domain]), "no_rule_yet": list(NO_RULE_YET[domain]),
            "gated_generic": list(GATED_GENERIC.get(domain, ())),
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
