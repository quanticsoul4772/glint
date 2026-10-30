"""Procedure gates for banking: the steps that must come before a write, judged on the DATABASE, never on whether the
agent says or seems to have done them.

- Verification: a write acting for a user needs that user's log_verification record dated today (every procedure in
  the knowledge base starts with "Customer must be verified").
- Required lookups: a write needs the hidden lookups its knowledge base procedure names, on the enforcer's own
  record of the hidden reads the agent ran in this conversation (`toolkit.glint_reads`, kept by
  glint.gates.enforce). Nothing here reads τ²'s read-log allowlist or the task's expected actions. Each requirement
  cites its document, and τ²'s TRAIN-task gold conversations all do the lookup before the write.
- Order: a credit card dispute waits while a credit limit increase is submitted and not decided
  (credit_card_account_logistics_007: a pending dispute fails the increase).
- transfer_to_human_agents: the reason code follows bank_accounts_(general)_042 where the database shows the
  situation, and a submitted increase is decided before the user is handed off.
- Dispute plan: a credit card dispute follows the user's current plan_disputes plan (its order, reason, merchant
  answer and provisional credit flag), on the enforcer's own record of the plans made in this conversation
  (`toolkit.glint_plans`, kept by glint.gates.enforce). Provisional credit needs at most 2 disputes in the past 12
  months (credit_cards_(general)_015 criterion 4), so the order decides which disputes get it; the order itself comes
  from τ²'s train gold (tasks 037-041 and 081), not from the policy text.
"""
from __future__ import annotations

import json
import re

from glint.assist import as_flag
from glint.assist.banking import CARD_DIGITS_TOOL, OVERCHARGE, PLAN_TOOL, TODAY, card_of_transaction, replacements_today
from glint.gates.bankdb import as_date, money, owner, undecided_increases, verified_today
from glint.gates.core import GateResult, allow, deny

DISPUTES_HISTORY, PENDING_ORDERS, CLOSURE_HISTORY = ("get_user_dispute_history_7291",
                                                      "get_pending_replacement_orders_5765",
                                                      "get_closure_reason_history_8293")
CLI_HISTORY, PAYMENTS = "get_credit_limit_increase_history_4829", "get_payment_history_6183"
ACCOUNTS, TRANSACTIONS, CARDS = ("get_all_user_accounts_by_user_id_3847", "get_bank_account_transactions_9173",
                                 "get_debit_cards_by_account_id_7823")
CLOSURE_LOOKUPS = ((CLOSURE_HISTORY, "credit_card_account_logistics_003 step 2"),
                   (PENDING_ORDERS, "credit_card_replacements_005 / logistics_003 step 1"),
                   (DISPUTES_HISTORY, "credit_cards_(general)_016 / logistics_003 step 1"))
CLI_LOOKUPS = ((DISPUTES_HISTORY, "credit_card_account_logistics_007"), (PENDING_ORDERS, "logistics_007"),
               (CLI_HISTORY, "logistics_007"), (PAYMENTS, "logistics_006"))
BANK_CREDIT_LOOKUPS = ((ACCOUNTS, "bank_accounts_(general)_043/_044"),
                       (TRANSACTIONS, "bank_accounts_(general)_017/_043/_044"))
# write -> the hidden lookups its procedure requires before it, each with its source document. Lookups the
# knowledge base mentions without making them a precondition are not required. The credit card dispute requires the
# dispute history: it is criterion 4's count of disputes on record, and τ² logs the read in the database it scores
# where its gold makes it (the read log allowlist), so a filing without it fails those tasks (12 of 37 failed
# banking-plan-v4-gated runs filed without it, 8 failing on it alone, banking.BEFORE_FILING; in banking-plan-v5-gated,
# with the hint, 081 trial 0 and 041 trial 2 still filed before any history read). τ² logs a read under an id made
# from the tool name alone and keeps the first record, so a repeated read changes nothing, and a read gold does not
# make is not logged. τ²'s train gold files without it in 031, 037 and
# 038; replays insert it there with the plan call (glint.gates.replayplan).
REQUIRED_LOOKUPS = {
    "approve_credit_limit_increase_5847": CLI_LOOKUPS,
    "deny_credit_limit_increase_5848": CLI_LOOKUPS,
    "log_credit_card_closure_reason_4521": CLOSURE_LOOKUPS,
    "close_credit_card_account_7834": CLOSURE_LOOKUPS,
    "apply_statement_credit_8472": CLOSURE_LOOKUPS,
    "apply_checking_account_credit_5829": BANK_CREDIT_LOOKUPS,
    "apply_savings_account_credit_6831": BANK_CREDIT_LOOKUPS,
    "submit_interest_discrepancy_report_7294": BANK_CREDIT_LOOKUPS,
    "close_debit_card_4721": ((TRANSACTIONS, "bank_accounts_(general)_025 requirements 4-5"),),
    "order_debit_card_5739": ((CARDS, "bank_accounts_(general)_029 replacement history"),
                              (ACCOUNTS, "bank_accounts_(general)_023 eligibility")),
    "file_debit_card_transaction_dispute_6281": ((TRANSACTIONS, "bank_accounts_(general)_031 argument 1"),),
    "file_credit_card_transaction_dispute_4829": ((DISPUTES_HISTORY, "credit_cards_(general)_015 criterion 4"),),
}
# writes that act for a user and so need that user verified today (the knowledge base's first step everywhere)
VERIFIED_WRITES = (
    "change_user_email", "update_transaction_rewards_3847", "file_credit_card_transaction_dispute_4829",
    "file_debit_card_transaction_dispute_6281", "set_debit_card_recurring_block_7382",
    "order_replacement_credit_card_7291", "log_credit_card_closure_reason_4521", "apply_statement_credit_8472",
    "apply_credit_card_account_flag_6147", "close_credit_card_account_7834", "pay_credit_card_from_checking_9182",
    "submit_credit_limit_increase_request_7392", "approve_credit_limit_increase_5847",
    "deny_credit_limit_increase_5848", "open_bank_account_4821", "close_bank_account_7392",
    "transfer_funds_between_bank_accounts_7291", "apply_checking_account_credit_5829",
    "apply_savings_account_credit_6831", "submit_interest_discrepancy_report_7294", "order_debit_card_5739",
    "activate_debit_card_8291", "activate_debit_card_8292", "activate_debit_card_8293", "close_debit_card_4721",
    "freeze_debit_card_3892", "unfreeze_debit_card_3893", "clear_debit_card_fraud_alert_4892",
    "reset_debit_card_pin_6284", "change_debit_card_pin_6285", "request_temporary_debit_card_limit_increase_8374",
)


def verified(db, a: dict) -> GateResult:
    source = "knowledge base: 'Customer must be verified' (every procedure's first step)"
    user = owner(db, a)
    if user is None:
        return allow("procedure.verified.no_user", source, reason="the write names no user, card or account the "
                     "database knows: τ²'s tool rejects or ignores it")
    if verified_today(db, user):
        return allow("procedure.verified", source, {"user_id": user})
    return deny("procedure.verified", source, f"the customer acting on {user}'s accounts has not been verified in "
                "this conversation", "Verify the user (2 of: date of birth, email, phone, address), log it with "
                "log_verification (time_verified from get_current_time), then retry this write.", {"user_id": user},
                consequence="Every account change for this customer is blocked until a verification is logged. "
                "One verification covers the rest of the conversation.",
                facts=[f"the write acts for user {user}", f"no log_verification record for {user} dated today"])


def lookups(tool: str):
    required = REQUIRED_LOOKUPS[tool]

    def gate(toolkit, a: dict) -> GateResult:
        source = "; ".join(sorted({s for _, s in required}))
        done = set(getattr(toolkit, "glint_reads", set()))
        missing = [r for r, _ in required if r not in done]
        details = {"required": [r for r, _ in required], "done": sorted(done & {r for r, _ in required})}
        if missing:
            return deny("procedure.lookups", source, "the procedure for this write starts with lookups not done "
                        f"in this conversation: {', '.join(missing)}", f"Unlock and call {', '.join(missing)} "
                        "(call_discoverable_agent_tool), check the result, then retry this write.",
                        details | {"missing": missing},
                        consequence="The lookups can change the decision itself (an open dispute or a pending "
                        "order can make the request ineligible), so read their results before retrying, and tell the "
                        "customer if they change the answer.",
                        facts=[f"required before {tool}: {', '.join(r for r, _ in required)} ({source})",
                               f"done in this conversation: {', '.join(details['done']) or 'none'}",
                               f"missing: {', '.join(missing)}"])
        return allow("procedure.lookups", source, details)
    return gate


CARD_SOURCE = "credit_cards_(general)_014 card_last_4_digits; the account record; the customer's card-digits tool"


def card_digits(db, a: dict) -> GateResult:
    """A credit card dispute carries the card's last 4 digits from a source the agent can see: the card's account
    record when the task puts them there (054, 080, 081; the account lookups show them), else the customer, who
    looks them up with get_card_last_4_digits once the agent gives it (031, 037-041, 053, as gold does). Glimmer filed
    card names or '0000' when it never gave the tool (banking-plan-v2-gated, the only difference in 6 runs,
    2026-10-29). The gate knows only what the agent can see: the tool was given, and the account's digits. The
    customer-side digits are not checked (only τ²'s own code knows them), and no block names a digit (2026-10-29:
    the first version computed them from τ²'s source and printed them)."""
    user, tid = str(a.get("user_id", "")), str(a.get("transaction_id", ""))
    txn = db.credit_card_transaction_history.data.get(tid)
    if txn is None or txn.get("user_id") != user:
        return allow("procedure.card_digits.unknown_transaction", CARD_SOURCE,
                     reason="no such transaction for this user: the dispute gate and τ²'s tool reject it")
    card = card_of_transaction(db, user, txn)
    on_file = (db.credit_card_accounts.data.get(card) or {}).get("card_last_4_digits")
    filed = str(a.get("card_last_4_digits") or "").strip()
    if on_file:
        details = {"credit_card_account_id": card, "on_file": True}
        if filed != str(on_file):
            return deny("procedure.card_digits.value", CARD_SOURCE, "the card digits are not the ones on the card's "
                        "account record", f"Look up the card's account ({card}) and file with the "
                        "card_last_4_digits its record shows.", details,
                        consequence="A dispute filed with the wrong card digits is a dispute on the wrong card.",
                        facts=[f"card_last_4_digits given: {filed!r}", f"{card}: its record carries the digits"])
        return allow("procedure.card_digits", CARD_SOURCE, details)
    given = any(r.get("tool_name") == CARD_DIGITS_TOOL for r in db.user_discoverable_tools.data.values())
    details = {"credit_card_account_id": card, "given": given}
    if not given:
        return deny("procedure.card_digits.give", CARD_SOURCE, "the customer was not given the tool that reports the "
                    "card's last 4 digits", f"Give the customer {CARD_DIGITS_TOOL} (give_discoverable_user_tool, "
                    f"arguments {{\"credit_card_account_id\": \"{card}\"}}), ask them to run it, then file with the "
                    "digits it reports.", details,
                    consequence="The digits are the customer's to look up; filing a guess files a dispute on the wrong "
                    "card. One tool covers every dispute on this card.",
                    facts=[f"{CARD_DIGITS_TOOL}: not given in this conversation", f"card: {card}",
                           "the card's account record carries no digits"])
    # a card's last 4 digits are 4 numerals: Glimmer filed "XXXX" once the tool was given (banking-plan-v5-gated
    # 041 t2, 2026-10-30), which the given-tool check let through (digits on file are held to their value above)
    if not re.fullmatch(r"\d{4}", filed):
        return deny("procedure.card_digits.format", CARD_SOURCE, "the card digits are not 4 numerals",
                    f"File with the 4 digits the customer's {CARD_DIGITS_TOOL} reports; ask the customer to run it "
                    "if they have not.", details,
                    consequence="A dispute filed without the card's digits is a dispute on no card.",
                    facts=[f"card_last_4_digits given: {filed!r}", f"{CARD_DIGITS_TOOL}: given"])
    return allow("procedure.card_digits", CARD_SOURCE, details)


def _meant_cards(toolkit, a: dict) -> set[str]:
    """The credit cards a give of the digits tool is for: the card its arguments name; else the cards of the
    transactions in this conversation's dispute plans; else the cards of the users verified today. Gold gives the
    tool with no arguments (031, 037-041, 053), so the arguments alone cannot say."""
    db = toolkit.db
    try:
        named = json.loads(a.get("arguments") or "{}").get("credit_card_account_id")
    except (ValueError, AttributeError):
        named = None
    if named:
        return {str(named)}
    planned = set()
    for user, plan in getattr(toolkit, "glint_plans", {}).items():
        for entry in plan.get("order", []):
            txn = db.credit_card_transaction_history.data.get(entry["transaction_id"])
            if txn is not None:
                planned.add(card_of_transaction(db, user, txn))
    if planned:
        return planned
    return {cid for cid, acc in db.credit_card_accounts.data.items() if verified_today(db, str(acc.get("user_id")))}


GIVE_SOURCE = "credit_cards_(general)_014 card_last_4_digits; the account record"


def give_card_digits(toolkit, a: dict) -> GateResult:
    """The customer gets the card-digits tool only for a card whose account record does not show the digits. When
    the record shows them (054, 2026-10-30: Glimmer gave the tool anyway in 2 of 4 runs), the agent files those, and
    the extra tool record in the database is a step the procedure does not have. Other user tools are allowed."""
    tool = str(a.get("discoverable_tool_name", ""))
    if tool != CARD_DIGITS_TOOL:
        return allow("banking.give_tool", GIVE_SOURCE, {"discoverable_tool_name": tool},
                     reason="no rule covers giving this tool")
    db, cards = toolkit.db, _meant_cards(toolkit, a)
    if not cards:
        return deny("banking.give_card_digits.unknown_card", GIVE_SOURCE, "no card this tool could be for: no card "
                    "named, no dispute plan, no customer verified today", "Verify the customer, plan the disputes "
                    f"({PLAN_TOOL}), then give {CARD_DIGITS_TOOL} for a card whose record shows no digits.",
                    {"discoverable_tool_name": tool},
                    consequence="The tool reports one card's digits; without a card it reports nothing usable.",
                    facts=["no credit_card_account_id in the arguments", "no dispute plan in this conversation",
                           "no customer verified today"])
    on_file = sorted(c for c in cards if (db.credit_card_accounts.data.get(c) or {}).get("card_last_4_digits"))
    details = {"cards": sorted(cards), "on_file": on_file}
    if len(on_file) == len(cards):
        return deny("banking.give_card_digits.on_file", GIVE_SOURCE, "the card's digits are on its account record",
                    "Do not give the tool: file the disputes with the card_last_4_digits the card's account record "
                    "shows (the dispute plan lists them).", details,
                    consequence="The customer would be asked for digits the bank already has on file.",
                    facts=[f"{c}: its record carries the digits" for c in on_file])
    return allow("banking.give_card_digits", GIVE_SOURCE, details)


def increase_first(toolkit, a: dict) -> GateResult:
    source = "credit_card_account_logistics_007"
    user = owner(toolkit.db, a)
    pending = undecided_increases(toolkit, user) if user else []
    if pending:
        return deny("procedure.increase_first", source, "a credit limit increase was submitted and not decided yet",
                    "Approve or deny the submitted credit limit increase first, then file the dispute.",
                    {"undecided_increase_requests": pending}, consequence=INCREASE_FIRST,
                    facts=[f"undecided increase requests for {user}: {', '.join(map(str, pending))}"])
    return allow("procedure.increase_first", source, {"undecided_increase_requests": []})


INCREASE_UNDECIDED = ("credit_card_account_logistics_005 (utilization at the request), _006 (payment history), _007 "
                      "(submit, the checks, then the decision, in that order)")


def increase_undecided(toolkit, a: dict) -> GateResult:
    """Any other change for a customer whose submitted credit limit increase is undecided: the checks read the account
    as submitted (task_051: a payment from checking before the decision lowered the utilization the denial reads)."""
    user = owner(toolkit.db, a)
    pending = undecided_increases(toolkit, user) if user else []
    if pending:
        return deny("procedure.increase_undecided", INCREASE_UNDECIDED, "a credit limit increase was submitted for "
                    "this customer and not decided yet", "Finish the increase first (the required lookups, then "
                    "approve or deny), then retry this write.", {"undecided_increase_requests": pending},
                    consequence="The increase is decided on the account as it was submitted; a payment, transfer or "
                    "other change before the decision changes what the checks read. Every other change for this "
                    "customer is blocked until the increase is approved or denied.",
                    facts=[f"undecided increase requests for {user}: {', '.join(map(str, pending))}"])
    return allow("procedure.increase_undecided", INCREASE_UNDECIDED, {"undecided_increase_requests": []})


DISPOSITION = ("customer_frustrated_demands_human", "supervisor_request_service_complaint",
               "customer_requests_human_no_specific_reason", "request_completed_customer_wants_human_followup")
FRAUD_DISPUTE = ("unauthorized_fraudulent_charge",)
FRAUD_DEBIT = ("card_present_fraud", "card_not_present_fraud")
FRAUD_CLOSURE = ("stolen", "fraud_suspected")
FRAUD_ORDER = ("stolen", "fraud_suspected")


def fraud_today(db) -> list[str]:
    """Fraud handled in this conversation, as the database shows it: a fraud dispute filed today, a debit card closed
    today as stolen or fraud_suspected, or a credit card replaced today for fraud or theft."""
    seen = [f"credit card dispute {k}" for k, r in db.transaction_disputes.data.items()
            if r.get("dispute_reason") in FRAUD_DISPUTE and as_date(r.get("submitted_at")) == TODAY]
    seen += [f"debit card dispute {k}" for k, r in db.debit_card_disputes.data.items()
             if r.get("dispute_category") in FRAUD_DEBIT and as_date(r.get("submitted_at")) == TODAY]
    seen += [f"debit card {k} closed {r.get('closure_reason')}" for k, r in db.debit_cards.data.items()
             if as_date(r.get("closed_date")) == TODAY and r.get("closure_reason") in FRAUD_CLOSURE]
    seen += [f"credit card order {k} ({r.get('reason')})" for k, r in db.credit_card_orders.data.items()
             if as_date(r.get("order_date")) == TODAY and r.get("reason") in FRAUD_ORDER]
    return seen


def transfer(toolkit, a: dict) -> GateResult:
    source = "bank_accounts_(general)_042 (transfer reason tiers); credit_card_account_logistics_007"
    reason = a.get("reason", "other")
    db, pending = toolkit.db, undecided_increases(toolkit)
    if pending:
        return deny("procedure.transfer.increase_undecided", source, "a submitted credit limit increase is not "
                    "decided yet, and a transfer would leave it open", "Finish the increase first (the required "
                    "lookups, then approve or deny), then transfer if the user still asks.",
                    {"undecided_increase_requests": pending},
                    consequence="The conversation would end with the customer's request unanswered. Tell the "
                    "customer the decision, then transfer if they still want a human agent.",
                    facts=[f"undecided increase requests: {', '.join(map(str, pending))}"])
    fraud = fraud_today(db)
    if fraud and reason in DISPOSITION:
        return deny("procedure.transfer.reason", source, "this conversation handled fraud or theft, so the transfer "
                    "carries the highest reason tier that applies, the fraud reason", "Transfer with reason "
                    "'fraud_or_security_concern'.", {"fraud": fraud, "reason": reason},
                    consequence="The reason code routes the customer to the right team; a fraud case sent with a "
                    "general reason reaches the wrong queue.",
                    facts=[f"reason given: '{reason}' (a tier-3 disposition code)",
                           *[f"handled in this conversation: {f}" for f in fraud]])
    return allow("procedure.transfer", source, {"reason": reason, "fraud": fraud})


PLAN_SOURCE = "credit_cards_(general)_015 criterion 4; filing order from τ² train gold (037-041, 081)"
# credit_cards_(general)_015 criterion 4: "has not filed more than 2 disputes in the past 12 months", counted before
# the dispute being filed (glint.assist.banking.check_dispute: len(filed) + planned_before <= 2)
CREDIT_ORDER = ("A dispute can get provisional credit only while the customer has 2 or fewer other disputes filed in "
                "the past 12 months, so the filing order decides which disputes get it, and a filed dispute cannot be "
                "taken back.")
INCREASE_FIRST = ("A pending dispute or replacement card makes the increase fail, so the increase is decided first; "
                  "the disputes and the replacement can follow once it is decided.")
REPLAN = (f"If the customer's list changed, call {PLAN_TOOL} again with every dispute not filed yet (it replaces the "
          "plan).")


def count_fact(db, user: str) -> str:
    """The count criterion 4 reads (glint.assist.banking.check_dispute): this user's disputes submitted in the past
    12 months, with the ones filed in this conversation named."""
    year_ago = TODAY.replace(year=TODAY.year - 1)
    rows = [r for r in db.transaction_disputes.data.values() if r.get("user_id") == user
            and (d := as_date(r.get("submitted_at"))) is not None and d > year_ago]
    today = [str(r.get("transaction_id")) for r in rows if as_date(r.get("submitted_at")) == TODAY]
    out = f"disputes counted for {user} in the past 12 months: {len(rows)}"
    return out + (f" ({len(today)} filed in this conversation: {', '.join(today)})" if today else "")


def dispute_plan(toolkit, a: dict) -> GateResult:
    """A credit card dispute must be the next filing of the user's current plan, with the plan's values.

    Every dispute needs a plan, a single one included: the gate sees the database and this call, not the customer's
    request, so it cannot tell a one-dispute request from the first of several, and a dispute filed out of order
    cannot be taken back (its credit slot is used). A one-transaction plan is one call and gives the flag
    check_dispute gives. A transaction that already has a dispute on record is allowed through: τ²'s tool rejects a
    second filing itself and nothing changes."""
    db, user, tid = toolkit.db, str(a.get("user_id", "")), str(a.get("transaction_id", ""))
    if any(r.get("user_id") == user and r.get("transaction_id") == tid for r in db.transaction_disputes.data.values()):
        return allow("procedure.dispute_plan.already_filed", PLAN_SOURCE, {"transaction_id": tid},
                     reason="this transaction already has a dispute on record: τ²'s tool rejects a second filing")
    plan = getattr(toolkit, "glint_plans", {}).get(user)
    if plan is None:
        return deny("procedure.dispute_plan.none", PLAN_SOURCE, "no dispute plan was made for this customer in this "
                    "conversation", f"Call {PLAN_TOOL}(user_id, transactions=[{{transaction_id, dispute_reason, "
                    "contacted_merchant, issue_noticed_date, card_action} for EVERY transaction the customer wants disputed]), then "
                    "file the "
                    "disputes in its order with its eligible_for_provisional_credit values.",
                    {"user_id": user, "transaction_id": tid},
                    consequence=f"{CREDIT_ORDER} Plan every dispute the customer asked for in one call.",
                    facts=[f"user {user} has no {PLAN_TOOL} result in this conversation", count_fact(db, user)])
    filed = {r.get("transaction_id") for r in db.transaction_disputes.data.values() if r.get("user_id") == user}
    remaining = [e for e in plan["order"] if e["transaction_id"] not in filed]
    details = {"user_id": user, "transaction_id": tid, "plan": [e["transaction_id"] for e in plan["order"]],
               "remaining": [e["transaction_id"] for e in remaining]}
    if tid not in details["plan"]:
        return deny("procedure.dispute_plan.not_planned", PLAN_SOURCE, f"transaction {tid} is not in the current "
                    "dispute plan", f"Call {PLAN_TOOL} again with every dispute not filed yet, this one included, "
                    "and file in its order.", details, consequence=f"{CREDIT_ORDER} A new plan replaces this one.",
                    facts=[f"plan order: {', '.join(details['plan'])}",
                           f"not filed yet: {', '.join(details['remaining']) or 'none'}", count_fact(db, user)])
    # the plan as the agent got it: τ² turns every bool in a tool result into the string "True"/"False"
    nxt = remaining[0] | {"eligible_for_provisional_credit": as_flag(remaining[0]["eligible_for_provisional_credit"])}
    if remaining[0]["contacted_merchant"] is not None:
        nxt["contacted_merchant"] = as_flag(remaining[0]["contacted_merchant"])
    # a replacement ordered today for the card after the plan was made sets cancel_and_reissue, as the plan does when
    # the order comes first (glint.assist.banking.card_actions); the plan's older answer would contradict
    # banking.dispute, which requires cancel_and_reissue once the card is being replaced
    txn = db.credit_card_transaction_history.data.get(nxt["transaction_id"]) or {}  # the next entry's own card
    if replacements_today(db, card_of_transaction(db, user, txn)):
        nxt["card_action"] = "cancel_and_reissue"
    refund = (f", resolution_requested {nxt['resolution_requested']!r}" if nxt.get("resolution_requested") else "")
    if nxt.get("partial_refund_amount") is not None:
        refund += f", partial_refund_amount {nxt['partial_refund_amount']}"
    if nxt.get("issue_noticed_date"):
        refund += f", issue_noticed_date {nxt['issue_noticed_date']!r}"
    if nxt.get("card_action"):
        refund += f", card_action {nxt['card_action']!r}"
    fix_next = (f"File {nxt['transaction_id']} next (dispute_reason {nxt['dispute_reason']!r}, "
                f"eligible_for_provisional_credit {nxt['eligible_for_provisional_credit']}{refund}). {REPLAN}")
    if tid != nxt["transaction_id"]:
        return deny("procedure.dispute_plan.order", PLAN_SOURCE, f"the plan files {nxt['transaction_id']} before "
                    f"{tid}", fix_next, details | {"next": nxt}, consequence=CREDIT_ORDER,
                    facts=[f"plan order: {', '.join(details['plan'])}",
                           f"not filed yet: {', '.join(details['remaining'])}", count_fact(db, user)])
    reasons = []
    if a.get("dispute_reason") != nxt["dispute_reason"]:
        reasons.append(f"dispute_reason {a.get('dispute_reason')!r} != the plan's {nxt['dispute_reason']!r}")
    if nxt["dispute_reason"] == OVERCHARGE and not nxt.get("resolution_requested"):
        return deny("procedure.dispute_plan.ask", PLAN_SOURCE, "the plan was made without the amount the customer "
                    f"should have been charged for {tid}", "Ask the customer what amount they should have been charged, "
                    f"then call {PLAN_TOOL} again with expected_amount (or partial_refund_amount, the overcharge) for "
                    "it.", details | {"next": nxt},
                    consequence="An incorrect_amount dispute is filed as a partial refund of the overcharge unless the "
                    "customer asks for a full refund (then plan resolution_requested 'full_refund'), so its amount "
                    "comes from the customer. Ask about every incorrect_amount dispute at once, without suggesting "
                    "either refund.", facts=[f"{tid}: dispute_reason incorrect_amount, no resolution in the plan"])
    if nxt.get("resolution_requested"):
        if a.get("resolution_requested") != nxt["resolution_requested"]:
            reasons.append(f"resolution_requested {a.get('resolution_requested')!r} != the plan's "
                           f"{nxt['resolution_requested']!r}")
        # the plan as the agent got it: τ² turns every value in a tool result into a string ("50.0")
        if money(a.get("partial_refund_amount")) != money(nxt.get("partial_refund_amount")):
            reasons.append(f"partial_refund_amount {a.get('partial_refund_amount')} != the plan's "
                           f"{nxt.get('partial_refund_amount')}")
    if nxt["contacted_merchant"] is None:
        return deny("procedure.dispute_plan.ask", PLAN_SOURCE, "the plan was made without the customer's answer on "
                    f"whether they contacted the merchant about {tid}", "Ask the customer whether they tried the "
                    f"merchant first, then call {PLAN_TOOL} again with contacted_merchant set.",
                    details | {"next": nxt},
                    consequence="Provisional credit depends on that answer (criterion 5), so the plan's credit "
                    "values are not final until it is given. Ask about every planned dispute missing it at once.",
                    facts=[f"{tid}: contacted_merchant not given in the plan"])
    if as_flag(a.get("contacted_merchant")) != nxt["contacted_merchant"]:
        reasons.append(f"contacted_merchant {a.get('contacted_merchant')} != the plan's {nxt['contacted_merchant']}")
    if nxt.get("issue_noticed_date") is None:
        return deny("procedure.dispute_plan.ask", PLAN_SOURCE, "the plan was made without the date the customer "
                    f"first noticed the {tid} charge", "Ask the customer when they first noticed the charge, then "
                    f"call {PLAN_TOOL} again with issue_noticed_date (MM/DD/YYYY) for it.", details | {"next": nxt},
                    consequence="The dispute records the date the customer gives (credit_cards_(general)_014 field "
                    "11); today's date is right only when the customer says they noticed it today. Ask about every "
                    "planned dispute missing it at once.",
                    facts=[f"{tid}: issue_noticed_date not given in the plan"])
    if nxt.get("card_action") is None:
        return deny("procedure.dispute_plan.ask", PLAN_SOURCE, "the plan was made without the customer's answer on "
                    f"whether to keep the card of {tid} active or cancel it and get a new one", "Ask the customer "
                    f"whether to keep the card active or cancel it and get a new one, then call {PLAN_TOOL} again "
                    "with card_action ('keep_active' or 'cancel_and_reissue') on every dispute for that card.",
                    details | {"next": nxt},
                    consequence="The dispute records the customer's choice for the card (credit_cards_(general)_014 "
                    "card_action), and cancel_and_reissue comes with a replacement order. One answer covers every "
                    "dispute on the card.", facts=[f"{tid}: card_action not given in the plan"])
    if a.get("card_action") != nxt["card_action"]:
        reasons.append(f"card_action {a.get('card_action')!r} != the plan's {nxt['card_action']!r}")
    if a.get("issue_noticed_date") != nxt["issue_noticed_date"]:
        reasons.append(f"issue_noticed_date {a.get('issue_noticed_date')!r} != the plan's "
                       f"{nxt['issue_noticed_date']!r}")
    if as_flag(a.get("eligible_for_provisional_credit")) != nxt["eligible_for_provisional_credit"]:
        reasons.append(f"eligible_for_provisional_credit {a.get('eligible_for_provisional_credit')} != the plan's "
                       f"{nxt['eligible_for_provisional_credit']}")
    if reasons:
        return deny("procedure.dispute_plan.values", PLAN_SOURCE, f"the filing for {tid} does not carry the plan's "
                    "values", fix_next, details | {"next": nxt}, consequence=CREDIT_ORDER,
                    facts=[*reasons, count_fact(db, user)])
    return allow("procedure.dispute_plan", PLAN_SOURCE, details | {"position": nxt["position"]})
