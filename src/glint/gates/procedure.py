"""Procedure gates for banking: the steps that must come before a write, judged on the DATABASE, never on whether the
agent says or seems to have done them.

- Verification: a write acting for a user needs that user's log_verification record dated today (every procedure in
  the knowledge base starts with "Customer must be verified").
- Required lookups: a write needs the hidden lookups its procedure names, recorded in agent_discoverable_tools. τ²
  records a hidden READ only when the environment's read-log allowlist names it (tau2.runner.build derives the list
  from the task's expected actions). A lookup the environment does not record can never show in the database, so the
  gate requires only lookups the environment records and says which ones it could not see.
- Order: a credit card dispute waits while a credit limit increase is submitted and not decided
  (credit_card_account_logistics_007: a pending dispute fails the increase).
- transfer_to_human_agents: the reason code follows bank_accounts_(general)_042 where the database shows the
  situation, and a submitted increase is decided before the user is handed off.
"""
from __future__ import annotations

from glint.assist.banking import LIMIT_FIRST, TODAY
from glint.gates.bankdb import as_date, owner, recorded_tools, undecided_increases, verified_today
from glint.gates.core import GateResult, allow, deny

DISPUTES_HISTORY, PENDING_ORDERS, CLOSURE_HISTORY = ("get_user_dispute_history_7291",
                                                      "get_pending_replacement_orders_5765",
                                                      "get_closure_reason_history_8293")
CLI_HISTORY, PAYMENTS = "get_credit_limit_increase_history_4829", "get_payment_history_6183"
ACCOUNTS, TRANSACTIONS, CARDS, DEBIT_DISPUTES = ("get_all_user_accounts_by_user_id_3847",
                                                 "get_bank_account_transactions_9173",
                                                 "get_debit_cards_by_account_id_7823", "get_debit_dispute_status_7483")
CLOSURE_LOOKUPS = ((CLOSURE_HISTORY, "credit_card_account_logistics_003 step 2"),
                   (PENDING_ORDERS, "credit_card_replacements_005 / logistics_003 step 1"),
                   (DISPUTES_HISTORY, "credit_cards_(general)_016 / logistics_003 step 1"))
CLI_LOOKUPS = ((DISPUTES_HISTORY, "credit_card_account_logistics_007"), (PENDING_ORDERS, "logistics_007"),
               (CLI_HISTORY, "logistics_007"), (PAYMENTS, "logistics_006"))
BANK_CREDIT_LOOKUPS = ((ACCOUNTS, "bank_accounts_(general)_043/_044"),
                       (TRANSACTIONS, "bank_accounts_(general)_017/_043/_044"))
# write -> the hidden lookups its procedure requires before it, each with its source document
REQUIRED_LOOKUPS = {
    "file_credit_card_transaction_dispute_4829": ((DISPUTES_HISTORY, "credit_cards_(general)_015/_016"),),
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
    "file_debit_card_transaction_dispute_6281": ((TRANSACTIONS, "bank_accounts_(general)_031 argument 1"),
                                                 (DEBIT_DISPUTES, "bank_accounts_(general)_031 requirement 4 / _035")),
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
    return deny("procedure.verified", source, f"user {user} has no verification logged in this conversation",
                "Verify the user (2 of: date of birth, email, phone, address), log it with log_verification "
                "(time_verified from get_current_time), then retry this write.", {"user_id": user})


def lookups(tool: str):
    required = REQUIRED_LOOKUPS[tool]

    def gate(toolkit, a: dict) -> GateResult:
        source = "; ".join(sorted({s for _, s in required}))
        recordable = set(getattr(toolkit, "_read_log_allowlist", set()) or set())
        done = recorded_tools(toolkit.db)
        checkable = [r for r, _ in required if r in recordable]
        missing = [r for r in checkable if r not in done]
        details = {"required": [r for r, _ in required], "recorded": sorted(done & {r for r, _ in required}),
                   "not_recordable": [r for r, _ in required if r not in recordable]}
        if missing:
            return deny("procedure.lookups", source, f"{tool} needs the lookup(s) {missing} first; the database has "
                        "no record of them in this conversation",
                        f"Unlock and call {', '.join(missing)} (call_discoverable_agent_tool), check the result, then "
                        "retry this write.", details | {"missing": missing})
        return allow("procedure.lookups", source, details)
    return gate


def increase_first(db, a: dict) -> GateResult:
    source = "credit_card_account_logistics_007"
    user = owner(db, a)
    pending = undecided_increases(db, user) if user else []
    if pending:
        return deny("procedure.increase_first", source, LIMIT_FIRST,
                    "Approve or deny the submitted credit limit increase first, then file the dispute.",
                    {"undecided_increase_requests": pending})
    return allow("procedure.increase_first", source, {"undecided_increase_requests": []})


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


def transfer(db, a: dict) -> GateResult:
    source = "bank_accounts_(general)_042 (transfer reason tiers); credit_card_account_logistics_007"
    reason = a.get("reason", "other")
    pending = undecided_increases(db)
    if pending:
        return deny("procedure.transfer.increase_undecided", source, f"credit limit increase request(s) {pending} "
                    "were submitted and not decided; a transfer would leave them open",
                    "Finish the increase first (the required lookups, then approve or deny), then transfer if the "
                    "user still asks.", {"undecided_increase_requests": pending})
    fraud = fraud_today(db)
    if fraud and reason in DISPOSITION:
        return deny("procedure.transfer.reason", source, f"reason '{reason}' is a tier-3 disposition code, but this "
                    f"conversation handled fraud or theft ({fraud}); _042 says to use the highest tier that applies",
                    "Transfer with reason 'fraud_or_security_concern'.", {"fraud": fraud, "reason": reason})
    return allow("procedure.transfer", source, {"reason": reason, "fraud": fraud})
