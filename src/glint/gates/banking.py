"""Banking write rules beyond the credit card dispute and credit limit increase (rules.py) and the debit cards
(debit.py). Each rule cites its knowledge base document.

- log_verification: the user's record on file and the domain clock (get_current_time).
- change_user_email: a new address that differs from the one on file.
- update_transaction_rewards_3847: only for a transaction whose cash back dispute is resolved and approved
  (credit_cards_(general)_004); otherwise the user files the dispute with their own tool.
- apply_checking_account_credit_5829: once per account per conversation, a 14-day cooldown, and a fee refund no
  larger than the fees on the account (bank_accounts_(general)_017).
- Credit card closure (credit_card_account_logistics_002/_003): the eligibility checks before a reason is logged or
  the account is closed; a logged reason before a closure, a fee waiver or a retention credit; the retention credit
  amount by card tier.
"""
from __future__ import annotations

from glint.assist.banking import FINAL_ORDER, OPEN_DISPUTE, TODAY, _tier
from glint.gates.bankdb import NOW_TEXT, as_date, days_since, money, within_year
from glint.gates.core import GateResult, allow, deny

VERIFY_FIELDS = (("name", "name"), ("address", "address"), ("email", "email"), ("phone_number", "phone_number"),
                 ("date_of_birth", "date_of_birth"))
RETENTION_CREDIT = {"entry": 5.0, "mid": 20.0, "premium": 50.0, "elite": 50.0, "invitation": 50.0}
CREDIT_TYPES = ("rebate_credit", "fee_refund")
FEE_TYPES = ("atm_fee", "fee", "debit_card_fee", "monthly_fee", "overdraft_fee")


def _norm(value) -> str:
    return " ".join(str(value or "").lower().split())


def log_verification(db, a: dict) -> GateResult:
    source = "log_verification docstring; get_current_time"
    user = db.users.data.get(a.get("user_id", ""))
    if user is None:
        return deny("banking.verification.user", source, f"no user {a.get('user_id')!r} on file",
                    "Look the user up (get_user_information_by_*) and log the verification with their user_id.",
                    {"user_id": a.get("user_id")})
    wrong = [arg for arg, field in VERIFY_FIELDS if arg in a and _norm(a[arg]) != _norm(user.get(field))]
    reasons = [f"{arg} {a[arg]!r} differs from the record on file" for arg in wrong]
    if str(a.get("time_verified", "")).strip() != NOW_TEXT:
        reasons.append(f"time_verified {a.get('time_verified')!r} is not the current time {NOW_TEXT!r}")
    if reasons:
        return deny("banking.verification", source, "; ".join(reasons),
                    "Log the verification with the user's details exactly as on file and time_verified from "
                    "get_current_time.", {"mismatched": wrong})
    return allow("banking.verification", source, {"user_id": a["user_id"]})


def change_email(db, a: dict) -> GateResult:
    source = "change_user_email docstring"
    user = db.users.data.get(a.get("user_id", ""))
    if user is not None and _norm(a.get("new_email")) == _norm(user.get("email")):
        return deny("banking.change_email.same", source, "the new email is already the email on file",
                    "Do not change it; tell the user the email on file already is that address.", {})
    return allow("banking.change_email", source, {"user_id": a.get("user_id")})


def update_rewards(db, a: dict) -> GateResult:
    source = "credit_cards_(general)_004"
    tid = a.get("transaction_id", "")
    disputes = [r for r in db.cash_back_disputes.data.values() if r.get("transaction_id") == tid]
    approved = [r for r in disputes if str(r.get("status", "")).upper() == "RESOLVED"
                and str(r.get("resolution", "")).upper() == "APPROVED"]
    details = {"transaction_id": tid, "disputes": [r.get("dispute_id") for r in disputes]}
    if not approved:
        return deny("banking.update_rewards.no_resolved_dispute", source, f"transaction {tid} has no resolved and "
                    "approved cash back dispute; rewards are corrected only after one",
                    "Give the user the submit_cash_back_dispute_0589 tool (give_discoverable_user_tool) so the user "
                    "files the dispute; update rewards only for a transaction whose dispute is RESOLVED/APPROVED.",
                    details)
    return allow("banking.update_rewards", source, details)


def checking_credit(db, a: dict) -> GateResult:
    source = "bank_accounts_(general)_017"
    acc_id = a.get("account_id", "")
    rows = [t for t in db.bank_account_transaction_history.data.values() if t.get("account_id") == acc_id]
    credits = [t for t in rows if t.get("type") in CREDIT_TYPES and (d := days_since(t.get("date"))) is not None
               and 0 <= d < 14]
    details = {"account_id": acc_id, "credits_past_14_days": [t.get("transaction_id") for t in credits]}
    if credits:
        today = [t for t in credits if as_date(t.get("date")) == TODAY]
        why = ("a credit was already applied to this account in this conversation; the tool may be called once per "
               "account per interaction") if today else "a credit was applied to this account within 14 days"
        return deny("banking.checking_credit.once", source, why,
                    "Do not apply another credit to this account. When several corrections apply, combine them into "
                    "ONE credit with the total amount.", details)
    amount = money(a.get("amount"))
    if a.get("credit_type") == "fee_refund":
        fees = round(-sum(money(t.get("amount")) for t in rows if t.get("type") in FEE_TYPES
                          and money(t.get("amount")) < 0), 2)
        details["fees_charged"] = fees
        if amount > fees:
            return deny("banking.checking_credit.amount", source, f"fee refund ${amount:,.2f} exceeds the ${fees:,.2f} "
                        "of fees charged to this account", "Refund only the fees that were charged wrongly (net "
                        "of any undercharges); recompute from the account's transactions.", details)
    return allow("banking.checking_credit", source, details)


def _card_disputes(db, card_id: str) -> list[str]:
    txns = {k for k, t in db.credit_card_transaction_history.data.items() if t.get("credit_card_account_id") == card_id}
    return [r.get("dispute_id") for r in db.transaction_disputes.data.values()
            if r.get("transaction_id") in txns and str(r.get("status", "")).upper() in OPEN_DISPUTE]


def closure_blockers(db, card_id: str) -> list[str]:
    """credit_card_account_logistics_002/_003 step 1: what keeps this credit card account from closing."""
    acc = db.credit_card_accounts.data.get(card_id, {})
    blockers = []
    if disputes := _card_disputes(db, card_id):
        blockers.append(f"pending disputes {disputes}")
    orders = [r.get("order_id") for r in db.credit_card_orders.data.values()
              if r.get("credit_card_account_id") == card_id and str(r.get("status", "")).upper() not in FINAL_ORDER]
    if orders:
        blockers.append(f"replacement card not yet received {orders}")
    age = days_since(acc.get("date_of_account_open"))
    if age is not None and age < 60:
        blockers.append(f"account open {age} days (needs 60)")
    if money(acc.get("current_balance")) > 0:
        blockers.append(f"outstanding balance ${money(acc.get('current_balance')):,.2f}")
    return blockers


def _reasons_logged(db, card_id: str) -> tuple[list[dict], list[dict]]:
    """(records logged today, earlier records within the past year) for this account."""
    rows = [r for r in db.credit_card_closure_reasons.data.values() if r.get("credit_card_account_id") == card_id]
    today = [r for r in rows if as_date(r.get("logged_at")) == TODAY]
    earlier = [r for r in rows if r not in today and within_year(r.get("logged_at"))]
    return today, earlier


def log_closure_reason(db, a: dict) -> GateResult:
    source = "credit_card_account_logistics_003 step 1"
    card = a.get("credit_card_account_id", "")
    if card not in db.credit_card_accounts.data:
        return allow("banking.closure_reason.unknown_card", source, reason="no such account: the tool decides")
    blockers = closure_blockers(db, card)
    if blockers:
        return deny("banking.closure_reason.ineligible", source, f"{card} cannot be closed: {blockers}",
                    "Do not log a closure reason or make retention offers; tell the user what must be resolved "
                    "first.", {"blockers": blockers})
    return allow("banking.closure_reason", source, {"credit_card_account_id": card})


def close_card(db, a: dict) -> GateResult:
    source = "credit_card_account_logistics_002/_003"
    card = a.get("credit_card_account_id", "")
    if card not in db.credit_card_accounts.data:
        return allow("banking.close_card.unknown_card", source, reason="no such account: the tool decides")
    blockers = closure_blockers(db, card)
    if blockers:
        return deny("banking.close_card.ineligible", source, f"{card} cannot be closed: {blockers}",
                    "Do not close it; tell the user what must be resolved first.", {"blockers": blockers})
    today, earlier = _reasons_logged(db, card)
    if not today and not earlier:
        return deny("banking.close_card.no_reason", source, "no closure reason is on record for this account",
                    "Ask why the user is closing and log it with log_credit_card_closure_reason_4521 (then the "
                    "retention steps), before closing.", {})
    return allow("banking.close_card", source, {"reasons_today": len(today), "reasons_past_year": len(earlier)})


def _retention(db, a: dict, rule: str, what: str) -> GateResult | None:
    """None when a retention offer may be made on this account; else the deny."""
    source = "credit_card_account_logistics_003"
    card = a.get("credit_card_account_id", "")
    today, earlier = _reasons_logged(db, card)
    if earlier:
        return deny(f"{rule}.prior_retention", source, f"{card} has a closure reason on record within the past year; "
                    f"retention offers are skipped for it ({what})", "Do not make a retention offer; proceed with "
                    "the closure the user asked for.", {"earlier": [r.get("record_id") for r in earlier]})
    if not today:
        return deny(f"{rule}.no_reason", source, f"no closure reason was logged for {card} before this {what}",
                    "Log the user's closure reason with log_credit_card_closure_reason_4521 first.", {})
    return None


def account_flag(db, a: dict) -> GateResult:
    source = "credit_card_account_logistics_003 step 4"
    if a.get("flag_type") != "annual_fee_waived":
        return allow("banking.account_flag.other", source, {"flag_type": a.get("flag_type")},
                     reason="no knowledge base rule covers this flag type")
    blocked = _retention(db, a, "banking.fee_waiver", "annual fee waiver")
    if blocked:
        return blocked
    today, _ = _reasons_logged(db, a.get("credit_card_account_id", ""))
    if not any(r.get("closure_reason") == "annual_fee" for r in today):
        return deny("banking.fee_waiver.reason", source, "the annual fee waiver answers an 'annual_fee' closure "
                    f"reason; the reason logged is {[r.get('closure_reason') for r in today]}",
                    "Offer the waiver only to a user closing over the annual fee.", {})
    return allow("banking.fee_waiver", source, {})


def statement_credit(db, a: dict) -> GateResult:
    source = "credit_card_account_logistics_003 step 5"
    if a.get("reason") != "retention_offer":
        return allow("banking.statement_credit.other", source, {"reason": a.get("reason")},
                     reason="not a retention credit: no amount rule applies")
    blocked = _retention(db, a, "banking.retention_credit", "retention credit")
    if blocked:
        return blocked
    acc = db.credit_card_accounts.data.get(a.get("credit_card_account_id", ""), {})
    tier = _tier(acc.get("card_type", ""))
    wanted = RETENTION_CREDIT.get(tier)
    if wanted is not None and money(a.get("amount")) != wanted:
        return deny("banking.retention_credit.amount", source, f"a {tier}-tier card's retention credit is "
                    f"${wanted:g}, not ${money(a.get('amount')):g}", f"Apply amount {wanted:g}.",
                    {"tier": tier, "card_type": acc.get("card_type")})
    return allow("banking.retention_credit", source, {"tier": tier})
