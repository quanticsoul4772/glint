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
from glint.gates import apy
from glint.gates.bankdb import NOW_TEXT, as_date, days_since, money, within_year
from glint.gates.core import GateResult, allow, deny

VERIFY_FIELDS = (("name", "name"), ("address", "address"), ("email", "email"), ("phone_number", "phone_number"),
                 ("date_of_birth", "date_of_birth"))
NOT_VERIFIED = ("Until a verification is logged, every account change for this customer is blocked; the details "
                "must match the record on file.")
CANNOT_CLOSE = ("The account stays open and no retention offer is made until each item below is resolved; tell the "
                "customer what they are and help with any of them they ask about.")
RETENTION_CREDIT = {"entry": 5.0, "mid": 20.0, "premium": 50.0, "elite": 50.0, "invitation": 50.0}
CREDIT_TYPES = ("rebate_credit", "fee_refund")
FEE_TYPES = ("atm_fee", "fee", "debit_card_fee", "monthly_fee", "overdraft_fee")


def _norm(value) -> str:
    return " ".join(str(value or "").lower().split())


def log_verification(db, a: dict) -> GateResult:
    source = "log_verification docstring; get_current_time"
    user = db.users.data.get(a.get("user_id", ""))
    if user is None:
        return deny("banking.verification.user", source, "the verification names a user who is not on file",
                    "Look the user up (get_user_information_by_*) and log the verification with their user_id.",
                    {"user_id": a.get("user_id")}, consequence=NOT_VERIFIED,
                    facts=[f"user_id {a.get('user_id')!r}: no such user"])
    wrong = [arg for arg, field in VERIFY_FIELDS if arg in a and _norm(a[arg]) != _norm(user.get(field))]
    reasons = [f"{arg} {a[arg]!r} differs from the record on file" for arg in wrong]
    if str(a.get("time_verified", "")).strip() != NOW_TEXT:
        reasons.append(f"time_verified {a.get('time_verified')!r} is not the current time {NOW_TEXT!r}")
    if reasons:
        return deny("banking.verification", source, "the verification does not match the customer's record or the "
                    "current time", "Log the verification with the user's details exactly as on file and "
                    "time_verified from get_current_time.", {"mismatched": wrong}, consequence=NOT_VERIFIED,
                    facts=reasons)
    return allow("banking.verification", source, {"user_id": a["user_id"]})


def change_email(db, a: dict) -> GateResult:
    source = "change_user_email docstring"
    user = db.users.data.get(a.get("user_id", ""))
    if user is not None and _norm(a.get("new_email")) == _norm(user.get("email")):
        return deny("banking.change_email.same", source, "the new email is already the email on file",
                    "Do not change it; tell the user the email on file already is that address.", {},
                    consequence="Nothing needs changing; continue with the rest of the customer's request.",
                    facts=[f"email on file for {a.get('user_id')}: {user.get('email')}"])
    return allow("banking.change_email", source, {"user_id": a.get("user_id")})


def update_rewards(db, a: dict) -> GateResult:
    source = "credit_cards_(general)_004"
    tid = a.get("transaction_id", "")
    disputes = [r for r in db.cash_back_disputes.data.values() if r.get("transaction_id") == tid]
    approved = [r for r in disputes if str(r.get("status", "")).upper() == "RESOLVED"
                and str(r.get("resolution", "")).upper() == "APPROVED"]
    details = {"transaction_id": tid, "disputes": [r.get("dispute_id") for r in disputes]}
    if not approved:
        return deny("banking.update_rewards.no_resolved_dispute", source, f"transaction {tid} has no approved cash "
                    "back dispute, and rewards are corrected only after one", "Give the user the "
                    "submit_cash_back_dispute_0589 tool (give_discoverable_user_tool) so the user files the dispute; "
                    "update rewards only for a transaction whose dispute is RESOLVED/APPROVED.", details,
                    consequence="The customer files the cash back dispute with their own tool; the rewards are "
                    "corrected after it is approved, not in this conversation.",
                    facts=[f"cash back disputes on {tid}: " + (", ".join(
                        f"{r.get('dispute_id')} {r.get('status')}/{r.get('resolution')}" for r in disputes)
                        or "none")])
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
                    "ONE credit with the total amount.", details,
                    consequence="This account takes no further credit now; tell the customer what was already "
                    "credited.",
                    facts=[f"credits on {acc_id} in the past 14 days: " + ", ".join(
                        f"{t.get('transaction_id')} ({t.get('type')}, {t.get('date')})" for t in credits)])
    amount = money(a.get("amount"))
    if a.get("credit_type") == "fee_refund":
        fees = round(-sum(money(t.get("amount")) for t in rows if t.get("type") in FEE_TYPES
                          and money(t.get("amount")) < 0), 2)
        details["fees_charged"] = fees
        if amount > fees:
            return deny("banking.checking_credit.amount", source, "the fee refund is larger than the fees charged to "
                        "this account", "Refund only the fees that were charged wrongly (net of any undercharges); "
                        "recompute from the account's transactions.", details,
                        consequence="The account takes one credit per conversation, so the single credit must "
                        "carry the full corrected amount.",
                        facts=[f"refund asked: ${amount:,.2f}", f"fees charged to {acc_id}: ${fees:,.2f}"])
    return allow("banking.checking_credit", source, details)


def _interest_basis(db, acc_id: str) -> tuple[float, float, list[dict]] | None:
    """The account's latest monthly interest credit, its balance before any interest correction applied today, and
    those corrections. None when the account has no interest credit on record."""
    rows = [t for t in db.bank_account_transaction_history.data.values() if t.get("account_id") == acc_id]
    interest = [t for t in rows if t.get("type") == "interest_credit"]
    if not interest:
        return None
    latest = max(interest, key=lambda t: (as_date(t.get("date")) or TODAY, str(t.get("transaction_id"))))
    today = [t for t in rows if t.get("type") == "interest_correction" and as_date(t.get("date")) == TODAY]
    holdings = money(db.accounts.data.get(acc_id, {}).get("current_holdings"))
    return money(latest.get("amount")), round(holdings - sum(money(t.get("amount")) for t in today), 2), today


def _components_fact(parts: dict) -> str:
    return (f"{parts['level']}: base {parts['base']}% + checking boost {parts['boost']}% + card bonus "
            f"{parts['card_bonus']}% + relationship {parts['relationship_bonus']}% = {parts['expected_apy']}%")


def savings_credit(db, a: dict) -> GateResult:
    """bank_accounts_(general)_043: an interest correction is the difference between the interest at the expected
    APY (glint.gates.apy) and the interest credited, less any correction already applied today. Fee refunds and
    goodwill credits are not checked."""
    source = "bank_accounts_(general)_043; glint.gates.apy"
    acc_id = a.get("account_id", "")
    if a.get("credit_type") != "interest_correction":
        return allow("banking.savings_credit.not_interest", source, {"account_id": acc_id},
                     reason="only interest corrections are checked")
    basis = _interest_basis(db, acc_id) if db.accounts.data.get(acc_id) else None
    computed = apy.expected_apy(db, acc_id) if basis else None
    if basis is None or computed is None:
        return allow("banking.savings_credit.not_compiled", source, {"account_id": acc_id},
                     reason="no interest credit on record, or the account level is not compiled")
    paid, balance, corrections = basis
    expected, parts = computed
    owed = round(round(balance * expected / 1200, 2) - paid, 2)
    applied = round(paid * 12 / balance * 100, 3) if balance > 0 else expected
    owed_alt = round(balance * (expected - applied) / 1200, 2)
    already = round(sum(money(t.get("amount")) for t in corrections), 2)
    remaining = round(owed - already, 2)
    amount = money(a.get("amount"))
    details = {"account_id": acc_id, "balance": balance, "interest_credited": paid, "owed": owed,
               "already_corrected_today": already, "remaining": remaining, **parts}
    facts = [_components_fact(parts), (f"interest at {expected}%: ${owed + paid:,.2f}; credited ${paid:,.2f}; "
             f"owed ${owed:,.2f}; already corrected today ${already:,.2f}")]
    if remaining <= 0.005:
        return deny("banking.savings_credit.nothing_owed", source, "no interest correction is owed on this account",
                    "Do not apply this credit. Tell the customer the interest credited matches the expected APY, or "
                    "that the correction was already applied.", details,
                    consequence="Nothing was credited.", facts=facts)
    if abs(amount - remaining) > 0.015 and abs(amount - round(owed_alt - already, 2)) > 0.015:
        return deny("banking.savings_credit.amount", source, "the interest correction is not the amount owed",
                    f"Apply ${remaining:,.2f} (the interest at {expected}% less the interest credited"
                    + (" and the correction already applied today" if already else "") + ").", details,
                    consequence="Nothing was credited. One correct credit, then the report with the same amount.",
                    facts=facts + [f"credit asked: ${amount:,.2f}"])
    return allow("banking.savings_credit", source, details)


def interest_report(db, a: dict) -> GateResult:
    """bank_accounts_(general)_043/_044: the report's actual APY is the one the account was paid at, its amount is the
    difference between the interest at the expected APY and the interest credited, and the credit is applied first,
    for that same amount. The expected APY itself is not checked (it needs the account type's rate documents)."""
    source = "bank_accounts_(general)_043/_044"
    acc_id = a.get("account_id", "")
    acc = db.accounts.data.get(acc_id)
    basis = _interest_basis(db, acc_id) if acc else None
    if basis is None:
        return allow("banking.interest_report.no_interest_on_record", source, {"account_id": acc_id},
                     reason="no interest credit on record for this account; nothing to check the report against")
    paid, balance, corrections = basis
    try:
        expected, actual, diff = (float(a.get(k)) for k in ("expected_apy", "actual_apy", "amount_difference"))
    except (TypeError, ValueError):
        return allow("banking.interest_report.not_numbers", source, {"account_id": acc_id},
                     reason="the APYs or amount are not numbers; the tool rejects them")
    applied = round(paid * 12 / balance * 100, 3) if balance > 0 else None
    owed = round(round(balance * expected / 1200, 2) - paid, 2)
    details = {"account_id": acc_id, "balance": balance, "interest_credited": paid, "applied_apy": applied,
               "amount_at_expected_apy": owed, "corrections_today": [money(t.get("amount")) for t in corrections]}
    paid_fact = f"latest monthly interest credited: ${paid:,.2f} on a balance of ${balance:,.2f}"
    computed = apy.expected_apy(db, acc_id)
    if computed is not None and abs(expected - computed[0]) > 0.0015:
        right, parts = computed
        return deny("banking.interest_report.expected_apy", source, "the expected APY is not the one the account's "
                    "products earn", f"Report expected_apy {right} and recompute the difference.",
                    details | parts, consequence="Nothing was reported. The expected APY sets the amount owed.",
                    facts=[_components_fact(parts), f"the report says {expected}%"])
    if applied is not None and abs(actual - applied) > 0.0015:
        return deny("banking.interest_report.actual_apy", source, "the actual APY is not the rate the account was paid at",
                    f"Report actual_apy {applied} (monthly interest x 12 / balance), then recompute the difference.",
                    details, consequence="Nothing was reported. The backend team needs the rate actually applied.",
                    facts=[paid_fact, f"rate applied: {applied}%; the report says {actual}%"])
    if abs(diff - owed) > 0.015 and abs(diff - round(balance * (expected - actual) / 1200, 2)) > 0.015:
        return deny("banking.interest_report.amount", source, "the amount difference does not follow from the APYs",
                    f"For expected_apy {expected}, the difference is ${owed:,.2f} (balance x expected APY / 12, minus "
                    "the interest credited). Check the expected APY against the account type's documents.",
                    details, consequence="Nothing was reported. The report and the credit must carry the same, "
                    "correctly computed amount.",
                    facts=[paid_fact, (f"interest at {expected}%: ${owed + paid:,.2f}; difference ${owed:,.2f}; "
                                       f"the report says ${diff:,.2f}")])
    if not any(abs(money(t.get("amount")) - diff) <= 0.015 for t in corrections):
        why = ("no interest correction was applied to this account in this conversation" if not corrections else
               "the interest correction applied does not equal the reported difference")
        return deny("banking.interest_report.credit_first", source, why,
                    f"Apply the interest correction of ${diff:,.2f} to this account first "
                    "(apply_savings_account_credit_6831), then submit the report with the same amount." if not
                    corrections else "The credit and the report must carry the same amount; recheck the expected APY "
                    "and tell the customer if the credit applied was wrong.",
                    details, consequence="Nothing was reported. The policy applies the credit before the report.",
                    facts=[f"interest corrections today: {details['corrections_today'] or 'none'}",
                           f"reported difference: ${diff:,.2f}"])
    return allow("banking.interest_report", source, details)


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
        return deny("banking.closure_reason.ineligible", source, f"credit card account {card} cannot be closed yet",
                    "Do not log a closure reason or make retention offers; tell the user what must be resolved "
                    "first.", {"blockers": blockers}, consequence=CANNOT_CLOSE, facts=blockers)
    return allow("banking.closure_reason", source, {"credit_card_account_id": card})


def close_card(db, a: dict) -> GateResult:
    source = "credit_card_account_logistics_002/_003"
    card = a.get("credit_card_account_id", "")
    if card not in db.credit_card_accounts.data:
        return allow("banking.close_card.unknown_card", source, reason="no such account: the tool decides")
    blockers = closure_blockers(db, card)
    if blockers:
        return deny("banking.close_card.ineligible", source, f"credit card account {card} cannot be closed yet",
                    "Do not close it; tell the user what must be resolved first.", {"blockers": blockers},
                    consequence=CANNOT_CLOSE, facts=blockers)
    today, earlier = _reasons_logged(db, card)
    if not today and not earlier:
        return deny("banking.close_card.no_reason", source, "no closure reason is on record for this account",
                    "Ask why the user is closing and log it with log_credit_card_closure_reason_4521 (then the "
                    "retention steps), before closing.", {},
                    consequence="The reason decides which retention offer applies, so it comes before the closure; "
                    "close only if the customer still wants it after the offer.",
                    facts=[f"closure reasons for {card} in the past year: none"])
    return allow("banking.close_card", source, {"reasons_today": len(today), "reasons_past_year": len(earlier)})


def _retention(db, a: dict, rule: str, what: str) -> GateResult | None:
    """None when a retention offer may be made on this account; else the deny."""
    source = "credit_card_account_logistics_003"
    card = a.get("credit_card_account_id", "")
    today, earlier = _reasons_logged(db, card)
    if earlier:
        return deny(f"{rule}.prior_retention", source, f"this account already had a closure reason logged within "
                    f"the past year, so no retention offer ({what}) is made", "Do not make a retention offer; proceed "
                    "with the closure the user asked for.", {"earlier": [r.get("record_id") for r in earlier]},
                    consequence="Retention offers are made once a year; the closure itself can go ahead.",
                    facts=[f"closure reasons for {card} in the past year: " + ", ".join(
                        f"{r.get('closure_reason')} ({r.get('logged_at')})" for r in earlier)])
    if not today:
        return deny(f"{rule}.no_reason", source, f"no closure reason was logged for {card} before this {what}",
                    "Log the user's closure reason with log_credit_card_closure_reason_4521 first.", {},
                    consequence="The logged reason decides which retention offer applies, if any.",
                    facts=[f"closure reasons for {card} today: none"])
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
        logged = [r.get("closure_reason") for r in today]
        return deny("banking.fee_waiver.reason", source, "the annual fee waiver is offered only to a customer "
                    "closing over the annual fee", "Offer the waiver only to a user closing over the annual fee.", {},
                    consequence="The retention offer must match the reason the customer gave; the other steps of "
                    "the closure are not affected.", facts=[f"closure reason logged today: {', '.join(logged)}"])
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
        return deny("banking.retention_credit.amount", source, "the retention credit amount does not match the "
                    "card's tier", f"Apply amount {wanted:g}.", {"tier": tier, "card_type": acc.get("card_type")},
                    consequence="The retention credit is fixed by card tier; offer the customer that amount.",
                    facts=[f"card type {acc.get('card_type')} ({tier} tier): ${wanted:g}",
                           f"amount given: ${money(a.get('amount')):g}"])
    return allow("banking.retention_credit", source, {"tier": tier})
