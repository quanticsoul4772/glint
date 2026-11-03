"""The checking-credit amount rule (bank_accounts_(general)_017, "Fee mischarges"): the credit equals the net correction
across every fee that differs from the documented fee schedule, plus rebates due and not applied.

    net = (ATM fees charged - ATM fees due) + (ATM rebates due - ATM rebates applied)

_017: "identify all fee discrepancies and apply a credit for the net correction"; "The credit amount must match the
exact rebate or fee amount - do not round or estimate". Only ATM fees are compiled, for the checking levels below; an
account with any other fee type, or at a level not compiled here, gets no expected amount (None), and the rule then
says nothing.

Fee schedules (knowledge base document ids, `doc_checking_accounts_*`):
- Bluest Account: out-of-network $2.00 per withdrawal (bluest_account_010); foreign $0.00 (bluest_account_003); the
  out-of-network fees rebated up to $50 per monthly cycle (bluest_account_003, _007).
- Light Green Account: 4 free out-of-network withdrawals a month, then $1.50 (light_green_account_001); foreign flat
  by amount: up to $100 $2.00, up to $300 $3.50, above $300 $5.00, a threshold amount takes the lower fee
  (light_green_account_013).
- Blue Account: out-of-network 1% of the withdrawal, at most $3.00 (blue_account_001); foreign 3%, at least $5.00
  (blue_account_012).
- Green Account (checking): out-of-network $3.00 (green_account_(checking)_001, _002, _010); foreign the greater of 3%
  and $5.00 (green_account_(checking)_012).

READINGS lists what the documents leave open and how each was settled.

Mode (CREDIT_AMOUNT_MODE): "deny" (default since the owner's decision of 2026-11-03, every fix on by default) refuses
a credit whose amount differs; "flag" allows it and logs it under `banking.checking_credit.amount_unmatched`; "off"
skips the rule. The off switch is Enforcer(credit_amount_mode="flag") or "off" (glint.gates.module_switches).
"""
from __future__ import annotations

from collections.abc import Callable

from glint.gates.bankdb import money
from glint.gates.core import GateResult, allow, deny

SOURCE = "bank_accounts_(general)_017 (net correction); checking account fee schedules (glint.gates.fees)"
CREDIT_AMOUNT_MODE = "deny"
MODES = ("off", "flag", "deny")
UNMATCHED = "banking.checking_credit.amount_unmatched"
WITHDRAWAL, FEE, REBATE = "atm_withdrawal", "atm_fee", "fee_rebate"
# fee types the model covers; any other fee on the account leaves the expected amount unknown
FEE_TYPES = ("atm_fee", "fee", "debit_card_fee", "monthly_fee", "overdraft_fee")
IN_NETWORK = "RHO-BANK"
US_STATES = frozenset((
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
    "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA",
    "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC", "PR"))
READINGS = (
    ("A withdrawal is in network when its description names RHO-BANK; it carries no fee and does not use a free "
     "out-of-network withdrawal (Glint reading)."),
    ("A withdrawal is foreign when its description ends in a two-letter code that is not a US state; a code that is "
     "both (DE, IN, CA) is foreign only when the fee row posted with it says FOREIGN (Glint reading of the database's "
     "descriptions; the DE case was found on training task 072's Berlin withdrawal)."),
    ("A foreign withdrawal carries the foreign fee only, not the out-of-network fee as well, and does not use one of "
     "Light Green's 4 free out-of-network withdrawals (reading; τ² training gold 072 and 073 agree with it)."),
    ("Bluest's rebate covers the out-of-network fees as due, not as charged, capped at $50 per calendar month (the "
     "database holds no statement cycle dates; reading)."),
    "Percent fees are rounded to the cent (Glint reading).",
    ("Checked: the net correction equals τ² gold's credit on all 5 training-gold credits (tasks 072, 073; in-sample, "
     "and the readings above were settled with those tasks in view)."),
)


def _cents(x: float) -> float:
    return round(x + 1e-9, 2)


def _light_green_foreign(amount: float) -> float:
    return 2.0 if amount <= 100 else 3.5 if amount <= 300 else 5.0


# level -> (out-of-network fee(amount), free out-of-network withdrawals a month, foreign fee(amount), rebate cap)
Schedule = tuple[Callable[[float], float], int, Callable[[float], float], float]
SCHEDULES: dict[str, Schedule] = {
    "bluest account": (lambda a: 2.0, 0, lambda a: 0.0, 50.0),
    "light green account": (lambda a: 1.5, 4, _light_green_foreign, 0.0),
    "blue account": (lambda a: min(_cents(a * 0.01), 3.0), 0, lambda a: max(_cents(a * 0.03), 5.0), 0.0),
    "green account": (lambda a: 3.0, 0, lambda a: max(_cents(a * 0.03), 5.0), 0.0),
    "green account (checking)": (lambda a: 3.0, 0, lambda a: max(_cents(a * 0.03), 5.0), 0.0),
}


def is_foreign(description: str, fee_label: str = "") -> bool:
    """A withdrawal abroad: its description ends in a two-letter country code. A code that is also a US state (DE
    Germany or Delaware, IN, CA) counts as foreign only when the fee row posted with it says FOREIGN."""
    last = str(description or "").split()[-1:] or [""]
    code = last[0]
    if not (len(code) == 2 and code.isalpha() and code.isupper()):
        return False
    return code not in US_STATES or "FOREIGN" in str(fee_label).upper()


def _month(date: str) -> str:
    parts = str(date or "").split("/")
    return f"{parts[2][:4]}-{parts[0]}" if len(parts) == 3 else ""


def net_correction(db, account_id: str) -> tuple[float, dict] | None:
    """(net credit due, the parts it is made of) for a checking account whose level and fee types are compiled, else
    None."""
    acc = db.accounts.data.get(account_id) or {}
    schedule = SCHEDULES.get(str(acc.get("level", "")).strip().lower())
    if acc.get("class") != "checking" or schedule is None:
        return None
    rows = [t for t in db.bank_account_transaction_history.data.values() if t.get("account_id") == account_id]
    if any(t.get("type") in FEE_TYPES and t.get("type") != FEE for t in rows):
        return None
    out_fee, free, foreign_fee, cap = schedule
    used: dict[str, int] = {}
    due_fees, rebate_due = 0.0, {}
    for i, t in enumerate(rows):  # stored order: the database keeps each account's history by date
        if t.get("type") != WITHDRAWAL:
            continue
        amount, text = abs(money(t.get("amount"))), str(t.get("description", ""))
        if IN_NETWORK in text.upper():
            continue
        after = rows[i + 1] if i + 1 < len(rows) else {}
        label = after.get("description", "") if after.get("type") == FEE and after.get("date") == t.get("date") else ""
        if is_foreign(text, label):
            due_fees += foreign_fee(amount)
            continue
        month = _month(t.get("date"))
        used[month] = used.get(month, 0) + 1
        fee = out_fee(amount) if used[month] > free else 0.0
        due_fees += fee
        rebate_due[month] = rebate_due.get(month, 0.0) + fee
    charged = -sum(money(t.get("amount")) for t in rows if t.get("type") == FEE)
    rebated = sum(money(t.get("amount")) for t in rows if t.get("type") == REBATE)
    rebates = sum(min(v, cap) for v in rebate_due.values())
    net = _cents((charged - due_fees) + (rebates - rebated))
    return net, {"level": acc.get("level"), "fees_charged": _cents(charged), "fees_due": _cents(due_fees),
                 "rebates_due": _cents(rebates), "rebates_applied": _cents(rebated), "net_correction": net}


def credit_amount(db, a: dict, mode: str | None = None) -> GateResult | None:
    """The amount rule on apply_checking_account_credit_5829: None when the amount matches, the rule is off, or the
    account's fees are not compiled; else a flagged allow (mode "flag") or a deny (mode "deny")."""
    mode = mode or CREDIT_AMOUNT_MODE
    if mode not in MODES:
        raise ValueError(f"unknown checking-credit amount mode {mode!r}; one of {MODES}")
    if mode == "off":
        return None
    found = net_correction(db, a.get("account_id", ""))
    if found is None:
        return None
    net, parts = found
    amount = money(a.get("amount"))
    if abs(amount - net) < 0.005:
        return None
    facts = [f"credit given: ${amount:,.2f}", f"net correction from the fee schedule: ${net:,.2f}",
             (f"ATM fees charged ${parts['fees_charged']:,.2f}, due ${parts['fees_due']:,.2f}; rebates due "
              f"${parts['rebates_due']:,.2f}, applied ${parts['rebates_applied']:,.2f} ({parts['level']})")]
    if mode == "flag":
        return allow(UNMATCHED, SOURCE, parts | {"amount": amount},
                     reason="the credit differs from the net correction the compiled fee schedule gives; allowed and "
                     "flagged for review (CREDIT_AMOUNT_MODE 'flag')")
    return deny(UNMATCHED, SOURCE, "the credit differs from the net correction across the account's fee "
                "discrepancies", f"Apply one credit of {net:.2f}: the net of every ATM fee that differs from the fee "
                "schedule, plus rebates due and not applied.", parts | {"amount": amount},
                consequence="The account takes one credit per conversation, so the single credit must carry the full "
                "net correction.", facts=facts)
