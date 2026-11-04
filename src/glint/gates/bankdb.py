"""Shared reads of τ²'s banking database for the gates: the domain's clock, checking-account tiers, who owns a record,
and what this conversation has already done (a verification, a recorded lookup, an undecided increase).

"Today" is the clock in effect (glint.clock): τ² banking_knowledge's fixed clock by default (`get_current_time`
returns 2025-11-14 03:40:00 EST), or the one passed as `Enforcer(clock=...)`. Under a given clock every date rule is
deterministic.
"""
from __future__ import annotations

from datetime import date, timedelta

from glint import clock
from glint.assist.banking import _date, _money

# bank_accounts_(general)_005 (personal checking) and _007 (business checking): the tier of each account level
CHECKING_TIERS = {
    "light blue account": "entry", "light green account": "entry", "green fee-free account": "entry",
    "blue account": "mid", "green account": "mid", "green account (checking)": "mid",
    "evergreen account": "premium", "bluest account": "elite",
    "navy blue": "entry", "true blue": "mid", "sky blue": "mid", "cobalt blue": "premium",
}


def checking_tier(account: dict) -> str | None:
    """The tier of a checking account from its level name; None for a level the knowledge base gives no tier."""
    level = str(account.get("level") or account.get("account_class") or "").strip().lower()
    return CHECKING_TIERS.get(level) or CHECKING_TIERS.get(level.removesuffix(" account"))


def days_since(value) -> int | None:
    d = _date(value) if value else None
    return (clock.today() - d).days if d else None


def within_year(value) -> bool:
    d = _date(value) if value else None
    return d is not None and d > clock.today() - timedelta(days=365)


def owner(db, args: dict) -> str | None:
    """The user a write acts for: its user_id, else the owner of the card, account or transaction it names."""
    if args.get("user_id"):
        return str(args["user_id"])
    for key, table in (("card_id", "debit_cards"), ("account_id", "accounts"),
                       ("credit_card_account_id", "credit_card_accounts"),
                       ("transaction_id", "credit_card_transaction_history")):
        if args.get(key):
            row = getattr(db, table).data.get(args[key]) or {}
            if row.get("user_id"):
                return str(row["user_id"])
    return None


def verified_today(db, user_id: str) -> bool:
    """A log_verification record for this user at today's date (earlier records are from earlier conversations)."""
    return any(r.get("user_id") == user_id and str(r.get("time_verified", "")).startswith(clock.today().isoformat())
               for r in db.verification_history.data.values())


def increase_request_ids(db) -> set[str]:
    return set(db.credit_limit_increase_requests.data)


def undecided_increases(toolkit, user_id: str | None = None, card: str | None = None) -> list[str]:
    """Credit limit increase requests submitted in this conversation and not yet approved or denied (of this user, on
    this card; of anyone when both are None), from the enforcer's record (`env.tools.glint_increases`: user -> the
    request ids every submit that ran created for them; an approve or deny that ran clears its card's requests;
    glint.gates.enforce).
    The database cannot show it: τ²'s approve record reuses the submit request's id, so a request stays PENDING after
    a decision, and τ² records each decision tool once, under its name alone, so a second decision in the conversation
    leaves no trace there."""
    waiting = getattr(toolkit, "glint_increases", {})
    rows = toolkit.db.credit_limit_increase_requests.data
    return [rid for user, ids in waiting.items() if user_id is None or user == user_id for rid in ids
            if card is None or (rows.get(rid) or {}).get("credit_card_account_id") == card]


def money(value) -> float:
    return _money(value)


def as_date(value) -> date | None:
    return _date(value) if value else None
