"""Shared reads of τ²'s banking database for the gates: the domain's clock, checking-account tiers, who owns a record,
and what this conversation has already done (a verification, a recorded lookup, an undecided increase).

"Today" is τ² banking_knowledge's fixed clock (`get_current_time` returns 2025-11-14 03:40:00 EST), so every
date rule is deterministic.
"""
from __future__ import annotations

from datetime import date, timedelta

from glint.assist.banking import TODAY, _date, _money

NOW_TEXT = "2025-11-14 03:40:00 EST"  # what get_current_time returns; log_verification's time_verified
APPROVE, DENY_CLI = "approve_credit_limit_increase_5847", "deny_credit_limit_increase_5848"
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
    return (TODAY - d).days if d else None


def within_year(value) -> bool:
    d = _date(value) if value else None
    return d is not None and d > TODAY - timedelta(days=365)


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
    return any(r.get("user_id") == user_id and str(r.get("time_verified", "")).startswith(TODAY.isoformat())
               for r in db.verification_history.data.values())


def recorded_tools(db) -> set[str]:
    """Hidden tools this conversation has called that the database recorded (every write; a read only when the
    environment records it)."""
    return {str(r.get("tool_name")) for r in db.agent_discoverable_tools.data.values()}


def undecided_increases(db, user_id: str | None = None) -> list[str]:
    """Credit limit increase requests submitted today (of this user, or of anyone when user_id is None) and not yet
    approved or denied. τ²'s approve record reuses the submit request's id, so the request stays PENDING after a
    decision; the decision shows only as the approve or deny call recorded in agent_discoverable_tools."""
    if {APPROVE, DENY_CLI} & recorded_tools(db):
        return []
    return [rid for rid, r in db.credit_limit_increase_requests.data.items()
            if (user_id is None or r.get("user_id") == user_id) and str(r.get("status", "")).upper() == "PENDING"
            and as_date(r.get("submitted_at")) == TODAY]


def money(value) -> float:
    return _money(value)


def as_date(value) -> date | None:
    return _date(value) if value else None
