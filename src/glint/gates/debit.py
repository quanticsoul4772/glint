"""Debit card write rules, from the knowledge base's debit card procedures. Each rule cites its source document.

- freeze_debit_card_3892 / unfreeze_debit_card_3893: bank_accounts_(general)_026 (status rules; a card that will be
  closed is unfrozen first, because close_debit_card_4721 only closes ACTIVE or PENDING cards).
- close_debit_card_4721: bank_accounts_(general)_025 (no pending transactions on the linked account; the
  14-day minimum card age unless the card is lost, stolen or fraud_suspected; a frozen card is unfrozen first).
- order_debit_card_5739: bank_accounts_(general)_023 (base fees) and _029 (replacement limits, the entry-tier waiting
  period, delivery and design fees by checking tier; tiers from _005).
- activate_debit_card_8291/8292/8293: bank_accounts_(general)_024 (the variant follows the card's issue_reason; the
  card must be PENDING, unexpired, on an OPEN account).

Whether a card is lost for certain or only misplaced (freeze first, then decide) is the user's statement, not the
database's, so no gate here requires a freeze before a close: τ²'s train tasks close without a freeze in 084 and
086-092 and freeze first in 081.
"""
from __future__ import annotations

from datetime import timedelta

from glint.assist.banking import TODAY
from glint.gates.bankdb import as_date, checking_tier, days_since, money
from glint.gates.core import GateResult, allow, deny

FREEZE, UNFREEZE, CLOSE, ORDER = ("freeze_debit_card_3892", "unfreeze_debit_card_3893", "close_debit_card_4721",
                                  "order_debit_card_5739")
ACTIVATE = {"activate_debit_card_8291": ("new_account", "first_card"),
            "activate_debit_card_8292": ("lost", "stolen", "fraud"),
            "activate_debit_card_8293": ("expired", "damaged", "upgrade", "bank_reissue")}
DEBIT_TOOLS = (FREEZE, UNFREEZE, CLOSE, ORDER, *ACTIVATE)
SECURITY_REASONS = ("lost", "stolen", "fraud_suspected")
REPLACEMENT_REASONS = ("lost", "stolen", "fraud", "damaged")
# _029: delivery fee by tier and option for a replacement (an option missing from a tier is not offered there)
REPLACEMENT_DELIVERY = {"entry": {"STANDARD": 0.0},
                        "mid": {"STANDARD": 0.0, "EXPEDITED": 15.0},
                        "premium": {"STANDARD": 0.0, "EXPEDITED": 0.0, "RUSH": 35.0},
                        "elite": {"STANDARD": 0.0, "EXPEDITED": 0.0, "RUSH": 0.0}}
BASE_DELIVERY = {"STANDARD": 0.0, "EXPEDITED": 15.0, "RUSH": 35.0}  # _023, a first card
DESIGN_FEE = {"entry": {"CLASSIC": 0.0, "PREMIUM": 10.0, "CUSTOM": 25.0},
              "mid": {"CLASSIC": 0.0, "PREMIUM": 10.0, "CUSTOM": 25.0},
              "premium": {"CLASSIC": 0.0, "PREMIUM": 0.0, "CUSTOM": 15.0},
              "elite": {"CLASSIC": 0.0, "PREMIUM": 0.0, "CUSTOM": 0.0}}
REPLACEMENT_LIMIT = {"entry": (2, 25.0), "mid": (3, 15.0), "premium": (5, None), "elite": (None, None)}
ENTRY_WAIT_DAYS = 2  # _029: 48 hours after a closure; the database dates closures by day


def _card(db, a: dict) -> dict | None:
    return db.debit_cards.data.get(a.get("card_id", ""))


def _unknown_card(rule: str, source: str, a: dict) -> GateResult:
    return allow(f"{rule}.unknown_card", source, {"card_id": a.get("card_id")},
                 reason="no such card: τ²'s tool rejects the call itself")


def freeze(db, a: dict) -> GateResult:
    source = "bank_accounts_(general)_026"
    card = _card(db, a)
    if card is None:
        return _unknown_card("debit.freeze", source, a)
    status = card.get("status")
    if status != "ACTIVE":
        return deny("debit.freeze.status", source, "only an ACTIVE card can be frozen", "Do not freeze it. A FROZEN "
                    "card is already frozen; a CLOSED or PENDING card cannot be frozen.", {"status": status},
                    consequence="The card's status is unchanged; tell the customer its current status.",
                    facts=[f"card {a['card_id']}: {status}"])
    return allow("debit.freeze", source, {"status": status})


def unfreeze(db, a: dict) -> GateResult:
    source = "bank_accounts_(general)_026"
    card = _card(db, a)
    if card is None:
        return _unknown_card("debit.unfreeze", source, a)
    status = card.get("status")
    account = db.accounts.data.get(card.get("account_id"), {})
    if status != "FROZEN":
        return deny("debit.unfreeze.status", source, "only a FROZEN card can be unfrozen", "Do not unfreeze it.",
                    {"status": status}, consequence="The card's status is unchanged; tell the customer its current "
                    "status.", facts=[f"card {a['card_id']}: {status}"])
    if account and account.get("status") != "OPEN":
        return deny("debit.unfreeze.account", source, "a card is unfrozen only while its linked account is OPEN",
                    "Do not unfreeze it; the linked account must be OPEN.", {"account_status": account.get("status")},
                    consequence="The card stays frozen; tell the customer why.",
                    facts=[f"card {a['card_id']}: FROZEN",
                           f"linked account {card.get('account_id')}: {account.get('status')}"])
    return allow("debit.unfreeze", source, {"status": status})


def close(db, a: dict) -> GateResult:
    source = "bank_accounts_(general)_025"
    card = _card(db, a)
    if card is None:
        return _unknown_card("debit.close", source, a)
    reason = str(a.get("reason", "")).lower()
    status = card.get("status")
    if status == "FROZEN":
        return deny("debit.close.frozen", source, "only an ACTIVE or PENDING card can be closed", "Unfreeze it with "
                    "unfreeze_debit_card_3893 first, then close it.", {"status": status},
                    consequence="The close tool rejects a frozen card; unfreezing is only the step before the "
                    "closure the customer asked for.",
                    facts=[f"card {a['card_id']}: FROZEN"])
    pending = [t for t in db.bank_account_transaction_history.data.values()
               if t.get("account_id") == card.get("account_id")
               and str(t.get("status", "")).lower() in ("pending", "processing")]
    # requirement 5 lets a pending refund through once the user acknowledges in writing that it goes to the checking
    # account; that acknowledgment is in the conversation only, so pending refunds are not judged here
    held = sorted(t["transaction_id"] for t in pending if t.get("type") != "refund")
    if held:
        return deny("debit.close.pending_transactions", source, "a card closes only after the pending transactions "
                    "on its account settle", "Do not close the card. Tell the user it can be closed once the pending "
                    "transactions settle; a frozen card can stay frozen meanwhile.", {"pending": held},
                    consequence="The card cannot be closed in this conversation; a replacement card that depends on "
                    "the closure waits too.",
                    facts=[f"pending transactions on {card.get('account_id')}: {', '.join(held)}"])
    age = days_since(card.get("issue_date"))
    if reason not in SECURITY_REASONS and age is not None and age < 14:
        eligible = as_date(card.get("issue_date")) + timedelta(days=14)
        return deny("debit.close.card_age", source, f"a card closed for '{reason}' must be at least 14 days old",
                    f"Do not close it yet; the earliest closure date is {eligible:%m/%d/%Y}.", {"age_days": age},
                    consequence="Lost, stolen and fraud_suspected closures skip the 14 days; for any other reason "
                    "the customer waits until the date above.",
                    facts=[f"card {a['card_id']} issued {card.get('issue_date')} ({age} days ago)",
                           f"closure reason given: '{reason}'"])
    return allow("debit.close", source, {"status": status, "reason": reason})


def _replacements_in_year(cards: list[dict]) -> list[dict]:
    return [c for c in cards if c.get("issue_reason") in REPLACEMENT_REASONS
            and (d := as_date(c.get("issue_date"))) is not None and d > TODAY - timedelta(days=365)]


def order(db, a: dict) -> GateResult:
    source = "bank_accounts_(general)_023/_029"
    account = db.accounts.data.get(a.get("account_id", ""))
    if account is None or account.get("class") != "checking":
        return allow("debit.order.not_checking", source, {"account_id": a.get("account_id")},
                     reason="not a checking account: τ²'s tool rejects the call itself")
    tier = checking_tier(account)
    if tier is None:
        return allow("debit.order.tier_unknown", source, {"level": account.get("level")},
                     reason="the knowledge base gives this account level no tier; fees are not checked")
    option, design = str(a.get("delivery_option", "")).upper(), str(a.get("card_design", "")).upper()
    delivery_fee, design_fee = money(a.get("delivery_fee")), money(a.get("design_fee"))
    excess = money(a.get("excess_replacement_fee"))
    cards = [c for c in db.debit_cards.data.values() if c.get("account_id") == a["account_id"]]
    closed = [c for c in cards if c.get("status") == "CLOSED"]
    details = {"tier": tier, "replacement": bool(closed), "level": account.get("level")}
    reasons, fixes = [], []
    wanted_design = DESIGN_FEE[tier].get(design)
    if wanted_design is not None and design_fee != wanted_design:
        reasons.append(f"design_fee {design_fee:g} for a {design} design on a {tier}-tier account must be "
                       f"{wanted_design:g}")
        fixes.append(f"design_fee {wanted_design:g}")
    if closed:
        offered = REPLACEMENT_DELIVERY[tier]
        if option not in offered:
            reasons.append(f"{option} delivery is not offered for a {tier}-tier replacement (offered: "
                           f"{sorted(offered)})")
            fixes.append(f"one of {sorted(offered)}")
        elif delivery_fee != offered[option]:
            reasons.append(f"delivery_fee {delivery_fee:g} for {option} on a {tier}-tier replacement must be "
                           f"{offered[option]:g}")
            fixes.append(f"delivery_fee {offered[option]:g}")
        recent = [c for c in closed if (d := days_since(c.get("closed_date"))) is not None and d < ENTRY_WAIT_DAYS
                  and c.get("closure_reason") != "fraud_suspected"]
        if tier == "entry" and recent:
            # τ²'s task_091 orders an entry-tier replacement right after a fraud_suspected closure, so the wait is
            # applied to the other closure reasons only
            when = max(as_date(c["closed_date"]) for c in recent) + timedelta(days=ENTRY_WAIT_DAYS)
            return deny("debit.order.entry_wait", source, "an entry-tier replacement waits 48 hours after the card "
                        "closure", f"Do not order it now; tell the user a replacement can be ordered from "
                        f"{when:%m/%d/%Y}.", details | {"closed": [c["card_id"] for c in recent]},
                        consequence="No replacement can be ordered in this conversation; the closure stands.",
                        facts=[f"account {a['account_id']}: {tier} tier", *[
                            f"card {c['card_id']} closed {c.get('closed_date')} ({c.get('closure_reason')})"
                            for c in recent]])
        limit, fee = REPLACEMENT_LIMIT[tier]
        used = _replacements_in_year(cards)
        details |= {"replacements_past_12_months": [c["card_id"] for c in used], "limit": limit}
        if limit is not None and len(used) >= limit:
            oldest = min(as_date(c["issue_date"]) for c in used) + timedelta(days=365)
            if fee is None:
                return deny("debit.order.replacement_limit", source, "this account has used its replacements for "
                            "the year", "Do not order it; tell the user a replacement can be ordered once the oldest "
                            f"ages out ({oldest:%m/%d/%Y}).", details,
                            consequence="This tier has no paid option, so no replacement can be ordered before the "
                            "date above.", facts=[f"account {a['account_id']}: {tier} tier, limit {limit} a year",
                                                  f"replacements in the past 12 months: {len(used)}"])
            if excess != fee:
                return deny("debit.order.replacement_limit", source, "this account has used its replacements for "
                            "the year, so another one carries the excess replacement fee", "Offer the user the "
                            f"choice: wait until {oldest:%m/%d/%Y}, or pay the ${fee:g} excess replacement fee "
                            f"(excess_replacement_fee {fee:g}). Order only if the user accepts the fee.", details,
                            consequence="The customer decides: order now with the fee, or wait without it.",
                            facts=[f"account {a['account_id']}: {tier} tier, limit {limit} a year",
                                   f"replacements in the past 12 months: {len(used)}",
                                   f"excess_replacement_fee given: {excess:g}, due: {fee:g}"])
        elif excess:
            reasons.append(f"excess_replacement_fee {excess:g}: the account is within its replacement limit")
            fixes.append("excess_replacement_fee 0")
    elif option in BASE_DELIVERY:
        allowed_fees = {BASE_DELIVERY[option], REPLACEMENT_DELIVERY[tier].get(option, BASE_DELIVERY[option])}
        if delivery_fee not in allowed_fees:
            reasons.append(f"delivery_fee {delivery_fee:g} for {option} must be one of {sorted(allowed_fees)}")
            fixes.append(f"delivery_fee {BASE_DELIVERY[option]:g}")
    if reasons:
        return deny("debit.order.fees", source, "the order's fees do not match the account's tier",
                    f"Order it with {', '.join(fixes)}.", details,
                    consequence="Fees follow the checking tier; tell the customer the correct fees before ordering.",
                    facts=[f"account {a['account_id']}: {tier} tier", *reasons])
    return allow("debit.order", source, details)


def activate(tool: str):
    source = "bank_accounts_(general)_024"

    def gate(db, a: dict) -> GateResult:
        card = _card(db, a)
        if card is None:
            return _unknown_card("debit.activate", source, a)
        reason = card.get("issue_reason", "new_account")
        right = next((t for t, reasons in ACTIVATE.items() if reason in reasons), None)
        details = {"issue_reason": reason, "status": card.get("status"), "tool": right}
        if right is not None and right != tool:
            return deny("debit.activate.variant", source, "this card is activated with a different activation tool",
                        f"Activate it with {right}.", details,
                        consequence="The activation tool follows why the card was issued; nothing else changes.",
                        facts=[f"card {a['card_id']}: issue_reason '{reason}'",
                               f"{right} activates '{reason}' cards"])
        if card.get("status") != "PENDING":
            return deny("debit.activate.status", source, "only a PENDING card can be activated", "Do not activate "
                        "it.", details, consequence="The card's status is unchanged; tell the customer its current "
                        "status.", facts=[f"card {a['card_id']}: {card.get('status')}"])
        expires = as_date(card.get("expiration_date"))
        if expires is not None and expires < TODAY:
            return deny("debit.activate.expired", source, "an expired card cannot be activated", "Do not activate "
                        "it; the customer needs a new card.", details,
                        consequence="The customer needs a replacement card; offer to order one.",
                        facts=[f"card {a['card_id']} expired {expires:%m/%d/%Y}"])
        return allow("debit.activate", source, details)
    return gate
