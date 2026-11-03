"""Debit card write rules, from the knowledge base's debit card procedures. Each rule cites its source document.

- freeze_debit_card_3892 / unfreeze_debit_card_3893: bank_accounts_(general)_026 (status rules; a card that will be
  closed is unfrozen first, because close_debit_card_4721 only closes ACTIVE or PENDING cards).
- close_debit_card_4721: bank_accounts_(general)_025 (no pending transactions on the linked account; the
  14-day minimum card age unless the card is lost, stolen or fraud_suspected; a frozen card is unfrozen first).
- order_debit_card_5739: bank_accounts_(general)_023 (base fees) and _029 (replacement limits, the entry-tier waiting
  period, delivery and design fees by checking tier; tiers from _005).
- activate_debit_card_8291/8292/8293: bank_accounts_(general)_024 (the variant follows the card's issue_reason; the
  card must be PENDING, unexpired, on an OPEN account).

- file_debit_card_transaction_dispute_6281: bank_accounts_(general)_031 (the open-dispute limit by checking tier,
  pre-filing requirement 4; the earliest of duplicate charges; card_in_possession on a card closed as lost or stolen;
  no liability cap of 0) and _032 (provisional credit where every REQUIRED condition holds), and the customer's
  answers to _031's two closed questions as recorded with record_dispute_answers (`dispute_answers`).

Whether a card is lost for certain or only misplaced (freeze first, then decide) is the user's statement, not the
database's, so no gate here requires a freeze before a close: τ²'s train tasks close without a freeze in 084 and
086-092 and freeze first in 081.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from glint.assist import as_flag
from glint.assist.banking import DISPUTE_ANSWERS_TOOL, TODAY
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


# ---- debit card disputes (bank_accounts_(general)_031/_032) -------------------------------------------------------

DISPUTE = "file_debit_card_transaction_dispute_6281"
DISPUTE_SOURCE = "bank_accounts_(general)_031"
OPEN_DISPUTE_MAX = {"entry": 2, "mid": 3, "premium": 4, "elite": 5}  # _031 pre-filing requirement 4, per account
# Glint convention: a dispute is open until it reaches a final status (τ²'s tool files every dispute as OPEN, and the
# training data holds only OPEN debit disputes)
FINAL_DISPUTE_STATUSES = ("RESOLVED", "CLOSED", "APPROVED", "DENIED", "REJECTED", "WITHDRAWN", "CANCELLED")
FRAUD_CATEGORIES = ("unauthorized_transaction", "card_present_fraud", "card_not_present_fraud")
# _032: the categories for which provisional credit is REQUIRED when every other condition holds
CREDIT_REQUIRED_CATEGORIES = (*FRAUD_CATEGORIES, "atm_cash_discrepancy", "duplicate_charge")
LOST_OR_STOLEN = ("lost", "stolen")
DUPLICATE_TIE_BREAK = ("same date with no time: the twin stored later is taken as the earlier one (fitted to τ² gold, "
                       "training tasks 084 and 085; τ²'s get_bank_account_transactions_9173 lists same-date records in "
                       "stored order under its most-recent-first contract)")
QUESTIONS = ("\"Do you still have your physical debit card in your possession?\" and \"Do you believe your PIN may "
             "have been compromised?\" (yes_shared, yes_observed, no or unknown)")


def _when(value) -> datetime | None:
    """A transaction date as the database holds it: MM/DD/YYYY, with or without HH:MM:SS."""
    for fmt in ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y"):
        try:
            return datetime.strptime(str(value or ""), fmt)
        except ValueError:
            continue
    return None


def _filed_with(a: dict, **changes) -> str:
    """The corrected call, as the changed arguments."""
    changed = ", ".join(f"{k} {v}" for k, v in changes.items())
    return f"{changed} (every other argument as in this call)"


def dispute_open_limit(db, a: dict) -> GateResult:
    """_031 pre-filing requirement 4: no more open disputes on the account than its checking tier allows. Counted
    before this filing, so the filing that reaches the maximum is allowed (training gold 084 and 086 file up to it)."""
    account_id = a.get("account_id", "")
    account = db.accounts.data.get(account_id)
    if account is None or account.get("class") != "checking":
        return allow("debit.dispute.open_limit.not_checking", DISPUTE_SOURCE, {"account_id": account_id},
                     reason="not a checking account: the limit is per checking account")
    tier = checking_tier(account)
    if tier is None:
        return allow("debit.dispute.open_limit.tier_unknown", DISPUTE_SOURCE, {"level": account.get("level")},
                     reason="the knowledge base gives this account level no tier")
    limit = OPEN_DISPUTE_MAX[tier]
    open_ = sorted(k for k, d in db.debit_card_disputes.data.items() if d.get("account_id") == account_id
                   and str(d.get("status", "")).upper() not in FINAL_DISPUTE_STATUSES)
    details = {"tier": tier, "limit": limit, "open": open_}
    if len(open_) >= limit:
        return deny("debit.dispute.open_limit", DISPUTE_SOURCE + " requirement 4", "the account already has the most "
                    "open disputes its checking tier allows", "Do not file it. Tell the customer the account has "
                    f"{len(open_)} open disputes, the maximum for a {tier}-tier account, and that this one can be "
                    "filed once an open dispute is resolved.", details,
                    consequence="No dispute can be filed on this account in this conversation; the limit is per "
                    "account, so a dispute on another of the customer's accounts is not affected.",
                    facts=[f"account {account_id}: {tier} tier, at most {limit} open disputes",
                           f"open disputes on it: {len(open_)} ({', '.join(open_)})"])
    return allow("debit.dispute.open_limit", DISPUTE_SOURCE, details)


def dispute_earliest_duplicate(db, a: dict) -> GateResult:
    """_031: "Dispute the earliest (first) transaction when multiple duplicates exist." Duplicates are the account's
    transactions with the same description and amount; a twin already disputed is not offered instead. On the same
    date the tie-break is fitted to τ² gold (DUPLICATE_TIE_BREAK)."""
    if a.get("dispute_category") != "duplicate_charge":
        return allow("debit.dispute.duplicate.not_duplicate", DISPUTE_SOURCE, {"category": a.get("dispute_category")})
    history = db.bank_account_transaction_history.data
    tx = history.get(a.get("transaction_id", ""))
    if tx is None:
        return allow("debit.dispute.duplicate.unknown_transaction", DISPUTE_SOURCE,
                     {"transaction_id": a.get("transaction_id")}, reason="no such transaction")
    disputed = {d.get("transaction_id") for d in db.debit_card_disputes.data.values()}
    twins = [(i, k, t) for i, (k, t) in enumerate(history.items())
             if t.get("account_id") == tx.get("account_id") and t.get("description") == tx.get("description")
             and money(t.get("amount")) == money(tx.get("amount")) and (k == a["transaction_id"] or k not in disputed)]
    if len(twins) < 2 or any(_when(t.get("date")) is None for _, _, t in twins):
        return allow("debit.dispute.duplicate", DISPUTE_SOURCE, {"twins": [k for _, k, _ in twins]})
    _, first, row = min(twins, key=lambda x: (_when(x[2].get("date")), -x[0]))
    details = {"twins": [k for _, k, _ in twins], "earliest": first}
    if first == a["transaction_id"]:
        return allow("debit.dispute.duplicate", DISPUTE_SOURCE, details)
    same_day = _when(row.get("date")) == _when(tx.get("date"))
    fix = _filed_with(a, transaction_id=first, transaction_date=str(row.get("date"))[:10])
    return deny("debit.dispute.duplicate.earliest", DISPUTE_SOURCE, "a duplicate charge is disputed on the earliest "
                "of the duplicates", f"File it with {fix}.", details,
                consequence="The same charge is disputed either way; only which of the duplicates carries the "
                "dispute changes.",
                facts=[f"duplicates of {tx.get('description')} {money(tx.get('amount')):g} on {tx.get('account_id')}: "
                       + ", ".join(f"{k} ({t.get('date')})" for _, k, t in twins),
                       f"earliest: {first}" + (f"; {DUPLICATE_TIE_BREAK}" if same_day else "")])


def dispute_card_in_possession(db, a: dict) -> GateResult:
    """_031 argument 10, by inference: a card the database shows CLOSED as lost or stolen is not in the customer's
    possession. A fraud_suspected closure is exempt: training task 091 files on such a card with possession true."""
    card = _card(db, a)
    if card is None:
        return _unknown_card("debit.dispute.card_in_possession", DISPUTE_SOURCE, a)
    reason = card.get("closure_reason")
    details = {"status": card.get("status"), "closure_reason": reason}
    if card.get("status") == "CLOSED" and reason in LOST_OR_STOLEN and as_flag(a.get("card_in_possession")):
        return deny("debit.dispute.card_in_possession", DISPUTE_SOURCE + " argument 10 (policy inference)",
                    f"the card was closed as {reason}, so it is not in the customer's possession",
                    f"File it with {_filed_with(a, card_in_possession='false')}.", details,
                    consequence="card_in_possession feeds the fraud classification of the dispute; nothing else "
                    "about the filing changes.",
                    facts=[f"card {a['card_id']}: CLOSED, closure_reason '{reason}'",
                           f"card_in_possession given: {a.get('card_in_possession')!r}"])
    return allow("debit.dispute.card_in_possession", DISPUTE_SOURCE, details)


def dispute_provisional_credit(db, a: dict) -> GateResult:
    """_032, one direction only: provisional credit is REQUIRED when all four conditions hold and no NOT REQUIRED
    case applies; anywhere else the credit is the bank's discretion and the flag is not judged. Timely reporting
    (condition 1) counts as met only when the transaction is at most 60 days old: the statement showing it is dated on
    or after it (policy inference; the database holds no statement dates)."""
    source = "bank_accounts_(general)_032"
    category = a.get("dispute_category")
    account = db.accounts.data.get(a.get("account_id", "")) or {}
    tx = db.bank_account_transaction_history.data.get(a.get("transaction_id", "")) or {}
    when = _when(tx.get("date")) or _when(a.get("transaction_date"))
    age = (TODAY - when.date()).days if when else None
    opened = days_since(account.get("date_opened"))
    pin = a.get("pin_compromised")
    online = category == "card_not_present_fraud" or a.get("transaction_type") == "online_purchase"
    conditions = [
        (category in CREDIT_REQUIRED_CATEGORIES, f"category '{category}' is one that requires it"),
        (age is not None and age <= 60, (f"the transaction is {age} days old, so reported within 60 days of its "
         "statement")),
        (as_flag(a.get("written_statement_provided")), "written_statement_provided true"),
        (account.get("status") == "OPEN", f"account {a.get('account_id')} is OPEN"),
        (category in FRAUD_CATEGORIES or as_flag(a.get("contacted_merchant")),
         "a fraud category, or the merchant was contacted (NOT REQUIRED case 2 does not apply)"),
        (pin != "yes_shared", f"pin_compromised '{pin}', not shared (NOT REQUIRED case 3 does not apply)"),
        (not (opened is not None and opened < 30 and online),
         "not a card-not-present dispute on an account under 30 days old (NOT REQUIRED case 4 does not apply)"),
    ]
    details = {"category": category, "transaction_age_days": age, "required": all(ok for ok, _ in conditions)}
    if details["required"] and not as_flag(a.get("provisional_credit_eligible")):
        return deny("debit.dispute.provisional_credit", source, "every condition that makes provisional credit "
                    "REQUIRED holds, and the filing says the customer is not eligible",
                    f"File it with {_filed_with(a, provisional_credit_eligible='true')}.", details,
                    consequence="Regulation E requires the provisional credit here; the rest of the filing stands.",
                    facts=[why for _, why in conditions])
    return allow("debit.dispute.provisional_credit", source, details)


def dispute_liability(db, a: dict) -> GateResult:
    """_031's liability tiers are $50, $500 and unlimited (the tool's -1, tools.py:978); the smallest transaction is
    $1.00, so a cap of 0 is none of them. Which of 50 and 500 applies is keyed on a statement date the database does
    not hold, so the refusal names the tiers, not a value."""
    value = a.get("customer_max_liability_amount")
    if value is None:
        return allow("debit.dispute.liability.missing", DISPUTE_SOURCE, reason="τ²'s tool rejects a missing cap itself")
    cap = money(value)
    if cap == 0:
        return deny("debit.dispute.liability_zero", DISPUTE_SOURCE + " liability tiers", "a liability cap of 0 is not "
                    "one of the policy's tiers", "File it with customer_max_liability_amount set by when the customer "
                    "reported it: 50 within 2 business days of the statement, 500 within 60 days, -1 (unlimited) "
                    "after 60 days.", {"customer_max_liability_amount": cap},
                    consequence="The cap tells the customer what they could lose; the rest of the filing stands.",
                    facts=[f"customer_max_liability_amount given: {value!r}",
                           "_031 tiers: $50, $500, unlimited (the tool's -1)"])
    return allow("debit.dispute.liability", DISPUTE_SOURCE, {"customer_max_liability_amount": cap})


def dispute_answers(toolkit, a: dict) -> GateResult:
    """The filing's card_in_possession and pin_compromised (_031 arguments 10 and 11, closed questions the agent asks
    the customer) equal the customer's latest answers for the card, as recorded with record_dispute_answers
    (`toolkit.glint_dispute_answers`: card id -> the recorded answers; glint.gates.enforce)."""
    source = DISPUTE_SOURCE + " arguments 10-11 (Glint record)"
    card_id = a.get("card_id", "")
    if toolkit.db.debit_cards.data.get(card_id) is None:
        return _unknown_card("debit.dispute.answers", source, a)
    answer = getattr(toolkit, "glint_dispute_answers", {}).get(card_id)
    if answer is None:
        return deny("debit.dispute.answers.not_recorded", source, "the customer's answers to the two dispute "
                    "questions are not recorded for this card", f"Ask the customer {QUESTIONS}, call "
                    f"{DISPUTE_ANSWERS_TOOL}(card_id, user_id, card_in_possession, pin_compromised, customer_words) "
                    "with their answers, then file with them.", {"card_id": card_id},
                    consequence="The filing waits only for the two answers; nothing else about the dispute changes.",
                    facts=[f"card {card_id}: no recorded answers in this conversation"])
    want = {"card_in_possession": answer["card_in_possession"], "pin_compromised": answer["pin_compromised"]}
    got = {"card_in_possession": as_flag(a.get("card_in_possession")), "pin_compromised": a.get("pin_compromised")}
    if got != want:
        fix = _filed_with(a, card_in_possession=str(want["card_in_possession"]).lower(),
                          pin_compromised=want["pin_compromised"])
        return deny("debit.dispute.answers.mismatch", source, "the filing's answers differ from the customer's "
                    "recorded answers", f"File it with {fix}; if the customer's answer for this transaction is "
                    f"different, record it with {DISPUTE_ANSWERS_TOOL} first.", {"recorded": want, "given": got},
                    consequence="Only the two answers change; the rest of the filing stands.",
                    facts=[f"recorded for {card_id}: {want}", f"given: {got}",
                           f"customer's words: {str(answer.get('customer_words', ''))[:200]}"])
    return allow("debit.dispute.answers", source, want)
