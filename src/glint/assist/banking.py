"""Decision tools for τ² banking_knowledge: credit limit increases and credit card disputes.

Rules come from the knowledge base (research/3-project/banking-procedure-rules.md): `credit_card_account_logistics_005`
to `_007` (CLI tiers, order of steps, the checks before a decision), `credit_cards_(general)_014` (`card_action`) and
`_015` (provisional credit). Reads the live `TransactionalDB`; "today" is the domain's fixed date. No language model
is used.
"""
from __future__ import annotations

from datetime import date, datetime

TODAY = date(2025, 11, 14)
CLI_TIERS = {  # account age days, cooldown days, utilization must be below, on-time payments, max increase share
    "entry": {"age": 120, "cooldown": 120, "utilization": 0.70, "on_time": 6, "max_increase": 0.25},
    "mid": {"age": 90, "cooldown": 90, "utilization": 0.80, "on_time": 3, "max_increase": 0.50},
    "premium": {"age": 60, "cooldown": 60, "utilization": 0.90, "on_time": 3, "max_increase": 0.50},
}
PROVISIONAL_MAX = {"entry": 2500.0, "mid": 5000.0, "premium": 10000.0, "elite": 15000.0, "invitation": 25000.0}
PROVISIONAL_REASONS = ("unauthorized_fraudulent_charge", "duplicate_charge", "goods_services_not_received")
OPEN_DISPUTE = ("SUBMITTED", "UNDER_REVIEW", "PENDING")
FINAL_ORDER = ("DELIVERED", "CANCELLED")
BAD_STANDING = ("PAST_DUE", "DELINQUENT", "COLLECTIONS", "CLOSED")
# The bank tools below are hidden (discoverable): each is unlocked once with unlock_discoverable_agent_tool, then
# called through call_discoverable_agent_tool, never by its own name
CLI_STEPS = [
    ("Each tool below is hidden: unlock it once with unlock_discoverable_agent_tool(agent_tool_name=...), then call it "
     "with call_discoverable_agent_tool(agent_tool_name=..., arguments=...). Never call it by its own name."),
    "1. submit_credit_limit_increase_request_7392 (requested_increase_amount must not exceed max_increase)",
    ("2-5. Required lookups, all four, before deciding (the policy requires them even though this verdict already "
     "read the same records): get_user_dispute_history_7291, get_pending_replacement_orders_5765, "
     "get_credit_limit_increase_history_4829, get_payment_history_6183 with months = payment_months"),
    ("6. approve_credit_limit_increase_5847 with new_credit_limit, or deny_credit_limit_increase_5848 with "
     "denial_reason, as this verdict says"),
]
LIMIT_FIRST = ("Do the credit limit increase first (submit, the required lookups, approve or deny) before filing any "
               "dispute or ordering a replacement card: a pending dispute or replacement makes the increase fail.")


def _tier(card_type: str) -> str | None:
    name = card_type.lower()
    for words, tier in ((("bronze", "eco", "crypto"), "entry"), (("silver", "green"), "mid"), (("gold",), "premium"),
                        (("platinum",), "elite"), (("diamond",), "invitation")):
        if any(w in name for w in words):
            return tier
    return None


def _money(value) -> float:
    return float(str(value or 0).replace("$", "").replace(",", ""))


def _date(value) -> date | None:
    """The database mixes MM/DD/YYYY and ISO dates."""
    if not value:
        return None
    return datetime.strptime(value[:10], "%Y-%m-%d" if value[4] == "-" else "%m/%d/%Y").date()


def _account(db, credit_card_account_id: str, user_id: str) -> tuple[dict | None, str | None]:
    acc = db.credit_card_accounts.data.get(credit_card_account_id)
    if acc is None:
        return None, f"credit card account {credit_card_account_id} does not exist"
    if acc.get("user_id") != user_id:
        return None, f"credit card account {credit_card_account_id} does not belong to user {user_id}"
    return acc, None


def _rows(table, **match) -> list[dict]:
    return [r for r in table.data.values() if all(r.get(k) == v for k, v in match.items())]


def check_credit_limit_increase(db, user_id: str, credit_card_account_id: str, requested_increase_amount: float,
                                other_requests: list[str] | None = None) -> dict:
    """The credit limit increase verdict: the maximum allowed increase, every Step-2 check with its value, and
    approve (with the new limit) or deny (with the first failing reason, in the policy's order)."""
    acc, error = _account(db, credit_card_account_id, user_id)
    if error:
        return {"verdict": "error", "reasons": [error]}
    tier = _tier(acc.get("card_type", ""))
    if tier not in CLI_TIERS:
        return {"verdict": "unknown", "reasons": [(f"the knowledge base defines no CLI tier for "
                                                   f"{acc.get('card_type')}; search it before deciding")]}
    if not acc.get("credit_limit"):
        return {"verdict": "unknown", "reasons": [(f"no credit limit is on record for {credit_card_account_id}, so the "
                                                   "maximum increase and utilization cannot be computed")]}
    rules = CLI_TIERS[tier]
    limit, balance = _money(acc.get("credit_limit")), _money(acc.get("current_balance"))
    max_increase = round(limit * rules["max_increase"], 2)
    out = {"tier": tier, "current_limit": limit, "max_increase": max_increase, "payment_months": rules["on_time"],
           "steps": CLI_STEPS}
    others = [r.lower() for r in other_requests or []]
    if any("dispute" in r or "replace" in r or "reissue" in r for r in others):
        out["sequencing"] = LIMIT_FIRST
    if requested_increase_amount <= 0:
        out |= {"verdict": "ask_amount", "reasons": [("ask the user how much increase they want (in dollars); then "
                                                      "call this again with that amount")]}
        return out
    if requested_increase_amount > max_increase:
        out |= {"verdict": "reduce_amount",
                "reasons": [(f"requested increase ${requested_increase_amount:,.2f} exceeds the {tier}-tier maximum "
                             f"${max_increase:,.2f} ({rules['max_increase']:.0%} of the current limit); do not submit "
                             "it, offer the maximum instead")]}
        return out

    opened = _date(acc.get("date_of_account_open"))
    age = (TODAY - opened).days if opened else None
    approved = [r for r in _rows(db.credit_limit_increase_requests, credit_card_account_id=credit_card_account_id)
                if str(r.get("status", "")).upper() == "APPROVED"]
    last_increase = max((d for r in approved
                         if (d := _date(r.get("decision_date") or r.get("submitted_at") or r.get("submitted_date")))),
                        default=None)
    disputes = [r["dispute_id"] for r in _rows(db.transaction_disputes, user_id=user_id)
                if str(r.get("status", "")).upper() in OPEN_DISPUTE]
    orders = [r["order_id"] for r in _rows(db.credit_card_orders, credit_card_account_id=credit_card_account_id)
              if str(r.get("status", "")).upper() not in FINAL_ORDER]
    past_due = _money(acc.get("past_due_amount"))
    standing = str(acc.get("account_status") or acc.get("status") or "").upper()
    utilization = balance / limit if limit else 1.0
    payments = sorted(_rows(db.payment_history, credit_card_account_id=credit_card_account_id),
                      key=lambda p: _date(p.get("payment_date")) or date.min, reverse=True)[:rules["on_time"]]
    on_time = next((i for i, p in enumerate(payments) if p.get("status") != "ON_TIME"), len(payments))
    checks = [  # Step 2, in the order the policy lists
        ("insufficient_account_age", age is not None and age >= rules["age"],
         f"account age {age} days, needs {rules['age']}"),
        ("cooldown_period_active", last_increase is None or (TODAY - last_increase).days >= rules["cooldown"],
         f"last approved increase {last_increase}, cooldown {rules['cooldown']} days"),
        ("pending_disputes", not disputes, f"open disputes {disputes}"),
        ("pending_replacement_card", not orders, f"replacement orders not delivered or cancelled {orders}"),
        ("past_due_balance", past_due <= 0 and standing not in BAD_STANDING,
         f"past due ${past_due:,.2f}, status {standing or 'unknown'}"),
        ("high_utilization", utilization < rules["utilization"],
         f"utilization {utilization:.1%}, must be below {rules['utilization']:.0%}"),
        ("insufficient_payment_history", on_time >= rules["on_time"],
         f"{on_time} consecutive on-time payments, needs {rules['on_time']}"),
    ]
    out["checks"] = {name: {"passed": ok, "detail": detail} for name, ok, detail in checks}
    failed = [name for name, ok, _ in checks if not ok]
    if failed:
        out |= {"verdict": "deny", "denial_reason": failed[0], "reasons": [out["checks"][failed[0]]["detail"]]}
    else:
        out |= {"verdict": "approve", "new_credit_limit": round(limit + requested_increase_amount),
                "reasons": ["every check passed"]}
    return out


def check_dispute(db, user_id: str, credit_card_account_id: str, transaction_id: str, dispute_reason: str,
                  contacted_merchant: bool | None = None, replacement_requested: bool = False,
                  other_requests: list[str] | None = None) -> dict:
    """The field values a credit card dispute must carry: `card_action`, `eligible_for_provisional_credit` (with each
    criterion), the purchase date and card digits, and what still has to be asked of the user. When the user also
    wants a credit limit increase, `sequencing` says to finish it before this dispute: Glimmer called this tool first
    and filed the dispute before the increase, which then failed (task_054 on Modal, 2026-10-27)."""
    acc, error = _account(db, credit_card_account_id, user_id)
    if error:
        return {"ready": False, "reasons": [error]}
    txn = db.credit_card_transaction_history.data.get(transaction_id)
    if txn is None or txn.get("user_id") != user_id:
        return {"ready": False, "reasons": [f"transaction {transaction_id} does not exist for user {user_id}"]}
    replaced_today = [r["order_id"] for r in _rows(db.credit_card_orders, credit_card_account_id=credit_card_account_id)
                      if _date(r.get("order_date")) == TODAY]
    reissue = replacement_requested or bool(replaced_today)
    out = {"card_action": "cancel_and_reissue" if reissue else "keep_active",
           "card_action_reason": (f"a replacement is ordered or requested in this conversation {replaced_today}: "
                                  "use cancel_and_reissue on EVERY dispute for this card") if reissue
           else "no replacement for this card in this conversation", "purchase_date": txn.get("transaction_date"),
           "card_last_4_digits": acc.get("card_last_4_digits"), "amount": _money(txn.get("transaction_amount")),
           "ask_user": []}
    fraud = dispute_reason == "unauthorized_fraudulent_charge"
    if contacted_merchant is None:
        # contacted_merchant is filed as the user's own answer, for every transaction, fraud included
        out["ask_user"].append("Did you try to resolve this with the merchant first? (ask for every transaction)")
    tier = _tier(acc.get("card_type", ""))
    opened, purchased = _date(acc.get("date_of_account_open")), _date(txn.get("transaction_date"))
    year_ago = TODAY.replace(year=TODAY.year - 1)
    filed = [r for r in _rows(db.transaction_disputes, user_id=user_id)
             if (d := _date(r.get("submitted_at"))) and d > year_ago and r.get("transaction_id") != transaction_id]
    criteria = {
        "account_open_60_days": opened is not None and (TODAY - opened).days >= 60,
        "reason_qualifies": dispute_reason in PROVISIONAL_REASONS and (
            dispute_reason != "goods_services_not_received"
            or (purchased is not None and (TODAY - purchased).days > 30)),
        "amount_in_range": tier in PROVISIONAL_MAX and 25.0 <= out["amount"] <= PROVISIONAL_MAX[tier],
        "at_most_2_disputes_past_12_months": len(filed) <= 2,
        "merchant_contacted_if_not_fraud": fraud or bool(contacted_merchant),
    }
    out["provisional_credit_criteria"] = criteria
    out["eligible_for_provisional_credit"] = all(criteria.values())
    # `other_requests` are the user's requests not done yet. The database cannot show that an increase was decided:
    # tau2's approve record reuses the submit request's id, so its insert is dropped and the request stays PENDING
    others = " ".join(other_requests or []).lower()
    if "limit" in others or "cli" in others.split():
        out["sequencing"] = LIMIT_FIRST  # this dispute waits until the increase is approved or denied
    out["ready"] = not out["ask_user"] and "sequencing" not in out
    return out
