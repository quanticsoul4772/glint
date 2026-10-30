"""Decision tools for τ² banking_knowledge: credit limit increases and credit card disputes.

Rules come from the knowledge base (research/3-project/banking-procedure-rules.md): `credit_card_account_logistics_005`
to `_007` (CLI tiers, order of steps, the checks before a decision), `credit_cards_(general)_014` (`card_action`) and
`_015` (provisional credit). Reads the live `TransactionalDB`; "today" is the domain's fixed date. No language model
is used.
"""
from __future__ import annotations

from datetime import date, datetime

from glint.assist import as_flag

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
    # A missing amount is $0 on purpose: τ²'s banking data omits zero fields (past_due_amount is on 1 of the 38 base
    # credit card accounts; tasks add it when it matters), so raising here would break the tool for most accounts.
    # A present but malformed amount still raises (float() of a non-number), naming the bad value.
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
    # Cooldown (logistics_005): counted from the submission date of the most recent request, and only when that
    # request was approved ("Denied requests do not trigger a cooldown period")
    dated = [(d, r) for r in _rows(db.credit_limit_increase_requests, credit_card_account_id=credit_card_account_id)
             if str(r.get("status", "")).upper() in ("APPROVED", "DENIED")
             and (d := _date(r.get("submitted_at") or r.get("submitted_date") or r.get("decision_date")))]
    latest = max(dated, key=lambda x: x[0], default=None)
    last_increase = latest[0] if latest and str(latest[1].get("status", "")).upper() == "APPROVED" else None
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


def replacements_today(db, credit_card_account_id: str) -> list[str]:
    """Replacement orders for this card placed today (in this conversation): its disputes use cancel_and_reissue
    (credit_cards_(general)_014 card_action)."""
    return [r["order_id"] for r in _rows(db.credit_card_orders, credit_card_account_id=credit_card_account_id)
            if _date(r.get("order_date")) == TODAY]


def check_dispute(db, user_id: str, credit_card_account_id: str, transaction_id: str, dispute_reason: str,
                  contacted_merchant: bool | None = None, replacement_requested: bool = False,
                  other_requests: list[str] | None = None, planned_before: int = 0) -> dict:
    """The field values a credit card dispute must carry: `card_action`, `eligible_for_provisional_credit` (with each
    criterion), the purchase date and card digits, and what still has to be asked of the user. When the user also
    wants a credit limit increase, `sequencing` says to finish it before this dispute: Glimmer called this tool first
    and filed the dispute before the increase, which then failed (task_054 on Modal, 2026-10-27). `planned_before`
    counts disputes not filed yet that a plan (plan_disputes) files ahead of this one, for criterion 4."""
    acc, error = _account(db, credit_card_account_id, user_id)
    if error:
        return {"ready": False, "reasons": [error]}
    txn = db.credit_card_transaction_history.data.get(transaction_id)
    if txn is None or txn.get("user_id") != user_id:
        return {"ready": False, "reasons": [f"transaction {transaction_id} does not exist for user {user_id}"]}
    replaced_today = replacements_today(db, credit_card_account_id)
    replacement_requested = as_flag(replacement_requested)
    if contacted_merchant is not None:
        contacted_merchant = as_flag(contacted_merchant)
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
        "at_most_2_disputes_past_12_months": len(filed) + planned_before <= 2,
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


PLAN_TOOL = "plan_disputes"
OVERCHARGE = "incorrect_amount"
CARD_DIGITS_TOOL = "get_card_last_4_digits"
CARD_DIGITS_HOW = (f"the customer's: give them {CARD_DIGITS_TOOL} (give_discoverable_user_tool) with the card's "
                   "credit_card_account_id, and file the digits it reports")
DISPUTE_HISTORY_TOOL = "get_user_dispute_history_7291"
# Glimmer filed without this lookup in 12 of the 37 failed banking-plan-v4-gated runs (039-041, 081; overflows count as
# failures), and τ² logs a read its gold makes in the database it scores (the read log allowlist), so 8 of those 12
# failed on it alone (counts corrected 2026-10-30; first written as "12 of 35 ... on it alone").
# The filing requires it (glint.gates.procedure.REQUIRED_LOOKUPS); an extra read is not logged where gold lacks it.
BEFORE_FILING = (f"Before filing, unlock and call {DISPUTE_HISTORY_TOOL} (call_discoverable_agent_tool) with this "
                 "user_id: the disputes already on record count toward the 2-dispute limit for provisional credit "
                 "(credit_cards_(general)_015 criterion 4). A filing is blocked until this lookup is done.")
OVERCHARGE_NOTE = ("Ask what the customer should have been charged, as written; do not offer or suggest a full refund. "
                   "Plan resolution_requested 'full_refund' only when the customer asks for the whole charge back "
                   "without being offered it.")


def overcharge(charged: float, item: dict) -> tuple[dict | None, str]:
    """(the resolution to file, rejection) for an incorrect_amount item. The customer's explicit
    `resolution_requested: 'full_refund'` wins: the policy says to ask the customer (credit_cards_(general)_014).
    Otherwise a partial refund of the overcharge, from `partial_refund_amount` (the overcharge) or `expected_amount`
    (what should have been charged); (None, "") when the customer has not been asked yet."""
    if item.get("resolution_requested") == "full_refund":
        return {"resolution_requested": "full_refund"}, ""
    given, expected = item.get("partial_refund_amount"), item.get("expected_amount")
    if given in (None, "") and expected in (None, ""):
        return None, ""
    refund = round(_money(given), 2) if given not in (None, "") else round(charged - _money(expected), 2)
    if not 0 < refund < charged:
        return None, (f"the overcharge must be more than $0 and less than the ${charged:,.2f} charged (got "
                      f"${refund:,.2f}); for the whole amount, plan resolution_requested 'full_refund'")
    return {"resolution_requested": "partial_refund", "partial_refund_amount": refund}, ""
DISPUTE_REASONS = ("unauthorized_fraudulent_charge", "duplicate_charge", "incorrect_amount",
                   "goods_services_not_received", "goods_services_not_as_described",
                   "canceled_subscription_still_charging", "refund_never_processed")
COUNT_CRITERION = "at_most_2_disputes_past_12_months"
PLAN_RULE = ("File the disputes in this order, each with its eligible_for_provisional_credit. Provisional credit "
             "needs at most 2 disputes filed in the past 12 months (credit_cards_(general)_015 criterion 4), so the "
             "disputes filed first take the credit; the disputes that meet every other criterion are filed first, "
             "largest amount first.")


def noticed_date(value) -> tuple[str | None, str]:
    """(the customer's issue_noticed_date as MM/DD/YYYY, rejection): the format the dispute record takes
    (credit_cards_(general)_014 field 11); (None, "") when the customer has not been asked yet."""
    if value in (None, ""):
        return None, ""
    text = str(value).strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).strftime("%m/%d/%Y"), ""
        except ValueError:
            continue
    return None, f"issue_noticed_date {text!r} is not a date (MM/DD/YYYY)"


CARD_ACTIONS = ("keep_active", "cancel_and_reissue")
ORDER_TOOL = "order_replacement_credit_card_7291"


def card_actions(db, given: dict[str, set[str]]) -> tuple[dict[str, dict], dict[str, str]]:
    """(card -> its planned card_action, card -> rejection) for the cards of a dispute plan, from the customer's
    answers on each card (credit_cards_(general)_014 card_action: keep the card active, or cancel it and get a
    replacement). One answer per card: the card is cancelled or it is not. A replacement already ordered today for a
    card wins (its disputes use cancel_and_reissue). A card the customer chose to cancel with no replacement ordered
    yet gets the order as the next step."""
    plans, rejected = {}, {}
    for card, answers in given.items():
        ordered = replacements_today(db, card)
        if ordered:
            plans[card] = {"credit_card_account_id": card, "card_action": "cancel_and_reissue",
                           "replacement_ordered_today": ordered}
        elif len(answers) > 1:
            rejected[card] = (f"card {card}: the customer gave both {' and '.join(sorted(answers))}; one answer per "
                              "card (the card is cancelled or it is not)")
        else:
            action = next(iter(answers), None)
            plans[card] = {"credit_card_account_id": card, "card_action": action, "replacement_ordered_today": []}
            if action == "cancel_and_reissue":
                plans[card]["next_step"] = (f"Call {ORDER_TOOL} for {card} first (the customer chose to cancel the "
                                            "card and get a new one), then file its disputes.")
    return plans, rejected


def card_of_transaction(db, user_id: str, txn: dict) -> str:
    """The credit card account a transaction was charged to (by its account id, else by the user's card type)."""
    return txn.get("credit_card_account_id") or next(
        (k for k, v in db.credit_card_accounts.data.items()
         if v.get("user_id") == user_id and v.get("card_type") == txn.get("credit_card_type")), "")


def plan_disputes(db, user_id: str, transactions: list[dict]) -> dict:
    """The filing order of a customer's credit card disputes and each one's provisional credit.

    Policy: `credit_cards_(general)_015` (provisional credit criteria 1-5: account open 60 days, a qualifying reason,
    the amount in the tier's range, at most 2 disputes filed in the past 12 months, the merchant contacted unless the
    charge is fraud). Criterion 4 makes credit depend on filing order, and the policy text gives no order. The order
    used here comes from τ²'s TRAIN gold, not from the policy: in tasks 039, 040 and 041 the expected actions file
    every dispute whose reason can qualify first (unauthorized_fraudulent_charge, duplicate_charge,
    goods_services_not_received older than 30 days), then the rest, and credit goes to the first qualifying ones until
    criterion 4 stops it. The first group is the disputes that meet every criterion but 4 (in 039-041 these are
    exactly the qualifying reasons; task_038's gold files a duplicate_charge whose merchant was not contacted after a
    non-qualifying reason, so a qualifying reason alone does not put a dispute first). Inside the first group the
    largest amount is filed first: τ²'s train gold files those disputes from the largest to the smallest amount in
    tasks 037, 038, 039, 040, 041 and 081. That is not in the policy text either, and those six tasks are tuned tasks
    (research/00-decisions.md, 2026-10-28). The second group keeps the given order (no credit is at stake there); equal
    amounts keep the given order too.

    An incorrect_amount dispute is filed as resolution_requested 'partial_refund' with partial_refund_amount = the
    overcharge (the amount charged minus the amount the customer should have been charged). The policy only says to
    ask the customer which resolution they want (credit_cards_(general)_014); τ²'s train gold files every
    incorrect_amount dispute that way (tasks 039, 040, 041; tuned tasks), and Glimmer, asking "full refund for all?",
    filed full_refund with no amount in 10 of 12 gated runs of those tasks (2026-10-29). So the plan asks the customer
    the amount they should have been charged, for each such dispute, until the item carries `expected_amount` or the
    overcharge itself as `partial_refund_amount`. The customer's own choice wins: an item with resolution_requested
    'full_refund' (the customer asked for the whole charge back) is planned as a full refund with no amount.

    `transactions`: the disputes the customer wants, each {transaction_id, dispute_reason, contacted_merchant,
    issue_noticed_date, card_action} (the last three None when not asked yet; the date is filed as MM/DD/YYYY; one
    card_action per card, see card_actions), and for incorrect_amount `expected_amount` or `partial_refund_amount`. Criterion 4
    counts the disputes on record in the past 12 months plus the planned ones ahead. A transaction already disputed is listed under `already_filed`; an unknown transaction, a reason
    outside τ²'s list or a repeated transaction under `rejected`. Reads the live database only; deterministic."""
    user_id = str(user_id or "")
    year_ago = TODAY.replace(year=TODAY.year - 1)
    on_record = [r for r in _rows(db.transaction_disputes, user_id=user_id)
                 if (d := _date(r.get("submitted_at"))) and d > year_ago]
    disputed = {r.get("transaction_id") for r in _rows(db.transaction_disputes, user_id=user_id)}
    wanted, rejected, already, seen = [], [], [], set()
    for item in transactions or []:
        item = item if isinstance(item, dict) else {}
        tid, reason = str(item.get("transaction_id") or ""), str(item.get("dispute_reason") or "")
        contacted = item.get("contacted_merchant")
        contacted = None if contacted is None or contacted == "" else as_flag(contacted)
        txn = db.credit_card_transaction_history.data.get(tid)
        why = ("listed twice" if tid in seen else
               f"transaction {tid!r} does not exist for user {user_id}" if txn is None or txn.get("user_id") != user_id
               else f"dispute_reason {reason!r} is not one of {list(DISPUTE_REASONS)}"
               if reason not in DISPUTE_REASONS else "")
        refund, noticed = None, None
        if not why and reason == OVERCHARGE:
            refund, why = overcharge(_money(txn.get("transaction_amount")), item)
        if not why:
            noticed, why = noticed_date(item.get("issue_noticed_date"))
        action = item.get("card_action") or None
        if not why and action is not None and action not in CARD_ACTIONS:
            why = f"card_action {action!r} is not one of {list(CARD_ACTIONS)}"
        seen.add(tid)
        if why:
            rejected.append({"transaction_id": tid, "reason": why})
        elif tid in disputed:
            already.append(tid)
        else:
            wanted.append((tid, reason, contacted, card_of_transaction(db, user_id, txn), refund, noticed, action))
    given: dict[str, set[str]] = {}
    for w in wanted:
        given.setdefault(w[3], set()).update({w[6]} - {None})
    cards, mixed = card_actions(db, given)
    rejected += [{"transaction_id": w[0], "reason": mixed[w[3]]} for w in wanted if w[3] in mixed]
    wanted = [w for w in wanted if w[3] not in mixed]
    first = [check_dispute(db, user_id, card, tid, reason, contacted) for tid, reason, contacted, card, *_ in wanted]
    qualifies = [all(ok for name, ok in v.get("provisional_credit_criteria", {"none": False}).items()
                     if name != COUNT_CRITERION) for v in first]
    amount = {tid: _money(db.credit_card_transaction_history.data[tid].get("transaction_amount"))
              for tid, *_ in wanted}
    credit_group = sorted((w for w, q in zip(wanted, qualifies) if q), key=lambda w: -amount[w[0]])  # stable
    ordered = credit_group + [w for w, q in zip(wanted, qualifies) if not q]
    order, ask = [], []
    for tid, reason, contacted, card, refund, noticed, _ in ordered:
        v = check_dispute(db, user_id, card, tid, reason, contacted, planned_before=len(order))
        if "provisional_credit_criteria" not in v:
            rejected.append({"transaction_id": tid, "reason": "; ".join(v["reasons"])})
            continue
        entry = {"position": len(order) + 1, "transaction_id": tid, "dispute_reason": reason,
                 "contacted_merchant": contacted, "issue_noticed_date": noticed,
                 "card_action": cards[card]["card_action"],
                 "eligible_for_provisional_credit": v["eligible_for_provisional_credit"],
                 "can_get_credit": all(ok for name, ok in v["provisional_credit_criteria"].items()
                                       if name != COUNT_CRITERION),
                 "disputes_counted_before": len(on_record) + len(order),
                 "provisional_credit_criteria": v["provisional_credit_criteria"]}
        if v.get("card_last_4_digits"):
            # the account record shows them: file those, do not give the customer the tool (054, 2026-10-30)
            entry["card_last_4_digits"] = v["card_last_4_digits"]
        else:
            # Glimmer filed card names or '0000' when it never gave the customer the digits tool (banking-plan-
            # v2-gated, the only difference in 6 runs, 2026-10-29); the digits are the customer's to look up
            entry["card_last_4_digits_from"] = CARD_DIGITS_HOW
        questions = []
        if contacted is None:
            questions.append(f"Did you try to resolve the {tid} charge with the merchant first?")
        if noticed is None:
            # credit_cards_(general)_014 field 11 is the customer's answer; banking-plan-v5-gated task_054 trial 2
            # filed today's 11/14/2025 for a charge the customer noticed on 11/12/2025 (its simulations file)
            questions.append(f"When did you first notice the {tid} charge?")
        if entry["card_action"] is None:
            # credit_cards_(general)_014 card_action is the customer's choice; in banking-plan-v5-gated task_039
            # trials 2-3 filed all 8 disputes with the wrong card_action and trials 0-1 never ordered the replacement
            # (the plan's next_step). One question per card.
            questions.append(f"For the {card} card: keep the card active, or cancel it and get a new one?")
        if reason == OVERCHARGE:
            # for the agent, not the customer: Glimmer's "full refund for all?" was agreed to and filed as the
            # customer's choice (039-041, 2026-10-29); a suggested full refund is not the customer asking for one
            entry["note"] = OVERCHARGE_NOTE
            if refund is None:
                questions.append(f"What amount should you have been charged for the {tid} charge "
                                 f"(${amount[tid]:,.2f})?")
            else:
                entry |= refund
        if questions:
            entry["ask_user"] = " ".join(questions)
            ask.extend(q for q in questions if q not in ask)
        order.append(entry)
    return {"user_id": user_id, "rule": PLAN_RULE, "before_filing": BEFORE_FILING,
            "disputes_on_record_past_12_months": len(on_record),
            "order": order, "card_actions": [cards[c] for c in dict.fromkeys(w[3] for w in ordered)],
            "already_filed": already, "rejected": rejected, "ask_user": ask,
            "ready": bool(order) and not ask and not rejected}
