"""The APY a personal savings account should earn, compiled by hand from τ²'s banking knowledge base.

expected = base rate (or balance tier) + the highest linked-checking boost + the highest credit card bonus
           + the relationship bonus, where one applies.

Sources (knowledge base document ids; `doc_savings_accounts_*` unless named otherwise):
- Base and tiers: bronze_001 2.0; silver_003 2.5 below $10,000, 4.0 at or above; silver_plus_002 3.0 below $15,000,
  4.5 at or above; gold_001 5.5; gold_plus_002 6.0; platinum_001 6.5; platinum_plus_002 7.0; diamond_elite_001 7.5;
  green_(savings)_005 4.0.
- Linked-checking boosts: the pairings are bank_accounts_(general)_012; the percentages are in the checking account
  documents (blue_003, bluest_003, dark_green_004, evergreen_001, gold_years_001, green_(checking)_001,
  green_fee-free_005, light_green_002, purple_001), not in the savings documents 012 points to. Only the highest
  boost applies (bank_accounts_(general)_046).
- Credit card bonuses: each savings level's table (bronze_006/_007, silver_006/_009, silver_plus_009, gold_014,
  gold_plus_009, platinum_010, platinum_plus_009, diamond_elite_011, green_(savings)_003/_009). Only the highest
  applies (bank_accounts_(general)_045).
- Relationship bonus 0.025%: silver_002/_005/_009, silver_plus_002, gold_013 (Gold Rewards Card holders).

CONVENTIONS lists every reading the documents leave open, each tagged with how it was settled. Two are settled by τ²'s
expected actions (tasks 093-097) rather than by the documents, and are labeled so.
"""
from __future__ import annotations

from glint.gates.bankdb import money

BASE = {"Bronze Account": 2.0, "Gold Account": 5.5, "Gold Plus Account": 6.0, "Platinum Account": 6.5,
        "Platinum Plus Account": 7.0, "Diamond Elite Account": 7.5, "Green Account": 4.0}
TIERS = {"Silver Account": (10_000, 2.5, 4.0), "Silver Plus Account": (15_000, 3.0, 4.5)}
BOOST = {  # (checking level, savings level) -> boost, from the checking account documents
    ("Green Account", "Gold Account"): 0.75, ("Green Account", "Silver Account"): 0.25,
    ("Green Fee-Free Account", "Bronze Account"): 0.4, ("Green Fee-Free Account", "Gold Plus Account"): 0.35,
    ("Evergreen Account", "Green Account"): 0.55, ("Evergreen Account", "Diamond Elite Account"): 0.15,
    ("Blue Account", "Silver Plus Account"): 0.35, ("Blue Account", "Platinum Account"): 0.8,
    ("Light Green Account", "Diamond Elite Account"): 0.2, ("Light Green Account", "Platinum Account"): 0.65,
    ("Dark Green Account", "Bronze Account"): 0.85, ("Dark Green Account", "Platinum Plus Account"): 0.45,
    ("Bluest Account", "Silver Account"): 0.45, ("Bluest Account", "Bronze Account"): 0.7,
    ("Purple Account", "Platinum Plus Account"): 0.3, ("Purple Account", "Gold Account"): 0.1,
    ("Gold Years Account", "Gold Plus Account"): 0.5, ("Gold Years Account", "Silver Account"): 0.6,
}
CARDS = ("Bronze Rewards Card", "Silver Rewards Card", "Gold Rewards Card", "Platinum Rewards Card",
         "Diamond Elite Card", "EcoCard", "Green Rewards Card", "Crypto-Cash Back Card")


def _row(*values: float) -> dict[str, float]:
    return dict(zip(CARDS, values, strict=True))


CARD_BONUS = {  # savings level -> card -> bonus (cards a level's documents do not name earn 0)
    "Bronze Account": _row(0.1, 0, 0.25, 0.55, 0.15, 0, 0.2, 0.3),
    "Silver Account": _row(0, 0.1, 0.5, 0.2, 0.25, 2.2, 0, 0.5),
    "Silver Plus Account": _row(0.15, 0.15, 0.2, 0.15, 0.4, 0.45, 0.1, 0),
    "Gold Account": _row(0.15, 0.2, 0.025, 0.15, 0.3, 0.6, 0.35, 0),
    "Gold Plus Account": _row(0.15, 0.1, 0.35, 0.2, 0.25, 0.1, 0.05, 0.3),
    "Platinum Account": _row(0, 0, 0.15, 0.25, 0.35, 0, 0, 0),
    "Platinum Plus Account": _row(0, 0, 0.1, 0.4, 0.6, 0.05, 0, 0.25),
    "Diamond Elite Account": _row(0, 0, 0, 0.1, 0.5, 0, 0, 0.15),
    "Green Account": _row(0, 0.45, 0, 0.35, 0.6, 0.5, 0.4, 0),
}
RELATIONSHIP = 0.025
RELATIONSHIP_LEVELS = {"Silver Account", "Silver Plus Account"}  # Gold: only with a Gold Rewards Card (gold_013)
CONVENTIONS = (
    "bronze_006 and _007 name different cards; their bonuses are combined (no card appears in both).",
    "silver_006 and _009 name different cards; combined (Platinum +0.2, Diamond +0.25 from _009).",
    ("Green (savings) EcoCard bonus 0.5 from _003/_009; the '4.0% + 0.5% = 4.75%' example is an arithmetic error and is "
    "not used."),
    ("gold_013 calls the Gold Rewards Card's 0.025% a relationship bonus (its '5.5% + 0.025% = 6.0%' example is an "
    "arithmetic error); it is read as that card's bonus in gold_014's table, so it does not stack with a higher card."),
    ("FITTED TO τ² GOLD: the relationship bonus applies only when no credit card bonus applies. bank_accounts_(general)"
    "_045 says the two stack; τ²'s expected actions include the 0.025% for a customer with no credit card (task_093) "
    "and leave it out for every customer holding one (tasks 094-097)."),
    ("FITTED TO τ² GOLD: Silver Plus's direct-deposit bonus (silver_plus_004/_006) is not applied; the database records "
    "no direct deposit into a savings account, and τ²'s expected APY for task_097 leaves it out."),
    ("Minimum balance requirements (Diamond Elite $250,000 and others) do not change the APY; the documents tie them to "
    "fees and eligibility, not to the rate."),
)


def expected_apy(db, account_id: str) -> tuple[float, dict] | None:
    """(expected APY, components) for a personal savings account, or None when its level is not compiled."""
    acc = db.accounts.data.get(account_id) or {}
    level, user = acc.get("level"), acc.get("user_id")
    if level not in BASE and level not in TIERS:
        return None
    balance = money(acc.get("current_holdings"))
    if level in TIERS:
        threshold, low, high = TIERS[level]
        base = high if balance >= threshold else low
    else:
        base = BASE[level]
    checking = [a.get("level") for a in db.accounts.data.values()
                if a.get("user_id") == user and a.get("class") == "checking" and a.get("status") in ("OPEN", "ACTIVE")]
    boosts = {c: BOOST.get((c, level), 0.0) for c in checking}
    cards = [c.get("card_type") for c in db.credit_card_accounts.data.values()
             if c.get("user_id") == user and str(c.get("account_status", "ACTIVE")).upper() == "ACTIVE"]
    bonuses = {c: CARD_BONUS[level].get(c, 0.0) for c in cards}
    boost, card = max(boosts.values(), default=0.0), max(bonuses.values(), default=0.0)
    related = level in RELATIONSHIP_LEVELS or (level == "Gold Account" and "Gold Rewards Card" in cards)
    relationship = RELATIONSHIP if related and card == 0 and level != "Gold Account" else 0.0
    total = round(base + boost + card + relationship, 4)
    return total, {"level": level, "base": base, "checking_boosts": boosts, "card_bonuses": bonuses,
                   "boost": boost, "card_bonus": card, "relationship_bonus": relationship, "expected_apy": total}
