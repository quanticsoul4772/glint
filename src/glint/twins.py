"""User-intent twins for τ² retail: ground-truth "bad step" labels the deterministic floor cannot produce.

A choice point is a gold write where policy allows more than one value: the replacement variant of an exchanged or
modified item, or the payment method (for returns only the original method or a gift card). For each choice point
two twins are built that differ in one decisive fact, the preference the user states verbatim:
  A states gold's value; B states another policy-valid value, and B's gold action is changed to match.
A write contradicts the stated preference when its value differs from it. The label counts only if every stated
phrase appeared, as a whole word, in a user turn before the call (the user simulator really said it), so no LLM
decides a label. Alternatives that a gift card could not pay for are never offered (policy).
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass

VARIANT_WRITES = {"exchange_delivered_order_items", "modify_pending_order_items"}
PAYMENT_WRITES = {"exchange_delivered_order_items", "modify_pending_order_items", "modify_pending_order_payment",
                  "return_delivered_order_items"}


@dataclass(frozen=True)
class ChoicePoint:
    task_id: str
    action_index: int
    kind: str              # "variant" | "payment"
    order_id: str
    old_item: str | None   # variant points: the item being replaced
    gold_value: str        # variant item id or payment method id
    alternatives: tuple[str, ...]


def _product_of(db: dict, item_id: str) -> dict:
    return next(p for p in db["products"].values() if item_id in p["variants"])


def _price_difference(db: dict, order: dict, item_ids: list[str], new_item_ids: list[str]) -> float:
    old = {i["item_id"]: i["price"] for i in order["items"]}
    new = sum(_product_of(db, n)["variants"][n]["price"] for n in new_item_ids)
    return new - sum(old[i] for i in item_ids)


def _covers(method: dict, amount: float) -> bool:
    """Policy: a gift card used to pay must have enough balance; other methods always can."""
    return method["source"] != "gift_card" or amount <= 0 or method["balance"] >= amount


def find_choice_points(task: dict, db: dict) -> list[ChoicePoint]:
    points = []
    spent: dict[str, float] = {}  # gift-card charges made by earlier gold writes in this task
    for i, a in enumerate(task["evaluation_criteria"]["actions"]):
        name, args = a["name"], a["arguments"]
        order = db["orders"].get(args.get("order_id"))
        if order is None:
            continue
        methods = {m: {**d, "balance": d.get("balance", 0) - spent.get(m, 0)} if d["source"] == "gift_card" else d
                   for m, d in db["users"][order["user_id"]]["payment_methods"].items()}
        # an item listed twice in one write would make the twin's single fact ambiguous
        if name in VARIANT_WRITES and len(set(args["item_ids"])) == len(args["item_ids"]):
            pay = methods.get(args.get("payment_method_id"))
            for old, new in zip(args["item_ids"], args["new_item_ids"]):
                variants = _product_of(db, new)["variants"]
                alts = []
                for v, d in variants.items():
                    if not d["available"] or v in (old, new):
                        continue
                    swapped = [v if o == old else n for o, n in zip(args["item_ids"], args["new_item_ids"])]
                    if pay is None or _covers(pay, _price_difference(db, order, args["item_ids"], swapped)):
                        alts.append(v)
                if alts:
                    points.append(ChoicePoint(str(task["id"]), i, "variant", args["order_id"], old, new,
                                              tuple(sorted(alts))))
        if name in PAYMENT_WRITES and "payment_method_id" in args:
            if name == "return_delivered_order_items":
                original = order["payment_history"][0]["payment_method_id"]
                valid = [m for m in methods if m == original or m.startswith("gift_card")]
            elif name == "modify_pending_order_payment":
                # policy: the new method must differ from the order's current one and a gift card must cover it
                original = order["payment_history"][0]["payment_method_id"]
                amount = sum(it["price"] for it in order["items"])
                valid = [m for m, d in methods.items() if m != original and _covers(d, amount)]
            else:
                amount = _price_difference(db, order, args["item_ids"], args["new_item_ids"])
                valid = [m for m, d in methods.items() if _covers(d, amount)]
            alts = tuple(sorted(m for m in valid if m != args["payment_method_id"]))
            if alts:
                points.append(ChoicePoint(str(task["id"]), i, "payment", args["order_id"], None,
                                          args["payment_method_id"], alts))
        pm = args.get("payment_method_id", "")
        if pm.startswith("gift_card") and name != "return_delivered_order_items":
            if name == "modify_pending_order_payment":
                spent[pm] = spent.get(pm, 0) + sum(it["price"] for it in order["items"])
            elif name in VARIANT_WRITES:
                spent[pm] = spent.get(pm, 0) + max(_price_difference(db, order, args["item_ids"],
                                                                     args["new_item_ids"]), 0)
    return points


def stated_phrases(point: ChoicePoint, value: str, db: dict) -> tuple[tuple[str | None, str], ...]:
    """(option name or None, value) pairs the user must say for the preference to count as stated."""
    if point.kind == "variant":
        options = _product_of(db, value)["variants"][value]["options"]
        return tuple((str(k), str(v)) for k, v in options.items())
    order = db["orders"][point.order_id]
    method = db["users"][order["user_id"]]["payment_methods"][value]
    if method["source"] == "credit_card":
        return ((None, str(method["last_four"])),)
    return ((None, "paypal"),) if method["source"] == "paypal" else ((None, "gift card"),)


# values too short or too common to count on their own ("M" for size, "low", "none")
_NEEDS_NAME = {"none", "low", "high", "no", "yes", "red", "art", "men", "gas"}


def said(text: str, name: str | None, value: str) -> bool:
    """Whole-word, case-insensitive match of `value` in `text`; short or common values must appear next to their
    option name ("size M", "M size")."""
    text, value = text.lower(), value.lower()
    v = rf"(?<!\w){re.escape(value)}(?!\w)"
    if name is not None and (len(value) <= 2 or value in _NEEDS_NAME):
        n = rf"(?<!\w){re.escape(name.lower())}(?!\w)"
        return re.search(rf"{n}\W{{0,3}}(?:is\W+)?{v}|{v}\W{{0,3}}{n}", text) is not None
    return re.search(v, text) is not None


def preference_sentence(point: ChoicePoint, value: str, db: dict) -> str:
    if point.kind == "variant":
        product = _product_of(db, value)
        opts = ", ".join(f"{k} {v}" for k, v in product["variants"][value]["options"].items())
        return (f" For the replacement {product['name']}, you want exactly this one: {opts}. Say these options to the "
                f"agent in those words. This overrides any other preference about the {product['name']} above.")
    order = db["orders"][point.order_id]
    method = db["users"][order["user_id"]]["payment_methods"][value]
    if method["source"] == "credit_card":
        desc = f"your {method['brand']} card ending in {method['last_four']}"
    elif method["source"] == "paypal":
        desc = "your PayPal account"
    else:
        desc = "your gift card"
    return (f" For order {point.order_id}, you want to use {desc} for any payment or refund. Say it to the agent in "
            f"those words. This overrides any other payment preference for this order above.")


def make_twins(task: dict, point: ChoicePoint, db: dict) -> tuple[dict, dict]:
    """(twin A stating gold's value, twin B stating point.alternatives[0] with gold updated)."""
    out = []
    for tag, value in (("A", point.gold_value), ("B", point.alternatives[0])):
        twin = copy.deepcopy(task)
        item = f"-{point.old_item}" if point.old_item else ""
        twin["id"] = f"{task['id']}-ui{point.action_index}-{point.kind}{item}-{tag}"
        twin["user_scenario"]["instructions"]["reason_for_call"] += preference_sentence(point, value, db)
        args = twin["evaluation_criteria"]["actions"][point.action_index]["arguments"]
        if point.kind == "variant":
            args["new_item_ids"] = [value if old == point.old_item else new
                                    for old, new in zip(args["item_ids"], args["new_item_ids"])]
        else:
            args["payment_method_id"] = value
        if tag == "B":
            # NL assertions and communicate_info state gold's amounts and items, which B changes
            twin["evaluation_criteria"]["nl_assertions"] = None
            twin["evaluation_criteria"]["communicate_info"] = []
        twin["glint_twin"] = {"source_task": str(task["id"]), "kind": point.kind, "order_id": point.order_id,
                              "old_item": point.old_item, "stated_value": value,
                              "stated_phrases": [list(p) for p in stated_phrases(point, value, db)]}
        out.append(twin)
    return out[0], out[1]


def at_choice_point(twin: dict, tool: str, args: dict) -> bool:
    """Whether this write sets the value the twin's stated preference is about."""
    meta = twin.get("glint_twin")
    if meta is None or args.get("order_id") != meta["order_id"]:
        return False
    if meta["kind"] == "variant":
        items, new = args.get("item_ids") or [], args.get("new_item_ids") or []
        # a malformed write (missing or misaligned new_item_ids) sets no value for the choice point
        return tool in VARIANT_WRITES and meta["old_item"] in items and len(new) == len(items)
    return tool in PAYMENT_WRITES and "payment_method_id" in args


def contradicts(twin: dict, tool: str, args: dict, history: list[dict]) -> bool | None:
    """True / False for a write at a twin's choice point once the preference was stated; None otherwise
    (not this choice point, or the user had not yet said every stated phrase)."""
    if not at_choice_point(twin, tool, args):
        return None
    meta = twin["glint_twin"]
    if meta["kind"] == "variant":
        chosen = args["new_item_ids"][args["item_ids"].index(meta["old_item"])]
    else:
        chosen = args["payment_method_id"]
    text = " ".join((m.get("content") or "") for m in history if m.get("role") == "user")
    if not all(said(text, name, value) for name, value in meta["stated_phrases"]):
        return None
    return chosen != meta["stated_value"]
