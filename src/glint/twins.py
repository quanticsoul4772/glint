"""User-intent twins for τ² retail: ground-truth "bad step" labels the deterministic floor cannot produce.

A choice point is a gold write where policy allows more than one value: the replacement variant of an exchanged or
modified item, or the payment method (for returns only the original method or a gift card). For each choice point
two twins are built that differ in one decisive fact, the preference the user states verbatim:
  A states gold's value; B states another policy-valid value, and B's gold action is changed to match.
A write contradicts the stated preference when its value differs from it. The label counts only if every stated
phrase appeared in a user turn before the call (the user simulator really said it), so no LLM decides a label.
Design and evidence: research/5-measurements/M2-twin-label-design.md.
"""
from __future__ import annotations

import copy
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


def find_choice_points(task: dict, db: dict) -> list[ChoicePoint]:
    points = []
    for i, a in enumerate(task["evaluation_criteria"]["actions"]):
        name, args = a["name"], a["arguments"]
        order = db["orders"].get(args.get("order_id"))
        if order is None:
            continue
        if name in VARIANT_WRITES:
            for old, new in zip(args["item_ids"], args["new_item_ids"]):
                variants = _product_of(db, new)["variants"]
                alts = tuple(sorted(v for v, d in variants.items() if d["available"] and v not in (old, new)))
                if alts:
                    points.append(ChoicePoint(str(task["id"]), i, "variant", args["order_id"], old, new, alts))
        if name in PAYMENT_WRITES and "payment_method_id" in args:
            methods = db["users"][order["user_id"]]["payment_methods"]
            if name == "return_delivered_order_items":
                original = order["payment_history"][0]["payment_method_id"]
                valid = [m for m in methods if m == original or m.startswith("gift_card")]
            else:
                valid = list(methods)
            alts = tuple(sorted(m for m in valid if m != args["payment_method_id"]))
            if alts:
                points.append(ChoicePoint(str(task["id"]), i, "payment", args["order_id"], None,
                                          args["payment_method_id"], alts))
    return points


def stated_phrases(point: ChoicePoint, value: str, db: dict) -> tuple[str, ...]:
    """What the user must say for the preference to count as stated (case-insensitive substrings)."""
    if point.kind == "variant":
        return tuple(str(v) for v in _product_of(db, value)["variants"][value]["options"].values())
    order = db["orders"][point.order_id]
    method = db["users"][order["user_id"]]["payment_methods"][value]
    if method["source"] == "credit_card":
        return (str(method["last_four"]),)
    return ("paypal",) if method["source"] == "paypal" else ("gift card",)


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
        twin["glint_twin"] = {"source_task": str(task["id"]), "kind": point.kind, "order_id": point.order_id,
                              "old_item": point.old_item, "stated_value": value,
                              "stated_phrases": list(stated_phrases(point, value, db))}
        out.append(twin)
    return out[0], out[1]


def contradicts(twin: dict, tool: str, args: dict, history: list[dict]) -> bool | None:
    """True / False for a write at a twin's choice point once the preference was stated; None otherwise
    (not this choice point, or the user had not yet said every stated phrase)."""
    meta = twin.get("glint_twin")
    if meta is None or args.get("order_id") != meta["order_id"]:
        return None
    if meta["kind"] == "variant":
        if tool not in VARIANT_WRITES or meta["old_item"] not in args.get("item_ids", []):
            return None
        chosen = args["new_item_ids"][args["item_ids"].index(meta["old_item"])]
    else:
        if tool not in PAYMENT_WRITES or "payment_method_id" not in args:
            return None
        chosen = args["payment_method_id"]
    said = " ".join((m.get("content") or "") for m in history if m.get("role") == "user").lower()
    if not all(p.lower() in said for p in meta["stated_phrases"]):
        return None
    return chosen != meta["stated_value"]
