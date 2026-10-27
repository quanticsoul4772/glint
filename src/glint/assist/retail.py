"""Decision tools for τ² retail: deterministic answers to the analytical questions Glimmer otherwise works out itself.

Every function reads the live retail database (`tau2.domains.retail.data_model.RetailDB`) and returns a plain dict:
a verdict, the policy rule it rests on (quoted from the retail policy), and the evidence. No language model is used.
"""
from __future__ import annotations

ACTIONS = ("cancel", "modify_address", "modify_payment", "modify_items", "return", "exchange")

RULES = {
    "pending": "An order can only be cancelled/modified if its status is 'pending'.",
    "delivered": "An order can only be returned/exchanged if its status is 'delivered'.",
    "same_product": "Each item can be changed to an available new item of the same product but of different "
                    "product option. There cannot be any change of product types.",
    "refund_method": "The refund must either go to the original payment method, or an existing gift card.",
    "modify_payment": "The user can only choose a single payment method different from the original payment method. "
                      "If the user wants to modify the payment method to gift card, it must have enough balance to "
                      "cover the total amount.",
    "difference": "The user must provide a payment method to pay or receive refund of the price difference. If the "
                  "user provides a gift card, it must have enough balance to cover the price difference.",
}


def _product_of(db, item_id: str):
    return next((p for p in db.products.values() if item_id in p.variants), None)


def _method(db, user_id: str, method_id: str):
    return db.users[user_id].payment_methods.get(method_id)


def _covers(method, amount: float) -> bool:
    return getattr(method, "source", "") != "gift_card" or amount <= 0 or method.balance >= amount


def check_eligibility(db, order_id: str, action: str, payment_method_id: str | None = None,
                      item_ids: list[str] | None = None, new_item_ids: list[str] | None = None) -> dict:
    """Whether `action` is allowed on `order_id` now, which payment methods may be used, and the price difference."""
    if action not in ACTIONS:
        return {"allowed": False, "reasons": [f"unknown action {action!r}; one of {list(ACTIONS)}"]}
    order = db.orders.get(order_id) or db.orders.get(f"#{order_id.lstrip('#')}")
    if order is None:
        return {"allowed": False, "reasons": [f"order {order_id} does not exist"]}
    reasons, rules, details = [], [], {"order_id": order.order_id, "status": order.status}
    needs = "pending" if action in ("cancel", "modify_address", "modify_payment", "modify_items") else "delivered"
    if order.status != needs:
        reasons.append(f"status is {order.status!r}, action requires {needs!r}")
        rules.append(RULES[needs])
    original = order.payment_history[0].payment_method_id if order.payment_history else None
    details["original_payment_method"] = original
    methods = db.users[order.user_id].payment_methods
    order_items = {i.item_id: i for i in order.items}

    amount = 0.0
    if action in ("modify_items", "exchange"):
        item_ids, new_item_ids = item_ids or [], new_item_ids or []
        item_errors = []
        if len(item_ids) != len(new_item_ids) or not item_ids:
            item_errors.append("item_ids and new_item_ids must be non-empty and the same length")
        for old, new in zip(item_ids, new_item_ids):
            if old not in order_items:
                item_errors.append(f"item {old} is not in order {order.order_id}")
                continue
            product = _product_of(db, new)
            if product is None or product.product_id != order_items[old].product_id:
                item_errors.append(f"{new} is not a variant of the same product as {old}")
                rules.append(RULES["same_product"])
            elif not product.variants[new].available:
                item_errors.append(f"variant {new} is not available")
            elif new == old:
                item_errors.append(f"{new} is the same item as {old}")
        reasons += item_errors
        if not item_errors:
            amount = sum(_product_of(db, n).variants[n].price for n in new_item_ids) - \
                sum(order_items[o].price for o in item_ids)
            details["price_difference"] = round(amount, 2)
        allowed_methods = [m for m, d in methods.items() if _covers(d, amount)]
        rules.append(RULES["difference"])
    elif action == "modify_payment":
        amount = sum(i.price for i in order.items)
        allowed_methods = [m for m, d in methods.items() if m != original and _covers(d, amount)]
        rules.append(RULES["modify_payment"])
    elif action == "return":
        allowed_methods = [m for m in methods if m == original or m.startswith("gift_card")]
        rules.append(RULES["refund_method"])
    else:
        allowed_methods = []
    if action in ("return", "exchange") and item_ids:
        missing = [i for i in item_ids if i not in order_items]
        if missing and action == "return":
            reasons.append(f"items {missing} are not in order {order.order_id}")
    details["allowed_payment_methods"] = allowed_methods
    if payment_method_id is not None and allowed_methods and payment_method_id not in allowed_methods:
        reasons.append(f"payment method {payment_method_id} is not allowed here; allowed: {allowed_methods}")
    return {"allowed": not reasons, "reasons": reasons, "rules": list(dict.fromkeys(rules)), "details": details}


def find_items(db, user_id: str, product_name: str | None = None, options: dict | None = None,
               order_status: str | None = None) -> dict:
    """Every item across the user's orders, filtered by product name (case-insensitive substring), exact option values
    and order status. Use it to find which item and order the user means."""
    user = db.users.get(user_id)
    if user is None:
        return {"items": [], "reasons": [f"user {user_id} does not exist"]}
    out = []
    for oid in user.orders:
        order = db.orders[oid]
        if order_status and order.status != order_status:
            continue
        for it in order.items:
            if product_name and product_name.lower() not in it.name.lower():
                continue
            if options and any(str(it.options.get(k, "")).lower() != str(v).lower() for k, v in options.items()):
                continue
            out.append({"order_id": oid, "order_status": order.status, "item_id": it.item_id,
                        "product_id": it.product_id, "name": it.name, "options": it.options, "price": it.price})
    return {"items": out, "count": len(out)}


def find_variants(db, item_id: str, require: dict | None = None, keep_same: list[str] | None = None,
                  available_only: bool = True, sort_by_price: str = "desc") -> dict:
    """Variants of `item_id`'s product that match `require` (exact option values) and keep the options named in
    `keep_same` equal to the current item's, sorted by price. The first result answers "most expensive"
    (sort_by_price="desc") or "cheapest" ("asc")."""
    product = _product_of(db, item_id)
    if product is None:
        return {"variants": [], "reasons": [f"item {item_id} is not in the catalogue"]}
    current = product.variants[item_id].options
    out = []
    for vid, v in product.variants.items():
        if vid == item_id or (available_only and not v.available):
            continue
        if require and any(str(v.options.get(k, "")).lower() != str(val).lower() for k, val in require.items()):
            continue
        if keep_same and any(v.options.get(k) != current.get(k) for k in keep_same):
            continue
        out.append({"item_id": vid, "options": v.options, "price": v.price, "available": v.available})
    out.sort(key=lambda x: x["price"], reverse=sort_by_price == "desc")
    return {"product": product.name, "current_item": {"item_id": item_id, "options": current}, "variants": out,
            "count": len(out)}
