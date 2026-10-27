import json
from pathlib import Path

import pytest

from glint.twins import contradicts, find_choice_points, make_twins

DATA = Path(__file__).resolve().parents[2] / "vendor" / "tau2-bench" / "data" / "tau2" / "domains" / "retail"


@pytest.fixture(scope="module")
def db():
    return json.loads((DATA / "db.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tasks():
    return {str(t["id"]): t for t in json.loads((DATA / "tasks.json").read_text(encoding="utf-8"))}


def test_task_98_choice_points(db, tasks):
    points = find_choice_points(tasks["98"], db)
    kinds = {(p.kind, p.order_id) for p in points}
    # the bicycle/puzzle exchange has variant and payment choices; Sofia Li has two cards and PayPal
    assert ("variant", "#W3916020") in kinds and ("payment", "#W3916020") in kinds
    pay = next(p for p in points if p.kind == "payment" and p.order_id == "#W3916020")
    assert pay.gold_value == "credit_card_8105988" and "credit_card_3951670" in pay.alternatives


def test_returns_only_offer_original_or_gift_card(db, tasks):
    for t in tasks.values():
        for p in find_choice_points(t, db):
            action = t["evaluation_criteria"]["actions"][p.action_index]
            if p.kind == "payment" and action["name"] == "return_delivered_order_items":
                original = db["orders"][p.order_id]["payment_history"][0]["payment_method_id"]
                assert all(a == original or a.startswith("gift_card") for a in p.alternatives)


def test_twins_differ_in_one_fact(db, tasks):
    pay = next(p for p in find_choice_points(tasks["98"], db) if p.kind == "payment" and p.order_id == "#W3916020")
    a, b = make_twins(tasks["98"], pay, db)
    assert a["glint_twin"]["stated_value"] == "credit_card_8105988"
    assert b["glint_twin"]["stated_value"] == "credit_card_3951670"
    assert b["evaluation_criteria"]["actions"][pay.action_index]["arguments"]["payment_method_id"] == "credit_card_3951670"
    assert "8484" in a["user_scenario"]["instructions"]["reason_for_call"]
    assert "6791" in b["user_scenario"]["instructions"]["reason_for_call"]
    # everything else is untouched
    for key in ("initial_state", "description"):
        assert a.get(key) == b.get(key) == tasks["98"].get(key)
    assert a["id"] != b["id"] and tasks["98"]["id"] == "98"


def test_contradiction_needs_the_user_to_have_said_it(db, tasks):
    pay = next(p for p in find_choice_points(tasks["98"], db) if p.kind == "payment" and p.order_id == "#W3916020")
    _, b = make_twins(tasks["98"], pay, db)  # user states the Visa ending in 6791
    args = {"order_id": "#W3916020", "item_ids": ["7758198585"], "new_item_ids": ["5606522780"],
            "payment_method_id": "credit_card_8105988"}
    silent = [{"role": "user", "content": "please exchange my bike"}]
    stated = [{"role": "user", "content": "use my visa card ending in 6791 please"}]
    assert contradicts(b, "exchange_delivered_order_items", args, silent) is None
    assert contradicts(b, "exchange_delivered_order_items", args, stated) is True
    assert contradicts(b, "exchange_delivered_order_items", {**args, "payment_method_id": "credit_card_3951670"},
                       stated) is False
    assert contradicts(b, "cancel_pending_order", {"order_id": "#W8855135"}, stated) is None


def test_variant_contradiction(db, tasks):
    point = next(p for p in find_choice_points(tasks["98"], db) if p.kind == "variant" and p.old_item == "7758198585")
    _, b = make_twins(tasks["98"], point, db)
    want = b["glint_twin"]["stated_value"]
    said = [{"role": "user", "content": " ".join(b["glint_twin"]["stated_phrases"])}]
    base = {"order_id": "#W3916020", "item_ids": ["7758198585", "4068787148"], "payment_method_id": "paypal_8194385"}
    assert contradicts(b, "exchange_delivered_order_items", {**base, "new_item_ids": [want, "6245746168"]}, said) is False
    assert contradicts(b, "exchange_delivered_order_items",
                       {**base, "new_item_ids": [point.gold_value, "6245746168"]}, said) is True
