"""M0 exit check: every label path in glint.labels must fire on an injected positive, and a clean
conversation must produce no labels. Entities are real rows from the τ² retail DB (user raj_sanchez_2970)."""
import pytest
from tau2.data_model.message import ToolCall
from tau2.domains.retail.environment import get_environment

from glint.labeler import TRANSFER_MESSAGE, Labeler, _fresh_db
from glint.labels import FAB, R0, RESID, VIOL, Label, is_resid_positive

EMAIL = "raj.sanchez2046@example.com"
PENDING = "#W4566809"  # pending; items 1569829406 (product 2985987096), 5826601160; paid credit_card_3362387
DELIVERED = "#W7736708"  # delivered; items 3104857380 (product 6992792935), ...; paid credit_card_3362387
CARD, GIFT = "credit_card_3362387", "gift_card_2259499"
OTHER_USER, OTHER_EMAIL = "noah_brown_6181", None
CANCEL_GOLD = {"action_id": "g0", "name": "cancel_pending_order", "arguments": {"order_id": PENDING, "reason": "no longer needed"}}


def transcript(steps):
    """Build τ²-shaped messages; tool results are produced by executing the calls on a scratch env."""
    env, msgs, n = get_environment(db=_fresh_db()), [], 0
    for step in steps:
        kind = step[0]
        if kind in ("user", "assistant"):
            msgs.append({"role": kind, "content": step[1], "tool_calls": None})
        elif kind == "call":
            name, args = step[1], step[2]
            content = step[3] if len(step) > 3 else None
            n += 1
            cid = f"c{n}"
            msgs.append({"role": "assistant", "content": content,
                         "tool_calls": [{"id": cid, "name": name, "arguments": args}]})
            resp = env.get_response(ToolCall(id=cid, name=name, arguments=args, requestor="assistant"))
            msgs.append({"role": "tool", "id": cid, "content": resp.content, "error": resp.error})
    return msgs


def auth_and_fetch(order=PENDING):
    return [
        ("assistant", "Hi! How can I help?"),
        ("user", f"Hi, my email is {EMAIL}. I need help with order {order}."),
        ("call", "find_user_id_by_email", {"email": EMAIL}),
        ("call", "get_order_details", {"order_id": order}),
    ]


def confirm(text="I will do this. Shall I proceed?"):
    return [("assistant", text), ("user", "Yes, please go ahead.")]


def labels_of(steps, gold=(CANCEL_GOLD,)):
    task = {"id": "t0", "evaluation_criteria": {"actions": list(gold)}}
    return Labeler(task).label(transcript(steps))


def last(steps, gold=(CANCEL_GOLD,)):
    return labels_of(steps, gold)[-1].labels


def test_clean_cancel_has_no_labels():
    rows = labels_of(auth_and_fetch() + confirm() +
                     [("call", "cancel_pending_order", {"order_id": PENDING, "reason": "no longer needed"})])
    assert all(not r.labels for r in rows), [(r.tool, r.labels) for r in rows]
    assert not is_resid_positive(rows[-1].labels)


@pytest.mark.parametrize("label,steps", [
    (Label.FAB_SCHEMA, auth_and_fetch() + confirm() + [("call", "cancel_pending_order", {"order_id": PENDING})]),
    (Label.FAB_ID_NONEXISTENT, auth_and_fetch() + [("call", "get_order_details", {"order_id": "#W0000000"})]),
    (Label.FAB_UNGROUNDED, auth_and_fetch() + [("call", "get_order_details", {"order_id": DELIVERED})]),
    (Label.VIOL_R1_AUTH, [("user", f"order {PENDING}"), ("call", "get_order_details", {"order_id": PENDING})]),
    (Label.VIOL_R2_OTHER_USER, auth_and_fetch() + [("user", f"look up {OTHER_USER}"),
                                                   ("call", "get_user_details", {"user_id": OTHER_USER})]),
    (Label.VIOL_R3_NO_STATUS_CHECK, auth_and_fetch()[:3] + [("user", f"cancel {PENDING}")] + confirm() +
     [("call", "cancel_pending_order", {"order_id": PENDING, "reason": "no longer needed"})]),
    (Label.VIOL_R4_CANCEL_NOT_PENDING, auth_and_fetch(DELIVERED) + confirm() +
     [("call", "cancel_pending_order", {"order_id": DELIVERED, "reason": "no longer needed"})]),
    (Label.VIOL_R5_MODIFY_NOT_PENDING, auth_and_fetch(DELIVERED) + confirm() +
     [("call", "modify_pending_order_payment", {"order_id": DELIVERED, "payment_method_id": GIFT})]),
    (Label.VIOL_R6_NOT_DELIVERED, auth_and_fetch() + confirm() +
     [("call", "return_delivered_order_items", {"order_id": PENDING, "item_ids": ["1569829406"],
                                                 "payment_method_id": CARD})]),
    (Label.VIOL_R8_CANCEL_REASON, auth_and_fetch() + [("user", "reason: changed my mind")] + confirm() +
     [("call", "cancel_pending_order", {"order_id": PENDING, "reason": "changed my mind"})]),
    (Label.VIOL_R9_PAYMENT, auth_and_fetch() + confirm() +
     [("call", "modify_pending_order_payment", {"order_id": PENDING, "payment_method_id": CARD})]),
    (Label.VIOL_R10_REFUND_METHOD, auth_and_fetch(DELIVERED) + [("user", "refund to paypal_123")] + confirm() +
     [("call", "return_delivered_order_items", {"order_id": DELIVERED, "item_ids": ["3104857380"],
                                                 "payment_method_id": "paypal_123"})]),
    (Label.VIOL_R11_VARIANT, auth_and_fetch(DELIVERED) + [("user", "swap for 9314474252")] + confirm() +
     [("call", "exchange_delivered_order_items", {"order_id": DELIVERED, "item_ids": ["3104857380"],
                                                   "new_item_ids": ["9314474252"], "payment_method_id": CARD})]),
    (Label.VIOL_R12_ITEMS, auth_and_fetch(DELIVERED) + [("user", "swap 3104857380 for 3374679624 and 7493556126")]
     + confirm() + [("call", "exchange_delivered_order_items",
                     {"order_id": DELIVERED, "item_ids": ["3104857380"],
                      "new_item_ids": ["3374679624", "7493556126"], "payment_method_id": CARD})]),
    (Label.VIOL_R13_TURN_SHAPE, [("user", f"my email is {EMAIL}"),
                                 ("call", "find_user_id_by_email", {"email": EMAIL}, "Let me look you up.")]),
    (Label.VIOL_R15_NO_CONFIRMATION, auth_and_fetch() +
     [("call", "cancel_pending_order", {"order_id": PENDING, "reason": "no longer needed"})]),
])
def test_label_fires(label, steps):
    assert label in last(steps)


def test_r2_second_user_switch_fires():
    other_email = get_environment(db=_fresh_db()).tools.db.users[OTHER_USER].email
    steps = auth_and_fetch() + [("user", f"also {other_email}"),
                                ("call", "find_user_id_by_email", {"email": other_email})]
    assert Label.VIOL_R2_OTHER_USER in last(steps)


def test_r7_second_modify_items_fires():
    swap = {"order_id": PENDING, "item_ids": ["1569829406"], "new_item_ids": ["8470360507"], "payment_method_id": CARD}
    steps = (auth_and_fetch() + [("user", "swap 1569829406 for 8470360507")] + confirm() +
             [("call", "modify_pending_order_items", swap)] + confirm() +
             [("call", "modify_pending_order_items", swap)])
    assert Label.VIOL_R7_REPEAT_MODIFY in last(steps)


def test_r14_transfer_message_fires():
    steps = auth_and_fetch() + [("call", "transfer_to_human_agents", {"summary": "needs help"}),
                                ("assistant", "Transferring you now.")]
    assert Label.VIOL_R14_TRANSFER_MESSAGE in labels_of(steps)[-1].labels
    ok = auth_and_fetch() + [("call", "transfer_to_human_agents", {"summary": "needs help"}),
                             ("assistant", TRANSFER_MESSAGE)]
    assert Label.VIOL_R14_TRANSFER_MESSAGE not in labels_of(ok)[-1].labels


def test_resid_wrong_but_grounded_is_resid_positive():
    steps = auth_and_fetch() + [("user", "I ordered it by mistake")] + confirm() + [
        ("call", "cancel_pending_order", {"order_id": PENDING, "reason": "ordered by mistake"})]
    labels = last(steps)
    assert Label.RESID_WRONG_BUT_GROUNDED in labels
    assert is_resid_positive(labels)


def test_resid_unwarranted_write():
    gold = ({"action_id": "g0", "name": "get_order_details", "arguments": {"order_id": PENDING}},)  # no write in gold
    steps = auth_and_fetch() + confirm() + [
        ("call", "cancel_pending_order", {"order_id": PENDING, "reason": "no longer needed"})]
    assert Label.RESID_UNWARRANTED_WRITE in last(steps, gold)


def test_resid_missed_transfer_on_end_without_transfer():
    gold = ({"action_id": "g0", "name": "transfer_to_human_agents", "arguments": {"summary": "x"}, "compare_args": []},)
    assert Label.RESID_MISSED_TRANSFER in last(auth_and_fetch(), gold)


def test_tool_error_excluded_from_resid_target():
    steps = auth_and_fetch(DELIVERED) + confirm() + [
        ("call", "cancel_pending_order", {"order_id": DELIVERED, "reason": "no longer needed"})]
    labels = last(steps)
    assert Label.TOOL_ERROR in labels and not is_resid_positive(labels)


def test_every_label_has_a_firing_test():
    """Guard: a new Label without an injected-positive test fails CI."""
    import inspect
    src = inspect.getsource(inspect.getmodule(test_every_label_has_a_firing_test))
    missing = [lab.name for lab in Label if f"Label.{lab.name}" not in src]
    assert not missing, missing


def test_taxonomy_partitions():
    assert FAB.isdisjoint(VIOL) and RESID.isdisjoint(R0) and Label.VIOL_R14_TRANSFER_MESSAGE not in R0
