"""glint.decisions on the recorded failure cases: each write the decision tool would overrule is a positive."""
import json

from tau2.data_model.tasks import Task
from tau2.domains.banking_knowledge.environment import get_environment as banking_env
from tau2.registry import registry

from glint import paths
from glint.decisions import label_run

DOMAINS = paths.tau2_data() / "tau2" / "domains"


def _task(domain, task_id):
    raw = json.loads((DOMAINS / domain / "tasks.json").read_text(encoding="utf-8"))
    return Task.model_validate(next(t for t in raw if str(t["id"]) == task_id))


def _sim(task_id, calls):
    return {"task_id": task_id, "trial": 0, "messages": [
        {"role": "assistant", "tool_calls": [{"id": str(i), "name": n, "arguments": a}]} for i, (n, a) in
        enumerate(calls)]}


def _hidden(name, args):
    return "call_discoverable_agent_tool", {"agent_tool_name": name, "arguments": json.dumps(args)}


def test_retail_82_refund_to_a_card_not_used_for_the_order():
    sim = _sim("82", [("return_delivered_order_items", {"order_id": "#W9571698", "item_ids": ["6065192424"],
                                                         "payment_method_id": "credit_card_1565124"})])
    [d] = label_run(sim, _task("retail", "82"), registry.get_env_constructor("retail")())
    assert d.label == "overruled" and "payment method" in d.reasons[0]
    ok = _sim("82", [("return_delivered_order_items", {"order_id": "#W9571698", "item_ids": ["6065192424"],
                                                        "payment_method_id": "gift_card_7250692"})])
    assert label_run(ok, _task("retail", "82"), registry.get_env_constructor("retail")())[0].label == "agreed"


def test_airline_7_cancel_basic_economy_without_grounds_then_after_upgrade():
    env = registry.get_env_constructor("airline")
    direct = _sim("7", [("cancel_reservation", {"reservation_id": "XEHM4B"})])
    assert label_run(direct, _task("airline", "7"), env())[0].label == "overruled"
    gold = _sim("7", [("update_reservation_flights", {
        "reservation_id": "XEHM4B", "cabin": "business", "payment_id": "credit_card_2408938",
        "flights": [{"flight_number": "HAT005", "date": "2024-05-20"}, {"flight_number": "HAT178", "date": "2024-05-30"}]}),
        ("cancel_reservation", {"reservation_id": "XEHM4B"})])
    assert [d.label for d in label_run(gold, _task("airline", "7"), env())] == ["agreed", "agreed"]


def test_banking_054_dispute_and_replacement_before_the_increase():
    args = {"credit_card_account_id": "cc_584f9c5d00_gold", "user_id": "584f9c5d00"}
    dispute = {"transaction_id": "txn_584f9c5d00_001", "card_action": "cancel_and_reissue",
               "card_last_4_digits": "7823", "full_name": "Sofia Papadopoulos", "user_id": "584f9c5d00",
               "phone": "347-555-0387", "email": "sofia.p@outlook.com", "address": "1456 Astoria Boulevard",
               "contacted_merchant": False, "purchase_date": "11/11/2025", "issue_noticed_date": "11/12/2025",
               "dispute_reason": "unauthorized_fraudulent_charge", "resolution_requested": "full_refund",
               "eligible_for_provisional_credit": True}
    sim = _sim("task_054", [
        ("unlock_discoverable_agent_tool", {"agent_tool_name": "file_credit_card_transaction_dispute_4829"}),
        _hidden("file_credit_card_transaction_dispute_4829", dispute),
        ("unlock_discoverable_agent_tool", {"agent_tool_name": "order_replacement_credit_card_7291"}),
        _hidden("order_replacement_credit_card_7291", args | {"shipping_address": "1456 Astoria Boulevard",
                                                              "reason": "fraud_suspected", "expedited_shipping": True}),
        ("unlock_discoverable_agent_tool", {"agent_tool_name": "submit_credit_limit_increase_request_7392"}),
        _hidden("submit_credit_limit_increase_request_7392", args | {"requested_increase_amount": 2500}),
    ])
    out = label_run(sim, _task("banking_knowledge", "task_054"), banking_env(retrieval_variant="bm25_grep"))
    assert [(d.tool, d.label) for d in out] == [
        ("file_credit_card_transaction_dispute_4829", "overruled"),
        ("order_replacement_credit_card_7291", "overruled"),
        ("submit_credit_limit_increase_request_7392", "agreed")]
    assert "credit limit increase first" in out[0].reasons[-1]




def _gold_calls(task):
    return [(a.name, a.arguments) for a in task.evaluation_criteria.actions if a.requestor == "assistant"]


def test_banking_approval_must_match_the_submitted_increase():
    # An approval was 'agreed' at any new limit, even with no submit first (review 2026-10-27)
    task = _task("banking_knowledge", "task_050")
    gold = _gold_calls(task)
    submit = next(a for n, a in gold if n == "call_discoverable_agent_tool"
                  and a["agent_tool_name"] == "submit_credit_limit_increase_request_7392")
    approve = next(a for n, a in gold if n == "call_discoverable_agent_tool"
                   and a["agent_tool_name"] == "approve_credit_limit_increase_5847")
    inner = json.loads(approve["arguments"])
    wrong = dict(approve, arguments=json.dumps(inner | {"new_credit_limit": inner["new_credit_limit"] - 500}))
    env = banking_env(retrieval_variant="bm25_grep")
    unlock = [("unlock_discoverable_agent_tool", {"agent_tool_name": n}) for n in
              ("submit_credit_limit_increase_request_7392", "approve_credit_limit_increase_5847")]
    run = _sim("task_050", unlock + [("call_discoverable_agent_tool", submit), ("call_discoverable_agent_tool", wrong)])
    labels = [(d.tool, d.label) for d in label_run(run, task, env)]
    assert labels[-1] == ("approve_credit_limit_increase_5847", "overruled")
    no_submit = _sim("task_050", unlock + [("call_discoverable_agent_tool", approve)])
    d = label_run(no_submit, task, banking_env(retrieval_variant="bm25_grep"))[-1]
    assert d.label == "overruled" and "no credit limit increase request was submitted" in d.reasons[0]


def test_every_gold_banking_write_is_agreed():
    # The labels must not overrule what gold does (the review replayed 68 of 68 gold banking writes as agreed)
    raw = json.loads((DOMAINS / "banking_knowledge" / "tasks.json").read_text(encoding="utf-8"))
    seen = 0
    for t in raw:
        task = Task.model_validate(t)
        if not task.evaluation_criteria or not task.evaluation_criteria.actions:
            continue
        sim = _sim(task.id, _gold_calls(task))
        for d in label_run(sim, task, banking_env(retrieval_variant="bm25_grep")):
            assert d.label == "agreed", (task.id, d.tool, d.reasons)
            seen += 1
    assert seen >= 60


def test_string_booleans_and_malformed_arguments():
    from glint.decisions import _flag

    assert _flag("False") is False and _flag("True") is True and _flag(False) is False and _flag(1) is True
    task = _task("banking_knowledge", "task_054")
    sim = _sim("task_054", [("call_discoverable_agent_tool", {
        "agent_tool_name": "file_credit_card_transaction_dispute_4829", "arguments": "{not json"})])
    assert label_run(sim, task, banking_env(retrieval_variant="bm25_grep")) == []


def test_airline_payment_only_update_is_not_judged():
    sim = _sim("7", [("update_reservation_flights", {
        "reservation_id": "XEHM4B", "cabin": "basic_economy", "payment_id": "credit_card_2408938",
        "flights": [{"flight_number": "HAT005", "date": "2024-05-20"}, {"flight_number": "HAT178", "date": "2024-05-30"}]})])
    assert label_run(sim, _task("airline", "7"), registry.get_env_constructor("airline")()) == []


def test_banking_later_writes_are_matched_by_card():
    # A replacement blocks a credit limit increase only for its own card (logistics_007 check 4); matching by name
    # alone overruled a replacement of card B before an increase on card A (review 2026-10-27)
    from glint.decisions import _banking

    env = banking_env(retrieval_variant="bm25_grep")
    task = _task("banking_knowledge", "task_054")
    env.tools.update_db(task.initial_state.initialization_data.agent_data)
    replace = {"credit_card_account_id": "cc_584f9c5d00_gold", "user_id": "584f9c5d00"}
    on_same = [("submit_credit_limit_increase_request_7392", {"credit_card_account_id": "cc_584f9c5d00_gold"})]
    on_other = [("submit_credit_limit_increase_request_7392", {"credit_card_account_id": "cc_other"})]
    assert _banking(env.tools.db, "order_replacement_credit_card_7291", replace, on_same)[0] == "overruled"
    assert _banking(env.tools.db, "order_replacement_credit_card_7291", replace, on_other)[0] == "agreed"
