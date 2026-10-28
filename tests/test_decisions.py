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


# --- Labels the policy decides, one per uncovered branch -----------------------------------------------------------
# Banking base: task_054 (card cc_584f9c5d00_gold passes every credit limit check; txn_584f9c5d00_001 is a $487.50
# fraud charge on a Gold card opened 03/20/2023, user has no prior disputes) and task_052 (cc_5e4c1a83b0_bronze:
# entry tier, an approved request submitted 2025-09-15, 60 days before 2025-11-14).


def _dispute_args(**over):
    return {"transaction_id": "txn_584f9c5d00_001", "card_action": "keep_active", "card_last_4_digits": "7823",
            "full_name": "Sofia Papadopoulos", "user_id": "584f9c5d00", "phone": "347-555-0387",
            "email": "sofia.p@outlook.com", "address": "1456 Astoria Boulevard", "contacted_merchant": False,
            "purchase_date": "11/11/2025", "issue_noticed_date": "11/12/2025",
            "dispute_reason": "unauthorized_fraudulent_charge", "resolution_requested": "full_refund",
            "eligible_for_provisional_credit": True} | over


def _file(dispute):
    sim = _sim("task_054", [
        ("unlock_discoverable_agent_tool", {"agent_tool_name": "file_credit_card_transaction_dispute_4829"}),
        _hidden("file_credit_card_transaction_dispute_4829", dispute)])
    [d] = label_run(sim, _task("banking_knowledge", "task_054"), banking_env(retrieval_variant="bm25_grep"))
    return d


def test_banking_dispute_with_the_wrong_card_action():
    # credit_cards_(general)_014 item 2: 'cancel_and_reissue' only "when user wants card replaced"; with no
    # replacement in the run the card stays 'keep_active'
    assert _file(_dispute_args()).label == "agreed"
    d = _file(_dispute_args(card_action="cancel_and_reissue"))
    assert d.label == "overruled" and d.reasons == ["card_action cancel_and_reissue != keep_active"]


def test_banking_dispute_with_the_wrong_provisional_credit_flag():
    # credit_cards_(general)_015: eligible if ALL hold; this charge meets each (open 60+ days, fraud reason,
    # $25.00 <= $487.50 <= $10,000 premium max, no prior disputes, merchant contact not needed for fraud)
    d = _file(_dispute_args(eligible_for_provisional_credit=False))
    assert d.label == "overruled" and d.reasons == ["eligible_for_provisional_credit False != True"]


def _decide(task_id, card, user, amount, decision, decision_args):
    unlock = [("unlock_discoverable_agent_tool", {"agent_tool_name": n}) for n in
              ("submit_credit_limit_increase_request_7392", decision)]
    base = {"credit_card_account_id": card, "user_id": user}
    sim = _sim(task_id, unlock + [_hidden("submit_credit_limit_increase_request_7392",
                                          base | {"requested_increase_amount": amount}),
                                  _hidden(decision, base | decision_args)])
    return label_run(sim, _task("banking_knowledge", task_id), banking_env(retrieval_variant="bm25_grep"))[-1]


def test_banking_approval_the_policy_denies():
    # credit_card_account_logistics_005 §2: entry-tier cooldown 120 days after an approved request; task_052's was
    # submitted 60 days ago, so _007 Step 4 says deny (cooldown_period_active), not approve
    d = _decide("task_052", "cc_5e4c1a83b0_bronze", "5e4c1a83b0", 500, "approve_credit_limit_increase_5847",
                {"new_credit_limit": 4500})
    assert d.tool == "approve_credit_limit_increase_5847" and d.label == "overruled"
    assert "cooldown 120 days" in d.reasons[0]
    agreed = _decide("task_052", "cc_5e4c1a83b0_bronze", "5e4c1a83b0", 500, "deny_credit_limit_increase_5848",
                     {"denial_reason": "cooldown_period_active"})
    assert agreed.label == "agreed"


def test_banking_denial_the_policy_approves():
    # credit_card_account_logistics_007 Step 4: "If all requirements are met ... approve"; task_054's Gold card passes
    # every check for +$2,500 (50% of $5,000, logistics_006)
    d = _decide("task_054", "cc_584f9c5d00_gold", "584f9c5d00", 2500, "deny_credit_limit_increase_5848",
                {"denial_reason": "high_utilization"})
    assert d.label == "overruled" and d.reasons == ["every check passes; the policy approves"]


def _airline(calls, unjudged=None):
    return label_run(_sim("7", calls), _task("airline", "7"), registry.get_env_constructor("airline")(),
                     unjudged)


def test_airline_cancel_allowed_only_by_an_insurance_reason_is_uncertain():
    # airline/policy.md "Cancel flight": allowed if "The user has travel insurance and the reason for cancellation is
    # covered by insurance". CDXEBS (economy, insured, booked 2024-05-08) has no other ground; the replay does not
    # see the user's reason, so the label is neither agreed nor overruled
    [d] = _airline([("cancel_reservation", {"reservation_id": "CDXEBS"})])
    assert d.label == "uncertain" and "covered by insurance" in d.reasons[0]


def test_airline_baggage_labels():
    # airline/policy.md "Change baggage and insurance": "The user can add but not remove checked bags." VAAOXJ has 1
    for total, label in ((0, "overruled"), (2, "agreed")):
        [d] = _airline([("update_reservation_baggages", {"reservation_id": "VAAOXJ", "total_baggages": total,
                                                         "nonfree_baggages": 0, "payment_id": "x"})])
        assert d.label == label
        assert d.reasons == (["bags can be added but not removed"] if label == "overruled" else [])


def test_airline_passenger_labels():
    # airline/policy.md "Change passengers": "The user can modify passengers but cannot modify the number of
    # passengers." VAAOXJ has one passenger
    one = [{"first_name": "Ana", "last_name": "Lee", "dob": "1980-01-01"}]
    [d] = _airline([("update_reservation_passengers", {"reservation_id": "VAAOXJ", "passengers": one})])
    assert d.label == "agreed"
    [d] = _airline([("update_reservation_passengers", {"reservation_id": "VAAOXJ", "passengers": one * 2})])
    assert d.label == "overruled" and d.reasons == ["the number of passengers cannot change"]


def test_airline_certificate_labels():
    # airline/policy.md "Refunds and Compensation": "Only compensate if the user is a silver/gold member or has travel
    # insurance or flies business." Regular members: olivia_jackson_4826 (no reservation qualifies),
    # lucas_hernandez_8985 (2 of 3 qualify), yusuf_thomas_7802 (all 8 qualify); aarav_nguyen_1055 is silver
    cases = {"olivia_jackson_4826": "overruled", "lucas_hernandez_8985": "uncertain", "yusuf_thomas_7802": "agreed",
             "aarav_nguyen_1055": "agreed"}
    for user, label in cases.items():
        [d] = _airline([("send_certificate", {"user_id": user, "amount": 100})])
        assert d.label == label, user


def test_unjudged_writes_are_counted():
    # A write no rule judges (a certificate to an unknown user, a payment-only flight update) is counted, not dropped
    unjudged = {}
    out = _airline([("send_certificate", {"user_id": "nobody_000", "amount": 100}),
                    ("update_reservation_flights", {
                        "reservation_id": "XEHM4B", "cabin": "basic_economy", "payment_id": "credit_card_2408938",
                        "flights": [{"flight_number": "HAT005", "date": "2024-05-20"},
                                    {"flight_number": "HAT178", "date": "2024-05-30"}]}),
                    ("get_user_details", {"user_id": "daiki_muller_1116"})], unjudged)
    assert out == [] and unjudged == {"send_certificate": 1, "update_reservation_flights": 1}


# --- Mutation-testing gaps (cosmic-ray on main, 2026-10-28: 144 of 419 mutants survived) ----------------------------
# Each test pins a comparison that a flipped operator (`<`, `<=`, `is`, `^`) or a shifted index passed unnoticed.
GOLD, USER = "cc_584f9c5d00_gold", "584f9c5d00"


def _db54():
    env = banking_env(retrieval_variant="bm25_grep")
    env.tools.update_db(_task("banking_knowledge", "task_054").initial_state.initialization_data.agent_data)
    return env.tools.db


def test_banking_dispute_mismatches_in_both_directions():
    # card_action keep_active vs the policy's cancel_and_reissue (a later replacement on the same card), and a True
    # provisional-credit flag where the policy says False: the reverse of the cases above
    from glint.decisions import _banking

    later = [("order_replacement_credit_card_7291", {"credit_card_account_id": GOLD})]
    v = _banking(_db54(), "file_credit_card_transaction_dispute_4829", _dispute_args(), later)
    assert v[0] == "overruled" and "card_action keep_active != cancel_and_reissue" in v[1]
    dup = _dispute_args(dispute_reason="duplicate_charge", contacted_merchant=False, eligible_for_provisional_credit=True)
    v = _banking(_db54(), "file_credit_card_transaction_dispute_4829", dup, [])
    assert v[0] == "overruled" and "eligible_for_provisional_credit True != False" in v[1]


def test_banking_later_writes_on_another_card_do_not_count():
    # another card's id that sorts before this one ("cc_000...") must not match it
    from glint.decisions import _banking

    other = "cc_000_other"
    v = _banking(_db54(), "order_replacement_credit_card_7291", {"credit_card_account_id": GOLD, "user_id": USER},
                 [("submit_credit_limit_increase_request_7392", {"credit_card_account_id": other})])
    assert v == ("agreed", [])
    v = _banking(_db54(), "file_credit_card_transaction_dispute_4829", _dispute_args(),
                 [("order_replacement_credit_card_7291", {"credit_card_account_id": other})])
    assert v == ("agreed", [])  # no replacement of this card later: keep_active stands


def _pending(db, *rows):
    for n, (card, amount) in enumerate(rows):
        db.credit_limit_increase_requests.data[f"cli_test_{n}"] = {
            "credit_card_account_id": card, "status": "PENDING", "requested_increase_amount": amount}
    return db


def test_banking_decision_reads_only_this_cards_latest_pending_request():
    from glint.decisions import _banking

    approve = {"credit_card_account_id": GOLD, "user_id": USER, "new_credit_limit": 7500}
    # a pending request on another card is not this card's request
    v = _banking(_pending(_db54(), ("cc_other", 2500)), "approve_credit_limit_increase_5847", approve, [])
    assert v == ("overruled", ["no credit limit increase request was submitted before this decision"])
    # two pending requests on this card: the latest (+2,500 on 5,000) is the one decided
    assert _banking(_pending(_db54(), (GOLD, 1000), (GOLD, 2500)), "approve_credit_limit_increase_5847", approve,
                    []) == ("agreed", [])


def test_banking_approval_above_the_submitted_increase_is_overruled():
    d = _decide("task_054", GOLD, USER, 2500, "approve_credit_limit_increase_5847", {"new_credit_limit": 9000})
    assert d.label == "overruled" and d.reasons == ["new_credit_limit 9,000 != current limit + submitted increase 7,500"]


def test_banking_denial_with_a_reason_that_sorts_first_is_overruled():
    d = _decide("task_052", "cc_5e4c1a83b0_bronze", "5e4c1a83b0", 500, "deny_credit_limit_increase_5848",
                {"denial_reason": "account_too_new"})
    assert d.label == "overruled" and d.reasons == ["denial_reason account_too_new != cooldown_period_active"]


def test_banking_a_write_before_this_one_is_not_later():
    # the increase submitted before the replacement is not a later request: the replacement is agreed
    unlock = [("unlock_discoverable_agent_tool", {"agent_tool_name": n}) for n in
              ("submit_credit_limit_increase_request_7392", "order_replacement_credit_card_7291")]
    sim = _sim("task_054", unlock + [
        _hidden("submit_credit_limit_increase_request_7392",
                {"credit_card_account_id": GOLD, "user_id": USER, "requested_increase_amount": 2500}),
        _hidden("order_replacement_credit_card_7291", {"credit_card_account_id": GOLD, "user_id": USER,
                                                       "shipping_address": "1456 Astoria Boulevard",
                                                       "reason": "fraud_suspected", "expedited_shipping": True})])
    d = label_run(sim, _task("banking_knowledge", "task_054"), banking_env(retrieval_variant="bm25_grep"))
    assert [(x.tool, x.label) for x in d][-1] == ("order_replacement_credit_card_7291", "agreed")


def test_calls_after_a_user_message_are_still_read():
    sim = _sim("82", [("return_delivered_order_items", {"order_id": "#W9571698", "item_ids": ["6065192424"],
                                                         "payment_method_id": "credit_card_1565124"})])
    sim["messages"].insert(0, {"role": "user", "content": "I want to return my item."})
    [d] = label_run(sim, _task("retail", "82"), registry.get_env_constructor("retail")())
    assert d.label == "overruled"


def test_unjudged_writes_of_the_same_tool_add_up():
    unjudged = {}
    _airline([("send_certificate", {"user_id": "nobody_000", "amount": 100}),
              ("send_certificate", {"user_id": "nobody_001", "amount": 50})], unjudged)
    assert unjudged == {"send_certificate": 2}
