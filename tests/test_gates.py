"""Enforced write gates (glint.gates) on real τ² databases: a deny path and an allow path for every gated write, the
boot check, fail-closed behavior, the hash chain, the tau2_tools wiring, and the false-refusal replay on gold."""
import json
import sqlite3

import pytest
from loguru import logger
from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import Task
from tau2.environment.toolkit import ToolType
from tau2.registry import registry

from glint import paths
from glint.gates import (
    ChainBroken,
    Enforcer,
    EventLog,
    Gate,
    GateRegistry,
    GateResult,
    allow,
    assert_required,
    core,
    deny,
    rules,
    run_gate,
    summary,
)
from glint.gates.core import NO_RULE

logger.remove()
BANKING_TASKS = paths.tau2_data() / "tau2" / "domains" / "banking_knowledge" / "tasks.json"


def _gated(domain: str, task=None, wording: str = "plain"):
    events = EventLog(":memory:")
    enforcer = Enforcer(domain, events, wording=wording)
    if domain == "banking_knowledge":
        env = registry.get_env_constructor(domain)(retrieval_variant="bm25_grep", task=task)
    else:
        env = registry.get_env_constructor(domain)()
    if task is not None:
        init = task.initial_state
        env.set_state(init.initialization_data if init else None, init.initialization_actions if init else None, [])
    return enforcer.attach(env), events


def _call(env, name: str, **args):
    return env.get_response(ToolCall(id="t", name=name, arguments=args, requestor="assistant"))


def _fired(events: EventLog) -> list[dict]:
    return [e["payload"] for e in events.events("gate.fired")]


def _denied(reply) -> bool:
    return reply.error and "was NOT executed" in reply.content


# ---- retail -------------------------------------------------------------------------------------------------------

def test_retail_cancel():
    env, events = _gated("retail")
    reply = _call(env, "cancel_pending_order", order_id="#W4817420", reason="no longer needed")  # delivered
    assert _denied(reply) and "status is 'delivered'" in reply.content
    assert env.tools.db.orders["#W4817420"].status == "delivered"
    reply = _call(env, "cancel_pending_order", order_id="#W5918442", reason="no longer needed")
    assert not reply.error and env.tools.db.orders["#W5918442"].status == "cancelled"
    assert [(e["rule_id"], e["decision"]) for e in _fired(events)] == [("retail.cancel", "deny"),
                                                                       ("retail.cancel", "allow")]


def test_retail_modify_address():
    env, _ = _gated("retail")
    address = {"address1": "1 Main St", "address2": "", "city": "Austin", "state": "TX", "country": "USA",
               "zip": "78701"}
    assert _denied(_call(env, "modify_pending_order_address", order_id="#W4817420", **address))
    assert env.tools.db.orders["#W4817420"].address.address1 != "1 Main St"
    assert not _call(env, "modify_pending_order_address", order_id="#W5918442", **address).error
    assert env.tools.db.orders["#W5918442"].address.address1 == "1 Main St"


def test_retail_modify_payment():
    env, _ = _gated("retail")
    # the original payment method is not a change
    reply = _call(env, "modify_pending_order_payment", order_id="#W6779827", payment_method_id="gift_card_7219486")
    assert _denied(reply) and "not allowed here" in reply.content
    reply = _call(env, "modify_pending_order_payment", order_id="#W6779827", payment_method_id="credit_card_9789590")
    assert not reply.error
    assert "credit_card_9789590" in [p.payment_method_id for p in env.tools.db.orders["#W6779827"].payment_history]


def test_retail_modify_items():
    env, _ = _gated("retail")
    # a variant of a different product
    reply = _call(env, "modify_pending_order_items", order_id="#W5918442", item_ids=["1586641416"],
                  new_item_ids=["4579334072"], payment_method_id="credit_card_5051208")
    assert _denied(reply) and "not a variant of the same product" in reply.content
    assert env.tools.db.orders["#W5918442"].status == "pending"
    product = env.tools.db.products["3377618313"]
    new = next(v for v, d in product.variants.items() if d.available and v not in ("1586641416", "6117189161"))
    reply = _call(env, "modify_pending_order_items", order_id="#W5918442", item_ids=["1586641416"],
                  new_item_ids=[new], payment_method_id="credit_card_5051208")
    assert not reply.error and env.tools.db.orders["#W5918442"].status == "pending (item modified)"


def test_retail_return():
    env, _ = _gated("retail")
    # task 82: the refund must go to the original payment method or a gift card
    reply = _call(env, "return_delivered_order_items", order_id="#W9571698", item_ids=["6065192424"],
                  payment_method_id="credit_card_1565124")
    assert _denied(reply) and env.tools.db.orders["#W9571698"].status == "delivered"
    reply = _call(env, "return_delivered_order_items", order_id="#W9571698", item_ids=["6065192424"],
                  payment_method_id="gift_card_7250692")
    assert not reply.error and env.tools.db.orders["#W9571698"].status == "return requested"


def test_retail_exchange():
    env, _ = _gated("retail")
    product = env.tools.db.products["8310926033"]
    new = next(v for v, d in product.variants.items() if d.available and v != "6777246137")
    reply = _call(env, "exchange_delivered_order_items", order_id="#W5918442", item_ids=["1586641416"],
                  new_item_ids=[new], payment_method_id="credit_card_5051208")  # pending, not delivered
    assert _denied(reply) and "requires 'delivered'" in reply.content
    reply = _call(env, "exchange_delivered_order_items", order_id="#W4817420", item_ids=["6777246137"],
                  new_item_ids=[new], payment_method_id="gift_card_8168843")
    assert not reply.error and env.tools.db.orders["#W4817420"].status == "exchange requested"


def test_a_write_with_no_rule_yet_is_allowed_and_logged():
    """Phase 2 gave retail's modify_user_address a rule; a banking write with no rule yet is still allowed and
    logged under no_rule_yet (with the procedure gates' verdicts beside it)."""
    env, events, _ = _banking("task_001")
    reply = _call(env, "example_agent_tool_0000")
    assert "was NOT executed" not in reply.content
    assert [(e["rule_id"], e["decision"]) for e in _fired(events) if e["tool"] == "example_agent_tool_0000"] == [
        (NO_RULE, "allow")]


# ---- airline ------------------------------------------------------------------------------------------------------

def test_airline_cancel():
    env, _ = _gated("airline")
    reply = _call(env, "cancel_reservation", reservation_id="EHGLP3")  # basic economy, no insurance, not recent
    assert _denied(reply) and env.tools.db.reservations["EHGLP3"].status != "cancelled"
    assert not _call(env, "cancel_reservation", reservation_id="4WQ150").error  # business
    assert env.tools.db.reservations["4WQ150"].status == "cancelled"


def test_airline_cancel_with_insurance_and_an_unseen_reason_is_allowed_and_logged():
    env, events = _gated("airline")
    db = env.tools.db
    rid = next(r.reservation_id for r in db.reservations.values()
               if r.insurance == "yes" and r.cabin != "business" and r.status != "cancelled"
               and not rules.airline.check_eligibility(db, r.reservation_id, "cancel", "change of plan")["allowed"]
               and rules.airline.check_eligibility(db, r.reservation_id, "cancel", "health")["allowed"])
    assert not _call(env, "cancel_reservation", reservation_id=rid).error
    assert _fired(events)[-1]["rule_id"] == "airline.cancel.insurance_reason_unseen"


def _flights(db, rid):
    return [{"flight_number": f.flight_number, "date": f.date} for f in db.reservations[rid].flights]


def test_airline_update_flights_and_cabin():
    env, events = _gated("airline")
    db = env.tools.db
    other = [{"flight_number": "HAT005", "date": "2024-05-21"}, {"flight_number": "HAT178", "date": "2024-05-30"}]
    pay = db.users["daiki_muller_1116"].payment_methods
    method = next(iter(pay))
    # basic economy flights cannot be modified
    reply = _call(env, "update_reservation_flights", reservation_id="XEHM4B", cabin="basic_economy", flights=other,
                  payment_id=method)
    assert _denied(reply) and "basic economy" in reply.content
    assert _flights(db, "XEHM4B")[0]["date"] == "2024-05-20"
    # its cabin can change without changing the flights
    reply = _call(env, "update_reservation_flights", reservation_id="XEHM4B", cabin="economy",
                  flights=_flights(db, "XEHM4B"), payment_id=method)
    assert not reply.error and db.reservations["XEHM4B"].cabin == "economy"
    assert [e["rule_id"] for e in _fired(events)] == ["airline.change_flights", "airline.change_cabin"]


def test_airline_cabin_change_after_a_flown_segment_is_denied():
    env, _ = _gated("airline")
    db = env.tools.db
    rid = next(r.reservation_id for r in db.reservations.values()
               if r.status != "cancelled" and rules.airline._flown(db, r) and r.cabin == "economy")
    method = next(iter(db.users[db.reservations[rid].user_id].payment_methods))
    reply = _call(env, "update_reservation_flights", reservation_id=rid, cabin="business",
                  flights=_flights(db, rid), payment_id=method)
    assert _denied(reply) and db.reservations[rid].cabin == "economy"


def test_airline_baggages():
    env, _ = _gated("airline")
    db = env.tools.db
    method = next(iter(db.users[db.reservations["4WQ150"].user_id].payment_methods))
    reply = _call(env, "update_reservation_baggages", reservation_id="4WQ150", total_baggages=4, nonfree_baggages=0,
                  payment_id=method)
    assert _denied(reply) and db.reservations["4WQ150"].total_baggages == 5
    assert not _call(env, "update_reservation_baggages", reservation_id="4WQ150", total_baggages=6,
                     nonfree_baggages=0, payment_id=method).error
    assert db.reservations["4WQ150"].total_baggages == 6


def test_airline_passengers():
    env, _ = _gated("airline")
    db = env.tools.db
    people = [p.model_dump() for p in db.reservations["XEHM4B"].passengers]
    reply = _call(env, "update_reservation_passengers", reservation_id="XEHM4B", passengers=people[:1])
    assert _denied(reply) and "number of passengers" in reply.content and len(db.reservations["XEHM4B"].passengers) == 2
    people[0]["first_name"] = "Renamed"
    assert not _call(env, "update_reservation_passengers", reservation_id="XEHM4B", passengers=people).error
    assert db.reservations["XEHM4B"].passengers[0].first_name == "Renamed"


# ---- banking ------------------------------------------------------------------------------------------------------

def _banking(task_id: str):
    task = Task.model_validate(next(t for t in json.loads(BANKING_TASKS.read_text(encoding="utf-8"))
                                    if t["id"] == task_id))
    env, events = _gated("banking_knowledge", task)
    gold = [(a.arguments["agent_tool_name"], json.loads(a.arguments["arguments"]))
            for a in task.evaluation_criteria.actions if a.name == "call_discoverable_agent_tool"]
    # phase 2 gates banking writes on the user's verification and on the lookups each procedure names: do them as
    # the gold conversation does (the hidden lookups are reads; they change nothing the phase-1 gates judge)
    for a in task.evaluation_criteria.actions:
        # the verification, and the customer tools gold gives (the card-digits tool, procedure.card_digits)
        if a.name in ("log_verification", "give_discoverable_user_tool"):
            env.get_response(ToolCall(id="v", name=a.name, arguments=a.arguments, requestor="assistant"))
    for tool, args in gold:
        if env.tools.tool_type(tool) == ToolType.READ:
            _call(env, "unlock_discoverable_agent_tool", agent_tool_name=tool)
            _call(env, "call_discoverable_agent_tool", agent_tool_name=tool, arguments=json.dumps(args))
    # the plan gate files disputes in a plan_disputes plan: plan gold's disputes, in gold's order, as an agent does
    disputes = [a for t, a in gold if t == rules.DISPUTE]
    if disputes:
        _plan(env, disputes)
    return env, events, gold


def _plan(env, disputes: list[dict]):
    items = [{k: d[k] for k in ("transaction_id", "dispute_reason", "contacted_merchant")} for d in disputes]
    reply = _call(env, "plan_disputes", user_id=disputes[0]["user_id"], transactions=items)
    assert not reply.error, reply.content
    return json.loads(reply.content)


def _hidden(env, tool: str, **args):
    _call(env, "unlock_discoverable_agent_tool", agent_tool_name=tool)
    return _call(env, "call_discoverable_agent_tool", agent_tool_name=tool, arguments=json.dumps(args))


def _gold(gold, tool: str, n: int = 0) -> dict:
    return [a for t, a in gold if t == tool][n]


def _disputes(env) -> int:
    return len(env.tools.db.transaction_disputes.data)


def test_banking_dispute_card_action():
    env, _, gold = _banking("task_038")
    _hidden(env, rules.REPLACE, **_gold(gold, rules.REPLACE))
    first = _gold(gold, rules.DISPUTE)
    before = _disputes(env)
    reply = _hidden(env, rules.DISPUTE, **(first | {"card_action": "keep_active"}))  # replaced today
    assert _denied(reply) and "cancel_and_reissue" in reply.content and _disputes(env) == before
    assert not _hidden(env, rules.DISPUTE, **first).error and _disputes(env) == before + 1


def test_banking_dispute_provisional_credit_flag():
    env, _, gold = _banking("task_040")
    first = _gold(gold, rules.DISPUTE)
    before = _disputes(env)
    reply = _hidden(env, rules.DISPUTE, **(first | {"eligible_for_provisional_credit": False}))
    assert _denied(reply) and "eligible_for_provisional_credit" in reply.content and _disputes(env) == before
    assert not _hidden(env, rules.DISPUTE, **first).error and _disputes(env) == before + 1


def test_banking_dispute_gate_counts_disputes_filed_earlier_in_the_conversation():
    """banking12 038-041 failed when Glimmer reused an early check_card_dispute verdict for later disputes. The gate
    reads the live database at write time, so a dispute after earlier filings sees them."""
    env, _, gold = _banking("task_039")
    disputes = [a for t, a in gold if t == rules.DISPUTE]
    assert [d["eligible_for_provisional_credit"] for d in disputes[:4]] == [True, True, True, False]
    fourth = disputes[3]
    # on the starting database the 4th transaction would qualify: only the filings before it make it ineligible
    fresh, _, _ = _banking("task_039")
    _plan(fresh, [fourth])  # a plan of the 4th alone, so the plan gate files it first
    assert not _hidden(fresh, rules.DISPUTE, **(fourth | {"eligible_for_provisional_credit": True})).error
    for d in disputes[:3]:
        assert not _hidden(env, rules.DISPUTE, **d).error
    reply = _hidden(env, rules.DISPUTE, **(fourth | {"eligible_for_provisional_credit": True}))
    assert _denied(reply) and "at_most_2_disputes_past_12_months" in reply.content
    assert not _hidden(env, rules.DISPUTE, **fourth).error


def test_banking_submit_increase():
    env, _, gold = _banking("task_054")
    submit = _gold(gold, rules.SUBMIT)
    requests = len(env.tools.db.credit_limit_increase_requests.data)
    reply = _hidden(env, rules.SUBMIT, **(submit | {"requested_increase_amount": 4000}))  # gold tier max is 2500
    assert _denied(reply) and "exceeds" in reply.content
    assert len(env.tools.db.credit_limit_increase_requests.data) == requests
    assert not _hidden(env, rules.SUBMIT, **submit).error
    assert len(env.tools.db.credit_limit_increase_requests.data) == requests + 1


def test_banking_approve_increase():
    env, _, gold = _banking("task_054")
    approve = _gold(gold, rules.APPROVE)
    reply = _hidden(env, rules.APPROVE, **approve)  # nothing submitted yet
    assert _denied(reply) and "a decision needs a submitted request" in reply.content
    _hidden(env, rules.SUBMIT, **_gold(gold, rules.SUBMIT))
    reply = _hidden(env, rules.APPROVE, **(approve | {"new_credit_limit": 9000}))
    assert _denied(reply) and "new_credit_limit" in reply.content
    card = env.tools.db.credit_card_accounts.data[approve["credit_card_account_id"]]
    limit = card["credit_limit"]
    assert not _hidden(env, rules.APPROVE, **approve).error
    assert card["credit_limit"] != limit


def test_banking_deny_increase():
    env, _, gold = _banking("task_052")
    deny_args = _gold(gold, rules.DENY)
    _hidden(env, rules.SUBMIT, **_gold(gold, rules.SUBMIT))
    reply = _hidden(env, rules.DENY, **(deny_args | {"denial_reason": "high_utilization"}))
    assert _denied(reply) and "cooldown_period_active" in reply.content
    assert not _hidden(env, rules.DENY, **deny_args).error
    # an increase every check passes cannot be denied
    env, _, gold = _banking("task_054")
    _hidden(env, rules.SUBMIT, **_gold(gold, rules.SUBMIT))
    reply = _hidden(env, rules.DENY, user_id="584f9c5d00", credit_card_account_id="cc_584f9c5d00_gold",
                    denial_reason="high_utilization")
    assert _denied(reply) and "every policy check passes" in reply.content


def test_banking_replacement_waits_for_the_increase_decision():
    env, _, gold = _banking("task_054")
    replace = _gold(gold, rules.REPLACE)
    _hidden(env, rules.SUBMIT, **_gold(gold, rules.SUBMIT))
    orders = len(env.tools.db.credit_card_orders.data)
    reply = _hidden(env, rules.REPLACE, **replace)
    assert _denied(reply) and "credit limit increase first" in reply.content
    assert len(env.tools.db.credit_card_orders.data) == orders
    assert not _hidden(env, rules.APPROVE, **_gold(gold, rules.APPROVE)).error
    assert not _hidden(env, rules.REPLACE, **replace).error
    assert len(env.tools.db.credit_card_orders.data) == orders + 1


def test_banking_unwraps_the_hidden_tool_and_gates_a_direct_call_too():
    env, events, gold = _banking("task_054")
    submit = _gold(gold, rules.SUBMIT) | {"requested_increase_amount": 4000}
    assert _denied(_call(env, rules.SUBMIT, **submit))  # by its own name, not through the wrapper
    reply = _call(env, "call_discoverable_agent_tool", agent_tool_name=rules.SUBMIT, arguments="{not json")
    assert _denied(reply) and _fired(events)[-1]["rule_id"] == "unparsed_arguments"
    # a hidden READ tool is not gated
    n = len(_fired(events))
    _hidden(env, "get_credit_limit_increase_history_4829", credit_card_account_id="cc_584f9c5d00_gold")
    assert len(_fired(events)) == n


def test_user_calls_and_gold_actions_are_not_gated():
    env, events = _gated("retail")
    env.get_response(ToolCall(id="u", name="cancel_pending_order", requestor="user",
                              arguments={"order_id": "#W4817420", "reason": "no longer needed"}))
    with pytest.raises(ValueError, match="Non-pending"):  # τ²'s own tool rejects it; the gate never saw it
        env.make_tool_call("cancel_pending_order", order_id="#W4817420", reason="no longer needed")
    env.make_tool_call("cancel_pending_order", order_id="#W5918442", reason="no longer needed")
    assert _fired(events) == []


# ---- boot check, fail-closed, registry ----------------------------------------------------------------------------

def test_boot_check_raises_when_a_required_write_has_no_gate(monkeypatch):
    reg = GateRegistry()
    reg.register(Gate("g", ("a",), "p", lambda db, a: allow("r", "s")))
    with pytest.raises(RuntimeError, match=r"write tools without a gate: \['b'\]"):
        assert_required(reg, {"a", "b"}, "test")
    monkeypatch.setitem(rules.NO_RULE_YET, "banking_knowledge", ())  # example_agent_tool_0000 loses its gate
    with pytest.raises(RuntimeError, match="example_agent_tool_0000"):
        Enforcer("banking_knowledge", EventLog(":memory:"))


def test_attach_raises_when_the_toolkit_has_a_write_no_gate_covers(monkeypatch):
    monkeypatch.setitem(rules.REQUIRED_GATES, "airline", rules.REQUIRED_GATES["airline"] - {"send_certificate"})
    monkeypatch.setattr(rules.travel, "send_certificate", None)
    real = rules.build_registry

    def without_certificate(domain):
        reg = real(domain)
        reg._gates.pop("airline_send_certificate_gate", None)
        return reg
    monkeypatch.setattr("glint.gates.enforce.build_registry", without_certificate)
    enforcer = Enforcer("airline", EventLog(":memory:"))
    with pytest.raises(RuntimeError, match="send_certificate"):
        enforcer.attach(registry.get_env_constructor("airline")())


def test_registration_rejects_a_gate_without_a_purpose():
    reg = GateRegistry()
    with pytest.raises(ValueError, match="no purpose"):
        reg.register(Gate("g", ("a",), "  ", lambda db, a: allow("r", "s")))
    with pytest.raises(ValueError, match="names no tool"):
        reg.register(Gate("g", (), "p", lambda db, a: allow("r", "s")))
    reg.register(Gate("g", ("a",), "p", lambda db, a: allow("r", "s")))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(Gate("g", ("a",), "p", lambda db, a: allow("r", "s")))
    with pytest.raises(ValueError, match="no gates"):
        rules.build_registry("telecom")


def test_a_gate_that_raises_or_hangs_fails_closed():
    import threading

    def boom(db, a):
        raise KeyError("order_id")

    result, _ = run_gate(Gate("boom", ("t",), "purpose", boom), None, {})
    assert not result.allowed and result.facts == ("error: KeyError: 'order_id'",) and result.purpose == "purpose"
    assert result.consequence and "failed with an error" in result.reason
    release = threading.Event()
    result, ms = run_gate(Gate("slow", ("t",), "p", lambda db, a: release.wait(5) or allow("r", "s"),
                               timeout_s=0.05), None, {})
    release.set()
    assert not result.allowed and "did not finish within" in result.reason and ms < 4000
    result, _ = run_gate(Gate("bad", ("t",), "p", lambda db, a: "yes"), None, {})
    assert not result.allowed and "returned no verdict" in result.reason and result.facts == ("returned: str",)
    result, _ = run_gate(Gate("odd", ("t",), "p", lambda db, a: GateResult("maybe")), None, {})
    assert not result.allowed and result.facts == ("decision: 'maybe'",)
    result, _ = run_gate(Gate("bare", ("t",), "p", lambda db, a: GateResult("deny")), None, {})
    assert not result.allowed and result.reason and result.fix


def test_a_raising_gate_blocks_the_write_in_the_environment(monkeypatch):
    def boom(db, a):
        raise RuntimeError("broken check")

    monkeypatch.setattr(rules, "_retail_gate", lambda action: boom)
    env, events = _gated("retail")
    reply = _call(env, "cancel_pending_order", order_id="#W5918442", reason="no longer needed")
    assert _denied(reply) and "broken check" in reply.content
    assert env.tools.db.orders["#W5918442"].status == "pending"
    assert _fired(events)[-1]["decision"] == "deny"


def test_an_event_that_cannot_be_logged_turns_the_write_into_a_deny():
    env, events = _gated("retail")
    events.close()
    reply = _call(env, "cancel_pending_order", order_id="#W5918442", reason="no longer needed")
    assert _denied(reply) and "could not be logged" in reply.content
    assert env.tools.db.orders["#W5918442"].status == "pending"


def test_the_deny_message_carries_finding_facts_consequence_fix_and_verdict_and_rewrites_nothing():
    env, _ = _gated("retail")
    args = {"order_id": "#W4817420", "reason": "no longer needed"}
    first, second = _call(env, "cancel_pending_order", **args), _call(env, "cancel_pending_order", **args)
    assert first.content == second.content
    lines = first.content.splitlines()
    assert [line.split(":", 1)[0] for line in lines[1:4]] == ["Rule", "What the check found", "Facts it used"]
    assert lines[1].startswith("Rule: retail.cancel (")
    facts = lines[4:lines.index(next(x for x in lines if x.startswith("Why it matters: ")))]
    assert facts and all(f.startswith("- ") for f in facts)
    for part in ("\nWhy it matters: ", "\nWhat to do next: Do not repeat this write.", "\nVerdict: {"):
        assert part in first.content
    assert args == {"order_id": "#W4817420", "reason": "no longer needed"}


def test_the_gate_log_keeps_each_denys_facts_and_consequence():
    env, events = _gated("retail")
    reply = _call(env, "cancel_pending_order", order_id="#W4817420", reason="no longer needed")
    denied = [f for f in _fired(events) if f["decision"] == "deny"]
    assert len(denied) == 1 and denied[0]["consequence"] and denied[0]["consequence"] in reply.content
    assert denied[0]["facts"] and all(f"\n- {f}" in reply.content for f in denied[0]["facts"])
    assert all(f["facts"] == [] and f["consequence"] == "" for f in _fired(events) if f["decision"] == "allow")


# ---- event log ----------------------------------------------------------------------------------------------------

def test_every_decision_and_the_manifest_are_logged(tmp_path):
    events = EventLog(tmp_path / "run" / "gates.sqlite")
    enforcer = Enforcer("retail", events, context=lambda: {"task_id": "7", "trial": "0"})
    env = enforcer.attach(registry.get_env_constructor("retail")())
    assert enforcer.attach(env) is env  # attaching twice is a no-op
    with pytest.raises(RuntimeError, match="another enforcer"):
        Enforcer("retail", EventLog(":memory:")).attach(env)
    _call(env, "cancel_pending_order", order_id="#W4817420", reason="no longer needed")
    _call(env, "get_order_details", order_id="#W4817420")  # a read: not gated, not logged
    manifest = events.events("gates.manifest")
    assert len(manifest) == 1 and manifest[0]["payload"]["no_rule_yet"] == []
    assert "modify_user_address" in manifest[0]["payload"]["required"]
    (fired,) = _fired(events)
    assert {k: fired[k] for k in ("tool", "decision", "rule_id", "task_id", "trial", "replay")} == {
        "tool": "cancel_pending_order", "decision": "deny", "rule_id": "retail.cancel", "task_id": "7",
        "trial": "0", "replay": False}
    assert fired["args"] == {"order_id": "#W4817420", "reason": "no longer needed"} and fired["duration_ms"] >= 0
    assert fired["source"] == "glint.assist.retail.check_eligibility"
    assert events.verify() == 2
    events.close()


def test_the_log_is_append_only_and_the_chain_catches_tampering(tmp_path):
    path = tmp_path / "gates.sqlite"
    events = EventLog(path)
    for i in range(3):
        events.emit("gate.fired", {"i": i})
    assert events.verify() == 3
    raw = sqlite3.connect(path)
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        raw.execute("UPDATE events SET payload = '{\"i\": 9}'")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        raw.execute("DELETE FROM events")
    raw.execute("DROP TRIGGER events_no_update")
    raw.execute("UPDATE events SET payload = '{\"i\": 9}' WHERE rowid = 2")
    raw.commit()
    with pytest.raises(ChainBroken, match="row 2: hash"):
        events.verify()
    raw.execute("CREATE TABLE t AS SELECT * FROM events")
    raw.execute("DROP TRIGGER events_no_delete")
    raw.execute("DELETE FROM events WHERE rowid = 1")
    raw.commit()
    raw.close()
    with pytest.raises(ChainBroken, match="prev_hash"):
        events.verify()
    events.close()


def test_a_failed_emit_rolls_back(tmp_path, monkeypatch):
    events = EventLog(tmp_path / "g.sqlite")
    events.emit("a", {})
    monkeypatch.setattr("glint.gates.eventlog.uuid.uuid4", lambda: type("U", (), {"hex": "same"})())
    events.emit("b", {})
    with pytest.raises(sqlite3.IntegrityError):
        events.emit("c", {})  # duplicate id
    assert [e["type"] for e in events.events()] == ["a", "b"] and events.verify() == 2
    monkeypatch.undo()
    events.emit("d", {})  # the failed write was rolled back: the writer still works
    assert events.count() == 3 and events.verify() == 3
    events.close()


# ---- wiring into tau2_tools and the evaluator's replay -------------------------------------------------------------

def test_install_with_an_enforcer_gates_every_environment_and_tau2s_replay():
    from glint.assist.tau2_tools import install

    events = EventLog(":memory:")
    enforcer = Enforcer("retail", events)
    install("retail")
    install("retail", enforcer=enforcer)  # replaces the ungated install
    constructor = registry.get_env_constructor("retail")
    assert constructor.glint_enforcer is enforcer
    install("retail", enforcer=enforcer)
    assert registry.get_env_constructor("retail") is constructor  # same enforcer: no-op
    env = constructor()
    assert "check_order_action" in env.get_tools() or env.tools.has_tool("check_order_action")
    deny_call = ToolCall(id="1", name="cancel_pending_order", requestor="assistant",
                         arguments={"order_id": "#W4817420", "reason": "no longer needed"})
    live = env.get_response(deny_call)
    assert _denied(live)
    # τ²'s evaluator: a fresh environment from the same constructor replays the trajectory through get_response
    from tau2.data_model.message import AssistantMessage
    replay = constructor()
    replay.set_state(None, None, [AssistantMessage(role="assistant", tool_calls=[deny_call]), live], strict=True)
    assert replay.tools.db.orders["#W4817420"].status == "delivered"
    assert [e["replay"] for e in _fired(events)] == [False, True]


def test_banking_install_with_an_enforcer():
    from glint.assist.tau2_tools import assist_environment, install

    enforcer = Enforcer("banking_knowledge", EventLog(":memory:"))
    install("banking_knowledge", enforcer=enforcer)
    env = registry.get_env_constructor("banking_knowledge")(retrieval_variant="bm25_grep")
    assert env.glint_enforcer is enforcer and env.tools.has_tool("check_card_dispute")
    assert assist_environment(env, "banking_knowledge", enforcer) is env  # already assisted and gated


def test_tools_ab_gates_only_where_the_set_says_so(tmp_path):
    import tools_ab

    assert tools_ab._enforcer({"gates": False}, "retail", tmp_path) is None
    enforcer = tools_ab._enforcer({}, "retail", tmp_path)
    assert enforcer.events.path.endswith("gates-retail.sqlite") and enforcer.context() == {
        "task_id": "unknown", "trial": "unknown"}
    assert "banking-test-gated" not in tools_ab.SETS  # the banking test split is read once (decision 2026-10-28)
    for name in ("banking-gated", "banking-p2-gated", "banking-plan-gated", "banking-plan-terse-gated",
                 "banking-plan-v2-gated", "banking-plan-v3-gated", "banking-plan-v4-gated",
                 "banking-plan-v5-gated", "heldout2-gated"):
        cfg = tools_ab.SETS[name]
        assert cfg["gates"] and cfg["trials"] == 4
        with pytest.raises(SystemExit, match="Modal only"):
            tools_ab.for_backend(cfg, False)
        assert tools_ab.for_backend(cfg, True)["data"].name.endswith("_gated_modal")
    assert tools_ab.SETS["banking-gated"]["tasks"] is tools_ab.banking_tasks
    # the same comparison as banking-gated, each in its own folder and result file
    later = [tools_ab.SETS[n] for n in ("banking-gated", "banking-p2-gated", "banking-plan-gated")]
    for key in ("data", "out", "save"):
        assert len({str(cfg[key]) for cfg in later}) == 3
    for cfg in later[1:]:
        assert cfg["tasks"] is tools_ab.banking_tasks
        assert {k: cfg[k] for k in ("trials", "glimmer_usd", "meta_usd", "gpu_usd")} == {
            k: later[0][k] for k in ("trials", "glimmer_usd", "meta_usd", "gpu_usd")}
    assert {k: tools_ab.SETS["banking-gated"][k] for k in ("glimmer_usd", "meta_usd", "gpu_usd")} == {
        k: tools_ab.SETS["banking"][k] for k in ("glimmer_usd", "meta_usd", "gpu_usd")}
    assert not any(cfg["gates"] for name, cfg in tools_ab.SETS.items() if not name.endswith("-gated"))


# ---- false refusals on gold ----------------------------------------------------------------------------------------

def test_no_gold_write_is_refused():
    """Every gold action of every TRAIN task (retail, airline, banking) through the gates: a deny of a gold write that
    τ²'s own tool would run is a false refusal; there must be none. τ²'s test splits stay unread."""
    import gate_replay

    events = EventLog(":memory:")
    reports = [gate_replay.replay(d, gate_replay.tasks_of(d), events) for d in gate_replay.DOMAINS]
    refusals = [f for r in reports for f in r["false_refusals"]]
    assert refusals == [], "\n".join(f"{f['domain']} {f['task_id']} {f['tool']} {f['rule_id']}: {f['reason']}"
                                     for f in refusals)
    # the gold write τ²'s tool rejects anyway (retail 105: gift card balance)
    assert sorted(f["task_id"] for r in reports for f in r["also_rejected"]) == ["105"]
    assert [r["tasks"] for r in reports] == [74, 30, 77]
    assert not {t.id for t in gate_replay.tasks_of("banking_knowledge")} & set(json.loads(
        (paths.RESULTS / "split-partition-banking.json").read_text(encoding="utf-8"))["test_untouched"])
    # task_005's gold verifies a placeholder user and changes an unverified user's email: denied by policy, listed
    assert sorted((f["task_id"], f["tool"]) for r in reports for f in r["placeholder_gold"]) == [
        ("task_005", "change_user_email"), ("task_005", "log_verification")]
    assert sum(r["gold_writes"] for r in reports) == 418 and events.verify() > 418


def test_gate_replay_main_reports_and_exits_by_the_count(tmp_path, monkeypatch, capsys):
    import gate_replay

    retail = [t for t in gate_replay.tasks_of("retail") if t.id in ("105", "0")]
    monkeypatch.setattr(gate_replay, "DOMAINS", ("retail",))
    monkeypatch.setattr(gate_replay, "tasks_of", lambda d: retail)
    out = tmp_path / "r.json"
    assert gate_replay.main(["--json", str(out), "--log", str(tmp_path / "g.sqlite")]) == 0
    text = capsys.readouterr().out
    assert "TOTAL false refusals on gold: 0" in text and "also rejected by τ² 105" in text
    assert json.loads(out.read_text(encoding="utf-8"))[0]["tasks"] == 2
    monkeypatch.setattr(rules, "_retail_gate", lambda action: lambda db, a: deny("x", "s", "no", "fix"))
    assert gate_replay.main([]) == 1
    assert "FALSE REFUSAL" in capsys.readouterr().out


def test_a_deny_without_its_consequence_or_facts_is_kept_as_a_deny_that_names_what_is_missing():
    bare = GateResult("deny", reason="r", fix="f", rule_id="x.rule")
    result, _ = run_gate(Gate("bare", ("t",), "purpose", lambda db, a: bare), None, {})
    assert not result.allowed and result.rule_id == "x.rule" and result.reason == "r" and result.fix == "f"
    assert result.facts == ("the deny is missing ['consequence', 'facts']",) and result.consequence
    full, _ = run_gate(Gate("full", ("t",), "purpose", lambda db, a: deny("x.rule", "s", "r", "f", consequence="c",
                                                                          facts=["a", "b"])), None, {})
    assert (full.consequence, full.facts, full.purpose) == ("c", ("a", "b"), "purpose")
    assert "\nFacts it used:\n- a\n- b\nWhy it matters: c\nWhat to do next: f\n" in full.message()


def test_the_terse_wording_keeps_the_decision_and_uses_the_layout_before_the_plain_messages(tmp_path):
    import tools_ab

    args = {"order_id": "#W4817420", "reason": "no longer needed"}
    (plain_env, plain_events), (terse_env, terse_events) = _gated("retail"), _gated("retail", wording="terse")
    plain, terse = _call(plain_env, "cancel_pending_order", **args), _call(terse_env, "cancel_pending_order", **args)
    assert plain.error and terse.error and terse.content != plain.content
    lines = terse.content.splitlines()
    assert lines[0].startswith(core.DENY_PREFIX + " (rule retail.cancel, check ")
    assert [line.split(":", 1)[0] for line in lines[1:]] == ["Reason", "Purpose", "Fix", "Verdict"]
    plain_deny, terse_deny = summary.parse_deny(plain.content), summary.parse_deny(terse.content)
    facts = [line[2:] for line in plain.content.splitlines() if line.startswith("- ")]
    assert facts and terse_deny["reason"] == "; ".join([plain_deny["reason"], *facts])
    assert {k: v for k, v in terse_deny.items() if k != "reason"} == {
        k: v for k, v in plain_deny.items() if k != "reason"}
    assert "Why it matters" not in terse.content
    assert [e["payload"]["wording"] for e in terse_events.events("gates.manifest")] == ["terse"]
    def decisions(events):
        return [{k: v for k, v in f.items() if k != "duration_ms"} for f in _fired(events)]

    assert decisions(terse_events) == decisions(plain_events)
    with pytest.raises(ValueError, match="unknown deny wording"):
        Enforcer("retail", EventLog(":memory:"), wording="loud")
    with pytest.raises(ValueError, match="unknown deny wording"):
        core.GateResult("deny").message("loud")
    terse_set, plan = tools_ab.SETS["banking-plan-terse-gated"], tools_ab.SETS["banking-plan-gated"]
    assert terse_set["arms"] == ("tools",) and terse_set["deny_wording"] == "terse" and "deny_wording" not in plan
    assert {k: terse_set[k] for k in ("tasks", "trials", "gates", "glimmer_usd")} == {
        k: plan[k] for k in ("tasks", "trials", "gates", "glimmer_usd")}
    assert all(terse_set[k] != plan[k] for k in ("data", "out", "save"))
    assert tools_ab._enforcer(terse_set, "retail", tmp_path).wording == "terse"
    (tmp_path / "plan").mkdir()
    assert tools_ab._enforcer(plan, "retail", tmp_path / "plan").wording == "plain"


@pytest.mark.parametrize("reason,expected", [
    ("payment method credit_card_1 is not allowed", "RETAIL_ARGS_NO"),
    ("item 123 is not in order #W1", "RETAIL_ARGS_NO"),
    ("status is 'delivered', action requires 'pending'", "POLICY_NO"),
    ("order #W1 does not exist", "RETAIL_ARGS_NO"),  # a mistyped order id: the right one can succeed
    ("payment method gift_card_1 is not allowed here; allowed: ['credit_card_2']", "RETAIL_ARGS_NO"),
    ("payment method gift_card_1 is not allowed here; allowed: none (no payment method covers it)", "POLICY_NO"),
    ("items ['1', '2'] are not in order #W1", "RETAIL_ARGS_NO"),
])
def test_a_retail_deny_on_the_arguments_says_a_corrected_call_can_succeed(monkeypatch, reason, expected):
    from glint.assist import retail as retail_checks

    monkeypatch.setattr(retail_checks, "check_eligibility",
                        lambda *a, **k: {"allowed": False, "reasons": [reason], "rules": []})
    assert rules._retail_gate("modify_payment")(None, {"order_id": "#W1"}).consequence == getattr(rules, expected)


def test_each_submit_verdict_has_its_own_consequence():
    assert set(rules.SUBMIT_CONSEQUENCE) == {"reduce_amount", "ask_amount", "error"}
    assert "largest increase" in rules.SUBMIT_CONSEQUENCE["reduce_amount"]
    assert "largest" not in rules.SUBMIT_CONSEQUENCE["ask_amount"] + rules.SUBMIT_CONSEQUENCE["error"]


def test_no_gate_message_carries_a_doubled_brace_from_a_plain_string_joined_to_an_f_string():
    # "...{{a, " f"..." "b}} ..." : the f-string part unescapes {{ but a plain string joined to it keeps }} as typed,
    # and the agent read "contacted_merchant}}" in procedure.dispute_plan.none's fix (2026-10-29)
    import ast
    from pathlib import Path

    for path in sorted((Path(__file__).parents[1] / "src" / "glint" / "gates").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.JoinedStr):
                for part in node.values:
                    if isinstance(part, ast.Constant):
                        assert "{{" not in part.value and "}}" not in part.value, (path.name, node.lineno, part.value)


@pytest.mark.parametrize("name", ["banking-plan-v2-gated", "banking-plan-v3-gated", "banking-plan-v4-gated",
                                  "banking-plan-v5-gated"])
def test_the_v2_and_v3_sets_are_the_plan_gated_tools_arm_again_in_their_own_folders(name):
    import tools_ab

    v2, plan = tools_ab.SETS[name], tools_ab.SETS["banking-plan-gated"]
    assert v2["arms"] == ("tools",) and "deny_wording" not in v2
    assert {k: v2[k] for k in ("tasks", "trials", "gates", "glimmer_usd", "requires_modal")} == {
        k: plan[k] for k in ("tasks", "trials", "gates", "glimmer_usd", "requires_modal")}
    others = [cfg for n, cfg in tools_ab.SETS.items() if n != name]
    assert all(v2[k] != cfg.get(k) for cfg in others for k in ("data", "out", "save"))
