"""Phase 2 write gates (glint.gates.debit, banking, procedure, travel) on real τ² databases: an allow path and a deny
path for every new rule. Banking environments are built the way τ²'s runner builds them (with the task's read-log
allowlist), and the conversation up to the write under test is τ²'s own gold actions."""
import json

import pytest
from loguru import logger
from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import Task
from tau2.registry import registry
from tau2.runner.build import _derive_read_log_allowlist

from glint import paths
from glint.gates import Enforcer, EventLog, Gate, GateRegistry, allow, rules
from glint.gates import procedure as proc

logger.remove()
BANKING = {t["id"]: t for t in json.loads((paths.tau2_data() / "tau2" / "domains" / "banking_knowledge" /
                                           "tasks.json").read_text(encoding="utf-8"))}


def _bank(task_id: str, allowlist: bool = True):
    task = Task.model_validate(BANKING[task_id])
    env = registry.get_env_constructor("banking_knowledge")(
        retrieval_variant="bm25_grep", task=task,
        read_log_allowlist=_derive_read_log_allowlist(task) if allowlist else None)
    init = task.initial_state
    env.set_state(init.initialization_data if init else None, init.initialization_actions if init else None, [])
    events = EventLog(":memory:")
    Enforcer("banking_knowledge", events).attach(env)
    return env, task, events


def _name(action) -> str:
    return action.arguments.get("agent_tool_name") if action.name == "call_discoverable_agent_tool" else action.name


def _play(env, task, tool: str, n: int = 0) -> dict:
    """Send τ²'s gold actions through the gated environment up to (not including) the n-th call of `tool`; return
    that call's arguments. Every gold action on the way must be allowed."""
    seen = 0
    for k, action in enumerate(task.evaluation_criteria.actions):
        if _name(action) == tool:
            if seen == n:
                args = action.arguments
                return json.loads(args["arguments"]) if action.name == "call_discoverable_agent_tool" else dict(args)
            seen += 1
        reply = env.get_response(ToolCall(id=f"g{k}", name=action.name, arguments=action.arguments,
                                          requestor=action.requestor))
        assert "was NOT executed" not in str(reply.content), (action.name, reply.content)
    raise AssertionError(f"{tool} #{n} is not in {task.id}'s gold actions")


def _call(env, name: str, /, **args):
    return env.get_response(ToolCall(id="t", name=name, arguments=args, requestor="assistant"))


def _hidden(env, tool: str, /, **args):
    _call(env, "unlock_discoverable_agent_tool", agent_tool_name=tool)
    return _call(env, "call_discoverable_agent_tool", agent_tool_name=tool, arguments=json.dumps(args))


def _denied(reply, text: str) -> bool:
    return reply.error and "was NOT executed" in reply.content and text in reply.content


def _ran(reply) -> bool:
    return "was NOT executed" not in reply.content and not str(reply.content).startswith("Error")


def _rule(events) -> str:
    return events.events("gate.fired")[-1]["payload"]["rule_id"]


def _card(env, card_id: str) -> dict:
    return env.tools.db.debit_cards.data[card_id]


# ---- debit cards (bank_accounts_(general)_023-_029) ----------------------------------------------------------------

def test_freeze_only_an_active_card():
    env, task, _ = _bank("task_087")
    _play(env, task, "unfreeze_debit_card_3893")
    assert _denied(_hidden(env, "freeze_debit_card_3892", card_id="dbc_mt35a7c9d2_blue"), "only an ACTIVE card")
    assert _ran(_hidden(env, "freeze_debit_card_3892", card_id="dbc_mt35a7c9d2_evergreen"))
    assert _card(env, "dbc_mt35a7c9d2_evergreen")["status"] == "FROZEN"


def test_unfreeze_only_a_frozen_card():
    env, task, _ = _bank("task_087")
    gold = _play(env, task, "unfreeze_debit_card_3893")
    assert _denied(_hidden(env, "unfreeze_debit_card_3893", card_id="dbc_mt35a7c9d2_evergreen"), "only a FROZEN")
    assert _ran(_hidden(env, "unfreeze_debit_card_3893", **gold))
    assert _card(env, gold["card_id"])["status"] == "ACTIVE"


def test_close_waits_for_pending_transactions_and_unfreezes_first():
    """task_078: the Light Green card has pending transactions; it stays frozen and is not closed."""
    env, task, _ = _bank("task_078")
    gold = _play(env, task, "close_debit_card_4721")
    lg = {"card_id": "dbc_mc78a5b9d2_lg", "reason": "lost"}
    assert _denied(_hidden(env, "close_debit_card_4721", **lg), "Unfreeze it")
    assert _ran(_hidden(env, "unfreeze_debit_card_3893", card_id=lg["card_id"]))
    assert _denied(_hidden(env, "close_debit_card_4721", **lg), "pending transactions")
    assert _card(env, lg["card_id"])["status"] == "ACTIVE"
    assert _ran(_hidden(env, "close_debit_card_4721", **gold))
    assert _card(env, gold["card_id"])["status"] == "CLOSED"


def test_close_a_new_card_needs_14_days_unless_lost_stolen_or_fraud():
    env, task, _ = _bank("task_077")
    assert _ran(_hidden(env, "order_debit_card_5739", **_play(env, task, "order_debit_card_5739")))
    new = next(k for k, c in env.tools.db.debit_cards.data.items()
               if c["account_id"] == "chk_lj82d4f1a9" and c["status"] == "PENDING")
    assert _denied(_hidden(env, "close_debit_card_4721", card_id=new, reason="no_longer_needed"), "at least 14 days")
    assert _ran(_hidden(env, "close_debit_card_4721", card_id=new, reason="lost"))


def test_close_with_a_pending_refund_is_allowed_and_logged_as_unseen():
    """task_079: a pending refund blocks the closure unless the user acknowledges it in writing (conversation only)."""
    env, task, events = _bank("task_079")
    gold = _play(env, task, "close_debit_card_4721", 2)
    assert gold["card_id"] == "dbc_cr89a2b3c4_green"
    assert _ran(_hidden(env, "close_debit_card_4721", **gold))
    assert any(e["payload"]["rule_id"] == "debit.close.refund_acknowledgment_unseen"
               for e in events.events("gate.fired"))


def test_order_fees_by_tier():
    env, task, _ = _bank("task_077")
    gold = _play(env, task, "order_debit_card_5739")  # Bluest (elite): RUSH and PREMIUM design are free
    orders = len(env.tools.db.debit_card_orders.data)
    assert _denied(_hidden(env, "order_debit_card_5739", **(gold | {"delivery_fee": 35})), "delivery_fee 35")
    assert _denied(_hidden(env, "order_debit_card_5739", **(gold | {"design_fee": 10})), "design_fee 10")
    assert len(env.tools.db.debit_card_orders.data) == orders
    assert _ran(_hidden(env, "order_debit_card_5739", **gold))
    assert len(env.tools.db.debit_card_orders.data) == orders + 1


def test_order_on_an_entry_tier_account_waits_48_hours_after_closure():
    env, task, _ = _bank("task_077")
    gold = _play(env, task, "order_debit_card_5739")
    light_green = gold | {"account_id": "chk_538bfb9cba", "delivery_option": "STANDARD", "card_design": "CLASSIC",
                          "delivery_fee": 0, "design_fee": 0}
    assert _denied(_hidden(env, "order_debit_card_5739", **light_green), "48 hours")
    # task_091: a fraud_suspected closure on an entry-tier account is replaced at once
    env, task, _ = _bank("task_091")
    gold = _play(env, task, "order_debit_card_5739", 1)
    assert gold["account_id"] == "chk_tw34a7d9e2_lightblue"
    assert _ran(_hidden(env, "order_debit_card_5739", **gold))


def test_order_past_the_replacement_limit_needs_the_excess_fee():
    """task_078: the Light Blue card is already the 2nd replacement in 12 months (entry limit 2)."""
    env, task, _ = _bank("task_078")
    _play(env, task, "close_debit_card_4721")
    assert _ran(_hidden(env, "close_debit_card_4721", card_id="dbc_mc78a5b9d2_lb", reason="fraud_suspected"))
    order = {"account_id": "chk_mc78a5b9d2_1", "user_id": "mc78a5b9d2", "delivery_option": "STANDARD",
             "delivery_fee": 0, "card_design": "CLASSIC", "design_fee": 0,
             "shipping_address": "3245 Sunset Boulevard, San Diego, CA 92103"}
    _hidden(env, "get_debit_cards_by_account_id_7823", account_id="chk_mc78a5b9d2_1")
    assert _denied(_hidden(env, "order_debit_card_5739", **order), "excess_replacement_fee 0 must be 25")
    assert _ran(_hidden(env, "order_debit_card_5739", **(order | {"excess_replacement_fee": 25})))


def test_activation_variant_follows_the_issue_reason():
    env, task, _ = _bank("task_077")
    gold = _play(env, task, "activate_debit_card_8292")
    assert _denied(_hidden(env, "activate_debit_card_8291", **gold), "Activate it with activate_debit_card_8292")
    assert _ran(_hidden(env, "activate_debit_card_8292", **gold))
    assert _card(env, gold["card_id"])["status"] == "ACTIVE"


# ---- other banking writes ------------------------------------------------------------------------------------------

def test_log_verification_matches_the_record_and_the_clock():
    env, task, _ = _bank("task_077")
    gold = _play(env, task, "log_verification")
    assert _denied(_call(env, "log_verification", **(gold | {"time_verified": "2025-11-14 12:00:00 EST"})),
                   "is not the current time")
    assert _denied(_call(env, "log_verification", **(gold | {"email": "someone@else.com"})), "email")
    assert _ran(_call(env, "log_verification", **gold))
    assert len(env.tools.db.verification_history.data) == 4


def test_change_email_needs_a_verified_user_and_a_new_address():
    env, _, _ = _bank("task_005")
    user = env.tools.db.users.data["6680a37184"]
    assert _denied(_call(env, "change_user_email", user_id="6680a37184", new_email="k@new.com"), "no verification")
    _call(env, "log_verification", time_verified="2025-11-14 03:40:00 EST",
          **{k: user[k] for k in ("name", "user_id", "address", "email", "phone_number", "date_of_birth")})
    assert _denied(_call(env, "change_user_email", user_id="6680a37184", new_email=user["email"]), "already")
    assert _ran(_call(env, "change_user_email", user_id="6680a37184", new_email="k@new.com"))


def test_rewards_update_only_after_an_approved_cash_back_dispute():
    env, task, _ = _bank("task_026")
    gold = _play(env, task, "update_transaction_rewards_3847")
    user = env.tools.db.credit_card_transaction_history.data[gold["transaction_id"]]["user_id"]
    disputed = {r["transaction_id"] for r in env.tools.db.cash_back_disputes.data.values()}
    other = next(k for k, t in env.tools.db.credit_card_transaction_history.data.items()
                 if t.get("user_id") == user and k not in disputed)
    assert _denied(_hidden(env, "update_transaction_rewards_3847", transaction_id=other, new_rewards_earned="1 points"),
                   "submit_cash_back_dispute_0589")
    assert _ran(_hidden(env, "update_transaction_rewards_3847", **gold))


def test_checking_credit_once_per_account_and_within_the_fees():
    env, task, _ = _bank("task_074")
    first = _play(env, task, "apply_checking_account_credit_5829")
    assert _ran(_hidden(env, "apply_checking_account_credit_5829", **first))
    assert _denied(_hidden(env, "apply_checking_account_credit_5829", **(first | {"amount": 1.0})), "ONE credit")
    assert _denied(_hidden(env, "apply_checking_account_credit_5829", account_id="chk_ar72c5d8e3_2", amount=500.0,
                           credit_type="fee_refund"), "exceeds")
    assert _ran(_hidden(env, "apply_checking_account_credit_5829", account_id="chk_ar72c5d8e3_2", amount=14.5,
                        credit_type="fee_refund"))


def test_closure_reason_only_for_an_account_that_can_close():
    env, task, _ = _bank("task_049")
    gold = _play(env, task, "log_credit_card_closure_reason_4521")
    blocked = next(k for k, v in env.tools.db.credit_card_accounts.data.items()
                   if v.get("user_id") == gold["user_id"] and "eco" in k.lower())
    assert _denied(_hidden(env, "log_credit_card_closure_reason_4521", **(gold | {"credit_card_account_id": blocked})),
                   "replacement card not yet received")
    assert _ran(_hidden(env, "log_credit_card_closure_reason_4521", **gold))


def test_close_credit_card_needs_a_logged_reason():
    env, task, _ = _bank("task_047")
    log = _play(env, task, "log_credit_card_closure_reason_4521")
    close = {"credit_card_account_id": log["credit_card_account_id"], "user_id": log["user_id"]}
    assert _denied(_hidden(env, "close_credit_card_account_7834", **close), "no closure reason")
    env, task, events = _bank("task_049")  # a reason logged within the past year: the closure goes ahead
    assert _ran(_hidden(env, "close_credit_card_account_7834", **_play(env, task, "close_credit_card_account_7834")))
    assert _rule(events) == "procedure.lookups"


def test_fee_waiver_answers_a_logged_annual_fee_reason():
    env, task, _ = _bank("task_043")
    log = _play(env, task, "log_credit_card_closure_reason_4521")
    flag = json.loads(next(a.arguments["arguments"] for a in task.evaluation_criteria.actions
                           if _name(a) == "apply_credit_card_account_flag_6147"))
    assert _denied(_hidden(env, "apply_credit_card_account_flag_6147", **flag), "no closure reason was logged")
    assert _ran(_hidden(env, "log_credit_card_closure_reason_4521", **log))
    assert _ran(_hidden(env, "apply_credit_card_account_flag_6147", **flag))


def test_retention_credit_amount_by_card_tier():
    env, task, _ = _bank("task_047")
    gold = _play(env, task, "apply_statement_credit_8472")
    assert _denied(_hidden(env, "apply_statement_credit_8472", **(gold | {"amount": 20.0})), "retention credit is $50")
    assert _ran(_hidden(env, "apply_statement_credit_8472", **gold))


# ---- procedure gates -----------------------------------------------------------------------------------------------

def test_a_write_needs_the_user_verified_in_this_conversation():
    env, task, _ = _bank("task_077")
    assert _denied(_hidden(env, "freeze_debit_card_3892", card_id="dbc_lj82d4f1a9_bluest"), "no verification")
    _play(env, task, "freeze_debit_card_3892")
    assert _ran(_hidden(env, "freeze_debit_card_3892", card_id="dbc_lj82d4f1a9_bluest"))


def test_a_dispute_needs_the_recorded_dispute_history_lookup():
    env, task, _ = _bank("task_040")
    _call(env, "log_verification", **_play(env, task, "log_verification"))
    dispute = json.loads(next(a.arguments["arguments"] for a in task.evaluation_criteria.actions
                              if _name(a) == rules.DISPUTE))
    before = len(env.tools.db.transaction_disputes.data)
    assert _denied(_hidden(env, rules.DISPUTE, **dispute), "get_user_dispute_history_7291")
    assert len(env.tools.db.transaction_disputes.data) == before
    _hidden(env, "get_user_dispute_history_7291", user_id=dispute["user_id"])
    assert _ran(_hidden(env, rules.DISPUTE, **dispute))
    # an environment that does not record the lookup cannot show it: the gate does not require it there
    env, task, events = _bank("task_040", allowlist=False)
    _call(env, "log_verification", **_play(env, task, "log_verification"))
    assert _ran(_hidden(env, rules.DISPUTE, **dispute))
    fired = [e["payload"] for e in events.events("gate.fired") if e["payload"]["gate"].startswith("procedure_lookups")]
    assert fired[-1]["decision"] == "allow"


def test_a_dispute_waits_for_the_submitted_increase():
    env, task, _ = _bank("task_054")
    approve = _play(env, task, rules.APPROVE)
    dispute = json.loads(next(a.arguments["arguments"] for a in task.evaluation_criteria.actions
                              if _name(a) == rules.DISPUTE))
    assert _denied(_hidden(env, rules.DISPUTE, **dispute), "credit limit increase first")
    assert _ran(_hidden(env, rules.APPROVE, **approve))
    assert _ran(_hidden(env, rules.DISPUTE, **dispute))


def test_transfer_waits_for_the_increase_decision():
    env, task, _ = _bank("task_054")
    approve = _play(env, task, rules.APPROVE)
    reply = _call(env, "transfer_to_human_agents", summary="s", reason="customer_requests_human_no_specific_reason")
    assert _denied(reply, "approve or deny")
    _hidden(env, rules.APPROVE, **approve)
    assert _ran(_call(env, "transfer_to_human_agents", summary="s", reason="customer_requests_human_no_specific_reason"))


def test_transfer_reason_after_fraud_is_the_fraud_code():
    env, task, _ = _bank("task_080")
    _play(env, task, "activate_debit_card_8292")  # the stolen cards are closed and the fraud disputes filed
    assert _denied(_call(env, "transfer_to_human_agents", summary="s", reason="customer_frustrated_demands_human"),
                   "fraud_or_security_concern")
    assert _ran(_call(env, "transfer_to_human_agents", summary="s", reason="fraud_or_security_concern"))


def test_transfer_is_gated_only_where_it_carries_a_reason_code():
    env = registry.get_env_constructor("retail")()
    events = EventLog(":memory:")
    Enforcer("retail", events).attach(env)
    _call(env, "transfer_to_human_agents", summary="s")
    assert events.events("gate.fired") == []


def test_a_toolkit_gate_given_no_toolkit_denies():
    events = EventLog(":memory:")
    enforcer = Enforcer("retail", events)
    enforcer.registry = GateRegistry()  # only the toolkit gate, so no other gate decides first
    enforcer.registry.register(Gate("needs_toolkit", ("cancel_pending_order",), "p",
                                    lambda tk, a: allow("r", "s"), toolkit=True))
    result = enforcer.check("cancel_pending_order", {"order_id": "#W0000000"}, None)
    assert not result.allowed and result.rule_id == "gate_error" and "needs the environment's toolkit" in result.reason
    assert enforcer.check("cancel_pending_order", {}, None, toolkit=object()).allowed


def test_every_lookup_rule_names_its_source_document():
    assert all(source for reqs in proc.REQUIRED_LOOKUPS.values() for _, source in reqs)
    assert set(rules.NO_RULE_YET["banking_knowledge"]) == set(rules.NO_RULE_REASONS)
    assert rules.NO_RULE_YET["retail"] == rules.NO_RULE_YET["airline"] == ()


# ---- airline and retail --------------------------------------------------------------------------------------------

def _travel(domain: str):
    env = registry.get_env_constructor(domain)()
    events = EventLog(":memory:")
    Enforcer(domain, events).attach(env)
    return env, events


def test_certificate_only_where_the_policy_compensates():
    env, events = _travel("airline")
    assert _denied(_call(env, "send_certificate", user_id="olivia_jackson_4826", amount=100), "no compensation")
    # a gold member whose delayed reservation is unchanged: no certificate for the delay alone (airline task 38)
    assert _denied(_call(env, "send_certificate", user_id="noah_muller_9847", amount=50), "changed or cancelled")
    assert _denied(_call(env, "send_certificate", user_id="noah_muller_9847", amount=75), "neither $100 nor $50")
    assert _ran(_call(env, "send_certificate", user_id="isabella_anderson_9682", amount=100))
    assert _rule(events) == "airline.certificate.cancelled_flight"
    assert _ran(_call(env, "send_certificate", user_id="noah_muller_9847", amount=100))
    assert _rule(events) == "no_rule_yet"


def test_book_reservation_bags_and_payment_methods():
    task = next(t for t in registry.get_tasks_loader("airline")("train") + registry.get_tasks_loader("airline")("test")
                if t.id == "20")
    gold = next(a.arguments for a in task.evaluation_criteria.actions if a.name == "book_reservation")
    env, _ = _travel("airline")
    assert _denied(_call(env, "book_reservation", **(gold | {"nonfree_baggages": 1})), "nonfree_baggages 1")
    six = (gold["passengers"] * 6)[:6]
    assert _denied(_call(env, "book_reservation", **(gold | {"passengers": six})), "at most five")
    assert _ran(_call(env, "book_reservation", **gold))


def test_user_address_needs_every_field():
    env, _ = _travel("retail")
    user = next(iter(env.tools.db.users))
    address = {"user_id": user, "address1": "1 Main St", "address2": "", "city": "Austin", "state": "TX",
               "country": "USA", "zip": "78701"}
    assert _denied(_call(env, "modify_user_address", **(address | {"city": " "})), "['city']")
    assert _ran(_call(env, "modify_user_address", **address))
    assert env.tools.db.users[user].address.city == "Austin"


@pytest.mark.parametrize("domain", ["retail", "airline", "banking_knowledge"])
def test_manifest_names_the_gated_generic_tools(domain):
    events = EventLog(":memory:")
    Enforcer(domain, events)
    manifest = events.events("gates.manifest")[0]["payload"]
    assert manifest["gated_generic"] == list(rules.GATED_GENERIC.get(domain, ()))
