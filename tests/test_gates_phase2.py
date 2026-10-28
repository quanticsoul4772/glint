"""Phase 2 write gates (glint.gates.debit, banking, procedure, travel) on real τ² databases: an allow path and a deny
path for every new rule. Only TRAIN tasks are used (banking: never the test_untouched 20 of
split-partition-banking.json). The conversation up to the write under test is τ²'s own gold actions, or calls built
here on the task's database."""
import json

import pytest
from loguru import logger
from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.data_model.tasks import Task
from tau2.registry import registry

from glint import paths
from glint.gates import Enforcer, EventLog, Gate, GateRegistry, allow, rules
from glint.gates import procedure as proc

logger.remove()
BANKING = {t["id"]: t for t in json.loads((paths.tau2_data() / "tau2" / "domains" / "banking_knowledge" /
                                           "tasks.json").read_text(encoding="utf-8"))}
TEST_SPLIT = set(json.loads((paths.RESULTS / "split-partition-banking.json").read_text(encoding="utf-8"))
                 ["test_untouched"])


def _bank(task_id: str, enforcer: Enforcer | None = None):
    assert task_id not in TEST_SPLIT, f"{task_id} is in the banking test split"
    task = Task.model_validate(BANKING[task_id])
    env = registry.get_env_constructor("banking_knowledge")(retrieval_variant="bm25_grep", task=task)
    init = task.initial_state
    env.set_state(init.initialization_data if init else None, init.initialization_actions if init else None, [])
    events = EventLog(":memory:")
    (enforcer or Enforcer("banking_knowledge", events)).attach(env)
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


def _verify(env, user_id: str) -> None:
    user = env.tools.db.users.data[user_id]
    assert _ran(_call(env, "log_verification", time_verified="2025-11-14 03:40:00 EST",
                      **{k: user[k] for k in ("name", "user_id", "address", "email", "phone_number",
                                              "date_of_birth")}))


def _lookups(env, user_id: str, account_id: str) -> None:
    _hidden(env, "get_all_user_accounts_by_user_id_3847", user_id=user_id)
    _hidden(env, "get_debit_cards_by_account_id_7823", account_id=account_id)
    _hidden(env, "get_bank_account_transactions_9173", account_id=account_id)


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


def test_close_a_frozen_card_is_unfrozen_first():
    env, task, _ = _bank("task_087")
    _play(env, task, "unfreeze_debit_card_3893")
    _lookups(env, "mt35a7c9d2", "chk_mt35a7c9d2_blue")
    blue = {"card_id": "dbc_mt35a7c9d2_blue", "reason": "lost"}
    assert _denied(_hidden(env, "close_debit_card_4721", **blue), "Unfreeze it")
    assert _ran(_hidden(env, "unfreeze_debit_card_3893", card_id=blue["card_id"]))
    assert _ran(_hidden(env, "close_debit_card_4721", **blue))


def test_close_waits_for_pending_transactions():
    """task_088: the Green account has a pending mobile deposit; the Evergreen card closes as the gold does."""
    env, task, _ = _bank("task_088")
    gold = _play(env, task, "close_debit_card_4721")
    _lookups(env, "cec8333d41", "chk_cec8333d41_green")
    assert _denied(_hidden(env, "close_debit_card_4721", card_id="dbc_31839d9c1891", reason="lost"),
                   "pending transactions")
    assert _card(env, "dbc_31839d9c1891")["status"] == "ACTIVE"
    assert _ran(_hidden(env, "close_debit_card_4721", **gold))
    assert _card(env, gold["card_id"])["status"] == "CLOSED"


def _new_card(env, account_id: str) -> str:
    return next(k for k, c in env.tools.db.debit_cards.data.items()
                if c["account_id"] == account_id and c["status"] == "PENDING")


def test_close_a_new_card_needs_14_days_unless_lost_stolen_or_fraud():
    env, task, _ = _bank("task_081")
    assert _ran(_hidden(env, "order_debit_card_5739", **_play(env, task, "order_debit_card_5739")))
    new = _new_card(env, "chk_tm92c4d7e8_blue")
    assert _denied(_hidden(env, "close_debit_card_4721", card_id=new, reason="no_longer_needed"), "at least 14 days")
    assert _ran(_hidden(env, "close_debit_card_4721", card_id=new, reason="lost"))


def test_order_fees_by_tier():
    env, task, _ = _bank("task_081")
    gold = _play(env, task, "order_debit_card_5739")  # Blue (mid): EXPEDITED is $15
    orders = len(env.tools.db.debit_card_orders.data)
    assert _denied(_hidden(env, "order_debit_card_5739", **(gold | {"delivery_fee": 0})), "delivery_fee 0")
    assert _denied(_hidden(env, "order_debit_card_5739", **(gold | {"delivery_option": "RUSH", "delivery_fee": 35})),
                   "RUSH delivery is not offered")
    assert _denied(_hidden(env, "order_debit_card_5739", **(gold | {"card_design": "PREMIUM"})), "design_fee 0")
    assert len(env.tools.db.debit_card_orders.data) == orders
    assert _ran(_hidden(env, "order_debit_card_5739", **gold))
    assert len(env.tools.db.debit_card_orders.data) == orders + 1


def test_order_on_an_entry_tier_account_waits_48_hours_after_closure():
    env, task, _ = _bank("task_081")  # the Green Fee-Free card (entry tier) is closed as stolen
    gold = _play(env, task, "order_debit_card_5739")
    gff = gold | {"account_id": "chk_tm92c4d7e8_gff", "delivery_option": "STANDARD", "delivery_fee": 0}
    assert _denied(_hidden(env, "order_debit_card_5739", **gff), "48 hours")
    # task_091: a fraud_suspected closure on an entry-tier account is replaced at once
    env, task, _ = _bank("task_091")
    gold = _play(env, task, "order_debit_card_5739", 1)
    assert gold["account_id"] == "chk_tw34a7d9e2_lightblue"
    assert _ran(_hidden(env, "order_debit_card_5739", **gold))


def test_order_past_the_replacement_limit_needs_the_excess_fee():
    """task_073's Light Green card (entry tier, limit 2) is a replacement from 08/2025; one more fraud replacement
    reaches the limit, so the next order needs the $25 excess fee."""
    env, _, events = _bank("task_073")
    user, account = "kj93a7b2e1", "chk_kj93a7b2e1_3"
    _verify(env, user)
    _lookups(env, user, account)
    order = {"account_id": account, "user_id": user, "delivery_option": "STANDARD", "delivery_fee": 0,
             "card_design": "CLASSIC", "design_fee": 0, "shipping_address": "1 Main St, Austin, TX 78701"}
    assert _ran(_hidden(env, "close_debit_card_4721", card_id="dbc_kj93a7b2e1_lightgreen", reason="fraud_suspected"))
    assert _ran(_hidden(env, "order_debit_card_5739", **order))  # the 2nd replacement in 12 months
    assert _ran(_hidden(env, "close_debit_card_4721", card_id=_new_card(env, account), reason="fraud_suspected"))
    assert _denied(_hidden(env, "order_debit_card_5739", **order), "excess_replacement_fee 0 must be 25")
    _hidden(env, "order_debit_card_5739", **(order | {"excess_replacement_fee": 25}))
    fired = [e["payload"] for e in events.events("gate.fired") if e["payload"]["tool"] == "order_debit_card_5739"]
    assert fired[-1]["decision"] == "allow"  # τ²'s own tool then refuses: the 2nd order is still pending


def test_activation_variant_follows_the_issue_reason():
    env, task, _ = _bank("task_081")
    assert _ran(_hidden(env, "order_debit_card_5739", **_play(env, task, "order_debit_card_5739")))
    card = _card(env, _new_card(env, "chk_tm92c4d7e8_blue"))  # replaces a stolen card
    details = {"card_id": card["card_id"], "last_4_digits": card["last_4_digits"], "cvv": card["cvv"],
               "expiration_date": card["expiration_date"][:2] + "/" + card["expiration_date"][-2:], "pin": "4826"}
    assert _denied(_hidden(env, "activate_debit_card_8291", **details), "Activate it with activate_debit_card_8292")
    assert _ran(_hidden(env, "activate_debit_card_8292", **details))
    assert card["status"] == "ACTIVE"


# ---- other banking writes ------------------------------------------------------------------------------------------

def test_log_verification_matches_the_record_and_the_clock():
    env, task, _ = _bank("task_081")
    gold = _play(env, task, "log_verification")
    assert _denied(_call(env, "log_verification", **(gold | {"time_verified": "2025-11-14 12:00:00 EST"})),
                   "is not the current time")
    assert _denied(_call(env, "log_verification", **(gold | {"email": "someone@else.com"})), "email")
    before = len(env.tools.db.verification_history.data)
    assert _ran(_call(env, "log_verification", **gold))
    assert len(env.tools.db.verification_history.data) == before + 1


def test_change_email_needs_a_verified_user_and_a_new_address():
    env, _, _ = _bank("task_005")
    assert _denied(_call(env, "change_user_email", user_id="6680a37184", new_email="k@new.com"), "no verification")
    _verify(env, "6680a37184")
    email = env.tools.db.users.data["6680a37184"]["email"]
    assert _denied(_call(env, "change_user_email", user_id="6680a37184", new_email=email), "already")
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
    env, task, _ = _bank("task_073")
    first = _play(env, task, "apply_checking_account_credit_5829")
    assert _ran(_hidden(env, "apply_checking_account_credit_5829", **first))
    assert _denied(_hidden(env, "apply_checking_account_credit_5829", **(first | {"amount": 1.0})), "ONE credit")
    other = next(k for k in ("chk_kj93a7b2e1_1", "chk_kj93a7b2e1_2", "chk_kj93a7b2e1_3") if k != first["account_id"])
    assert _denied(_hidden(env, "apply_checking_account_credit_5829", account_id=other, amount=5000.0,
                           credit_type="fee_refund"), "exceeds")
    assert _ran(_hidden(env, "apply_checking_account_credit_5829", account_id=other, amount=1.0,
                        credit_type="rebate_credit"))


def test_closure_reason_only_for_an_account_that_can_close():
    env, task, _ = _bank("task_049")
    gold = _play(env, task, "log_credit_card_closure_reason_4521")
    blocked = next(k for k, v in env.tools.db.credit_card_accounts.data.items()
                   if v.get("user_id") == gold["user_id"] and "eco" in k.lower())
    assert _denied(_hidden(env, "log_credit_card_closure_reason_4521", **(gold | {"credit_card_account_id": blocked})),
                   "replacement card not yet received")
    assert _ran(_hidden(env, "log_credit_card_closure_reason_4521", **gold))


def test_close_credit_card_needs_a_logged_reason():
    env, task, _ = _bank("task_048")
    log = _play(env, task, "log_credit_card_closure_reason_4521")  # the Gold card, its balance paid
    close = {"credit_card_account_id": log["credit_card_account_id"], "user_id": log["user_id"]}
    assert _denied(_hidden(env, "close_credit_card_account_7834", **close), "no closure reason")
    assert _ran(_hidden(env, "log_credit_card_closure_reason_4521", **log))
    assert _ran(_hidden(env, "close_credit_card_account_7834", **close))
    env, task, _ = _bank("task_048")  # the Green card has a reason logged within the past year: it closes
    assert _ran(_hidden(env, "close_credit_card_account_7834", **_play(env, task, "close_credit_card_account_7834")))


def test_fee_waiver_answers_a_logged_annual_fee_reason():
    env, task, _ = _bank("task_044")
    log = _play(env, task, "log_credit_card_closure_reason_4521")
    flag = {"credit_card_account_id": log["credit_card_account_id"], "user_id": log["user_id"],
            "flag_type": "annual_fee_waived", "expiration_date": "11/14/2026", "reason": "loyalty_benefit"}
    assert _denied(_hidden(env, "apply_credit_card_account_flag_6147", **flag), "no closure reason was logged")
    assert _ran(_hidden(env, "log_credit_card_closure_reason_4521", **(log | {"closure_reason": "annual_fee"})))
    assert _ran(_hidden(env, "apply_credit_card_account_flag_6147", **flag))


def test_fee_waiver_needs_the_annual_fee_reason():
    env, task, _ = _bank("task_044")
    log = _play(env, task, "log_credit_card_closure_reason_4521")
    assert _ran(_hidden(env, "log_credit_card_closure_reason_4521", **log))  # found_better_card
    flag = {"credit_card_account_id": log["credit_card_account_id"], "user_id": log["user_id"],
            "flag_type": "annual_fee_waived", "expiration_date": "11/14/2026", "reason": "loyalty_benefit"}
    assert _denied(_hidden(env, "apply_credit_card_account_flag_6147", **flag), "'annual_fee' closure reason")


def test_retention_credit_amount_by_card_tier():
    env, task, _ = _bank("task_045")
    gold = _play(env, task, "apply_statement_credit_8472")  # a Silver (mid-tier) card: $20
    assert _denied(_hidden(env, "apply_statement_credit_8472", **(gold | {"amount": 50.0})), "retention credit is $20")
    assert _ran(_hidden(env, "apply_statement_credit_8472", **gold))


# ---- procedure gates -----------------------------------------------------------------------------------------------

def test_a_write_needs_the_user_verified_in_this_conversation():
    env, task, _ = _bank("task_081")
    assert _denied(_hidden(env, "freeze_debit_card_3892", card_id="dbc_tm92c4d7e8_blue"), "no verification")
    _play(env, task, "freeze_debit_card_3892")
    assert _ran(_hidden(env, "freeze_debit_card_3892", card_id="dbc_tm92c4d7e8_blue"))


def test_a_close_needs_the_transactions_lookup_on_the_gates_own_record():
    env, _, _ = _bank("task_084")
    _verify(env, "a7c2f85d91")
    close = {"card_id": "dbc_82b5d1a6f3", "reason": "fraud_suspected"}
    assert _denied(_hidden(env, "close_debit_card_4721", **close), "get_bank_account_transactions_9173")
    assert env.tools.glint_reads == set()
    assert _call(env, "call_discoverable_agent_tool", agent_tool_name="get_bank_account_transactions_9173",
                 arguments=json.dumps({"account_id": "chk_e5a31c7d82"})).error is False
    assert env.tools.glint_reads == set()  # not unlocked: the call failed and is not a lookup
    _hidden(env, "get_bank_account_transactions_9173", account_id="chk_e5a31c7d82")
    assert env.tools.glint_reads == {"get_bank_account_transactions_9173"}
    assert _ran(_hidden(env, "close_debit_card_4721", **close))


def test_lookup_gate_decisions_replay_unchanged_in_taus_strict_replay():
    """τ²'s evaluator replays only mutating calls (the hidden-tool wrapper is one) on a fresh environment from the same
    constructor; the read record is rebuilt there, so a deny and the later allow replay unchanged (strict=True)."""
    enforcer = Enforcer("banking_knowledge", EventLog(":memory:"))
    env, task, _ = _bank("task_084", enforcer)
    history = []

    def send(name, /, **args):
        call = ToolCall(id=f"c{len(history)}", name=name, arguments=args, requestor="assistant")
        reply = env.get_response(call)
        history.extend([AssistantMessage(role="assistant", tool_calls=[call]), reply])
        return reply

    user = env.tools.db.users.data["a7c2f85d91"]
    send("log_verification", time_verified="2025-11-14 03:40:00 EST",
         **{k: user[k] for k in ("name", "user_id", "address", "email", "phone_number", "date_of_birth")})
    close = json.dumps({"card_id": "dbc_82b5d1a6f3", "reason": "fraud_suspected"})
    send("unlock_discoverable_agent_tool", agent_tool_name="close_debit_card_4721")
    assert _denied(send("call_discoverable_agent_tool", agent_tool_name="close_debit_card_4721", arguments=close),
                   "lookup")
    send("unlock_discoverable_agent_tool", agent_tool_name="get_bank_account_transactions_9173")
    send("call_discoverable_agent_tool", agent_tool_name="get_bank_account_transactions_9173",
         arguments=json.dumps({"account_id": "chk_e5a31c7d82"}))
    assert _ran(send("call_discoverable_agent_tool", agent_tool_name="close_debit_card_4721", arguments=close))
    replay, _, _ = _bank("task_084", enforcer)
    init = task.initial_state
    replay.set_state(init.initialization_data if init else None, init.initialization_actions if init else None,
                     history, strict=True)
    assert replay.tools.db.debit_cards.data["dbc_82b5d1a6f3"]["status"] == "CLOSED"


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
    env, task, _ = _bank("task_081")
    _play(env, task, "transfer_to_human_agents")  # the stolen cards are closed and the fraud disputes filed
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


def test_replay_waits_for_a_slow_gate_instead_of_timing_out():
    """Live, a gate over its timeout fails closed (logged as an error); in τ²'s replay it runs to its own decision, so
    the replayed decision depends only on the database and the arguments."""
    import threading

    from glint.gates import run_gate

    release = threading.Event()
    slow = Gate("slow", ("t",), "p", lambda db, a: release.wait(5) and allow("r", "s"), timeout_s=0.05)
    live, _ = run_gate(slow, None, {})
    assert live.rule_id == "gate_timeout"
    threading.Timer(0.2, release.set).start()
    replayed, ms = run_gate(slow, None, {}, wait=True)
    assert replayed.allowed and ms >= 150
