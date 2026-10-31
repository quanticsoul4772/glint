"""Enforced dispute planning (decision 2026-10-28): plan_disputes (glint.assist.banking), the plan gate on credit card
disputes (glint.gates.procedure.dispute_plan), the enforcer's plan record, τ²'s strict replay of it, glint-bank
publishing the tool, and the replay helper (glint.gates.replayplan). Real τ² databases of TRAIN tasks only (never the
test_untouched 20 of split-partition-banking.json); tasks 039-041 are tuned tasks from here on."""
import json

import anyio
import pytest
from loguru import logger
from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.data_model.tasks import Task
from tau2.registry import registry

from glint import paths
from glint.assist import banking
from glint.assist.tau2_tools import BankingDecisionTools, DisputePlanTools, add_plan_tool, install
from glint.gates import Enforcer, EventLog, enforce, rules
from glint.gates.core import run_gate
from glint.gates.replayplan import PLAN_FIELDS, filing, plan_arguments, with_plans

logger.remove()
TASKS = {t["id"]: t for t in json.loads((paths.tau2_data() / "tau2" / "domains" / "banking_knowledge" /
                                         "tasks.json").read_text(encoding="utf-8"))}
TEST_SPLIT = set(json.loads((paths.RESULTS / "split-partition-banking.json").read_text(encoding="utf-8"))
                 ["test_untouched"])
WRAPPER = "call_discoverable_agent_tool"
FIELDS = PLAN_FIELDS


def _task(task_id: str) -> Task:
    assert task_id not in TEST_SPLIT, f"{task_id} is in the banking test split"
    return Task.model_validate(TASKS[task_id])


def _env(task, enforcer: Enforcer | None = None):
    env = registry.get_env_constructor("banking_knowledge")(retrieval_variant="bm25_grep", task=task)
    init = task.initial_state
    env.set_state(init.initialization_data if init else None, init.initialization_actions if init else None, [])
    if enforcer is not None:
        enforcer.attach(env)
    return env


def _gold_disputes(task) -> list[dict]:
    return [json.loads(a.arguments["arguments"]) for a in task.evaluation_criteria.actions
            if a.name == WRAPPER and a.arguments.get("agent_tool_name") == rules.DISPUTE]


def _items(disputes: list[dict]) -> list[dict]:
    return [{k: d[k] for k in FIELDS if d.get(k) is not None} for d in disputes]


def _call(env, name: str, /, **args):
    return env.get_response(ToolCall(id="t", name=name, arguments=args, requestor="assistant"))


def _file(env, args: dict):
    _call(env, "unlock_discoverable_agent_tool", agent_tool_name=rules.DISPUTE)
    return _call(env, WRAPPER, agent_tool_name=rules.DISPUTE, arguments=json.dumps(args))


def _verify(env, user_id: str) -> None:
    u = env.tools.db.users.data[user_id]
    _call(env, "log_verification", time_verified="2025-11-14 03:40:00 EST",
          **{k: u[k] for k in ("name", "user_id", "address", "email", "phone_number", "date_of_birth")})


def test_the_plan_denies_state_the_credit_count_the_rule_uses():
    # 015 criterion 4 counts the disputes before this one: with none on record, the first 3 filings can get credit
    from glint.gates import procedure

    assert "2 or fewer other disputes" in procedure.CREDIT_ORDER and "at most 2" not in procedure.CREDIT_ORDER
    env, _, gold = _gated("task_040")
    reply = _file(env, gold[0])
    assert _denied(reply, "procedure.dispute_plan.none") and procedure.CREDIT_ORDER in reply.content


def _denied(reply, rule: str) -> bool:
    return reply.error and "was NOT executed" in reply.content and f"Rule: {rule} (" in reply.content


def _ran(reply) -> bool:
    return "was NOT executed" not in reply.content and not str(reply.content).startswith("Error")


def _plan_gate(env, args: dict):
    """The plan gate's own verdict (the dispute gate before it may deny first on the live database)."""
    gate = next(g for g in env.glint_enforcer.registry.for_tool(rules.DISPUTE) if g.name == "procedure_dispute_plan_gate")
    return run_gate(gate, env.tools, args)[0]


def _give_card_digits(env) -> None:
    """Give the customer the card-digits tool, as gold does where the card has no digits on file
    (procedure.card_digits)."""
    _call(env, "give_discoverable_user_tool", discoverable_tool_name=banking.CARD_DIGITS_TOOL)


def _gated(task_id: str, digits: bool = True):
    task = _task(task_id)
    env = _env(task, Enforcer("banking_knowledge", EventLog(":memory:")))
    disputes = _gold_disputes(task)
    _verify(env, disputes[0]["user_id"])
    _read_history(env, disputes[0]["user_id"])
    if digits:
        _give_card_digits(env)
    return env, task, disputes


def _read_history(env, user_id: str) -> None:
    """The dispute-history read every filing requires (procedure.REQUIRED_LOOKUPS, credit_cards_(general)_015
    criterion 4)."""
    _call(env, "unlock_discoverable_agent_tool", agent_tool_name=banking.DISPUTE_HISTORY_TOOL)
    _call(env, WRAPPER, agent_tool_name=banking.DISPUTE_HISTORY_TOOL, arguments=json.dumps({"user_id": user_id}))


# ---- plan_disputes on the tuned tasks ------------------------------------------------------------------------------

GOLD_FLAGS = {"task_039": "TTTFFFFF", "task_040": "TTFFFFFF", "task_041": "TT" + "F" * 14}


@pytest.mark.parametrize("task_id", sorted(GOLD_FLAGS))
def test_plan_files_the_qualifying_disputes_first_with_golds_credit_flags(task_id):
    task = _task(task_id)
    db, gold = _env(task).tools.db, _gold_disputes(task)
    user = gold[0]["user_id"]
    flags = "".join("T" if d["eligible_for_provisional_credit"] else "F" for d in gold)
    assert flags == GOLD_FLAGS[task_id]
    # gold's own order comes back unchanged, with gold's flag on every transaction
    plan = banking.plan_disputes(db, user, _items(gold))
    assert [(e["transaction_id"], e["eligible_for_provisional_credit"]) for e in plan["order"]] == \
        [(d["transaction_id"], d["eligible_for_provisional_credit"]) for d in gold]
    assert plan["ready"] and plan["rejected"] == plan["already_filed"] == []
    # the never-qualifying disputes named first are moved behind the rest; inside each group the given order is kept,
    # so every transaction gets gold's flag
    first = [e["transaction_id"] for e in plan["order"] if e["can_get_credit"]]
    shuffled = [i for i in _items(gold) if i["transaction_id"] not in first] + \
               [i for i in _items(gold) if i["transaction_id"] in first]
    again = banking.plan_disputes(db, user, shuffled)
    assert [e["transaction_id"] for e in again["order"]] == [d["transaction_id"] for d in gold]
    # any given order: gold's sequence of credit flags, credit only to the first qualifying ones
    backwards = banking.plan_disputes(db, user, list(reversed(_items(gold))))
    assert "".join("T" if e["eligible_for_provisional_credit"] else "F" for e in backwards["order"]) == flags
    counted = [e["disputes_counted_before"] for e in backwards["order"]]
    assert counted == list(range(counted[0], counted[0] + len(gold)))


@pytest.mark.parametrize("task_id", ["task_037", "task_038", "task_039", "task_040", "task_041", "task_081"])
def test_plan_files_the_credit_group_largest_amount_first_as_gold_does(task_id):
    """Whatever order the customer gives, the credit group comes out largest amount first, as τ²'s train gold files it
    in these six tasks, so every transaction gets gold's flag (the given order alone does not: 0 of 12 recorded
    039-041 runs, scripts/plan_sim.py)."""
    task = _task(task_id)
    db, gold = _env(task).tools.db, _gold_disputes(task)
    want = {d["transaction_id"]: d["eligible_for_provisional_credit"] for d in gold}
    for given in (_items(gold), list(reversed(_items(gold))), sorted(_items(gold), key=lambda i: i["transaction_id"])):
        plan = banking.plan_disputes(db, gold[0]["user_id"], given)
        credit = [e for e in plan["order"] if e["can_get_credit"]]
        amounts = [banking._money(db.credit_card_transaction_history.data[e["transaction_id"]]["transaction_amount"])
                   for e in credit]
        assert amounts == sorted(amounts, reverse=True)
        assert [e["transaction_id"] for e in credit] == [d["transaction_id"] for d in gold][:len(credit)]
        assert {e["transaction_id"]: e["eligible_for_provisional_credit"] for e in plan["order"]} == want


def test_plan_lists_rejected_and_already_filed_disputes_and_asks_the_merchant_question():
    env, _, gold = _gated("task_039")
    user = gold[0]["user_id"]
    assert _ran(_call(env, "plan_disputes", other_requests=[], user_id=user, transactions=_items(gold[:1])))
    assert _ran(_file(env, gold[0]))
    asked = gold[1] | {"contacted_merchant": None}
    plan = banking.plan_disputes(env.tools.db, user, [
        _items(gold[:1])[0], _items([asked])[0], {"transaction_id": "txn_nope", "dispute_reason": "x"},
        {"transaction_id": gold[2]["transaction_id"], "dispute_reason": "bogus"}, _items(gold[3:4])[0],
        _items(gold[3:4])[0], "not a dict"])
    assert plan["already_filed"] == [gold[0]["transaction_id"]]
    assert [r["transaction_id"] for r in plan["rejected"]] == ["txn_nope", gold[2]["transaction_id"],
                                                               gold[3]["transaction_id"], ""]
    assert "listed twice" in plan["rejected"][2]["reason"]
    assert plan["disputes_on_record_past_12_months"] == 1 and plan["order"][0]["disputes_counted_before"] == 1
    assert plan["ask_user"] and "ask_user" in plan["order"][0] and not plan["ready"]


# ---- the plan gate ------------------------------------------------------------------------------------------------

def test_a_dispute_without_a_plan_is_denied_and_the_plan_is_the_enforcers_record():
    env, _, gold = _gated("task_040")
    assert _denied(_file(env, gold[0]), "procedure.dispute_plan.none")
    assert env.tools.glint_plans == {}
    plan = json.loads(_call(env, "plan_disputes", other_requests=[], user_id=gold[0]["user_id"], transactions=_items(gold)).content)
    assert env.tools.glint_plans[gold[0]["user_id"]] == plan
    assert _ran(_file(env, gold[0]))


def test_filings_out_of_the_plan_are_denied_and_the_plan_order_is_allowed():
    env, _, gold = _gated("task_039")
    user = gold[0]["user_id"]
    _call(env, "plan_disputes", other_requests=[], user_id=user, transactions=list(reversed(_items(gold))))
    plan = env.tools.glint_plans[user]["order"]
    by_id = {d["transaction_id"]: d for d in gold}
    first, second = by_id[plan[0]["transaction_id"]], by_id[plan[1]["transaction_id"]]
    first_flag = plan[0]["eligible_for_provisional_credit"] == "True"  # τ² shows tool-result bools as strings
    reply = _file(env, second | {"eligible_for_provisional_credit": True})
    assert _denied(reply, "procedure.dispute_plan.order") and f"File {first['transaction_id']} next" in reply.content
    # the dispute gate passes the next filing either way (the live database allows both flags for a first dispute);
    # the plan gate holds it to the plan's values
    for wrong in ({"dispute_reason": "incorrect_amount"}, {"contacted_merchant": not first["contacted_merchant"]}):
        reply = _file(env, first | {"eligible_for_provisional_credit": first_flag} | wrong)
        assert reply.error and "was NOT executed" in reply.content
    assert _denied(_file(env, {**first, "transaction_id": "txn_not_planned", "eligible_for_provisional_credit": True}),
                   "banking.dispute.record")
    for entry in plan:
        args = by_id[entry["transaction_id"]] | {
            "eligible_for_provisional_credit": entry["eligible_for_provisional_credit"] == "True"}
        assert _ran(_file(env, args)), entry
    flags = {r["transaction_id"]: r["eligible_for_provisional_credit"]
             for r in env.tools.db.transaction_disputes.data.values() if r.get("user_id") == user}
    assert "".join("T" if flags[e["transaction_id"]] else "F" for e in plan) == GOLD_FLAGS["task_039"]
    # a second filing of a transaction already on record: τ²'s tool rejects it, so the plan gate lets it through
    assert _plan_gate(env, first).rule_id == "procedure.dispute_plan.already_filed"


def test_a_flag_off_the_plan_and_a_transaction_off_the_plan_are_denied():
    env, _, gold = _gated("task_037")
    user = gold[0]["user_id"]
    _call(env, "plan_disputes", other_requests=[], user_id=user, transactions=_items(gold[:1]))
    assert _denied(_file(env, gold[1]), "procedure.dispute_plan.not_planned")
    flipped = gold[0] | {"eligible_for_provisional_credit": not gold[0]["eligible_for_provisional_credit"]}
    verdict = _plan_gate(env, flipped)
    assert verdict.rule_id == "procedure.dispute_plan.values" and any(f.startswith("eligible_for_provisional_credit") for f in verdict.facts)
    assert not verdict.allowed and _plan_gate(env, gold[0]).allowed


def test_a_plan_made_before_the_merchant_answer_holds_the_filing_until_replanned():
    env, _, gold = _gated("task_037")
    user, item = gold[0]["user_id"], _items(gold[:1])[0]
    _call(env, "plan_disputes", other_requests=[], user_id=user, transactions=[{k: v for k, v in item.items() if k != FIELDS[2]}])
    assert _denied(_file(env, gold[0]), "procedure.dispute_plan.ask")
    _call(env, "plan_disputes", other_requests=[], user_id=user, transactions=[item])  # calling it again replaces the plan
    assert _ran(_file(env, gold[0]))


def test_the_noticed_date_is_the_customers_answer_filed_as_planned():
    # task_054: the customer noticed the charge on 11/12/2025, two days before τ²'s clock (credit_cards_(general)_014
    # field 11: ask the customer)
    env, _, gold = _gated("task_054", digits=False)
    user, item = gold[0]["user_id"], _items(gold)[0]
    assert item["issue_noticed_date"] == "11/12/2025"
    unasked = json.loads(_call(env, "plan_disputes", other_requests=[], user_id=user, transactions=[
        {k: v for k, v in item.items() if k != "issue_noticed_date"}]).content)
    question = f"When did you first notice the {item['transaction_id']} charge? (Today is 11/14/2025.)"
    assert unasked["order"][0]["ask_user"] == question and unasked["ready"] == "False"  # τ² stringifies bools
    verdict = _plan_gate(env, gold[0])
    assert verdict.rule_id == "procedure.dispute_plan.ask" and "first noticed" in verdict.reason
    # an ISO answer is planned in the record's MM/DD/YYYY; a filing with today's date is off the plan
    plan = json.loads(_call(env, "plan_disputes", other_requests=[], user_id=user,
                            transactions=[item | {"issue_noticed_date": "2025-11-12"}]).content)
    assert plan["order"][0]["issue_noticed_date"] == "11/12/2025" and plan["ready"] == "True"
    verdict = _plan_gate(env, gold[0] | {"issue_noticed_date": "11/14/2025"})
    assert verdict.rule_id == "procedure.dispute_plan.values"
    assert "issue_noticed_date '11/14/2025' != the plan's '11/12/2025'" in verdict.facts
    assert _plan_gate(env, gold[0]).allowed
    bad = banking.plan_disputes(env.tools.db, user, [item | {"issue_noticed_date": "last week"}])
    assert bad["order"] == [] and "is not a date" in bad["rejected"][0]["reason"]
    # the v6 user simulator gave the real-world date for "today": a date after τ²'s today is refused, naming today
    future = banking.plan_disputes(env.tools.db, user, [item | {"issue_noticed_date": "10/30/2026"}])
    assert future["order"] == [] and "after today (11/14/2025" in future["rejected"][0]["reason"]
    assert banking.plan_disputes(env.tools.db, user, [item | {"issue_noticed_date": "11/14/2025"}])["order"]


def test_the_card_action_is_the_customers_answer_one_per_card():
    # task_039: the customer wants the card cancelled and reissued; gold files all 8 disputes with cancel_and_reissue
    # and orders the replacement after them (credit_cards_(general)_014 card_action)
    env, _, gold = _gated("task_039")
    user, items = gold[0]["user_id"], _items(gold)
    assert {i["card_action"] for i in items} == {"cancel_and_reissue"}
    unasked = json.loads(_call(env, "plan_disputes", other_requests=[], user_id=user, transactions=[
        {k: v for k, v in i.items() if k != "card_action"} for i in items]).content)
    card = unasked["card_actions"][0]["credit_card_account_id"]
    question = f"For the {card} card: keep the card active, or cancel it and get a new one?"
    assert unasked["ask_user"].count(question) == 1 and len(unasked["card_actions"]) == 1
    verdict = _plan_gate(env, gold[0])
    assert verdict.rule_id == "procedure.dispute_plan.ask" and "cancel it and get a new one" in verdict.reason
    # two answers for one card: those disputes are rejected, nothing is planned for the card
    mixed = banking.plan_disputes(env.tools.db, user, [items[0] | {"card_action": "keep_active"}] + items[1:])
    assert mixed["order"] == [] and len(mixed["rejected"]) == len(items)
    assert "one answer per card" in mixed["rejected"][0]["reason"]
    assert "not one of" in banking.plan_disputes(env.tools.db, user, [items[0] | {"card_action": "cancel"}])[
        "rejected"][0]["reason"]
    # cancel with no replacement ordered yet: the plan names the order as the next step; filings follow the plan
    plan = json.loads(_call(env, "plan_disputes", other_requests=[], user_id=user, transactions=items).content)
    assert plan["card_actions"][0]["card_action"] == "cancel_and_reissue"
    step = plan["card_actions"][0]["next_step"]
    # not "first": that word made all 4 v6 runs of 054 order the replacement before the credit limit increase
    assert step.startswith(f"Order the replacement with {banking.ORDER_TOOL} for {card}") and " first" not in step
    assert "credit limit increase, submit and decide it before ordering" in step
    verdict = _plan_gate(env, gold[0] | {"card_action": "keep_active"})
    assert verdict.rule_id == "procedure.dispute_plan.values"
    assert "card_action 'keep_active' != the plan's 'cancel_and_reissue'" in verdict.facts
    assert _ran(_file(env, gold[0]))  # gold's filing before the order: allowed


def test_a_replacement_ordered_after_the_plan_sets_the_card_action():
    # the customer first says keep the card, then asks for a new one: the plan's keep_active answer gives way to the
    # order, as banking.dispute requires, instead of two gates demanding opposite values
    task = _task("task_039")
    env, _, gold = _gated("task_039")
    user, items = gold[0]["user_id"], [i | {"card_action": "keep_active"} for i in _items(gold)]
    _call(env, "plan_disputes", other_requests=[], user_id=user, transactions=items)
    order = next(json.loads(a.arguments["arguments"]) for a in task.evaluation_criteria.actions
                 if a.name == WRAPPER and a.arguments.get("agent_tool_name") == banking.ORDER_TOOL)
    _call(env, "unlock_discoverable_agent_tool", agent_tool_name=banking.ORDER_TOOL)
    # 039 ships to the address on file: address_ask refuses the first order once
    assert "address_ask" in _call(env, WRAPPER, agent_tool_name=banking.ORDER_TOOL, arguments=json.dumps(order)).content
    assert _ran(_call(env, WRAPPER, agent_tool_name=banking.ORDER_TOOL, arguments=json.dumps(order)))
    verdict = _plan_gate(env, gold[0] | {"card_action": "keep_active"})
    assert verdict.rule_id == "procedure.dispute_plan.values" and "'cancel_and_reissue'" in verdict.fix
    assert _ran(_file(env, gold[0]))  # gold's cancel_and_reissue filing


def test_a_replacement_ordered_today_sets_the_card_action():
    # task_037: gold orders the replacement before the disputes; the order wins over a keep_active answer
    task = _task("task_037")
    env, _, gold = _gated("task_037")
    order = next(json.loads(a.arguments["arguments"]) for a in task.evaluation_criteria.actions
                 if a.name == WRAPPER and a.arguments.get("agent_tool_name") == banking.ORDER_TOOL)
    _call(env, "unlock_discoverable_agent_tool", agent_tool_name=banking.ORDER_TOOL)
    assert _ran(_call(env, WRAPPER, agent_tool_name=banking.ORDER_TOOL, arguments=json.dumps(order)))
    items = [i | {"card_action": "keep_active"} for i in _items(gold)]
    plan = banking.plan_disputes(env.tools.db, gold[0]["user_id"], items)
    entry = plan["card_actions"][0]
    assert entry["card_action"] == "cancel_and_reissue" and entry["replacement_ordered_today"]
    assert "next_step" not in entry and {e["card_action"] for e in plan["order"]} == {"cancel_and_reissue"}


def test_a_filing_needs_the_dispute_history_read_first():
    # credit_cards_(general)_015 criterion 4 counts the disputes on record; task_037's gold files without the read
    task = _task("task_037")
    env = _env(task, Enforcer("banking_knowledge", EventLog(":memory:")))
    gold = _gold_disputes(task)
    _verify(env, gold[0]["user_id"])
    _give_card_digits(env)
    _call(env, "plan_disputes", other_requests=[], user_id=gold[0]["user_id"], transactions=_items(gold))
    reply = _file(env, gold[0])
    assert _denied(reply, "procedure.lookups") and banking.DISPUTE_HISTORY_TOOL in reply.content
    _read_history(env, gold[0]["user_id"])
    assert _ran(_file(env, gold[0]))


def test_replies_carry_the_open_items_and_replay_them_unchanged():
    # task_039: the customer chose cancel_and_reissue and gold orders the replacement after the 8 disputes; until then
    # every reply that is not an error names what is still open (banking-plan-v5-gated 031 t1, 037 t3, 038 t3 planned
    # and filed nothing)
    task = _task("task_039")
    install("banking_knowledge", Enforcer("banking_knowledge", EventLog(":memory:")))
    constructor = registry.get_env_constructor("banking_knowledge")
    env = constructor(retrieval_variant="bm25_grep", task=task)
    init = task.initial_state
    env.set_state(init.initialization_data if init else None, init.initialization_actions if init else None, [])
    history, gold = [], _gold_disputes(task)
    user = gold[0]["user_id"]

    def send(name, /, **args):
        call = ToolCall(id=f"c{len(history)}", name=name, arguments=args, requestor="assistant")
        reply = env.get_response(call)
        history.extend([AssistantMessage(role="assistant", tool_calls=[call]), reply])
        return reply

    assert "[open]" not in send("get_current_time").content  # nothing planned yet
    u = env.tools.db.users.data[user]
    send("log_verification", time_verified="2025-11-14 03:40:00 EST",
         **{k: u[k] for k in ("name", "user_id", "address", "email", "phone_number", "date_of_birth")})
    send("give_discoverable_user_tool", discoverable_tool_name=banking.CARD_DIGITS_TOOL)
    send("unlock_discoverable_agent_tool", agent_tool_name=banking.DISPUTE_HISTORY_TOOL)
    send(WRAPPER, agent_tool_name=banking.DISPUTE_HISTORY_TOOL, arguments=json.dumps({"user_id": user}))
    plan = send("plan_disputes", other_requests=[], user_id=user, transactions=_items(gold))
    assert "[open]" not in plan.content  # the plan lists them itself
    card = json.loads(plan.content)["card_actions"][0]["credit_card_account_id"]
    read = send("get_current_time").content
    assert f"[open] disputes planned for {user} and not filed yet, in the plan's order: " + ", ".join(
        d["transaction_id"] for d in gold) in read
    assert f"[open] {card}: the customer chose to cancel it and get a new one; {banking.ORDER_TOOL}" in read
    send("unlock_discoverable_agent_tool", agent_tool_name=rules.DISPUTE)
    filed = send(WRAPPER, agent_tool_name=rules.DISPUTE, arguments=json.dumps(gold[0]))
    assert _ran(filed) and ", ".join(d["transaction_id"] for d in gold[1:]) + "\n" in filed.content
    denied = send(WRAPPER, agent_tool_name=rules.DISPUTE, arguments=json.dumps(gold[5]))
    assert _denied(denied, "procedure.dispute_plan.order") and "[open]" not in denied.content
    replay = constructor(retrieval_variant="bm25_grep", task=task)
    replay.set_state(init.initialization_data if init else None, init.initialization_actions if init else None,
                     history, strict=True)  # raises if a replayed reply, footer included, differs
    assert enforce._open_items(replay.tools) == enforce._open_items(env.tools)


def test_a_new_plan_replaces_the_old_one():
    env, _, gold = _gated("task_037")
    user = gold[0]["user_id"]
    _call(env, "plan_disputes", other_requests=[], user_id=user, transactions=_items(gold[:1]))
    assert _denied(_file(env, gold[1]), "procedure.dispute_plan.not_planned")
    _call(env, "plan_disputes", other_requests=[], user_id=user, transactions=_items(gold[1:2]))
    assert _ran(_file(env, gold[1]))


def test_a_failed_plan_call_records_nothing():
    env, _, gold = _gated("task_037")
    assert _call(env, "plan_disputes", other_requests=[], user_id=gold[0]["user_id"]).error  # transactions missing: τ² raises
    assert env.tools.glint_plans == {}


# ---- replay determinism ---------------------------------------------------------------------------------------------

def test_plan_disputes_is_replayed_by_tau2_and_never_changes_the_database():
    env, _, gold = _gated("task_037")
    assert env.tools.tool_mutates_state(banking.PLAN_TOOL) and not env.tools.has_tool("check_card_dispute")
    before = env.get_db_hash()
    _call(env, "plan_disputes", other_requests=[], user_id=gold[0]["user_id"], transactions=_items(gold))
    assert env.get_db_hash() == before


def test_plan_gate_decisions_replay_unchanged_in_taus_strict_replay():
    """τ²'s evaluator replays the mutating calls on a fresh environment from the same constructor, skipping READ
    tools, and compares each result (strict=True raises on a difference). The plan is rebuilt there from the replayed
    plan_disputes call, so the deny without a plan, the allowed filings and the out-of-order deny replay unchanged."""
    task = _task("task_039")
    install("banking_knowledge", Enforcer("banking_knowledge", EventLog(":memory:")))
    constructor = registry.get_env_constructor("banking_knowledge")
    env = constructor(retrieval_variant="bm25_grep", task=task)
    assert isinstance(env.tools, BankingDecisionTools)
    init = task.initial_state
    env.set_state(init.initialization_data if init else None, init.initialization_actions if init else None, [])
    history, gold = [], _gold_disputes(task)

    def send(name, /, **args):
        call = ToolCall(id=f"c{len(history)}", name=name, arguments=args, requestor="assistant")
        reply = env.get_response(call)
        history.extend([AssistantMessage(role="assistant", tool_calls=[call]), reply])
        return reply

    u = env.tools.db.users.data[gold[0]["user_id"]]
    send("log_verification", time_verified="2025-11-14 03:40:00 EST",
         **{k: u[k] for k in ("name", "user_id", "address", "email", "phone_number", "date_of_birth")})
    send("give_discoverable_user_tool", discoverable_tool_name=banking.CARD_DIGITS_TOOL)
    send("unlock_discoverable_agent_tool", agent_tool_name=banking.DISPUTE_HISTORY_TOOL)
    send(WRAPPER, agent_tool_name=banking.DISPUTE_HISTORY_TOOL, arguments=json.dumps({"user_id": gold[0]["user_id"]}))
    send("unlock_discoverable_agent_tool", agent_tool_name=rules.DISPUTE)
    assert _denied(send(WRAPPER, agent_tool_name=rules.DISPUTE, arguments=json.dumps(gold[0])),
                   "procedure.dispute_plan.none")
    send("plan_disputes", other_requests=[], user_id=gold[0]["user_id"], transactions=_items(gold))
    for d in gold[:3]:
        assert _ran(send(WRAPPER, agent_tool_name=rules.DISPUTE, arguments=json.dumps(d)))
    assert _denied(send(WRAPPER, agent_tool_name=rules.DISPUTE, arguments=json.dumps(gold[5])),
                   "procedure.dispute_plan.order")
    replay = constructor(retrieval_variant="bm25_grep", task=task)
    replay.set_state(init.initialization_data if init else None, init.initialization_actions if init else None,
                     history, strict=True)
    assert replay.tools.glint_plans == env.tools.glint_plans
    assert replay.get_db_hash() == env.get_db_hash()


def test_add_plan_tool_puts_the_tool_on_a_bare_banking_toolkit_once():
    env = _env(_task("task_037"))
    assert not env.tools.has_tool(banking.PLAN_TOOL)
    cls = type(add_plan_tool(env).tools)
    assert issubclass(cls, DisputePlanTools) and env.tools.has_tool(banking.PLAN_TOOL)
    assert type(add_plan_tool(env).tools) is cls


# ---- glint-bank ---------------------------------------------------------------------------------------------------

def test_glint_bank_always_publishes_plan_disputes_and_its_plan_gates_the_filing(tmp_path):
    from mcp.client.client import Client

    from glint.bank.server import BankServer

    server = BankServer(tmp_path, sid="plan")
    gold = _gold_disputes(_task("task_037"))

    async def body():
        async with Client(server.mcp_server()) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            await client.call_tool("load_scenario", {"task_id": "task_037"})
            env = server.session().env
            _verify(env, gold[0]["user_id"])
            await client.call_tool("give_discoverable_user_tool",
                                   {"discoverable_tool_name": banking.CARD_DIGITS_TOOL})
            await client.call_tool("unlock_discoverable_agent_tool",
                                   {"agent_tool_name": banking.DISPUTE_HISTORY_TOOL})
            await client.call_tool(WRAPPER, {"agent_tool_name": banking.DISPUTE_HISTORY_TOOL,
                                             "arguments": json.dumps({"user_id": gold[0]["user_id"]})})
            await client.call_tool("unlock_discoverable_agent_tool", {"agent_tool_name": rules.DISPUTE})
            unplanned = await client.call_tool(WRAPPER, {"agent_tool_name": rules.DISPUTE,
                                                         "arguments": json.dumps(gold[0])})
            plan = await client.call_tool(banking.PLAN_TOOL, {"user_id": gold[0]["user_id"],
                                                               "transactions": _items(gold), "other_requests": []})
            planned = await client.call_tool(WRAPPER, {"agent_tool_name": rules.DISPUTE,
                                                       "arguments": json.dumps(gold[0])})
            return names, unplanned, plan, planned

    names, unplanned, plan, planned = anyio.run(body)
    assert banking.PLAN_TOOL in names and "check_card_dispute" not in names
    assert unplanned.is_error and "procedure.dispute_plan.none" in unplanned.content[0].text
    assert not plan.is_error and json.loads(plan.content[0].text)["ready"]
    assert not planned.is_error
    server.session().events.close()


# ---- the replay helper --------------------------------------------------------------------------------------------

def test_with_plans_inserts_one_plan_per_user_before_the_first_filing():
    a = {"user_id": "u1", "transaction_id": "t1", "dispute_reason": "duplicate_charge", "contacted_merchant": True}
    b = {"user_id": "u1", "transaction_id": "t2", "dispute_reason": "incorrect_amount"}
    c = {"user_id": "u2", "transaction_id": "t3", "dispute_reason": "duplicate_charge"}
    calls = [ToolCall(id="0", name="log_verification", arguments={}, requestor="assistant"),
             ToolCall(id="1", name=WRAPPER, arguments={"agent_tool_name": rules.DISPUTE, "arguments": json.dumps(a)},
                      requestor="assistant"),
             ToolCall(id="2", name=rules.DISPUTE, arguments=a, requestor="assistant"),  # a retry, by name
             ToolCall(id="3", name=rules.DISPUTE, arguments=b, requestor="user"),  # not the agent's: ignored
             ToolCall(id="4", name=rules.DISPUTE, arguments=b, requestor="assistant"),
             ToolCall(id="5", name=rules.DISPUTE, arguments=c, requestor="assistant")]
    out = with_plans(calls)
    assert [c.id for c in out] == ["0", "1-history-unlock", "1-history", "1-plan", "1", "2", "3", "4", "5-plan", "5"]
    assert out[2].arguments == {"agent_tool_name": banking.DISPUTE_HISTORY_TOOL,
                                "arguments": json.dumps({"user_id": "u1"})}
    read = ToolCall(id="r", name=WRAPPER, requestor="assistant",
                    arguments={"agent_tool_name": banking.DISPUTE_HISTORY_TOOL, "arguments": "{}"})
    assert [c.id for c in with_plans([read] + calls)] == ["r", "0", "1-plan", "1", "2", "3", "4", "5-plan", "5"]
    out = out[2:]
    assert out[1].arguments == {"user_id": "u1", "other_requests": [], "transactions": [
        _items([a])[0], {"transaction_id": "t2", "dispute_reason": "incorrect_amount"}]}
    assert filing(calls[0]) is None and filing(calls[3]) is None
    assert plan_arguments([c])["u2"]["transactions"] == [{"transaction_id": "t3", "dispute_reason": "duplicate_charge"}]


def test_an_incorrect_amount_dispute_is_planned_as_a_partial_refund_of_the_overcharge():
    # task_039's Costco charge: $198.23 charged, $148.23 expected; gold files partial_refund 50 (tuned)
    env, _, gold = _gated("task_039")
    over = next(d for d in gold if d["dispute_reason"] == "incorrect_amount")
    user = over["user_id"]
    base = {k: over[k] for k in ("transaction_id", "dispute_reason", "contacted_merchant", "issue_noticed_date", "card_action")}
    asked = json.loads(_call(env, "plan_disputes", other_requests=[], user_id=user, transactions=[base]).content)
    entry = asked["order"][0]
    assert "resolution_requested" not in entry and "should you have been charged" in entry["ask_user"]
    assert str(asked["ready"]) == "False"
    for item in (base | {"expected_amount": 148.23}, base | {"partial_refund_amount": 50}):
        plan = json.loads(_call(env, "plan_disputes", other_requests=[], user_id=user, transactions=[item]).content)
        assert (plan["order"][0]["resolution_requested"], float(plan["order"][0]["partial_refund_amount"])) == (
            "partial_refund", 50.0)
    for bad in ({"expected_amount": 198.23}, {"partial_refund_amount": 250}):
        plan = json.loads(_call(env, "plan_disputes", other_requests=[], user_id=user, transactions=[base | bad]).content)
        assert plan["order"] == [] and "overcharge must be more than $0" in plan["rejected"][0]["reason"]


def test_the_plan_gate_holds_an_overcharge_filing_to_the_planned_partial_refund():
    env, _, gold = _gated("task_039")
    over = next(d for d in gold if d["dispute_reason"] == "incorrect_amount")
    user = over["user_id"]
    base = {k: over[k] for k in ("transaction_id", "dispute_reason", "contacted_merchant", "issue_noticed_date", "card_action")}
    _call(env, "plan_disputes", other_requests=[], user_id=user, transactions=[base])
    reply = _file(env, over)
    assert _denied(reply, "procedure.dispute_plan.ask") and "should have been charged" in reply.content
    _call(env, "plan_disputes", other_requests=[], user_id=user, transactions=[base | {"partial_refund_amount": 50}])
    full = over | {"resolution_requested": "full_refund", "partial_refund_amount": None}
    reply = _file(env, full)
    assert _denied(reply, "procedure.dispute_plan.values") and "resolution_requested 'full_refund'" in reply.content
    assert "partial_refund_amount 50.0" in reply.content
    assert _ran(_file(env, over))


def test_a_customer_who_asks_for_a_full_refund_gets_one():
    # the policy says to ask the customer (credit_cards_(general)_014); the partial default must not override them
    env, _, gold = _gated("task_039")
    over = next(d for d in gold if d["dispute_reason"] == "incorrect_amount")
    user = over["user_id"]
    base = {k: over[k] for k in ("transaction_id", "dispute_reason", "contacted_merchant", "issue_noticed_date", "card_action")}
    plan = json.loads(_call(env, "plan_disputes", other_requests=[], user_id=user,
                            transactions=[base | {"resolution_requested": "full_refund"}]).content)
    entry = plan["order"][0]
    assert entry["resolution_requested"] == "full_refund" and "partial_refund_amount" not in entry
    assert "ask_user" not in entry
    full = over | {"resolution_requested": "full_refund", "partial_refund_amount": None}
    partial = _file(env, over)  # against a full-refund plan, the partial filing is the one refused
    assert _denied(partial, "procedure.dispute_plan.values") and "the plan's 'full_refund'" in partial.content
    assert _ran(_file(env, full))


def test_every_overcharge_entry_tells_the_agent_not_to_suggest_a_full_refund():
    from glint.assist import banking

    env, _, gold = _gated("task_039")
    over = next(d for d in gold if d["dispute_reason"] == "incorrect_amount")
    plan = json.loads(_call(env, "plan_disputes", other_requests=[], user_id=over["user_id"], transactions=_items(gold)).content)
    notes = {e["transaction_id"]: e.get("note") for e in plan["order"]}
    assert notes[over["transaction_id"]] == banking.OVERCHARGE_NOTE and "do not offer or suggest" in notes[
        over["transaction_id"]]
    assert all(n is None for t, n in notes.items() if t != over["transaction_id"])
    assert "full refund" not in str(plan["ask_user"]).lower()  # the question for the customer stays neutral


def test_the_plan_says_to_look_up_the_dispute_history_before_filing():
    # Glimmer filed without get_user_dispute_history_7291 in 12 failed banking-plan-v4-gated runs (039-041, 081);
    # τ² logs that read in the database it scores when its gold makes it, so those runs failed on it alone (2026-10-29)
    env, _, gold = _gated("task_039")
    plan = json.loads(_call(env, "plan_disputes", other_requests=[], user_id=gold[0]["user_id"], transactions=_items(gold)).content)
    assert plan["before_filing"] == banking.BEFORE_FILING and banking.DISPUTE_HISTORY_TOOL in plan["before_filing"]


@pytest.mark.parametrize("task_id, scored", [("task_039", True), ("task_031", False)])
def test_the_dispute_history_read_is_scored_only_where_gold_makes_it(task_id, scored):
    # why the plan hints instead of a gate requiring the lookup: τ²'s read log allowlist logs the read in 039 (gold
    # makes it), not in 031 (gold files without it), so the hint cannot cost 031 its database match
    from tau2.runner.build import _derive_read_log_allowlist

    task = _task(task_id)
    env = registry.get_env_constructor("banking_knowledge")(retrieval_variant="bm25_grep", task=task,
                                                             read_log_allowlist=_derive_read_log_allowlist(task))
    init = task.initial_state
    env.set_state(init.initialization_data if init else None, init.initialization_actions if init else None, [])
    before = env.get_db_hash()
    user_id = _gold_disputes(task)[0]["user_id"]
    _call(env, "unlock_discoverable_agent_tool", agent_tool_name=banking.DISPUTE_HISTORY_TOOL)
    reply = _call(env, WRAPPER, agent_tool_name=banking.DISPUTE_HISTORY_TOOL, arguments=json.dumps({"user_id": user_id}))
    assert not str(reply.content).startswith("Error")
    assert (env.get_db_hash() != before) is scored


def test_a_dispute_needs_the_customer_given_the_digits_tool_and_no_block_names_a_digit():
    # Glimmer filed 'Gold Rewards Card', 'bgold' or '0000' when it never gave the customer get_card_last_4_digits
    # (banking-plan-v2-gated, 2026-10-29); gold gives it where the card has no digits on file. The gate does not know
    # the customer-side digits (only τ²'s code does), so once the tool is given the filing goes through.
    env, _, gold = _gated("task_039", digits=False)
    _call(env, "plan_disputes", other_requests=[], user_id=gold[0]["user_id"], transactions=_items(gold))
    first = env.tools.glint_plans[gold[0]["user_id"]]["order"][0]
    assert banking.CARD_DIGITS_TOOL in first["card_last_4_digits_from"] and "card_last_4_digits" not in first
    filing = next(d for d in gold if d["transaction_id"] == first["transaction_id"])
    flag = {"eligible_for_provisional_credit": first["eligible_for_provisional_credit"] == "True"}
    not_given = _file(env, filing | flag)
    assert _denied(not_given, "procedure.card_digits.give") and banking.CARD_DIGITS_TOOL in not_given.content
    assert filing["card_last_4_digits"] not in not_given.content  # gold's digits (1652) never appear in a block
    _give_card_digits(env)
    assert _ran(_file(env, filing | flag))


def test_a_card_with_digits_on_file_is_held_to_them_without_naming_them():
    # task_054's card account record carries card_last_4_digits 7823, which the agent's account lookups show
    env, _, gold = _gated("task_054", digits=False)
    filing = gold[0]
    _call(env, "plan_disputes", other_requests=[], user_id=filing["user_id"], transactions=_items(gold))
    flag = {"eligible_for_provisional_credit": env.tools.glint_plans[filing["user_id"]]["order"][0][
        "eligible_for_provisional_credit"] == "True"}
    wrong = _file(env, filing | flag | {"card_last_4_digits": "0791"})
    assert _denied(wrong, "procedure.card_digits.value") and "record shows" in wrong.content
    assert "7823" not in wrong.content
    assert _ran(_file(env, filing | flag))


def test_card_digits_must_be_4_numerals_even_after_the_tool_was_given():
    # Glimmer filed "XXXX" once the tool was given (banking-plan-v5-gated 041 t2, 2026-10-30)
    env, _, gold = _gated("task_039")
    _call(env, "plan_disputes", other_requests=[], user_id=gold[0]["user_id"], transactions=_items(gold))
    first = env.tools.glint_plans[gold[0]["user_id"]]["order"][0]
    filing = next(d for d in gold if d["transaction_id"] == first["transaction_id"])
    flag = {"eligible_for_provisional_credit": first["eligible_for_provisional_credit"] == "True"}
    for bad in ("XXXX", "165", "1652a", ""):
        assert _denied(_file(env, filing | flag | {"card_last_4_digits": bad}), "procedure.card_digits.format")
    assert _ran(_file(env, filing | flag))


def test_the_plan_lists_digits_on_file_and_the_digits_tool_is_not_given_for_that_card():
    # 054: the card's record shows 7823; Glimmer gave the customer the tool anyway in 2 of 4 v5 runs, and the extra
    # tool record broke the database match (2026-10-30)
    env, _, gold = _gated("task_054", digits=False)
    user = gold[0]["user_id"]
    _call(env, "plan_disputes", other_requests=[], user_id=user, transactions=_items(gold))
    entry = env.tools.glint_plans[user]["order"][0]
    assert entry["card_last_4_digits"] == "7823" and "card_last_4_digits_from" not in entry
    before = env.get_db_hash()
    reply = _call(env, "give_discoverable_user_tool", discoverable_tool_name=banking.CARD_DIGITS_TOOL)
    assert _denied(reply, "banking.give_card_digits.on_file") and "7823" not in reply.content
    assert env.get_db_hash() == before


def test_the_digits_tool_is_given_where_the_record_shows_no_digits_and_other_tools_pass():
    env, _, gold = _gated("task_039", digits=False)
    assert _ran(_call(env, "give_discoverable_user_tool", discoverable_tool_name=banking.CARD_DIGITS_TOOL))
    assert _ran(_call(env, "give_discoverable_user_tool", discoverable_tool_name="submit_cash_back_dispute_0589",
                      arguments=json.dumps({"user_id": gold[0]["user_id"], "transaction_id": "t"})))


def test_the_digits_tool_for_no_known_card_is_denied():
    # nobody verified, no plan, no card named: the gate cannot tell which card the tool is for (fail closed)
    env = _env(_task("task_039"), Enforcer("banking_knowledge", EventLog(":memory:")))
    reply = _call(env, "give_discoverable_user_tool", discoverable_tool_name=banking.CARD_DIGITS_TOOL)
    assert _denied(reply, "banking.give_card_digits.unknown_card")
    named = _call(env, "give_discoverable_user_tool", discoverable_tool_name=banking.CARD_DIGITS_TOOL,
                  arguments="not json")
    assert _denied(named, "banking.give_card_digits.unknown_card")
