"""Fix research 2026-11-02 rows 8, 14, 15 and appendix N2, N10: the address confirmation record, the one-credit fact
at the account reads, the search-loop guard (and its firing count), the duplicate-write deny and the identifier-
provenance deny. Banking TRAIN tasks only (never the test_untouched 20 of split-partition-banking.json)."""
import json
from types import SimpleNamespace

import pytest
import search_guard_count
from loguru import logger
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage
from tau2.data_model.tasks import Task
from tau2.registry import registry

import splits
from glint import paths
from glint.assist import banking
from glint.assist.tau2_tools import assist_environment
from glint.gates import Enforcer, EventLog, aids, rules
from glint.gates.replayplan import LOOKUP_SUFFIX, UNLOCK, with_address_confirmations, with_lookups, with_plans

logger.remove()
WRAPPER = "call_discoverable_agent_tool"
TASKS = {t["id"]: t for t in json.loads((paths.tau2_data() / "tau2" / "domains" / "banking_knowledge" /
                                         "tasks.json").read_text(encoding="utf-8"))}
TEST_SPLIT = set(json.loads(splits.BANKING.read_text(encoding="utf-8"))["test_untouched"])


def _bank(task_id: str, enforcer: Enforcer | None = None, history=None):
    assert task_id not in TEST_SPLIT
    task = Task.model_validate(TASKS[task_id])
    env = registry.get_env_constructor("banking_knowledge")(retrieval_variant="bm25_grep", task=task)
    events = EventLog(":memory:")
    (enforcer or Enforcer("banking_knowledge", events)).attach(env)
    init = task.initial_state
    env.set_state(init.initialization_data if init else None, init.initialization_actions if init else None,
                  history or [], strict=True)
    return env, task, events


def _name(call) -> str:
    return (call.arguments or {}).get("agent_tool_name") if call.name == WRAPPER else call.name


def _play(env, task, tool: str, send=None) -> ToolCall:
    """Send gold (with the replays' inserted plans, address records and lookups) up to the first call of `tool`;
    return that call unsent."""
    send = send or env.get_response
    gold = [ToolCall(id=f"g{k}", name=a.name, arguments=a.arguments, requestor=a.requestor)
            for k, a in enumerate(task.evaluation_criteria.actions)]
    for call in with_lookups(with_address_confirmations(with_plans(gold)), env.tools.db):
        if _name(call) == tool and call.name != UNLOCK:
            return call
        assert "was NOT executed" not in str(send(call).content), call
    raise AssertionError(f"{tool} not in {task.id}'s gold")


def _hidden(tool: str, cid: str = "x", **args) -> ToolCall:
    return ToolCall(id=cid, name=WRAPPER, requestor="assistant",
                    arguments={"agent_tool_name": tool, "arguments": json.dumps(args)})


def _rule(events) -> str:
    return [e for e in events.events("gate.fired") if e["payload"]["decision"] == "deny"][-1]["payload"]["rule_id"]


# ---- row 8: address confirmation record ---------------------------------------------------------------------------

def test_the_address_record_validates_the_account_and_the_answer():
    env, _, _ = _bank("task_054")
    db = env.tools.db
    card = next(k for k, a in db.credit_card_accounts.data.items())
    owner = db.credit_card_accounts.data[card]["user_id"]
    assert not banking.record_address_confirmation(db, card, "nobody", "1 Main St", "yes")["recorded"]
    assert "address" in banking.record_address_confirmation(db, card, owner, " ", "yes")["error"]
    home = db.users.data[owner]["address"]
    ok = banking.record_address_confirmation(db, card, owner, home.upper(), "yes, home")
    assert ok["recorded"] and ok["address_on_file"] and banking.ORDER_TOOL in ok["next"]


def test_a_recorded_confirmation_satisfies_the_address_ask_with_no_refused_order():
    env, task, events = _bank("task_054")
    order = _play(env, task, rules.REPLACE)  # the replay plan recorded the confirmation before the order
    card = json.loads(order.arguments["arguments"])["credit_card_account_id"]
    assert card in env.tools.glint_address_confirmed and not env.tools.glint_address_asked
    reply = env.get_response(order)
    assert "was NOT executed" not in reply.content
    assert any(e["payload"]["credit_card_account_id"] == card for e in events.events("address.recorded"))


def test_an_order_to_another_address_than_the_confirmed_one_is_refused():
    env, task, events = _bank("task_054")
    order = _play(env, task, rules.REPLACE)
    args = json.loads(order.arguments["arguments"]) | {"shipping_address": "9 Elsewhere Rd, Nowhere"}
    reply = env.get_response(_hidden(rules.REPLACE, **args))
    assert reply.error and _rule(events) == rules.ADDRESS_MISMATCH and banking.ADDRESS_TOOL in reply.content


def test_the_address_ask_names_the_record_tool():
    env, task, events = _bank("task_054")
    order = _play(env, task, rules.REPLACE)
    env.tools.glint_address_confirmed.clear()  # as if no confirmation was recorded
    reply = env.get_response(order)
    assert reply.error and _rule(events) == rules.ADDRESS_ASK and f"{banking.ADDRESS_TOOL}(" in reply.content


def test_the_address_record_is_ignored_when_it_failed_or_is_not_json():
    env, _, _ = _bank("task_054")
    enforcer = env.glint_enforcer
    call = ToolCall(id="a", name=banking.ADDRESS_TOOL, requestor="assistant", arguments={})
    for content in ("not json", json.dumps({"recorded": False})):
        enforcer._record_address(env, call, ToolMessage(id="a", role="tool", content=content), False)
    assert env.tools.glint_address_confirmed == {}


def test_replays_insert_one_confirmation_before_each_cards_first_order():
    task = Task.model_validate(TASKS["task_054"])
    gold = [ToolCall(id=f"gold-{k}", name=a.name, arguments=a.arguments, requestor=a.requestor)
            for k, a in enumerate(task.evaluation_criteria.actions)]
    calls = with_address_confirmations(gold + gold[-3:])  # a second order of the same card gets none
    inserted = [k for k, c in enumerate(calls) if c.name == banking.ADDRESS_TOOL]
    assert len(inserted) == 1 and _name(calls[inserted[0] + 1]) == rules.REPLACE
    order = json.loads(calls[inserted[0] + 1].arguments["arguments"])
    assert calls[inserted[0]].arguments["shipping_address"] == order["shipping_address"]


# ---- row 14: one credit per checking account, at the account reads -----------------------------------------------

def test_the_account_read_shows_the_one_credit_fact_and_which_accounts_were_credited_today():
    env, task, _ = _bank("task_072")
    credit = _play(env, task, aids.CREDIT_TOOL)
    user = json.loads(credit.arguments["arguments"])
    account = user["account_id"]
    owner = env.tools.db.accounts.data[account]["user_id"]
    read = env.get_response(_hidden(aids.ACCOUNTS_READ, "r1", user_id=owner))
    assert "[policy] bank_accounts_(general)_017" in read.content and "Credited today: none" in read.content
    assert "was NOT executed" not in env.get_response(credit).content
    again = env.get_response(_hidden(aids.TRANSACTIONS_READ, "r2", account_id=account))
    assert f"Credited today: {account}" in again.content


def test_the_credit_note_is_only_for_account_reads_of_a_customer_with_checking():
    env, _, _ = _bank("task_072")
    db = env.tools.db
    assert aids.credit_note(db, "get_user_dispute_history_7291", {"user_id": "x"}) is None
    assert aids.credit_note(db, aids.ACCOUNTS_READ, {"user_id": "no-such-user"}) is None
    assert aids.credit_note(db, aids.TRANSACTIONS_READ, {"account_id": "no-such-account"}) is None


# ---- row 15: the search-loop guard ---------------------------------------------------------------------------------

def test_the_search_guard_adds_the_open_items_every_n_calls_with_no_write(monkeypatch):
    from glint.assist import tau2_tools

    monkeypatch.setattr(tau2_tools, "SEARCH_GUARD", 3)
    tools = assist_environment(registry.get_env_constructor("banking_knowledge")(retrieval_variant="bm25_grep"),
                               "banking_knowledge").tools
    assert "[guard]" not in tools.use_tool("KB_search", query="stolen card replacement")
    assert "[guard]" not in tools.use_tool("grep", pattern="fraud")
    third = tools.use_tool("KB_read", doc_id="doc_credit_cards_credit_cards_(general)_018")
    assert "[guard] 3 knowledge-base calls" in third and "No open items are recorded." in third
    tools.glint_verdicts = {"cc_1": {"verdict": "deny", "new_credit_limit": None, "denial_reason": "x"}}
    for _ in range(2):
        tools.use_tool("KB_search", query="credit limit increase")
    sixth = tools.use_tool("KB_search", query="provisional credit criteria")
    assert "[guard] 6 knowledge-base calls" in sixth and "Open items:" in sixth and "cc_1" in sixth
    user = next(iter(tools.db.users.data))
    tools.use_tool("change_user_email", user_id=user, new_email="new.address@example.com")  # a write ran
    assert tools.__dict__["_glint_since_write"] == 0
    assert tools.use_tool("change_user_email", user_id="nobody", new_email="x@example.com").startswith("Error")
    tools.use_tool("KB_search", query="card last 4 digits")
    assert tools.__dict__["_glint_since_write"] == 1  # a failed write does not reset the count


def test_the_firing_count_follows_the_guard_rule():
    writes = {"change_user_email", "freeze_debit_card_3892"}

    def kb(i):
        return {"role": "assistant", "tool_calls": [{"id": f"k{i}", "name": "KB_search", "arguments": {"query": "q"}}]}

    def write(i, tool, error=False):
        return [{"role": "assistant", "tool_calls": [{"id": f"w{i}", "name": WRAPPER, "arguments": {
                    "agent_tool_name": tool, "arguments": "{}"}}]},
                {"role": "tool", "id": f"w{i}", "content": "Error: no" if error else "done", "error": False}]

    msgs = [kb(i) for i in range(4)] + write(1, "freeze_debit_card_3892") + [kb(i) for i in range(4, 7)]
    assert search_guard_count.firings(msgs, writes, 2) == 3  # calls 2, 4, then 2 after the write
    failed = [kb(i) for i in range(2)] + write(2, "freeze_debit_card_3892", error=True) + [kb(9), kb(10)]
    assert search_guard_count.firings(failed, writes, 4) == 1
    assert "change_user_email" in search_guard_count.write_tools()
    assert WRAPPER not in search_guard_count.write_tools()


def test_the_firing_count_reads_arm_files_and_skips_the_test_split(tmp_path, capsys):
    test_task = min(TEST_SPLIT)
    msgs = [{"role": "assistant", "tool_calls": [{"id": f"k{i}", "name": "grep", "arguments": {}}]} for i in range(10)]
    sims = [{"task_id": t, "messages": msgs, "reward_info": {"reward": r}, "termination_reason": "user_stop"}
            for t, r in (("task_072", 1.0), ("task_037", 0.0), (test_task, 0.0))]
    folder = tmp_path / "set"
    folder.mkdir()
    (folder / "tools-banking_knowledge.json").write_text(json.dumps({"simulations": sims}), encoding="utf-8")
    rows = search_guard_count.count([folder], [5], {"x"})
    assert rows[("set:tools", "untuned", 5)]["runs"] == 1 and rows[("set:tools", "untuned", 5)]["pass_fired"] == 1
    assert rows[("set:tools", "tuned", 5)]["fail_fired"] == 1 and rows[("set:tools", "tuned", 5)]["firings"] == 2
    assert search_guard_count.main([str(folder), "--n", "5"]) == 0
    assert "set:tools | tuned | 5 | 1 | 1 (0 / 1) | 2" in capsys.readouterr().out
    assert search_guard_count.main([]) == 2


# ---- N2: duplicate writes -----------------------------------------------------------------------------------------

def test_an_identical_write_that_already_ran_is_refused_and_a_different_one_is_not():
    env, task, events = _bank("task_072")
    credit = _play(env, task, aids.CREDIT_TOOL)
    assert "was NOT executed" not in env.get_response(credit).content
    again = env.get_response(credit.model_copy(update={"id": "again"}))
    # the checking-credit rule refuses it first (once per account); the duplicate deny stands behind it
    assert again.error and _rule(events) == "banking.checking_credit.once"
    gate = aids.duplicate(aids.CREDIT_TOOL)
    args = json.loads(credit.arguments["arguments"])
    assert not gate(env.tools, args).allowed and gate(env.tools, args | {"amount": 1.0}).allowed


def test_the_duplicate_deny_is_first_where_no_policy_rule_refuses():
    env, task, events = _bank("task_072")
    _play(env, task, aids.ACCOUNTS_READ)  # the customer is looked up and verified
    user = next(a.arguments["user_id"] for a in task.evaluation_criteria.actions if a.name == "log_verification")
    env.get_response(_hidden(aids.ACCOUNTS_READ, "r", user_id=user))  # shows the account ids
    accounts = sorted(k for k, a in env.tools.db.accounts.data.items() if a.get("user_id") == user)
    env.get_response(ToolCall(id="u", name=UNLOCK, requestor="assistant",
                              arguments={"agent_tool_name": "transfer_funds_between_bank_accounts_7291"}))
    move = _hidden("transfer_funds_between_bank_accounts_7291", "m1", source_account_id=accounts[0],
                   destination_account_id=accounts[1], amount=1.0)
    first = env.get_response(move)
    assert not first.error and not first.content.startswith("Error"), first.content
    again = env.get_response(move.model_copy(update={"id": "m2"}))
    assert again.error and _rule(events) == "glint.duplicate_write" and "already ran" in again.content


def test_transfers_and_the_increase_submit_have_no_duplicate_gate():
    registry_ = rules.build_registry("banking_knowledge")
    names = {g.name for g in registry_.gates()}
    for tool in aids.DUPLICATE_EXEMPT:
        assert f"procedure_duplicate_{tool}" not in names and f"procedure_provenance_{tool}" in names
    assert "procedure_duplicate_change_user_email" in names
    # an allow by these gates is not a rule check
    from glint.gates.enforce import attestation_classes
    assert attestation_classes(["procedure_duplicate_x", "procedure_provenance_x"]) == ["uncovered"]


# ---- N10: identifier provenance -----------------------------------------------------------------------------------

def test_identifiers_are_the_id_arguments_only():
    assert aids.identifiers({"user_id": "u1", "card_last_4_digits": "1234", "account_id": " ", "amount": 3.0,
                             "transaction_id": 5.0, "card_id": "dbc_1"}) == {"card_id": "dbc_1", "user_id": "u1"}


def test_a_write_naming_an_identifier_no_lookup_showed_is_refused_until_it_is_looked_up():
    env, task, events = _bank("task_072")
    verify = next(a for a in task.evaluation_criteria.actions if a.name == "log_verification")
    call = ToolCall(id="v", name="log_verification", arguments=verify.arguments, requestor="assistant")
    reply = env.get_response(call)
    assert reply.error and _rule(events) == "glint.identifier_provenance" and "user_id" in reply.content
    user = env.tools.db.users.data[verify.arguments["user_id"]]
    env.get_response(ToolCall(id="l", name="get_user_information_by_name", requestor="assistant",
                              arguments={"customer_name": user["name"]}))
    assert "was NOT executed" not in env.get_response(call.model_copy(update={"id": "v2"})).content


def test_provenance_allows_when_no_record_is_kept():
    assert aids.provenance("x")(SimpleNamespace(), {"user_id": "u"}).allowed
    assert not aids.provenance("x")(SimpleNamespace(glint_seen=["other"]), {"user_id": "u"}).allowed


def test_the_seen_record_takes_only_assistant_results_that_are_not_errors():
    msgs = [ToolMessage(id="a", role="tool", content="u1 shown"),
            ToolMessage(id="b", role="tool", content="Error: no", error=False),
            ToolMessage(id="c", role="tool", content="u2", requestor="user"),
            ToolMessage(id="d", role="tool", content="x", error=True),
            AssistantMessage(role="assistant", content="hi"),
            ToolMessage(id="e", role="tool", content="u3")]
    before, texts = aids.seen_before(msgs)
    assert texts == ("u1 shown", "u3") and before["a"] == () and before["e"] == ("u1 shown",)


def test_provenance_and_duplicate_decisions_replay_unchanged_in_taus_strict_replay():
    """READ calls are skipped in τ²'s replay; each write is judged on the results before it in the history, so a
    refusal and the allowed retry replay with the same content (strict=True raises on any difference)."""
    enforcer = Enforcer("banking_knowledge", EventLog(":memory:"))
    env, task, _ = _bank("task_072", enforcer)
    history = []

    def send(call: ToolCall):
        reply = env.get_response(call)
        history.extend([AssistantMessage(role="assistant", tool_calls=[call]), reply])
        return reply

    verify = next(a for a in task.evaluation_criteria.actions if a.name == "log_verification")
    call = ToolCall(id="v0", name="log_verification", arguments=verify.arguments, requestor="assistant")
    assert send(call).error  # nothing looked up yet
    credit = _play(env, task, aids.CREDIT_TOOL, send=send)  # gold with its lookups: the verification now runs
    assert "was NOT executed" not in send(credit).content
    replay, _, _ = _bank("task_072", enforcer, history)
    assert replay.tools.glint_ran == env.tools.glint_ran != set()
    assert replay.tools.glint_seen == env.tools.glint_seen


def test_replays_insert_the_lookup_that_shows_each_identifier_once():
    env, task, _ = _bank("task_072")
    gold = [ToolCall(id=f"gold-{k}", name=a.name, arguments=a.arguments, requestor=a.requestor)
            for k, a in enumerate(task.evaluation_criteria.actions)]
    calls = with_lookups(gold, env.tools.db)
    inserted = [c for c in calls if LOOKUP_SUFFIX in c.id]
    assert inserted[0].name == "get_user_information_by_name"
    unlocks = [c.arguments["agent_tool_name"] for c in inserted if c.name == UNLOCK]
    assert len(unlocks) == len(set(unlocks))
    assert with_lookups([ToolCall(id="n", name="x", arguments={"card_id": "no-such"}, requestor="assistant")],
                        env.tools.db)[0].id == "n"


@pytest.mark.parametrize("table", ["credit_card_accounts", "credit_card_transaction_history", "debit_cards",
                                   "bank_account_transaction_history"])
def test_every_table_has_a_lookup(table):
    from glint.gates.replayplan import _lookup
    env, _, _ = _bank("task_072")
    value = next(iter(getattr(env.tools.db, table).data))
    tool, args, _ = _lookup(env.tools.db, value)
    assert tool and all(args.values())
