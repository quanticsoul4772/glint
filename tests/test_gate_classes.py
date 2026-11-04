"""What each gate rests on (glint.gates.core.CHECKS, docs/gate-classes.md): the class on every registered gate, its
counts in the registry and the gates.manifest event, refusals split by class (summary.denies_by_checks,
scripts/gate_classes.py), and the limit of the agent-made class: a record the agent makes up unlocks the write."""
import json

import gate_classes
import pytest
from tau2.data_model.message import ToolCall
from test_gates_phase2 import _bank, _call, _denied, _hidden, _play, _ran, _verify

from glint.gates import EventLog, Gate, GateRegistry, allow, summary
from glint.gates import procedure as proc
from glint.gates.core import ACCOUNT_RECORDS, AGENT_RECORDS, CHECKS
from glint.gates.replayplan import lookups
from glint.gates.rules import REQUEST_TOOLS, build_registry, manifest

# the banking gates a record_* call, the customer's answers in a plan_disputes call, a log_verification call made from
# the agent's own lookup or a repeated call can pass
AGENT_GATES = ({f"banking_request_{t}" for t in REQUEST_TOOLS}
               | {"banking_retention_answer_gate", "banking_replacement_gate", "procedure_dispute_plan_gate",
                  "banking_log_verification_gate", "procedure_verified_gate"})


def test_every_gate_is_classed_and_the_agent_made_class_is_exactly_the_record_gates():
    reg = build_registry("banking_knowledge")
    assert {g.checks for g in reg.gates()} <= set(CHECKS)
    assert {g.name for g in reg.gates() if g.checks == AGENT_RECORDS} == AGENT_GATES
    counts = reg.checks_counts()
    assert list(counts) == list(CHECKS) and sum(counts.values()) == len(reg.gates())
    assert counts[AGENT_RECORDS] == len(AGENT_GATES)
    # with the request gate off its gates are gone from the class too
    off = build_registry("banking_knowledge", request_gate=False).checks_counts()
    assert off[AGENT_RECORDS] == len(AGENT_GATES) - len(REQUEST_TOOLS)
    for domain in ("retail", "airline"):
        assert build_registry(domain).checks_counts()[AGENT_RECORDS] == 0


def test_the_manifest_carries_each_gates_class_and_the_counts():
    reg = build_registry("banking_knowledge")
    m = manifest("banking_knowledge", reg)
    assert m["checks_counts"] == reg.checks_counts()
    assert {g["name"]: g["checks"] for g in m["gates"]} == {g.name: g.checks for g in reg.gates()}


def test_the_registry_refuses_an_unknown_class_and_a_gate_with_no_class_cannot_be_registered():
    with pytest.raises(ValueError, match="checks 'vibes'"):
        GateRegistry().register(Gate("g", ("t",), "a purpose", lambda db, a: allow("r", "s"), checks="vibes"))
    # no default: an unlabeled gate used to count as account records (technical judge, round 2, 2.1)
    with pytest.raises(TypeError, match="checks"):
        GateRegistry().register(Gate("g", ("t",), "a purpose", lambda db, a: allow("r", "s")))
    assert GateRegistry().register(Gate("g", ("t",), "a purpose", lambda db, a: allow("r", "s"),
                                        checks=ACCOUNT_RECORDS)).checks == ACCOUNT_RECORDS


def test_the_gate_kinds_split_policy_rules_from_generated_checks_and_no_rule_gates():
    reg = build_registry("banking_knowledge")
    kinds = {g.name: gate_classes.gate_kind(g.name) for g in reg.gates()}
    assert kinds["procedure_duplicate_close_debit_card_4721"] == (gate_classes.GENERATED, "duplicate")
    assert kinds["procedure_provenance_close_debit_card_4721"] == (gate_classes.GENERATED, "provenance")
    assert kinds["no_rule_transfer_funds_between_bank_accounts_7291"] == (gate_classes.NO_RULE, "")
    assert kinds["banking_cli_submit_gate"] == kinds["procedure_verified_gate"] == (gate_classes.POLICY, "")
    # every agent-made gate is a policy rule; the generated and no-rule gates all rest on account records
    assert all(kinds[g.name][0] == gate_classes.POLICY for g in reg.gates() if g.checks == AGENT_RECORDS)
    n = {k: sum(v[0] == k for v in kinds.values()) for k in gate_classes.KINDS}
    assert sum(n.values()) == len(reg.gates()) and n[gate_classes.NO_RULE] == 9
    line = gate_classes.kinds_line()
    assert line.startswith(f"KINDS banking_knowledge: policy rule {n[gate_classes.POLICY]} (account_records ")
    assert line.endswith(f"no rule {n[gate_classes.NO_RULE]}")


def test_a_verification_with_no_customer_turn_passes_and_unlocks_writes():
    """The documented limit of the verification gates (docs/gate-classes.md): the conversation holds no customer
    message at all. The agent looks the user up by name and copies the record into log_verification; the
    verification gate allows it and the verified gate then allows that customer's writes. Neither gate can tell
    whether the customer supplied those details, so both are agent-made."""
    env, _, events = _bank("task_084")
    user = "a7c2f85d91"
    write = {"user_id": user}
    assert not proc.verified(env.tools.db, write).allowed
    _verify(env, user)  # one lookup and one log_verification call; no customer turn
    fired = [e["payload"]["decision"] for e in events.events("gate.fired")
             if e["payload"]["gate"] == "banking_log_verification_gate"]
    assert fired == ["allow"]
    assert proc.verified(env.tools.db, write).allowed


def _log(path):
    log = EventLog(path)
    ctx = {"task_id": "task_x", "trial": 0}
    log.emit("gates.manifest", {"gates": []})
    for gate, rule, decision, replay in (("banking_request_close_debit_card_4721", "banking.request.no_record", "deny",
                                          False),
                                         ("banking_request_close_debit_card_4721", "banking.request.no_record", "deny",
                                          True),  # replay: skipped
                                         ("banking_cli_submit_gate", "banking.cli.submit", "deny", False),
                                         ("banking_cli_submit_gate", "banking.cli.submit", "allow", False),
                                         ("unparsed", "unparsed_arguments", "deny", False)):
        log.emit("gate.fired", {"gate": gate, "rule_id": rule, "decision": decision, "replay": replay, **ctx})
    log.close()


def test_refusals_are_split_by_the_class_of_the_gate_that_refused(tmp_path):
    _log(tmp_path / "g.sqlite")
    checks = {g.name: g.checks for g in build_registry("banking_knowledge").gates()}
    by = summary.denies_by_checks(summary.read_log(tmp_path / "g.sqlite"), checks)
    assert by == {AGENT_RECORDS: {("banking_request_close_debit_card_4721", "banking.request.no_record"): 1},
                  ACCOUNT_RECORDS: {("banking_cli_submit_gate", "banking.cli.submit"): 1},
                  summary.OTHER_CHECKS: {("unparsed", "unparsed_arguments"): 1}}


def test_the_gate_classes_script_prints_the_registry_and_each_logs_split(tmp_path, capsys):
    _log(tmp_path / "a.sqlite")
    _log(tmp_path / "b.sqlite")
    assert gate_classes.main([str(tmp_path / "a.sqlite"), str(tmp_path / "b.sqlite")]) == 0
    out = capsys.readouterr().out.splitlines()
    counts = build_registry("banking_knowledge").checks_counts()
    assert out[0] == (f"GATES banking_knowledge: {sum(counts.values())} registered, account_records "
                      f"{counts[ACCOUNT_RECORDS]}, agent_records {counts[AGENT_RECORDS]}")
    assert out[1] == gate_classes.kinds_line()
    assert out[2].endswith(": 3 live, account_records 1, agent_records 1, other 1")
    assert out[3].endswith(": policy rule 2, generated check 0, no rule 0, other 1")
    assert "banking_request_close_debit_card_4721  banking.request.no_record" in out[4]
    assert out[-2] == "REFUSALS total: 6 live, account_records 2, agent_records 2, other 2"
    assert out[-1] == "BY KIND total: policy rule 4, generated check 0, no rule 0, other 2"


def _send_lookups(env, tool: str, args: dict) -> None:
    """The reads the procedure gates require before `tool` (glint.gates.replayplan.lookups), as a careful agent makes
    them: real reads, which the account-record gates check ran."""
    write = ToolCall(id="c", name="call_discoverable_agent_tool", requestor="assistant",
                     arguments={"agent_tool_name": tool, "arguments": json.dumps(args)})
    for read in lookups([write], env.tools.db):
        env.get_response(read)


def _allowed_by(events, tool: str) -> set[str]:
    return {e["payload"]["gate"] for e in events.events("gate.fired")[-40:]
            if e["payload"]["tool"] == tool and e["payload"]["decision"] == "allow"}


def test_a_made_up_retention_answer_closes_a_card_the_customer_chose_to_keep():
    """The documented limit (docs/gate-classes.md). In task_045's gold the customer asks to close, accepts the retention
    credit and keeps the card. One record_retention_answer call with a decline the customer never gave passes the
    retention gate, and the card closes: the gate never sees the conversation."""
    env, task, events = _bank("task_045")
    credit = _play(env, task, "apply_statement_credit_8472")
    card, user = credit["credit_card_account_id"], credit["user_id"]
    close = {"credit_card_account_id": card, "user_id": user}
    _send_lookups(env, "close_credit_card_account_7834", close)
    assert _denied(_hidden(env, "close_credit_card_account_7834", **close), "record_retention_answer")
    _call(env, "record_retention_answer", **close, offer="$50 statement credit", accepted=False,
          customer_words="made up: no thanks, close it")
    assert _ran(_hidden(env, "close_credit_card_account_7834", **close))
    assert str(env.tools.db.credit_card_accounts.data[card]["status"]).upper() == "CLOSED"
    assert "banking_retention_answer_gate" in _allowed_by(events, "close_credit_card_account_7834")
    ran = events.events("write.ran")[-1]["payload"]
    assert (ran["tool"], ran["status"], ran["db_hash"]) == ("close_credit_card_account_7834", "ok", env.get_db_hash())


def test_a_made_up_request_record_closes_a_card_nobody_asked_to_close():
    """The same limit for the request gate (the judge's example): task_084's customer never names this debit card. A
    record_requests call with words the customer never said unlocks its close."""
    env, _, events = _bank("task_084")
    _verify(env, "a7c2f85d91")
    named = "dbc_82b5d1a6f3"  # the card the customer's request is about
    card = next(k for k, c in env.tools.db.debit_cards.data.items() if c.get("user_id") == "a7c2f85d91" and k != named)
    close = {"card_id": card, "reason": "fraud_suspected"}
    _send_lookups(env, "close_debit_card_4721", close)
    _hidden(env, "get_bank_account_transactions_9173", account_id=env.tools.db.debit_cards.data[card]["account_id"])
    assert _denied(_hidden(env, "close_debit_card_4721", **close), "record_requests")
    _call(env, "record_requests", user_id="a7c2f85d91", requests=[
        {"action": "close_debit_card", "target_id": card, "customer_words": "made up: close that card"}])
    before = env.get_db_hash()
    reply = _hidden(env, "close_debit_card_4721", **close)
    assert _ran(reply), reply.content
    assert env.get_db_hash() != before and events.events("write.ran")[-1]["payload"]["db_hash"] == env.get_db_hash()
    assert "banking_request_close_debit_card_4721" in _allowed_by(events, "close_debit_card_4721")
