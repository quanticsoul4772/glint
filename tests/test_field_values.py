"""The field-value mechanisms from fix-research 2026-11-02's appendix that failed the Nov 4 gate or were deferred:
the checking-credit amount rule (glint.gates.fees), the 48-hour entry-wait exemption as a flagged allow
(glint.gates.debit), constrained values (glint.gates.constraints), conversation-text checks (glint.convcheck), LLM
stages with fakes (glint.verifiers) and the conversation audit script. No test calls a model or a paid
service."""
import json
from types import SimpleNamespace

import pytest
from test_gates_phase2 import BANKING, TEST_SPLIT, _bank, _hidden, _play, _ran

from glint import convcheck, paths, verifiers
from glint.gates import Enforcer, EventLog, GateRegistry, constraints, debit, fees, rules

DB = json.loads((paths.tau2_data() / "tau2" / "domains" / "banking_knowledge" / "db.json").read_text(encoding="utf-8"))
CREDIT = "apply_checking_account_credit_5829"


def _db(**tables):
    """A stand-in database: the vendored db.json, with tables replaced as given."""
    data = {k: v["data"] for k, v in DB.items()} | tables
    return SimpleNamespace(**{k: SimpleNamespace(data=v) for k, v in data.items()})


def _gold(tool: str):
    """(task id, arguments) of every training-gold call of `tool` (the test split is never read)."""
    for tid, t in BANKING.items():
        if tid in TEST_SPLIT:
            continue
        for a in (t.get("evaluation_criteria") or {}).get("actions") or []:
            args = a.get("arguments") or {}
            name = a["name"]
            if name == "call_discoverable_agent_tool":
                name, args = args.get("agent_tool_name"), json.loads(args.get("arguments") or "{}")
            if tool is None or name == tool:
                yield tid, name, args


def _gate_rule(events, gate: str) -> str:
    return [e for e in events.events("gate.fired") if e["payload"]["gate"] == gate][-1]["payload"]["rule_id"]


# ---- checking-credit amount (bank_accounts_(general)_017) ----------------------------------------------------------

def test_net_correction_equals_every_training_gold_credit():
    db = _db()
    gold = [(tid, a) for tid, _, a in _gold(CREDIT) if "amount" in a]
    assert len(gold) == 5
    for tid, a in gold:
        net, parts = fees.net_correction(db, a["account_id"])
        assert net == pytest.approx(float(a["amount"])), (tid, parts)
        assert fees.credit_amount(db, a, "deny") is None


def test_credit_amount_modes():
    db, a = _db(), {"account_id": "chk_kj93a7b2e1_1", "amount": 12.0, "credit_type": "fee_refund"}
    assert fees.credit_amount(db, a, "off") is None
    flagged = fees.credit_amount(db, a, "flag")
    assert flagged.allowed and flagged.rule_id == fees.UNMATCHED and flagged.details["net_correction"] == 9.5
    refused = fees.credit_amount(db, a, "deny")
    assert not refused.allowed and "9.50" in refused.fix and refused.facts
    assert not fees.credit_amount(db, a).allowed  # the default is "deny" (owner's decision, 2026-11-03)
    with pytest.raises(ValueError):
        fees.credit_amount(db, a, "strict")


def test_net_correction_is_none_outside_the_compiled_schedules():
    db = _db()
    assert fees.net_correction(db, "no_such_account") is None
    for kind, level in (("saving", "Bronze Account"), ("checking", "Light Blue Account")):  # not checking; not compiled
        other = next(k for k, v in DB["accounts"]["data"].items() if (v.get("class"), v.get("level")) == (kind, level))
        assert fees.net_correction(db, other) is None
    rows = dict(DB["bank_account_transaction_history"]["data"])
    rows["x1"] = {"account_id": "chk_kj93a7b2e1_1", "type": "monthly_fee", "amount": -20.0, "date": "11/01/2025"}
    assert fees.net_correction(_db(bank_account_transaction_history=rows), "chk_kj93a7b2e1_1") is None
    assert fees.credit_amount(_db(bank_account_transaction_history=rows), {"account_id": "chk_kj93a7b2e1_1",
                                                                          "amount": 1}) is None


def test_foreign_withdrawals_and_the_monthly_rebate_cap():
    assert fees.is_foreign("ATM WITHDRAWAL - HSBC LONDON UK")
    assert not fees.is_foreign("ATM WITHDRAWAL - CHASE BANK #4521 CHICAGO IL")
    assert not fees.is_foreign("ATM WITHDRAWAL - SOMEBANK DOVER DE")
    assert fees.is_foreign("ATM WITHDRAWAL - DEUTSCHE BANK BERLIN DE", "FOREIGN ATM FEE - TIER 3")
    assert not fees.is_foreign("")
    acc = {"z": {"account_id": "z", "class": "checking", "level": "Bluest Account", "status": "OPEN"}}
    rows = {f"w{i}": {"account_id": "z", "type": "atm_withdrawal", "amount": -100.0, "description": "ATM - PNC X IL",
                      "date": f"11/0{i % 9 + 1}/2025"} for i in range(30)}
    db = _db(accounts=acc, bank_account_transaction_history=rows)
    net, parts = fees.net_correction(db, "z")
    assert parts["fees_due"] == 60.0 and parts["rebates_due"] == 50.0 and net == -10.0


def test_checking_credit_gate_refuses_an_unmatched_amount_and_flags_it_when_off(monkeypatch):
    env, task, events = _bank("task_073")
    first = _play(env, task, CREDIT)
    assert _ran(_hidden(env, CREDIT, **first))
    assert _gate_rule(events, f"banking_{CREDIT}_gate") == "banking.checking_credit"
    other = next(k for k in ("chk_kj93a7b2e1_1", "chk_kj93a7b2e1_2") if k != first["account_id"])
    # the default "deny" (owner's decision, 2026-11-03)
    assert not _ran(_hidden(env, CREDIT, account_id=other, amount=1.0, credit_type="fee_refund"))
    assert _gate_rule(events, f"banking_{CREDIT}_gate") == fees.UNMATCHED
    # the off switch, Enforcer(credit_amount_mode="flag"): allowed and flagged
    monkeypatch.setattr(fees, "CREDIT_AMOUNT_MODE", fees.CREDIT_AMOUNT_MODE)  # restored after the test
    events = EventLog(":memory:")
    env, task, _ = _bank("task_073", Enforcer("banking_knowledge", events, credit_amount_mode="flag"))
    first = _play(env, task, CREDIT)
    assert _ran(_hidden(env, CREDIT, **first))
    assert _ran(_hidden(env, CREDIT, account_id=other, amount=1.0, credit_type="fee_refund"))
    assert _gate_rule(events, f"banking_{CREDIT}_gate") == fees.UNMATCHED


# ---- 48-hour entry wait after a fraud_suspected closure (doc 029 vs 031) ------------------------------------------

def test_entry_wait_after_a_fraud_closure_is_refused_by_default_and_flagged_when_off(monkeypatch):
    # the default "deny" (owner's decision, 2026-11-03) refuses task_091's gold
    env, task, _ = _bank("task_091")
    gold = _play(env, task, debit.ORDER, 1)
    assert debit.ENTRY_WAIT_FRAUD == "deny"
    assert not debit.order(env.tools.db, gold).allowed
    # the off switch, Enforcer(entry_wait_fraud="flag"): allowed and flagged
    monkeypatch.setattr(debit, "ENTRY_WAIT_FRAUD", debit.ENTRY_WAIT_FRAUD)  # restored after the test
    events = EventLog(":memory:")
    env, task, _ = _bank("task_091", Enforcer("banking_knowledge", events, entry_wait_fraud="flag"))
    gold = _play(env, task, debit.ORDER, 1)
    assert _ran(_hidden(env, debit.ORDER, **gold))
    assert _gate_rule(events, f"debit_{debit.ORDER}_gate") == debit.ENTRY_WAIT_FLAG
    fired = [e for e in events.events("gate.fired") if e["payload"]["rule_id"] == debit.ENTRY_WAIT_FLAG][-1]
    assert "029 vs 031" in fired["payload"]["reason"]


# ---- constrained values (τ² tool docstrings) ----------------------------------------------------------------------

def test_constraints_parse_the_docstrings_and_no_training_gold_call_violates_them():
    table = constraints.tool_constraints()
    dispute = table[debit.DISPUTE]
    assert "card_present_fraud" in dispute["dispute_category"]["enum"]
    assert dispute["pin_compromised"]["enum"] == ["yes_shared", "yes_observed", "no", "unknown"]
    assert table[CREDIT]["amount"] == {"type": "number", "exclusiveMinimum": 0}
    checked = [(tid, name, constraints.violations(name, a)) for tid, name, a in _gold(None) if name in table]
    assert len(checked) > 100 and not [c for c in checked if c[2]]


def test_constraint_violations_schema_and_gate():
    table = {"t": {"kind": {"type": "string", "enum": ["a", "b"]}, "n": {"type": "number", "exclusiveMinimum": 0}}}
    assert constraints.violations("t", {"kind": "a", "n": 2}, table) == []
    assert len(constraints.violations("t", {"kind": "c", "n": 0}, table)) == 2
    assert constraints.violations("t", {"n": "x"}, table) == ["n 'x' is not a number"]
    assert constraints.violations("t", {"kind": None}, table) == []
    schema = constraints.guided_decoding("t", table)["guided_json"]
    assert schema["properties"]["kind"]["enum"] == ["a", "b"] and schema["additionalProperties"] is True
    assert constraints.parse_docstring("x (string): free text\nn (number): an amount (must be greater than 0)") == {
        "n": {"type": "number", "exclusiveMinimum": 0}}
    gate = constraints.gate(CREDIT)
    assert gate(None, {"account_id": "a", "amount": 5, "credit_type": "fee_refund"}).allowed
    refused = gate(None, {"account_id": "a", "amount": -5, "credit_type": "fee_refund"})
    assert not refused.allowed and refused.facts == ("amount -5 must be greater than 0",)


def test_constraint_gates_register_only_when_enabled(monkeypatch):
    assert constraints.ENABLED  # on by default (owner's decision, 2026-11-03)
    assert [g for g in rules.build_registry("banking_knowledge").gates() if g.name.startswith("constraints_")]
    monkeypatch.setattr(constraints, "ENABLED", False)
    reg = GateRegistry()
    constraints.register(reg, "banking_knowledge")
    assert reg.gates() == []
    monkeypatch.setattr(constraints, "ENABLED", True)
    constraints.register(reg, "retail")
    assert reg.gates() == []
    constraints.register(reg, "banking_knowledge")
    assert f"constraints_{CREDIT}" in {g.name for g in reg.gates()}


# ---- conversation-text checks (offline) ---------------------------------------------------------------------------

def _tc(name, args, hidden=False):
    if hidden:
        return {"name": "call_discoverable_agent_tool", "arguments": {"agent_tool_name": name,
                                                                      "arguments": json.dumps(args)}}
    return {"name": name, "arguments": args}


def _sim(*turns):
    """turns: ("user", text) | ("call", name, args, reply) with the call hidden."""
    msgs = []
    for t in turns:
        if t[0] == "user":
            msgs.append({"role": "user", "content": t[1]})
        else:
            msgs.append({"role": "assistant", "content": None, "tool_calls": [_tc(t[1], t[2], hidden=True)]})
            msgs.append({"role": "tool", "content": t[3]})
    return {"task_id": "task_x", "trial": 0, "messages": msgs}


FILING = {"card_id": "dbc_1", "transaction_id": "tx_9", "card_in_possession": True, "disputed_amount": 42.5}


def test_possession_conflicts_against_what_the_customer_said():
    sim = _sim(("user", "My debit card was stolen yesterday and there is a charge of $42.50 on tx_9."),
               ("call", debit.DISPUTE, FILING, "dispute dsp_1 filed for dbc_1"))
    assert convcheck.possession_conflicts(sim)[0]["customer_said_has_card"] is False
    held = _sim(("user", "I still have my card, it was not stolen."), ("call", debit.DISPUTE, FILING, "filed"))
    assert convcheck.possession_conflicts(held) == []
    assert convcheck.possession_said("nothing about it") is None


def test_unsupplied_values_and_cited_records():
    sim = _sim(("user", "Please dispute tx_9 on card dbc_1. I still have it."),
               ("call", debit.DISPUTE, FILING, "filed"),
               ("call", "record_requests", {"user_id": "u", "requests": [
                   {"action": "close_debit_card", "target_id": "dbc_1", "customer_words": "dispute tx_9 on card dbc_1"},
                   {"action": "order_debit_card", "target_id": "", "customer_words": "send me a new card"}]},
                "recorded"))
    missing = convcheck.unsupplied_values(sim)
    assert [m["arg"] for m in missing] == ["disputed_amount"]
    assert convcheck.unsupplied_values(sim, tools=("other_tool",)) == []
    uncited = convcheck.uncited_records(sim)
    assert [u["customer_words"] for u in uncited] == ["send me a new card"]
    as_text = _sim(("user", "hi"), ("call", "record_requests", {"requests": "not json"}, "recorded"))
    assert convcheck.uncited_records(as_text) == []


def test_dispute_without_filing_and_audit():
    talk = _sim(("user", "I was charged twice for the same coffee."))
    assert convcheck.dispute_without_filing(talk) == {"customer_turns": [0]}
    assert convcheck.dispute_without_filing(_sim(("user", "What is my balance?"))) is None
    filed = _sim(("user", "This charge is fraudulent."), ("call", debit.DISPUTE, FILING, "filed"))
    assert convcheck.dispute_without_filing(filed) is None
    report = convcheck.audit(talk)
    assert "possession_extracted" not in report and report["dispute_without_filing"]


def test_the_fact_extractor_is_off_without_a_model():
    sim = _sim(("user", "Please close my card dbc_1, I lost it."))
    assert convcheck.extract_fact(sim, "Does the customer have the card?", ("yes", "no", "unknown")) is None
    assert convcheck.extract_fact(sim, "Card held?", ("yes", "no", "unknown"), lambda p: " No.") == "no"
    assert convcheck.extract_fact(sim, "Card held?", ("yes", "no"), lambda p: "maybe") is None
    assert convcheck.audit(sim, lambda p: "unknown")["possession_extracted"] == "unknown"


# ---- LLM stages (fakes only) --------------------------------------------------------------------------------------

def test_llm_stages_are_off_without_a_model():
    assert verifiers.provenance_guard([], 1.0, "schedule").status == "off"
    assert verifiers.draft_rules("text", "t") == []


def test_provenance_guard_with_a_fake_model():
    rows = [{"type": "atm_fee", "amount": -3.0}]
    assert verifiers.provenance_guard(rows, 3.0, "s", lambda p: '{"expected": 3.0, "why": "fee"}').status == "pass"
    assert verifiers.provenance_guard(rows, 3.0, "s", lambda p: '{"expected": 1.5}').status == "fail"
    assert verifiers.provenance_guard(rows, 3.0, "s", lambda p: "no idea").status == "unsure"
    assert verifiers.provenance_guard(rows, 3.0, "s", lambda p: '{"expected": "x"}').status == "unsure"
    assert verifiers.provenance_guard(rows, 3.0, "s", lambda p: '{"expected": 3.0,}').status == "unsure"


def test_drafted_rules_compile_and_are_accepted_only_without_gold_refusals():
    assert verifiers.compile_rule({"tool": "t", "arg": "a", "op": "bogus", "value": 1}) is None
    assert verifiers.compile_rule("not a dict") is None
    rule = verifiers.compile_rule({"tool": CREDIT, "arg": "amount", "op": "le", "value": 50})
    assert rule.allows(CREDIT, {"amount": 10}) and not rule.allows(CREDIT, {"amount": 60})
    assert rule.allows("other", {"amount": 60}) and rule.allows(CREDIT, {})
    assert not rule.allows(CREDIT, {"amount": "x"})
    gold = [(CREDIT, a) for _, _, a in _gold(CREDIT) if "amount" in a]
    probes = [(CREDIT, {"amount": 75.0})]
    verdict = verifiers.accept_drafted_rule(rule, gold, lambda t, a: True, probes)
    assert verdict["accepted"] and verdict["adds"] == probes
    strict = verifiers.compile_rule({"tool": CREDIT, "arg": "amount", "op": "lt", "value": 5})
    assert not verifiers.accept_drafted_rule(strict, gold, lambda t, a: True, probes)["accepted"]
    reply = 'Here: [{"tool": "' + CREDIT + '", "arg": "amount", "op": "gt", "value": 0, "source": "_017"}, {"op": 1}]'
    drafted = verifiers.draft_rules("text", CREDIT, lambda p: reply)
    assert len(drafted) == 1 and drafted[0].source == "_017"
    assert verifiers.draft_rules("text", CREDIT, lambda p: "[not json") == []


def test_conversation_audit_script(tmp_path, capsys):
    import conversation_audit
    data = tmp_path / "set"
    data.mkdir()
    talk = _sim(("user", "I was charged twice."))
    (data / "gated-banking_knowledge.json").write_text(json.dumps({"simulations": [talk, _sim(("user", "hi"))]}))
    assert conversation_audit.main([str(data)]) == 0
    out = capsys.readouterr().out
    assert "2 runs" in out and "dispute_without_filing 1" in out
    assert conversation_audit.main([]) == 2
    assert conversation_audit.main([str(tmp_path), "--domain", "retail"]) == 1
