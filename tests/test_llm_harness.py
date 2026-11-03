"""The LLM-stage harnesses, all off by default and tested with fake model calls only (glint.modelcall, glint.extract,
glint.second_sim, glint.denystate, glint.draft_rules, scripts/llm_harness.py). glint.conformal's test is in archive/."""
import json
import sys
import types

import llm_harness
import pytest

from glint import denystate, draft_rules, extract, modelcall, second_sim
from glint.gates.core import DENY_PREFIX

SIM = {"task_id": "task_1", "trial": 0, "messages": [
    {"role": "user", "content": "Hi, I lost my debit card yesterday."},
    {"role": "assistant", "content": "Sorry to hear that. Do you still have the card?"},
    {"role": "user", "content": "No, it is gone. Please close my debit card. ###STOP###"},
]}


def fake(reply):
    return lambda prompt: reply if isinstance(reply, str) else json.dumps(reply)


# ---- modelcall ------------------------------------------------------------------------------------------------

def test_model_calls_are_off_unless_named(monkeypatch):
    assert modelcall.resolve(None) is None and modelcall.resolve("") is None
    with pytest.raises(modelcall.ModelOff):
        modelcall.require(None, "stage")
    with pytest.raises(ValueError):
        modelcall.resolve("no_colon")
    mod = types.ModuleType("fake_llm_mod")
    mod.call, mod.value = (lambda p: "ok"), 3
    monkeypatch.setitem(sys.modules, "fake_llm_mod", mod)
    assert modelcall.resolve("fake_llm_mod:call")("x") == "ok"
    with pytest.raises(TypeError):
        modelcall.resolve("fake_llm_mod:value")
    assert modelcall.parse_object("```json\n{\"a\": 1}\n```") == {"a": 1}
    assert modelcall.parse_object("none") is None and modelcall.parse_object("{bad}") is None
    assert modelcall.parse_object("[1]") is None


# ---- T7 extraction ------------------------------------------------------------------------------------------------

def test_extraction_checks_the_closed_set_and_the_quote():
    with pytest.raises(modelcall.ModelOff):
        extract.extract(SIM, "card_in_possession", None)
    with pytest.raises(ValueError):
        extract.extract(SIM, "nope", fake({}))
    ok = extract.extract(SIM, "card_in_possession", fake({"answer": "no", "turn": 2, "quote": "it is  GONE"}))
    assert ok["valid"] and ok["answer"] == "no"
    assert extract.extract(SIM, "card_in_possession", fake({"answer": "maybe", "turn": 1, "quote": "x"}))[
        "why"].startswith("answer 'maybe'")
    assert "not in turn 1" in extract.extract(SIM, "card_in_possession",
                                              fake({"answer": "no", "turn": 1, "quote": "it is gone"}))["why"]
    assert "does not exist" in extract.extract(SIM, "card_in_possession",
                                               fake({"answer": "no", "turn": 9, "quote": "x"}))["why"]
    assert extract.extract(SIM, "card_in_possession", fake("no json"))["why"] == "no JSON object in the reply"
    assert extract.extract(SIM, "card_in_possession", fake({"answer": "no", "turn": "x"}))["why"] == \
        "turn is not a number"
    assert extract.extract(SIM, "pin_compromised", fake({"answer": "not_said", "turn": 0}))["valid"]
    assert "[2] No, it is gone." in extract.prompt("card_in_possession", extract.customer_turns(SIM))


def test_request_record_and_scoring(tmp_path):
    def model(prompt):
        if "close debit card" in prompt:
            return json.dumps({"answer": "yes", "turn": 2, "quote": "Please close my debit card"})
        return json.dumps({"answer": "not_said", "turn": 0, "quote": ""})
    assert extract.request_record(SIM, model) == [{"action": "close_debit_card", "target_id": "",
                                                   "customer_words": "Please close my debit card", "turn": 2}]
    with pytest.raises(modelcall.ModelOff):
        extract.request_record(SIM, None)
    preds = [{"task_id": "task_1", "trial": 0, "fact": "card_in_possession", "answer": "no"},
             {"task_id": "task_1", "trial": 0, "fact": "pin_compromised", "answer": "invalid"},
             {"task_id": "task_2", "trial": 0, "fact": "card_in_possession", "answer": "no"}]
    labels = [{"task_id": "task_1", "trial": 0, "fact": "card_in_possession", "answer": "no"},
              {"task_id": "task_1", "fact": "pin_compromised", "answer": "no"},
              {"task_id": "task_3", "fact": "card_in_possession", "answer": "yes"}]
    s = extract.score(preds, labels)
    assert s["n"] == 2 and s["accuracy"] == 0.5 and s["invalid"] == 1 and s["labels_unmatched"] == 1
    assert s["confusion"] == {"no -> invalid": 1, "no -> no": 1}
    assert extract.score([], [])["accuracy"] is None
    p = tmp_path / "l.jsonl"
    p.write_text(json.dumps(labels[0]) + "\n\n", encoding="utf-8")
    assert extract.load_jsonl(p) == labels[:1]


# ---- second simulator -----------------------------------------------------------------------------------------

def test_second_simulator_cross_check():
    with pytest.raises(modelcall.ModelOff):
        second_sim.cross_check(SIM, "lost card", None)
    rows = second_sim.cross_check(SIM, "lost card", fake("It is gone, close my card please."))
    assert len(rows) == 1 and rows[0]["recorded_stops"] and not rows[0]["second_stops"]
    assert not rows[0]["agree_stop"] and 0 < rows[0]["overlap"] < 1
    s = second_sim.summary({"task_1 t0": rows, "task_2 t0": []})
    assert s["stop_disagreements"] == 1 and s["runs_with_stop_disagreement"] == ["task_1 t0"]
    assert second_sim.summary({})["mean_overlap"] is None
    assert second_sim.jaccard("", "") == 1.0
    assert second_sim.instructions_of({"user_scenario": {"instructions": "x"}}) == "x"
    assert second_sim.instructions_of({"user_scenario": {"instructions": {"a": 1}}}) == "{'a': 1}"
    obj = types.SimpleNamespace(user_scenario=types.SimpleNamespace(instructions="y"))
    assert second_sim.instructions_of(obj) == "y"
    assert "Customer: Hi" in second_sim.prompt("ins", SIM["messages"][:2])


# ---- deny states (T11a) ----------------------------------------------------------------------------------------

DENY_SIM = {"task_id": "task_1", "trial": 0, "messages": [
    {"role": "user", "content": "pay my card"},
    {"role": "assistant", "tool_calls": [{"id": "1", "name": "pay", "arguments": {"card_id": "c1", "amount": 50}},
                                         {"id": "2", "name": "other", "arguments": {}}]},
    {"role": "tool", "content": f"{DENY_PREFIX} (rule r, check s) and was NOT executed.", "error": True},
    {"role": "tool", "content": f"{DENY_PREFIX} (rule r2, check s) and was NOT executed.", "error": True},
]}
GOLD = [{"name": "pay", "arguments": {"card_id": "c0", "amount": 1}},
        {"name": "pay", "arguments": {"card_id": "c1", "amount": 40}}]


def test_deny_states_name_the_gold_fault(monkeypatch):
    monkeypatch.setattr(denystate, "run_calls", lambda sim: [
        {"index": 2, "tool": "pay", "args": {"card_id": "c1", "amount": 50}, "deny": {"rule_id": "r"}},
        {"index": 3, "tool": "other", "args": {}, "deny": {"rule_id": "r2"}},
        {"index": 4, "tool": "pay", "args": {"card_id": "c1"}, "deny": None}])
    st = denystate.states(DENY_SIM, GOLD)
    assert st[0]["gold"] == {"card_id": "c1", "amount": 40} and st[0]["fault"] == {"amount": [50, 40]}
    assert st[0]["oracle"] == "pay was NOT executed. Fix these arguments and call it again: amount should be 40 " \
                              "(sent 50)."
    assert st[1]["no_gold"] and st[1]["oracle"] is None
    with pytest.raises(modelcall.ModelOff):
        denystate.resume(DENY_SIM, st[0], None)
    with pytest.raises(ValueError):
        denystate.resume(DENY_SIM, st[1], fake("{}"))
    good = denystate.resume(DENY_SIM, st[0], fake({"name": "pay", "arguments": {"card_id": "c1", "amount": 40}}))
    assert denystate.recovered(good, st[0])
    hidden = fake({"name": "call_discoverable_agent_tool",
                   "arguments": {"agent_tool_name": "pay", "arguments": json.dumps({"card_id": "c1", "amount": 50})}})
    assert not denystate.recovered(denystate.resume(DENY_SIM, st[0], hidden), st[0])
    assert denystate.resume(DENY_SIM, st[0], fake("no call")) is None
    assert denystate.oracle_message("pay", {}).endswith("not the arguments.")
    assert "assistant tool call: pay" in denystate.prompt(DENY_SIM["messages"][:2], "refused")
    assert "tool: Error: this write was blocked" in denystate.prompt(DENY_SIM["messages"], "refused")


# ---- drafted rules (T8) -----------------------------------------------------------------------------------------

def test_drafted_rules_are_validated_and_measured(tmp_path):
    reply = {"clauses": [{"template": "precedence", "a": "close", "b": "log_reason"},
                         {"template": "nope", "a": "close"}, {"template": "absence", "a": "unknown_tool"}, "x"]}
    r = draft_rules.draft({"id": "doc_1", "title": "T", "content": "C"}, ["close", "log_reason"], fake(reply))
    assert [c["id"] for c in r["clauses"]] == ["D-doc_1-01"] and r["invalid"] == 3
    assert r["clauses"][0]["source"] == "drafted"
    with pytest.raises(modelcall.ModelOff):
        draft_rules.draft({"id": "d"}, [], None)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "doc_1.json").write_text(json.dumps({"id": "doc_1", "title": "T", "content": "C"}), encoding="utf-8")
    (docs / "doc_2.json").write_text(json.dumps({"title": "U", "content": "D"}), encoding="utf-8")
    out = draft_rules.draft_all(docs, ["close", "log_reason"], fake(reply), only=["doc_2"])
    assert out["documents"] == 1 and out["clauses"][0]["doc"] == "doc_2"
    assert "close, log_reason" in draft_rules.prompt("d", "t", "c", ["close", "log_reason"])


# ---- the CLI ------------------------------------------------------------------------------------------------------

def test_llm_harness_cli_is_off_without_a_model(tmp_path, monkeypatch, capsys):
    data = tmp_path / "set"
    data.mkdir()
    (data / "tools-banking_knowledge.json").write_text(json.dumps({"simulations": [SIM]}), encoding="utf-8")
    (data / "base-banking_knowledge.json").write_text(json.dumps({"simulations": [DENY_SIM]}), encoding="utf-8")
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([{"id": "task_1", "user_scenario": {"instructions": "lost card"},
                                  "evaluation_criteria": {"actions": GOLD}}]), encoding="utf-8")
    assert llm_harness.main([]) == 2
    assert llm_harness.main(["extract", str(data)]) == 2
    assert "off: T7 extraction" in capsys.readouterr().out
    assert llm_harness.main(["second-sim", str(data), "--tasks", str(tasks)]) == 2
    assert llm_harness.main(["draft", "--docs", str(tmp_path)]) == 2
    out = tmp_path / "deny.json"
    assert llm_harness.main(["deny-states", str(data), "--tasks", str(tasks), "--out", str(out)]) == 0
    assert "recovered: None" in capsys.readouterr().out and json.loads(out.read_text(encoding="utf-8"))["refusals"] == 2

    mod = types.ModuleType("fake_cli_llm")
    mod.call = lambda p: json.dumps({"answer": "not_said", "turn": 0, "quote": "", "name": "pay",
                                     "arguments": {"card_id": "c1", "amount": 40}, "clauses": []})
    monkeypatch.setitem(sys.modules, "fake_cli_llm", mod)
    assert llm_harness.main(["deny-states", str(data), "--tasks", str(tasks), "--model", "fake_cli_llm:call"]) == 0
    assert "recovered: 1" in capsys.readouterr().out
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"task_id": "task_1", "trial": 0, "fact": "card_in_possession", "answer": "no"}),
                      encoding="utf-8")
    assert llm_harness.main(["extract", str(data), "--model", "fake_cli_llm:call", "--facts", "card_in_possession",
                             "--labels", str(labels)]) == 0
    assert "'accuracy': 0.0" in capsys.readouterr().out
    assert llm_harness.main(["second-sim", str(data), "--tasks", str(tasks), "--model", "fake_cli_llm:call"]) == 0
    assert "'runs': 2, 'turns': 1" in capsys.readouterr().out
    rec = tmp_path / "rec.json"
    rec.write_text(json.dumps({"agreed": [{"id": "a", "template": "absence", "a": "close_debit_card_4721"}]}),
                   encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "d1.json").write_text(json.dumps({"id": "d1", "title": "t", "content": "c"}), encoding="utf-8")
    assert llm_harness.main(["draft", "--model", "fake_cli_llm:call", "--docs", str(docs), "--clauses", str(rec),
                             "--only", "d1"]) == 0
    assert "recall 0/1 exact" in capsys.readouterr().out
    assert "close_debit_card_4721" in llm_harness._tools()
