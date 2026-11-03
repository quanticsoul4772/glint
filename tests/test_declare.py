"""glint.declare (the Declare grader, runtime states, reconciling two blind clause lists), glint.mine (mined
follow-up rules, the negative-trace filter, the call-sequence automaton) and scripts/declare_grade.py."""
import json

import declare_grade
import pytest

from glint import declare, mine


def ev(tool, **args):
    return {"tool": tool, "args": args}


def cl(cid, template, a, b=None, **kw):
    return declare.validate({"id": cid, "template": template, "a": a, "b": b, "doc": "doc_x", **kw})


TRACE = [ev("log_verification", user_id="u1"), ev("log_reason", card="c1"), ev("close", card="c1", amount=5),
         ev("close", card="c2", amount="12.5")]


def test_validate_rejects_bad_clauses_and_marks_conditions_uncheckable():
    for bad in ({"id": "x", "template": "nope", "a": "t"}, {"id": "x", "template": "response", "a": "t"},
                {"id": "x", "template": "at_most", "a": "t"}, {"id": "x", "template": "existence"},
                {"id": "x", "template": "value", "a": "t", "a_where": [{"field": "f", "op": "~"}]},
                {"id": "x", "template": "response", "a": "t", "b": "u", "correlate": [["a"]]}):
        with pytest.raises(ValueError):
            declare.validate(bad)
    c = cl("c", "existence", "t", condition_text="when the customer asks")
    assert c["checkable"] is False
    assert declare.check(c, [])["status"] == "not_checkable"


def test_predicates_compare_as_tau2_does():
    args = {"amount": "12.5", "flag": "True", "name": "x", "empty": ""}
    assert declare.holds({"field": "amount", "op": "gt", "value": 12}, args)
    assert declare.holds({"field": "amount", "op": "le", "value": "12.5"}, args)
    assert not declare.holds({"field": "name", "op": "lt", "value": 3}, args)
    assert declare.holds({"field": "flag", "op": "eq", "value": True}, args)
    assert declare.holds({"field": "name", "op": "in", "value": ["x", "y"]}, args)
    assert declare.holds({"field": "name", "op": "not_in", "value": "y"}, args)
    assert declare.holds({"field": "name", "op": "ne", "value": "z"}, args)
    assert declare.holds({"field": "missing", "op": "ne", "value": "z"}, args)
    assert not declare.holds({"field": "missing", "op": "eq", "value": "z"}, args)
    assert not declare.holds({"field": "empty", "op": "present"}, args)
    assert declare.holds({"field": "amount", "op": "ge", "value": 12.5}, args)
    assert declare.holds({"field": "amount", "op": "lt", "value": 13}, args)


def test_every_template_on_a_complete_trace():
    corr = [["card", "card"]]
    assert declare.check(cl("e", "existence", "close"), TRACE)["status"] == "satisfied"
    assert declare.check(cl("e2", "existence", "refund"), TRACE)["violations"] == [-1]
    assert declare.check(cl("ab", "absence", "close"), TRACE)["violations"] == [2, 3]
    assert declare.check(cl("am", "at_most", "close", n=1), TRACE)["status"] == "violated"
    assert declare.check(cl("am2", "at_most", "close", n=2), TRACE)["status"] == "satisfied"
    assert declare.check(cl("v", "value", "close", a_where=[{"field": "amount", "op": "le", "value": 10}]),
                         TRACE)["violations"] == [3]
    # precedence with correlation: c2 was closed with no reason logged for c2
    assert declare.check(cl("p", "precedence", "close", "log_reason", correlate=corr), TRACE)["violations"] == [3]
    assert declare.check(cl("r", "response", "log_reason", "close", correlate=corr), TRACE)["status"] == "satisfied"
    assert declare.check(cl("r2", "response", "close", "log_reason"), TRACE)["violations"] == [2, 3]
    assert declare.check(cl("ns", "not_succession", "log_reason", "close"), TRACE)["violations"] == [1]
    assert declare.check(cl("nc", "not_coexistence", "refund", "close"), TRACE)["status"] == "vacuous"
    assert declare.check(cl("nc2", "not_coexistence", "log_reason", "close"), TRACE)["violations"] == [1]
    assert declare.check(cl("re", "responded_existence", "close", "log_verification"), TRACE)["status"] == \
        "satisfied"
    assert declare.check(cl("re2", "responded_existence", "close", "refund"), TRACE)["violations"] == [2, 3]
    # a where-filtered activation: only closes above 10 must be preceded by a verification of u9
    c = cl("w", "precedence", "close", "log_verification", a_where=[{"field": "amount", "op": "gt", "value": 10}],
           b_where=[{"field": "user_id", "op": "eq", "value": "u9"}])
    assert declare.check(c, TRACE)["violations"] == [3]


def test_runtime_states_and_pending_obligations():
    resp = cl("r", "response", "log_reason", "close", correlate=[["card", "card"]])
    assert declare.state(resp, TRACE[:2]) == "possibly_violated"
    assert declare.state(resp, TRACE[:3]) == "possibly_satisfied"
    assert declare.state(cl("e", "existence", "close"), TRACE[:1]) == "possibly_violated"
    assert declare.state(cl("e", "existence", "close"), TRACE) == "permanently_satisfied"
    assert declare.state(cl("ab", "absence", "close"), TRACE) == "permanently_violated"
    assert declare.state(cl("ab", "absence", "close"), TRACE[:2]) == "possibly_satisfied"
    pend = declare.pending([resp, cl("e", "existence", "refund"),
                            cl("x", "existence", "refund", condition_text="db")], TRACE[:2])
    assert [p["id"] for p in pend] == ["r", "e"]
    assert declare.reminder(pend[0]) == ("[open] policy doc_x (r): after log_reason, close is required and not "
                                         "called yet")
    assert "refund is required" in declare.reminder(pend[1])
    owed = declare.pending([cl("r3", "response", "log_verification", "close")], TRACE[:1])
    assert "for u1" in declare.reminder(owed[0])
    g = declare.grade([resp, cl("ab", "absence", "close")], TRACE)
    assert g["violated"] == ["ab"] and g["counts"] == {"satisfied": 1, "violated": 1}


def test_traces_from_runs_and_from_gold_unwrap_hidden_tools():
    hidden = {"name": "call_discoverable_agent_tool",
              "arguments": {"agent_tool_name": "close", "arguments": json.dumps({"card": "c1"})}}
    assert declare.gold_trace([hidden, {"name": "log_verification", "arguments": {"user_id": "u1"}},
                               {"name": "call_discoverable_agent_tool",
                                "arguments": {"agent_tool_name": "x", "arguments": "{bad"}}]) == [
        ev("close", card="c1"), ev("log_verification", user_id="u1")]

    class Action:
        name, arguments = "log_verification", {"user_id": "u2"}
    assert declare.gold_trace([Action()]) == [ev("log_verification", user_id="u2")]
    sim = {"messages": [
        {"role": "assistant", "tool_calls": [{"id": "1", "name": "call_discoverable_agent_tool",
                                              "arguments": hidden["arguments"]},
                                             {"id": "2", "name": "close", "arguments": {"card": "c9"}}]},
        {"role": "tool", "content": "{\"ok\": true}"}, {"role": "tool", "content": "Error: no", "error": True}]}
    assert declare.trace_of(sim) == [ev("close", card="c1")]


def test_load_reads_a_clause_file_or_a_bare_list(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"clauses": [{"id": "a", "template": "existence", "a": "t"}]}), encoding="utf-8")
    assert declare.load(p)[0]["checkable"] is True
    p.write_text(json.dumps([{"id": "a", "template": "absence", "a": "t"}]), encoding="utf-8")
    assert declare.load(p)[0]["template"] == "absence"


def test_reconcile_sorts_agreed_conflicting_and_single_pass_rules():
    a = [cl("A-1", "precedence", "close", "log_reason", quote="q1"), cl("A-2", "response", "x", "y"),
         cl("A-3", "absence", "z")]
    b = [cl("B-1", "precedence", "close", "log_reason", quote="other words"), cl("B-2", "precedence", "y", "x"),
         cl("B-3", "existence", "w")]
    r = declare.reconcile(a, b)
    assert [c["id"] for c in r["agreed"]] == ["A-1+B-1"] and r["agreed"][0]["status"] == "agreed"
    assert r["conflicts"][0]["activities"] == ["x", "y"]
    assert [c["id"] for c in r["a_only"]] == ["A-3"] and [c["id"] for c in r["b_only"]] == ["B-3"]
    rec = declare.recall(a, [cl("M", "precedence", "close", "log_reason"), cl("N", "precedence", "y", "x")])
    assert rec["exact"] == 1 and rec["activities_only"] == 1 and rec["recall"] == round(1 / 3, 4)
    assert declare.recall([], [])["recall"] is None


GOLD = [[ev("verify", user_id="u1"), ev("log_reason", card_id="c1", user_id="u1"), ev("close", card_id="c1")],
        [ev("verify", user_id="u2"), ev("log_reason", card_id="c2", user_id="u2"), ev("close", card_id="c2")],
        [ev("verify", user_id="u3"), ev("pay", card_id="c3")]]


def test_mining_finds_follow_up_rules_with_correlations():
    rules = {(r["template"], r["a"], r["b"]): r for r in mine.mine(GOLD, min_support=2)}
    assert rules[("precedence", "close", "log_reason")]["correlate"] == [["card_id", "card_id"]]
    assert rules[("response", "log_reason", "close")]["support"] == 2
    assert ("response", "verify", "log_reason") not in rules  # task 3 verifies and logs nothing
    assert all(r["source"] == "mined" and r["confidence"] == "high" for r in rules.values())
    loose = mine.mine(GOLD, min_support=2, min_confidence=0.6)
    assert any(r["a"] == "verify" and r["b"] == "log_reason" and r["confidence"] == "low" for r in loose)


def test_negative_filter_keeps_rules_extra_writes_break():
    rules = mine.mine(GOLD, min_support=2)
    rows = [{"writes": [{"role": "assistant", "name": "close", "args": {"card_id": "c9"}, "st": "extra"},
                        {"role": "user", "name": "U:x", "args": {}, "st": "ok"}]}]
    negatives, vocabulary = mine.negative_traces(rows)
    assert negatives == [[ev("close", card_id="c9")]] and vocabulary == {"close"}
    kept = mine.negative_filter(rules, negatives, require_negative=True)
    assert {(r["template"], r["a"], r["b"]) for r in kept} >= {("precedence", "close", "log_reason")}
    judged = mine.negative_filter(rules, negatives, vocabulary=vocabulary)
    assert all(r["negative_violations"] is None for r in judged)  # no rule's two tools are both recorded


def test_automaton_flags_orders_gold_never_shows():
    auto = mine.Automaton(GOLD)
    assert auto.accepts(GOLD[0])
    assert auto.unseen_steps([ev("verify"), ev("close")]) == [{"index": 1, "from": "verify", "to": "close"}]
    writes = mine.Automaton(GOLD, vocabulary={"log_reason", "close"})
    assert writes.accepts([ev("search"), ev("log_reason"), ev("close")])
    assert not writes.accepts([ev("close")])


def test_declare_grade_script(tmp_path, monkeypatch, capsys):
    clauses = tmp_path / "rec.json"
    agreed = {"id": "A-1+B-1", "template": "precedence", "a": "close", "b": "log_reason", "doc": "d",
              "correlate": [["card_id", "card_id"]], "status": "agreed"}
    single = {"id": "A-2", "template": "absence", "a": "pay", "doc": "d", "status": "a_only"}
    clauses.write_text(json.dumps({"agreed": [agreed], "conflicts": [], "a_only": [single], "b_only": []}),
                       encoding="utf-8")
    assert [c["id"] for c in declare_grade.clauses_of(clauses)] == ["A-1+B-1"]
    assert len(declare_grade.clauses_of(clauses, include_all=True)) == 2
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps({"clauses": [single]}), encoding="utf-8")
    assert declare_grade.clauses_of(plain)[0]["id"] == "A-2"

    monkeypatch.setattr(declare_grade, "gold_traces", lambda: {f"task_{i}": t for i, t in enumerate(GOLD)})
    assert declare_grade.main(["gold", "--clauses", str(clauses), "--all"]) == 0
    out = capsys.readouterr().out
    assert "3 tasks, 2 clauses (2 checkable), 1 clauses violated by gold" in out and "A-2: 1 tasks: task_2" in out

    data = tmp_path / "set"
    data.mkdir()
    sim = {"task_id": "task_9", "trial": 0, "messages": [
        {"role": "assistant", "tool_calls": [{"id": "1", "name": "close", "arguments": {"card_id": "c1"}}]},
        {"role": "tool", "content": "ok"}]}
    (data / "tools-banking_knowledge.json").write_text(json.dumps({"simulations": [sim]}), encoding="utf-8")
    assert declare_grade.main(["runs", str(data), "--clauses", str(clauses)]) == 0
    out = capsys.readouterr().out
    assert "task_9 t0: A-1+B-1" in out and "per clause: A-1+B-1 1" in out
    assert declare_grade.main(["runs", "--clauses", str(clauses)]) == 2
    assert declare_grade.main([]) == 2

    neg = tmp_path / "neg.json"
    neg.write_text(json.dumps([{"writes": [{"role": "assistant", "name": "close", "args": {"card_id": "c9"},
                                            "st": "extra"}]}]), encoding="utf-8")
    monkeypatch.setattr(declare_grade, "NEGATIVES", neg)
    out_file = tmp_path / "mined.json"
    assert declare_grade.main(["mine", "--clauses", str(clauses), "--out", str(out_file)]) == 0
    assert "fitted to τ² gold" in capsys.readouterr().out
    mined = json.loads(out_file.read_text(encoding="utf-8"))
    assert mined["recall_of_clause_list"]["exact"] == 1  # the mined precedence matches the agreed clause
    assert mined["negative_traces"] == 1


def test_reconcile_script_and_gold_violations_file(tmp_path, monkeypatch, capsys):
    import reconcile_clauses
    a, b, out = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "rec.json"
    a.write_text(json.dumps({"clauses": [{"id": "A-1", "template": "absence", "a": "t", "ambiguity": "unclear"}]}),
                 encoding="utf-8")
    b.write_text(json.dumps({"clauses": [{"id": "B-1", "template": "absence", "a": "t"}]}), encoding="utf-8")
    assert reconcile_clauses.main([str(a), str(b), str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["counts"]["agreed"] == 1 and data["agreed"][0]["other"]["id"] == "B-1"
    assert data["ambiguity_notes"] == [{"id": "A-1", "doc": "", "ambiguity": "unclear"}]
    monkeypatch.setattr(declare_grade, "gold_traces", lambda: {"task_1": [ev("t")]})
    viol = tmp_path / "gv.json"
    assert declare_grade.main(["gold", "--clauses", str(out), "--out", str(viol)]) == 0
    assert json.loads(viol.read_text(encoding="utf-8")) == {"A-1+B-1": ["task_1"]}
    assert "reconciled:" in capsys.readouterr().out


def test_the_shipped_clause_lists_load_and_reconcile():
    d = declare.CLAUSE_DIR
    a, b = declare.load(d / "pass_a.json"), declare.load(d / "pass_b.json")
    rec = json.loads((d / "reconciled.json").read_text(encoding="utf-8"))
    assert rec["counts"]["pass_a"] == len(a) and rec["counts"]["pass_b"] == len(b)
    assert len(declare.reconcile(a, b)["agreed"]) == rec["counts"]["agreed"]
