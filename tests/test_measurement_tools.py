"""The measurement tools of fix-research 2026-11-02: the "held, then allowed" attestation class from the gate log (N1,
glint.gates.summary.held_then_allowed), the offline requested-vs-done grader (row 16, glint.grade and
scripts/requested_done.py) and the read-only measures (N4, N5, N9, N12, glint.measures). Runs are fixture
simulations in τ²'s saved format; gate logs are temporary EventLogs."""
import json

import pytest
import requested_done

from glint import grade, measures
from glint.gates import EventLog, GateResult, summary

WRAPPER = "call_discoverable_agent_tool"
DENY = GateResult("deny", "r", "p", "f", "banking.request.no_record", "s").message()


def hidden(tool: str, **args) -> dict:
    return {"name": WRAPPER, "arguments": {"agent_tool_name": tool, "arguments": json.dumps(args)}}


def plain(tool: str, **args) -> dict:
    return {"name": tool, "arguments": args}


def turn(call: dict, reply: str = "ok", error: bool = False) -> list[dict]:
    return [{"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "content": reply, "error": error}]


def sim(*turns, term="user_stop", reward=0.0, task="task_1", trial=0) -> dict:
    return {"task_id": task, "trial": trial, "termination_reason": term, "reward_info": {"reward": reward},
            "messages": [m for t in turns for m in t]}


def recorded(*requests) -> str:
    return json.dumps({"recorded": True, "user_id": "u1", "requests": [
        {"action": a, "target_id": t, "customer_words": "w"} for a, t in requests]})


# ---- N1: held, then allowed ----

def fired(tool, args, decision="allow", rule="r", replay=False, sim_id="s1"):
    return ("gate.fired", {"tool": tool, "args": args, "decision": decision, "rule_id": rule, "replay": replay,
                           "simulation_id": sim_id, "task_id": "task_1", "trial": 0, "gate": "g"})


def attested(tool, classes=("customer_requested",), replay=False, sim_id="s1"):
    return ("write.attested", {"tool": tool, "classes": list(classes), "replay": replay, "simulation_id": sim_id,
                               "task_id": "task_1", "trial": 0})


def events(*rows):
    return [{"type": t, "payload": p} for t, p in rows]


def test_a_write_allowed_after_a_refusal_on_the_same_record_is_held_then_allowed():
    card = {"card_id": "c1", "reason": "lost"}
    rows = events(fired("close", card, "deny", "banking.request.no_record"),
                  fired("close", {"card_id": "c1", "reason": "stolen"}), attested("close"),
                  fired("close", {"card_id": "c2"}), attested("close"))  # another record: not held
    out = summary.held_then_allowed(rows)
    assert len(out) == 1
    assert out[0]["classes"] == ["customer_requested", "held_then_allowed"]
    assert out[0]["refused_by"] == ["banking.request.no_record"] and out[0]["task_id"] == "task_1"


def test_replays_other_runs_and_unparsed_refusals_do_not_count():
    rows = events(fired("close", {"card_id": "c1"}, "deny", replay=True),
                  fired("close", {"card_id": "c1"}, "deny", sim_id="other"),
                  fired("close", None, "deny"),
                  ("gates.manifest", {"gates": []}),
                  fired("close", {"card_id": "c1"}), attested("close"),
                  attested("close", replay=True))
    assert summary.held_then_allowed(rows) == []


def test_the_summary_and_the_gates_line_count_held_then_allowed(tmp_path):
    log = EventLog(tmp_path / "g.sqlite")
    log.emit("gates.manifest", {"gates": [{"name": "g", "tools": ["close"]}]})
    for t, p in (fired("close", {"card_id": "c1"}, "deny", "x"), fired("close", {"card_id": "c1"}),
                 attested("close")):
        log.emit(t, p)
    s = summary.summarize(log.events(), [])
    assert s["held_then_allowed"] == 1
    assert summary.line("tools", s).endswith("; held, then allowed 1")


# ---- row 16: requested vs done ----

def test_requested_done_and_extra_writes_against_the_record():
    run = sim(turn(plain("record_requests", user_id="u1", requests=[]),
                   recorded(("close_debit_card", "c1"), ("order_debit_card", "a1")) + "\n\n[open] x"),
              turn(hidden("close_debit_card_4721", card_id="c1")),
              turn(hidden("pay_credit_card_from_checking_9182", credit_card_account_id="cc1")),  # nobody asked
              turn(hidden("order_debit_card_5739", account_id="a1"), DENY, error=True))  # refused: not done
    g = grade.grade(run)
    assert not g["no_record"] and g["done"] == 2
    assert [r["action"] for r in g["requested_not_done"]] == ["order_debit_card"]
    assert [w["tool"] for w in g["done_not_requested"]] == ["pay_credit_card_from_checking_9182"]


def test_a_run_without_a_record_lists_its_request_tool_writes_as_unrecorded():
    run = sim(turn(plain("record_requests"), json.dumps({"recorded": False, "error": "x"})),
              turn(plain("record_requests"), "not json"),
              turn(hidden("close_debit_card_4721", card_id="c1")), turn(hidden("log_verification", user_id="u1")))
    g = grade.grade(run)
    assert g["no_record"] and [w["tool"] for w in g["unrecorded"]] == ["close_debit_card_4721"]


def test_card_closure_needs_the_reason_and_the_closure_unless_the_offer_was_accepted():
    req = turn(plain("record_requests"), recorded(("close_credit_card", "cc1"), ("close_credit_card", "cc2"),
                                                  ("correct_rewards", "")))
    accept = turn(plain("record_retention_answer"), json.dumps({"recorded": True, "credit_card_account_id": "cc2",
                                                                "accepted": "True", "offer": "$20 credit"}))
    run = sim(req, turn(hidden("log_credit_card_closure_reason_4521", credit_card_account_id="cc1")), accept,
              turn(hidden("log_credit_card_closure_reason_4521", credit_card_account_id="cc2")),
              turn(hidden("update_transaction_rewards_3847", transaction_id="t9")))  # untargeted request covers it
    g = grade.grade(run)
    assert [(r["target_id"], r["tool"]) for r in g["requested_not_done"]] == [("cc1", "close_credit_card_account_7834")]
    assert g["required_not_done"] == [{"rule": "retention offer accepted and not applied "
                                       "(credit_card_account_logistics_003 step 5)", "target_id": "cc2"}]
    applied = sim(*[req, accept, turn(hidden("apply_statement_credit_8472", credit_card_account_id="cc2"))])
    assert grade.grade(applied)["required_not_done"] == []


def test_planned_disputes_must_be_filed():
    plan = json.dumps({"user_id": "u1", "order": [{"transaction_id": "t1"}, {"transaction_id": "t2"}, "junk"]})
    run = sim(turn(plain("plan_disputes"), plan), turn(plain("plan_disputes"), "[1]"),
              turn(hidden("file_credit_card_transaction_dispute_4829", transaction_id="t1")))
    g = grade.grade(run)
    assert [r["target_id"] for r in g["required_not_done"]] == ["t2"]
    t = grade.totals([g, grade.grade(sim())])
    assert t["runs"] == 2 and t["required_not_done"] == 1 and t["runs_required_not_done"] == 1
    assert t["runs_with_record"] == 0


def test_the_script_prints_each_arm_and_its_findings(tmp_path, capsys):
    folder = tmp_path / "set"
    folder.mkdir()
    run = sim(turn(hidden("close_debit_card_4721", card_id="c1")))
    (folder / "tools-banking_knowledge.json").write_text(json.dumps({"simulations": [run, sim(task="task_2")]}),
                                                         encoding="utf-8")
    assert requested_done.main([str(folder)]) == 0
    out = capsys.readouterr().out
    assert "'runs': 2" in out and "task_1 t0: unrecorded 1: close_debit_card_4721" in out
    assert requested_done.main([str(folder), "--quiet", "--domain", "banking_knowledge"]) == 0
    assert "task_1 t0" not in capsys.readouterr().out
    assert requested_done.main([]) == 2
    assert requested_done.main([str(tmp_path)]) == 1


# ---- N4, N5, N9, N12 ----

def test_deny_costs_count_calls_until_the_refused_write_ran():
    close = hidden("close_debit_card_4721", card_id="c1")
    run = sim(turn(close, DENY, error=True), turn(plain("KB_search", query="q")), turn(close, DENY, error=True),
              turn(close), turn(hidden("order_debit_card_5739", account_id="a1"), DENY, error=True),
              turn(plain("KB_search", query="q")))
    costs = measures.deny_costs(run)
    assert [(c["resolved"], c["calls"]) for c in costs] == [("ran", 2), ("ran", 0), ("not_ran", 1)]
    assert costs[0]["messages"] == 5 and costs[2]["messages"] == 2


def test_clopper_pearson_upper_bound():
    assert measures.clopper_pearson_upper(0, 289) == pytest.approx(1 - 0.05 ** (1 / 289), abs=1e-9)
    assert measures.clopper_pearson_upper(1, 407) == pytest.approx(0.0116, abs=2e-4)
    assert measures.clopper_pearson_upper(0, 0) == 1.0 and measures.clopper_pearson_upper(3, 3) == 1.0
    with pytest.raises(ValueError):
        measures.clopper_pearson_upper(4, 3)


def test_writes_before_their_required_lookup():
    txns = "get_bank_account_transactions_9173"
    run = sim(turn(hidden("close_debit_card_4721", card_id="c1")),
              turn(hidden(txns, account_id="a"), "Error: no", error=True),
              turn(hidden("close_debit_card_4721", card_id="c2"), DENY, error=True),
              turn(hidden(txns, account_id="a")), turn(hidden("close_debit_card_4721", card_id="c3")))
    out = measures.writes_before_lookups(run)
    assert [(w["tool"], w["missing"]) for w in out] == [("close_debit_card_4721", [txns])]
    assert out[0]["sources"] == ["bank_accounts_(general)_025 requirements 4-5"]
    assert measures.writes_before_lookups(run, {"x": ()}) == []


def test_endings_and_completion_under_policy():
    assert measures.ending(sim(term="TerminationReason.MAX_STEPS")) == "max_steps"
    assert measures.ending(sim(term="user_stop", reward=1.0)) == "user_stop"
    overflow = sim(term="infrastructure_error")
    overflow["info"] = {"error": "litellm.ContextWindowExceededError: too long"}
    assert measures.ending(overflow) == "context_overflow"
    rows = [{"ending": "user_stop", "passed": True, "offpolicy": 0}, {"ending": "user_stop", "passed": True,
                                                                       "offpolicy": 1},
            {"ending": "agent_stop", "passed": False, "offpolicy": 0}, {"ending": "max_steps", "passed": False,
                                                                        "offpolicy": 0}]
    assert measures.completion_under_policy(rows) == {"runs": 4, "completed": 3, "passed": 2, "cup": 1, "endings": {
        "agent_stop": 1, "max_steps": 1, "user_stop": 2}}
