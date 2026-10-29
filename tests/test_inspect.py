"""glint.inspect (a run's writes against τ²'s expected actions, the deny follow-ups), the per-run diff in the run log,
the GATE OUTCOMES lines, and the CLI. Runs are built from τ²'s own task data (task_041 of banking_knowledge, read-only)
with known changes; gate logs are temporary EventLogs. No model, no network, nothing outside tmp_path is written."""
import json
import types

import gate_summary
import pytest
import tools_ab
from loguru import logger
from test_generate import stand_in  # noqa: F401  (the fixture: tools_ab.generate with τ²'s batch replaced)

from glint import inspect
from glint.gates import EventLog, GateResult, summary
from glint.inspect import __main__ as cli
from glint.runlog import RunEvents

logger.remove()

DOMAIN, TASK = "banking_knowledge", "task_041"
DISPUTE = "file_credit_card_transaction_dispute_4829"
DISPUTE_FIX = "File the dispute with the card_action and eligible_for_provisional_credit values in the verdict."


def call(name, args, i):
    return {"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}", "name": name, "arguments": args}]}


def hidden(tool, args, i):
    """A hidden banking write as the agent sends it: arguments.arguments is a JSON string."""
    return call("call_discoverable_agent_tool", {"agent_tool_name": tool, "arguments": json.dumps(args)}, i)


def reply(i, content="ok", error=False):
    return {"role": "tool", "id": f"c{i}", "content": content, "error": error}


def deny(rule, reason, fix, details=None):
    return GateResult("deny", reason, "p", fix, rule, "s", details or {}).message()


def gold():
    return inspect.gold_writes(DOMAIN, TASK)


def failed_run(trial=0):
    """task_041 with known faults: the first dispute is denied once and refiled with the fix values; the second has the
    wrong dispute_reason; the last gold write is never made; one dispute on a transaction gold does not have; one
    call with arguments that do not parse (the environment rejects it)."""
    writes, msgs, i = gold(), [], 0
    for k, w in enumerate(writes[:-1]):
        args = dict(w["args"])
        if w["tool"] != DISPUTE:
            msgs += [call(w["tool"], args, i), reply(i)]
            i += 1
            continue
        if k == 1:  # the first dispute: denied with the flag flipped, then refiled as the fix says
            wrong = args | {"eligible_for_provisional_credit": not args["eligible_for_provisional_credit"]}
            msgs += [hidden(DISPUTE, wrong, i), reply(i, deny("banking.dispute", "eligible differs", DISPUTE_FIX, {
                "card_action": args["card_action"],
                "eligible_for_provisional_credit": args["eligible_for_provisional_credit"]}), error=True)]
            i += 1
        if k == 2:
            args["dispute_reason"] = "duplicate_charge"
        msgs += [hidden(DISPUTE, args, i), reply(i)]
        i += 1
    msgs += [hidden(DISPUTE, dict(writes[1]["args"], transaction_id="txn_not_in_gold"), i), reply(i)]
    msgs += [call("call_discoverable_agent_tool", {"agent_tool_name": DISPUTE, "arguments": "{not json"}, i + 1),
             reply(i + 1, "Error: arguments do not parse", error=True)]
    reward = {"reward": 0.0, "db_check": {"db_match": False, "db_reward": 0.0}, "reward_basis": ["DB"],
              "action_checks": [{"action": {"action_id": "041_9", "name": "call_discoverable_agent_tool",
                                            "arguments": {"agent_tool_name": DISPUTE, "arguments": "{}"}},
                                 "action_match": False}, {"action": {"action_id": "041_0", "name": "log_verification",
                                                                     "arguments": {}}, "action_match": True}],
              "nl_assertions": [{"nl_assertion": "agent explains the dispute", "met": False}],
              "info": {"note": "graded"}}
    return {"task_id": TASK, "trial": trial, "termination_reason": "user_stop", "info": {}, "reward_info": reward,
            "messages": msgs, "duration": 1.0}


def test_gold_writes_unwrap_hidden_writes_and_leave_out_reads_unlocks_and_user_actions():
    writes = gold()
    tools = [w["tool"] for w in writes]
    assert tools[0] == "log_verification" and tools.count(DISPUTE) == 16 and len(tools) == 17
    assert "get_user_dispute_history_7291" not in tools and "unlock_discoverable_agent_tool" not in tools
    assert writes[1]["args"]["transaction_id"] == "txn_645286a3dd13" and writes[1]["action_id"] == "041_9"
    assert writes[1]["args"]["eligible_for_provisional_credit"] is True  # parsed from the JSON string


def test_task_env_is_at_the_tasks_initial_state():
    env = inspect.task_env(DOMAIN, TASK)
    txn = env.tools.db.credit_card_transaction_history.data["txn_645286a3dd13"]
    assert txn["transaction_date"] == "10/14/2025" and txn["transaction_amount"]
    # the earlier dispute exists only in task_041's initialization data
    assert env.tools.db.transaction_disputes.data["dsp_dfab0685d198"]["user_id"] == "a6a7d745b2"


def test_run_writes_split_executed_from_gate_and_environment_rejections():
    w = inspect.run_writes(failed_run(), DOMAIN)
    assert len(w["executed"]) == 17
    assert [(r["by"], r["tool"]) for r in w["rejected"]] == [("gate", DISPUTE), ("env", DISPUTE)]
    assert w["rejected"][0]["rule_id"] == "banking.dispute" and w["rejected"][1]["args"] is None


def test_diff_run_names_missing_extra_and_changed_writes_with_taus_breakdown():
    d = inspect.diff_run(DOMAIN, failed_run())
    assert [w["action_id"] for w in d["missing"]] == [gold()[-1]["action_id"]]
    assert d["missing"][0]["context"]["transaction_amount"]  # the transaction's amount and date beside the write
    assert [w["args"]["transaction_id"] for w in d["extra"]] == ["txn_not_in_gold"] and d["extra"][0]["context"] == {}
    assert [(w["target"]["transaction_id"], w["fields"]) for w in d["changed"]] == [
        (gold()[2]["args"]["transaction_id"],
         [{"field": "dispute_reason", "gold": gold()[2]["args"]["dispute_reason"], "run": "duplicate_charge"}])]
    assert d["changed"][0]["context"]["transaction_date"]
    assert d["reward"] == {"reward": 0.0, "basis": ["DB"], "breakdown": None, "db_match": False,
                           "actions": {"met": 1, "total": 2, "unmet": [f"041_9 {DISPUTE}"]},
                           "nl_assertions": {"met": 0, "total": 1, "unmet": ["agent explains the dispute"]},
                           "note": "graded"}
    c = inspect.compact(d)
    assert c["text"] == (f"diff: 1 field (dispute_reason x1); missing 1 ({DISPUTE} x1), extra 1 ({DISPUTE} x1); "
                         "rejected 2")
    assert (c["fields"], c["missing"], c["extra"], c["rejected"]) == ({"dispute_reason": 1}, 1, 1, 2)


def test_a_write_matching_gold_after_type_coercion_is_not_a_difference():
    args = {"credit_card_account_id": "a", "new_credit_limit": 5000, "flag": "True"}
    missing, extra, changed = inspect.pair_writes([{"tool": "t", "args": args}], [{"tool": "t", "index": 1, "args": {
        "credit_card_account_id": "a", "new_credit_limit": 5000.0, "flag": True}}])
    assert (missing, extra, changed) == ([], [], [])
    assert inspect.reward_breakdown(None) is None


def test_fix_followed_by_kind_of_fix():
    writes = {"close_card", "unfreeze_card_3893", "approve_5847", "deny_5848", "transfer_to_human_agents"}

    def c(tool, args=None, error=False, named=None):
        return {"tool": tool, "args": args or {}, "error": error, "named": named}

    def judged(fix, tool, args, later, verdict=None):
        denied = c(tool, args) | {"deny": {"fix": fix, "verdict": verdict or {}}}
        return summary.fix_followed(denied, later, writes)

    card = {"card_id": "c1"}
    assert judged("Do not close it.", "close_card", card, [c("close_card", {"card_id": "c2"})]) == (True, "not retried")
    assert judged("Do not close it.", "close_card", card, [c("close_card", card)])[0] is False
    cli = {"credit_card_account_id": "a"}
    assert judged("Approve it with new_credit_limit 5000.", "deny_5848", cli,
                  [c("approve_5847", cli | {"new_credit_limit": 5000})]) == (True, "approve_5847 with the fix values")
    assert judged("Approve it with new_credit_limit 5000.", "deny_5848", cli,
                  [c("approve_5847", cli | {"new_credit_limit": 6000})])[0] is False
    assert judged("Approve it with new_credit_limit 5000.", "deny_5848", cli, [])[0] is False
    first = "Unfreeze it with unfreeze_card_3893 first, then close it."
    assert judged(first, "close_card", card, [c("unfreeze_card_3893", card), c("close_card", card)])[0] is True
    assert judged(first, "close_card", card, [c("close_card", card), c("unfreeze_card_3893", card)])[0] is False
    assert judged("Unlock and call get_x_9173 (call_discoverable_agent_tool), check the result, then retry this write.",
                  "close_card", card, [c("unlock_discoverable_agent_tool", named="get_x_9173")])[0] is True
    log = "Log the verification with the user's details exactly as on file."
    assert judged(log, "close_card", card, [c("close_card", card)]) == (True, "retry of close_card executed")
    assert judged(log, "close_card", card, [c("close_card", card, error=True)])[0] is False
    assert judged(log, "close_card", card, []) == (False, "not retried")


def test_expected_values_read_with_pairs_and_the_verdict_fields_the_fix_names():
    assert summary.expected_values({"fix": "Order it with delivery_fee 15, design_fee 0."}, {}) == {
        "delivery_fee": 15.0, "design_fee": 0.0}
    assert summary.expected_values({"fix": "Transfer with reason 'fraud_or_security_concern'."}, {}) == {
        "reason": "fraud_or_security_concern"}
    assert summary.expected_values({"fix": "Unfreeze it with unfreeze_debit_card_3893 first, then close it."}, {}) == {}
    assert summary.expected_values({"fix": DISPUTE_FIX, "verdict": {
        "card_action": "keep_active", "eligible_for_provisional_credit": False, "amount": 3.0, "checks": {}}},
        {"card_action": "x", "eligible_for_provisional_credit": True, "checks": 1}) == {
        "card_action": "keep_active", "eligible_for_provisional_credit": False}


def gate_log(path, denies) -> EventLog:
    log = EventLog(path)
    log.emit("gates.manifest", {"domain": DOMAIN, "gated_generic": ["transfer_to_human_agents"],
                                "gates": [{"name": "g", "tools": [DISPUTE, "log_verification"]}]})
    for task, trial, reason in denies:
        log.emit("gate.fired", {"tool": DISPUTE, "gate": "g", "args": {}, "decision": "deny", "replay": False,
                                "rule_id": "banking.dispute", "reason": reason, "task_id": task, "trial": str(trial)})
    return log


def test_run_denies_pairs_the_log_with_the_saved_run_and_judges_the_fix(tmp_path):
    db = tmp_path / f"gates-{DOMAIN}.sqlite"
    gate_log(db, [(TASK, 0, "eligible differs"), (TASK, 0, "an earlier attempt"), (TASK, 1, "other run")]).close()
    (tmp_path / f"tools-{DOMAIN}.json").write_text(json.dumps({"simulations": [failed_run()]}), encoding="utf-8")
    d = inspect.run_denies(db, TASK, 0)
    assert [(x["rule_id"], x["reason"], x["fix_followed"], x["saved_run"], x["logged"]) for x in d] == [
        ("banking.dispute", "eligible differs", True, True, True),
        ("banking.dispute", "an earlier attempt", None, False, True)]
    assert d[0]["fix"] == DISPUTE_FIX and d[0]["next_write"] == {"tool": DISPUTE, "args": gold()[1]["args"],
                                                                 "refused": False}
    assert d[0]["how"] == f"{DISPUTE} with the fix values"
    with pytest.raises(KeyError, match="no run of task_041 trial 7"):
        inspect.run_denies(db, TASK, 7)


def test_rule_outcomes_count_denies_followed_fixes_and_how_the_runs_ended(tmp_path):
    events = gate_log(tmp_path / "g.sqlite", []).events()
    passed = failed_run(trial=1) | {"reward_info": {"reward": 1.0}}
    sims = [failed_run(), passed, {"task_id": "x", "trial": 0, "termination_reason": "infrastructure_error",
                                   "info": {"error": "ConnectionError"}, "messages": failed_run()["messages"]}]
    table = summary.rule_outcomes(events, sims, inspect.arm_diffs(DOMAIN, sims[:2]))
    row = table["banking.dispute"]
    assert (row["denies"], row["fix_followed"], row["passed"], row["failed"], row["not_result"]) == (3, 3, 1, 1, 1)
    assert summary.outcome_lines("tools banking", table) == [(
        "GATE OUTCOMES: tools banking: banking.dispute: 3 denies, fix followed 3; runs with this deny: 1 passed, "
        "1 failed, 1 not results (failed runs differ on: dispute_reason x1, missing 1, extra 1)")]
    unknown = failed_run() | {"task_id": "task_not_in_tau2"}
    assert inspect.arm_diffs(DOMAIN, [unknown]) == {}
    assert summary.outcome_lines("x", summary.rule_outcomes(events, [unknown], {})) == [(
        "GATE OUTCOMES: x: banking.dispute: 1 denies, fix followed 1; runs with this deny: 0 passed, 1 failed; 1 failed "
        "runs not diffed (task not in τ²'s tasks)")]
    assert summary.outcome_lines("x", {}) == ["GATE OUTCOMES: x: no denies in the saved runs"]


def test_run_events_put_a_failed_runs_diff_in_its_record_and_line_and_print_diffs_at_the_end(tmp_path, capsys):
    results, events = tmp_path / "results.json", tmp_path / "events.jsonl"
    unknown = failed_run(trial=2) | {"task_id": "task_not_in_tau2"}
    with RunEvents(results, events, label="tools banking", poll_s=60, domain=DOMAIN) as watch:
        results.write_text(json.dumps({"simulations": [failed_run(), failed_run(1) | {"reward_info": {"reward": 1.0}},
                                                       unknown]}), encoding="utf-8")  # landed while watched
    records = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    assert records[0]["diff"]["text"].startswith("diff: 1 field (dispute_reason x1); missing 1")
    assert "diff" not in records[1]  # a passed run is not diffed
    assert records[2]["diff_error"].startswith("KeyError") and "diff" not in records[2]
    err = capsys.readouterr().err
    assert "task_041 trial 0: FAIL (user_stop, 1 s, " in err and "diff: 1 field (dispute_reason x1)" in err
    assert "WARNING: no diff (KeyError" in err
    assert "DIFFS: 1 failed runs diffed against τ²'s expected writes; top fields {'dispute_reason': 1}; top tools " \
           f"{{'{DISPUTE}': 3}}" in err
    assert list(watch.diffs) == [(TASK, 0)]


def test_cli_prints_each_run_and_the_arms_diffs_line(tmp_path, capsys):
    not_result = {"task_id": TASK, "trial": 5, "termination_reason": "infrastructure_error",
                  "info": {"error": "ConnectionError"}, "messages": []}
    (tmp_path / f"tools-{DOMAIN}.json").write_text(json.dumps({"simulations": [failed_run(), not_result]}),
                                                   encoding="utf-8")
    assert cli.main(["diff", str(tmp_path), "--task", TASK]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"== tools {DOMAIN}: 2 runs"
    assert out[1].startswith(f"{TASK} trial 0 FAIL: diff: 1 field (dispute_reason x1)")
    assert out[1].endswith("τ² reward 0.0, db_match False, actions 1/2, nl_assertions 0/1, graded")
    assert any(line.startswith(f"  changed {DISPUTE} (transaction_id ") and "dispute_reason gold " in line
               for line in out)
    assert any(line.startswith(f"  rejected by gate {DISPUTE} (rule banking.dispute)") for line in out)
    assert any(line.startswith(f"  rejected by env {DISPUTE}: Error: arguments") for line in out)
    assert f"  τ² unmet actions: 041_9 {DISPUTE}" in out
    assert f"{TASK} trial 5: not a result (infrastructure_error)" in out
    assert out[-1].startswith("DIFFS: 1 failed runs diffed")
    assert cli._reward(None) == "τ²: no reward"
    with pytest.raises(FileNotFoundError, match="no base arm file"):
        cli.folder_lines(tmp_path, "base")


def test_gate_summary_script_prints_gate_outcomes(tmp_path, capsys):
    gate_log(tmp_path / f"gates-{DOMAIN}.sqlite", [(TASK, 0, "eligible differs")]).close()
    (tmp_path / f"tools-{DOMAIN}.json").write_text(json.dumps({"simulations": [failed_run()]}), encoding="utf-8")
    assert gate_summary.main([str(tmp_path)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[1] == (f"GATE OUTCOMES: tools {DOMAIN}: banking.dispute: 1 denies, fix followed 1; runs with this deny: "
                      "0 passed, 1 failed (failed runs differ on: dispute_reason x1, missing 1, extra 1)")


def test_generate_prints_gate_outcomes_and_watches_with_the_domain(stand_in, make_sim, monkeypatch, tmp_path,  # noqa: F811
                                                                   capsys):
    cfg, outcomes = stand_in
    outcomes["base"] = [make_sim("1", 0), make_sim("1", 1)]
    outcomes["tools"] = [make_sim("1", 0), make_sim("1", 1)]
    log = gate_log(tmp_path / "gates.sqlite", [])
    monkeypatch.setattr(tools_ab, "_enforcer", lambda cfg_, domain, data: types.SimpleNamespace(events=log))
    domains = []
    real = tools_ab.RunEvents

    def run_events(*args, **kwargs):
        domains.append(kwargs.get("domain"))
        return real(*args, **kwargs)

    monkeypatch.setattr(tools_ab, "RunEvents", run_events)
    tools_ab.generate(cfg)
    assert "GATE OUTCOMES: tools retail: no denies in the saved runs" in capsys.readouterr().out.splitlines()
    assert domains == ["retail", "retail"]
