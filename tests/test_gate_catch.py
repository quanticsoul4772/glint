"""scripts/gate_catch.py on a fixture of recorded banking12 (train) runs, one of each verdict: the runs' own tool calls
replayed through the gates, labeled with the failure review's write labels."""
import json
from pathlib import Path

import gate_catch
from loguru import logger

from glint.gates import EventLog

logger.remove()
FIXTURE = Path(__file__).parent / "data" / "gate_catch"
RUNS = FIXTURE / "tools_ab_fixture_modal"


def _inputs():
    recs = json.loads((FIXTURE / "recs.json").read_text(encoding="utf-8"))
    final = json.loads((FIXTURE / "final.json").read_text(encoding="utf-8"))
    return recs, {(f["set"], f["arm"], f["task"], f["trial"]): f["cause"] for f in final}


def test_gate_catch_classifies_each_recorded_run():
    recs, causes = _inputs()
    rows = {(r["task"], r["trial"]): r for r in gate_catch.run(recs, causes, [RUNS], EventLog(":memory:"))}
    # task_037: the agent transferred with a disposition code after the fraud replacement
    caught = rows[("task_037", 0)]
    assert caught["verdict"] == "caught" and caught["cause"] == "f"
    assert caught["denied"]["rule_id"] == "procedure.transfer.reason"
    assert caught["denied"]["index"] == caught["first_bad"]["index"]
    # task_050: the approval (labeled ok) came before the lookups the procedure names
    early = rows[("task_050", 0)]
    assert early["verdict"] == "denied_earlier" and early["denied"]["rule_id"] == "procedure.lookups"
    # task_031 t1 was missed until the card-digits gate (2026-10-29): its first wrong write files the dispute with
    # guessed digits, never having given the customer get_card_last_4_digits
    late = rows[("task_031", 1)]
    assert late["verdict"] == "caught" and late["denied"]["rule_id"] == "procedure.card_digits.give"
    assert late["denied"]["index"] == late["first_bad"]["index"]
    assert rows[("task_054", 1)]["verdict"] == "no_wrong_write"
    assert rows[("task_031", 0)]["verdict"] == "clean"
    # the same calls as task_037 t0, labeled a pass: a deny there is a false refusal on real traffic
    assert rows[("task_037", 9)]["verdict"] == "false_refusal"


def test_gate_catch_report_and_exit_code(tmp_path, capsys):
    out = tmp_path / "rows.json"
    argv = [str(FIXTURE / "recs.json"), str(FIXTURE / "final.json"), "--runs", str(RUNS)]
    assert gate_catch.main(argv + ["--json", str(out)]) == 1
    text = capsys.readouterr().out
    assert "TOTAL: caught 2, denied earlier 1, missed 0, no wrong write 1" in text
    assert "ended or transferred early: 1 / 0 / 0 / 0 (of 1)" in text and "FALSE REFUSAL fixture" in text
    assert "PASSING runs: 2; false refusals 1" in text
    assert len(json.loads(out.read_text(encoding="utf-8"))) == 6
    recs = [r for r in json.loads((FIXTURE / "recs.json").read_text(encoding="utf-8")) if r["trial"] != 9]
    only = tmp_path / "recs.json"
    only.write_text(json.dumps(recs), encoding="utf-8")
    assert gate_catch.main([str(only), str(FIXTURE / "final.json"), "--runs", str(RUNS)]) == 0


def test_gate_catch_reads_calls_and_labels():
    sim = {"messages": [{"role": "user"}, {"role": "assistant", "tool_calls": [
        {"name": "call_discoverable_agent_tool", "arguments": {"agent_tool_name": "x_1", "arguments": "{}"}}]}]}
    (call,) = gate_catch.trajectory(sim)
    assert call == {"i": 1, "name": "call_discoverable_agent_tool", "requestor": "assistant",
                    "arguments": {"agent_tool_name": "x_1", "arguments": "{}"}}
    assert gate_catch.label_of(call, {(1, "x_1"): "extra"}) == "extra"
    assert gate_catch.label_of(call, {}) is None
    assert gate_catch.set_name(Path("tools_ab_banking12_modal")) == "banking12"


def test_classify_names_every_verdict_including_a_deny_that_came_too_late():
    failed, passed = {"outcome": "fail"}, {"outcome": "pass"}
    bad, early, late = {"index": 5}, {"index": 3}, {"index": 8}
    assert gate_catch.classify(failed, {"first_bad": bad, "denied": {"index": 5}}) == "caught"
    assert gate_catch.classify(failed, {"first_bad": bad, "denied": early}) == "denied_earlier"
    assert gate_catch.classify(failed, {"first_bad": None, "denied": early}) == "denied_earlier"
    assert gate_catch.classify(failed, {"first_bad": bad, "denied": late}) == "missed"
    assert gate_catch.classify(failed, {"first_bad": bad, "denied": None}) == "missed"
    assert gate_catch.classify(failed, {"first_bad": None, "denied": None}) == "no_wrong_write"
    assert gate_catch.classify(passed, {"first_bad": None, "denied": early}) == "false_refusal"
    assert gate_catch.classify(passed, {"first_bad": None, "denied": None}) == "clean"
