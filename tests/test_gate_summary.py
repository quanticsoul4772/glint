"""glint.gates.summary: the GATES line per arm, the error-limit ALERT while an arm runs, scripts/gate_summary.py for a
finished set's folder, and the tools_ab wiring. Every gate log is a temporary EventLog; the runs are fixture
simulations."""
import json
import types

import gate_summary
import pytest
import tools_ab
from loguru import logger
from test_generate import stand_in  # noqa: F401  (the fixture: tools_ab.generate with τ²'s batch replaced)

from glint.gates import EventLog, GateResult, summary
from glint.runlog import RunEvents

logger.remove()

MANIFEST = {"domain": "banking_knowledge", "gates": [
    {"name": "dispute_gate", "tools": ["file_dispute"]}, {"name": "dispute_amount_gate", "tools": ["file_dispute"]},
    {"name": "cli_gate", "tools": ["approve_cli"]}]}


def fired(gate, tool, decision="allow", rule="r", replay=False, task="task_1", trial="0"):
    return {"domain": "banking_knowledge", "gate": gate, "tool": tool, "args": {}, "decision": decision,
            "rule_id": rule, "replay": replay, "task_id": task, "trial": trial}


def gate_log(path, decisions) -> EventLog:
    log = EventLog(path)
    log.emit("gates.manifest", MANIFEST)
    for payload in decisions:
        log.emit("gate.fired", payload)
    return log


DECISIONS = [
    fired("dispute_gate", "file_dispute"), fired("dispute_amount_gate", "file_dispute"),  # one write, two gates
    fired("dispute_gate", "file_dispute", "deny", "banking.dispute"),  # the first gate denies: one event
    fired("dispute_gate", "file_dispute", "deny", "banking.dispute", replay=True),  # τ²'s replay: not counted
    fired("cli_gate", "approve_cli", "deny", "banking.cli.approve"),
    fired("unparsed", "file_dispute", "deny", "unparsed_arguments"),
]


def deny_msg(rule="banking.dispute"):
    return {"role": "tool", "error": True, "content": GateResult("deny", "r", "p", "f", rule, "s").message()}


def tool_error():
    return {"role": "tool", "error": True, "content": "Error: user not found"}


def run(task, trial, messages, term="user_stop"):
    return {"task_id": task, "trial": trial, "termination_reason": term, "reward_info": {"reward": 0.0},
            "messages": messages}


SIMS = [
    run("task_1", 0, [{"role": "assistant", "content": "hi"}, deny_msg(), {"role": "tool", "content": "ok"}]),
    run("task_2", 0, [deny_msg()] * 6 + [tool_error()] * 4, term="too_many_errors"),
    run("task_3", 0, [tool_error()] * 10, term="TerminationReason.TOO_MANY_ERRORS"),  # the enum text in memory
    run("task_4", 0, []),
]


def test_summary_counts_writes_denies_by_rule_and_error_limit_endings(tmp_path):
    s = summary.summarize(gate_log(tmp_path / "g.sqlite", DECISIONS).events(), SIMS)
    assert s == {"writes": 4, "denies": 3, "denies_by_rule": {"banking.cli.approve": 1, "banking.dispute": 1,
                                                              "unparsed_arguments": 1},
                 "runs": 4, "runs_with_deny": 2, "error_limit_runs": 2, "error_limit_runs_with_deny": 1,
                 "error_limit_errors": 20, "error_limit_deny_errors": 6}
    assert summary.line("tools banking_knowledge", s) == (
        "GATES: tools banking_knowledge: 4 writes checked, 3 denied {'banking.cli.approve': 1, 'banking.dispute': 1, "
        "'unparsed_arguments': 1}; 2 of 4 runs had at least one deny; 2 runs ended on τ²'s error limit "
        "(too_many_errors), 1 of them with gate denies; 6 of their 20 errors were gate denies")


def test_a_quiet_arm_says_zero_and_a_log_without_a_manifest_is_refused(tmp_path):
    s = summary.summarize(gate_log(tmp_path / "g.sqlite", []).events(), [run("t", 0, [])])
    assert summary.line("x", s) == ("GATES: x: 0 writes checked, 0 denied; 0 of 1 runs had at least one deny; 0 runs "
                                    "ended on τ²'s error limit (too_many_errors)")
    with pytest.raises(ValueError, match="before any gates.manifest"):
        summary.decisions([{"type": "gate.fired", "payload": fired("cli_gate", "approve_cli")}])


def test_the_latest_manifest_decides_which_gate_counts_a_write():
    second = {"gates": [{"name": "new_gate", "tools": ["file_dispute"]}]}
    events = [{"type": "gates.manifest", "payload": MANIFEST}, {"type": "gate.fired", "payload": fired(
        "dispute_gate", "file_dispute")}, {"type": "gates.manifest", "payload": second},
              {"type": "gate.fired", "payload": fired("new_gate", "file_dispute")},
              {"type": "gate.fired", "payload": fired("dispute_gate", "file_dispute")}]
    assert summary.decisions(events)[0] == 2


def test_error_limit_alert_at_two_deny_driven_endings_then_each_doubling():
    alerts = summary.ErrorLimitAlerts()
    deny_end = lambda task: run(task, 1, [deny_msg()] * 3 + [tool_error()] * 7, term="too_many_errors")
    assert alerts(run("a", 0, [deny_msg()])) is None  # a deny, but the run did not end on the limit
    assert alerts(run("b", 0, [tool_error()] * 10, term="too_many_errors")) is None  # the limit, no deny
    assert alerts(deny_end("c")) is None  # 1: below 2
    assert alerts(deny_end("d")) == ("ALERT: 2 runs ended on τ²'s error limit (too_many_errors) with gate denies "
                                     "among their errors, tasks {'c': 1, 'd': 1}; latest d trial 1: 3 of 10 errors "
                                     "were gate denies")
    assert alerts(deny_end("e")) is None  # 3: not yet double 2
    assert alerts(deny_end("e")).startswith("ALERT: 4 runs")


def test_run_events_print_what_on_run_returns(tmp_path, capsys):
    results = tmp_path / "results.json"
    deny_end = run("task_9", 0, [deny_msg()] * 10, term="too_many_errors")
    results.write_text(json.dumps({"simulations": [deny_end, dict(deny_end, trial=1)]}), encoding="utf-8")
    RunEvents(results, tmp_path / "e.jsonl", label="tools banking", poll_s=60,
              on_run=summary.ErrorLimitAlerts()).poll()
    assert "] ALERT: 2 runs ended on τ²'s error limit" in capsys.readouterr().err


def test_read_log_opens_read_only(tmp_path):
    path = tmp_path / "g.sqlite"
    gate_log(path, DECISIONS).close()
    before = path.read_bytes()
    events = summary.read_log(path)
    assert [e["type"] for e in events] == ["gates.manifest"] + ["gate.fired"] * len(DECISIONS)
    assert path.read_bytes() == before
    import sqlite3

    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("CREATE TABLE x (a)")
    conn.close()


def test_gate_summary_script_prints_the_line_for_a_finished_folder_and_writes_nothing(tmp_path, capsys):
    data = tmp_path / "set"
    gate_log(data / "gates-banking_knowledge.sqlite", DECISIONS).close()
    (data / "tools-banking_knowledge.json").write_text(json.dumps({"simulations": SIMS}), encoding="utf-8")
    before = {p.name: p.read_bytes() for p in data.iterdir()}
    assert gate_summary.main([str(data)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("GATES: tools banking_knowledge: 4 writes checked, 3 denied")
    assert {p.name: p.read_bytes() for p in data.iterdir()} == before
    assert gate_summary.main([]) == 2


def test_gate_summary_script_refuses_a_folder_without_a_log_or_a_tools_arm(tmp_path):
    with pytest.raises(FileNotFoundError, match="no gate log"):
        gate_summary.folder_lines(tmp_path)
    gate_log(tmp_path / "gates-retail.sqlite", []).close()
    with pytest.raises(FileNotFoundError, match="tools-retail.json is missing"):
        gate_summary.folder_lines(tmp_path)


def test_generate_prints_gates_for_the_gated_arm_only(stand_in, make_sim, monkeypatch, tmp_path, capsys):  # noqa: F811
    cfg, outcomes = stand_in
    outcomes["base"] = [make_sim("1", 0), make_sim("1", 1)]
    outcomes["tools"] = [run("1", 0, [deny_msg()] * 10, term="too_many_errors"), make_sim("1", 1)]
    log = gate_log(tmp_path / "gates.sqlite", DECISIONS)
    made = []

    def enforcer(cfg_, domain, data):
        made.append(domain)
        return types.SimpleNamespace(events=log)

    monkeypatch.setattr(tools_ab, "_enforcer", enforcer)
    watched = []
    real = tools_ab.RunEvents

    def run_events(*args, **kwargs):
        watched.append(kwargs.get("on_run"))
        return real(*args, **kwargs)

    monkeypatch.setattr(tools_ab, "RunEvents", run_events)
    tools_ab.generate(cfg)
    out = capsys.readouterr().out
    gates = [line for line in out.splitlines() if line.startswith("GATES:")]
    assert gates == [("GATES: tools retail: 4 writes checked, 3 denied {'banking.cli.approve': 1, 'banking.dispute': 1, "
                     "'unparsed_arguments': 1}; 1 of 2 runs had at least one deny; 1 runs ended on τ²'s error limit "
                     "(too_many_errors), 1 of them with gate denies; 10 of their 10 errors were gate denies")]
    assert made == ["retail"]  # the gates are built for the tools arm only
    assert watched[0] is None and isinstance(watched[1], summary.ErrorLimitAlerts)


def test_parse_deny_reads_the_plain_language_format_and_the_one_older_runs_recorded():
    new = GateResult("deny", "found", "p", "Approve it with new_credit_limit 5000.", "banking.cli.approve", "s",
                     {"tier": "gold"}, "matters", ("a fact",)).message()
    old = ("Error: this write was blocked by a policy gate (rule banking.cli.approve, check s) and was NOT executed."
           "\nReason: found\nPurpose: p\nFix: Approve it with new_credit_limit 5000.\nVerdict: {\"tier\": \"gold\"}")
    for content in (new, old):
        assert summary.parse_deny(content) == {"rule_id": "banking.cli.approve", "reason": "found",
                                               "fix": "Approve it with new_credit_limit 5000.",
                                               "verdict": {"tier": "gold"}}
