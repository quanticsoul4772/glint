"""The measurement pipeline on known inputs: tools_ab.analyze (the counting behind every published number),
export_results (the Results tab) and gate_labels' run selection. The fixtures are in conftest.py."""
import json

import pytest
from loguru import logger

logger.remove()  # τ² logs its registry at DEBUG on import

import tools_ab


def _report(cfg):
    tools_ab.analyze(cfg)
    return json.loads(cfg["out"].read_text(encoding="utf-8"))


def test_analyze_counts_each_outcome_as_the_project_rules_say(run_set, make_sim, capsys):
    base = [make_sim("1", 0, "pass"), make_sim("1", 1, "fail"), make_sim("1", 2, "context_overflow"),
            make_sim("1", 3, "infrastructure_error")]
    tools = [make_sim("1", 0, "pass"), make_sim("1", 1, "glint_error"), make_sim("1", 2, "eval_error"),
             make_sim("1", 3, "budget_stop")]
    r = _report(run_set({("base", "retail"): base, ("tools", "retail"): tools}, k=4))["retail-1"]
    # a pass, a fail and an overflow are runs; an infrastructure error is excluded and counted
    assert (r["base"]["runs"], r["base"]["passes"], r["base"]["context_overflows"], r["base"]["infra_errors"]) == \
        (3, 1, 1, 1)
    # a Glint error is a failed run charged to the tools; an evaluation error and a cap stop are not results
    assert (r["tools"]["runs"], r["tools"]["passes"], r["tools"]["glint_errors"], r["tools"]["eval_errors"],
            r["tools"]["budget_stops"]) == (2, 1, 1, 1, 1)
    out = capsys.readouterr().out
    assert "WARNING: retail-1: runs per arm {'base': 3, 'tools': 2}" in out and "Glint errors" in out


def test_analyze_leaves_out_trials_beyond_k_and_says_so(run_set, make_sim, capsys):
    sims = [make_sim("1", t, "pass") for t in range(3)]
    r = _report(run_set({("base", "airline"): sims, ("tools", "airline"): sims}, k=4, per_domain={"airline": 2}))
    assert r["airline-1"]["base"]["runs"] == 2
    assert "NOTE: base airline: 1 runs beyond k=2 left out" in capsys.readouterr().out


def test_analyze_warns_about_a_missing_arm_and_one_armed_tasks(run_set, make_sim, capsys):
    cfg = run_set({("base", "retail"): [make_sim("1", 0)], ("tools", "retail"): [make_sim("1", 0)],
                   ("base", "airline"): [make_sim("7", 0)]}, k=1)
    _report(cfg)
    out = capsys.readouterr().out
    assert "WARNING: no tools arm for airline" in out
    assert "WARNING: airline-7: only the base arm has runs" in out


def test_analyze_warns_when_a_tools_arm_never_called_a_tool(run_set, make_sim, capsys):
    _report(run_set({("base", "retail"): [make_sim("1", 0)], ("tools", "retail"): [make_sim("1", 0)]}, k=1))
    assert "the tools arm for retail made 0 Glint tool calls" in capsys.readouterr().out


def test_analyze_counts_glint_tool_calls(run_set, make_sim):
    call = {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "name": "check_order_action",
                                                                  "arguments": {}}]}
    r = _report(run_set({("base", "retail"): [make_sim("1", 0)],
                         ("tools", "retail"): [make_sim("1", 0, messages=[call])]}, k=1))
    assert r["retail-1"]["tools"]["tool_calls"] == {"check_order_action": 1}


def test_analyze_refuses_an_empty_set(run_set):
    with pytest.raises(FileNotFoundError, match="no results"):
        tools_ab.analyze(run_set({}, k=1))


def test_export_results_tallies_by_task_and_totals_only_compared_tasks(run_set, make_sim, monkeypatch, tmp_path,
                                                                        capsys):
    import export_results

    arms = {("base", "banking_knowledge"): [make_sim("a", 0, "fail"), make_sim("a", 1, "fail"),
                                            make_sim("b", 0, "pass"), make_sim("c", 0, "context_overflow"),
                                            make_sim("d", 0, "infrastructure_error"), make_sim("e", 0, "pass")],
            ("tools", "banking_knowledge"): [make_sim("a", 0, "pass"), make_sim("a", 1, "fail"),
                                             make_sim("b", 0, "fail"), make_sim("c", 0, "context_overflow"),
                                             make_sim("d", 0, "pass")]}
    cfg = run_set(arms, k=2)
    monkeypatch.setattr(tools_ab, "SETS", {"x": {}})
    monkeypatch.setattr(tools_ab, "for_backend", lambda c, modal: cfg)
    monkeypatch.setattr(export_results, "SETS", [("x", True, "X", "untuned", "note")])
    monkeypatch.setattr(export_results, "OUT", tmp_path / "results.json")
    export_results.main()
    (s,) = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    rows = {r["task"].split("-")[-1]: r for r in s["tasks"]}
    assert rows["a"]["change"] == "improved" and rows["b"]["change"] == "worse"
    assert rows["c"]["change"] == "no data" and rows["c"]["no_data_reason"] == "every run overflowed the context"
    assert rows["d"]["change"] == "no data" and rows["d"]["no_data_reason"].startswith("no counted runs")
    assert "e" not in rows  # one arm only: left off, and said so
    assert "banking_knowledge-e has only the base arm" in capsys.readouterr().out
    assert s["by_task"] == {"improved": 1, "tied": 0, "worse": 1, "no data": 2}
    # totals come from the compared tasks only (e's base pass is not in them)
    assert s["runs"] == {"alone": [1, 4], "tools": [2, 5]}


def test_export_results_says_when_a_set_has_no_data(monkeypatch, tmp_path, capsys):
    import export_results

    monkeypatch.setattr(tools_ab, "SETS", {"x": {}})
    monkeypatch.setattr(tools_ab, "for_backend", lambda c, modal: {"data": tmp_path / "missing"})
    monkeypatch.setattr(export_results, "SETS", [("x", True, "X", "untuned", "note")])
    monkeypatch.setattr(export_results, "OUT", tmp_path / "results.json")
    export_results.main()
    assert "WARNING: x: no data folder" in capsys.readouterr().out
    assert json.loads((tmp_path / "results.json").read_text(encoding="utf-8")) == []


def test_gate_labels_counts_every_run_it_leaves_out(make_sim, monkeypatch, tmp_path, capsys):
    import gate_labels

    sims = [make_sim("0", 0, "infrastructure_error"), make_sim("0", 1, "eval_error"),
            make_sim("no-such-task", 0, "fail")]
    (tmp_path / "runs.json").write_text(json.dumps({"simulations": sims}), encoding="utf-8")
    monkeypatch.setattr(gate_labels, "DATA", tmp_path)
    monkeypatch.setattr(gate_labels, "SOURCES", {"runs.json": "retail", "gone.json": "retail"})
    monkeypatch.setattr(gate_labels, "OUT", tmp_path / "labels.json")
    gate_labels.main()
    out = capsys.readouterr().out
    assert "sources are not on disk" in out and "gone.json" in out  # a missing source is named
    assert "('retail', 'infrastructure_error'): 1" in out and "('retail', 'eval_error'): 1" in out
    assert "('retail', 'unknown task id'): 1" in out


def test_gate_labels_labels_the_writes_of_kept_runs(make_sim, monkeypatch, tmp_path, capsys):
    import gate_labels

    # retail task 0 at its gold actions: every write is one the policy allows, in a passed run
    task = gate_labels.tasks("retail")["0"]
    calls = [{"id": str(i), "name": a.name, "arguments": a.arguments} for i, a in
             enumerate(task.evaluation_criteria.actions)]
    gold = make_sim("0", 0, "pass", messages=[{"role": "assistant", "content": None, "tool_calls": calls}])
    malformed = make_sim("0", 1, "fail", messages=[{"role": "assistant", "content": None, "tool_calls": [
        {"id": "x", "name": "cancel_pending_order", "arguments": {"order_id": "#no-such-order"}}]}])
    (tmp_path / "runs.json").write_text(json.dumps({"simulations": [gold, malformed]}), encoding="utf-8")
    monkeypatch.setattr(gate_labels, "DATA", tmp_path)
    monkeypatch.setattr(gate_labels, "SOURCES", {"runs.json": "retail"})
    monkeypatch.setattr(gate_labels, "OUT", tmp_path / "labels.json")
    gate_labels.main()
    rows = {r["trial"]: r for r in json.loads((tmp_path / "labels.json").read_text(encoding="utf-8"))}
    assert rows[0]["label"] == "agreed" and rows[0]["run_passed"]  # the gold write
    assert rows[1]["label"] == "overruled" and not rows[1]["run_passed"] and not rows[1]["executed"]
    assert "retail: 2 writes, 1 overruled (50.0%)" in capsys.readouterr().out


def test_gate_labels_keeps_overflows_and_glint_errors_as_failed_runs(make_sim, monkeypatch, tmp_path, capsys):
    import gate_labels

    # a context overflow is a model failure and a Glint error is Glint's: both are failed runs, labeled, never dropped
    task = gate_labels.tasks("retail")["0"]
    calls = [{"id": str(i), "name": a.name, "arguments": a.arguments} for i, a in
             enumerate(task.evaluation_criteria.actions)]
    msgs = [{"role": "assistant", "content": None, "tool_calls": calls}]
    sims = [make_sim("0", 0, "context_overflow", messages=msgs), make_sim("0", 1, "glint_error", messages=msgs)]
    (tmp_path / "runs.json").write_text(json.dumps({"simulations": sims}), encoding="utf-8")
    monkeypatch.setattr(gate_labels, "DATA", tmp_path)
    monkeypatch.setattr(gate_labels, "SOURCES", {"runs.json": "retail"})
    monkeypatch.setattr(gate_labels, "OUT", tmp_path / "labels.json")
    gate_labels.main()
    rows = json.loads((tmp_path / "labels.json").read_text(encoding="utf-8"))
    assert sorted(r["trial"] for r in rows) == [0, 1]  # the one write of each run
    assert not any(r["run_passed"] for r in rows)
    out = capsys.readouterr().out
    assert "runs left out" not in out and "retail: 2 writes" in out


def test_export_results_reports_tuned_and_untuned_tasks_apart(run_set, make_sim, monkeypatch, tmp_path):
    import export_results

    # one run set holds a tuned task (worse with the tools) and an untuned one (improved): pooled, they would tie out
    arms = {("base", "banking_knowledge"): [make_sim("task_038", 0, "pass"), make_sim("task_001", 0, "fail")],
            ("tools", "banking_knowledge"): [make_sim("task_038", 0, "fail"), make_sim("task_001", 0, "pass")]}
    cfg = run_set(arms, k=1)
    monkeypatch.setattr(tools_ab, "SETS", {"banking": {}})
    monkeypatch.setattr(tools_ab, "for_backend", lambda c, modal: cfg)
    monkeypatch.setattr(export_results, "SETS", [  # the filters the real SETS use
        ("banking", True, "Untuned", "untuned", "note", lambda task: task not in export_results.TUNED_BANKING),
        ("banking", True, "Tuned", "tuned", "note", lambda task: task in export_results.TUNED_BANKING)])
    monkeypatch.setattr(export_results, "OUT", tmp_path / "results.json")
    export_results.main()
    untuned, tuned = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert (untuned["role"], tuned["role"]) == ("untuned", "tuned")
    assert [r["task"] for r in untuned["tasks"]] == ["banking_knowledge-task_001"]
    assert [r["task"] for r in tuned["tasks"]] == ["banking_knowledge-task_038"]
    assert untuned["by_task"] == {"improved": 1, "tied": 0, "worse": 0, "no data": 0}
    assert tuned["by_task"] == {"improved": 0, "tied": 0, "worse": 1, "no data": 0}
    assert untuned["runs"] == {"alone": [0, 1], "tools": [1, 1]} and tuned["runs"] == {"alone": [1, 1], "tools": [0, 1]}
    # the shipped SETS keep tuned tasks out of every banking set that has a filter
    for name, _, _, role, _, *only in export_results.SETS:
        if only:
            assert only[0]("task_038") == only[0]("task_054") == (role == "tuned")
            assert only[0]("task_001") == (role != "tuned")
