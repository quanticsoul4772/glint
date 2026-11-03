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


def test_analyze_does_not_read_a_reruns_released_side_files_as_domains(run_set, make_sim, capsys):
    # banking-test's rerun left released-<arm>-<domain>-<time>.json beside the arm files, and analyze warned that
    # "base-banking_knowledge-20261028-121251" had no tools arm (2026-10-29)
    cfg = run_set({("base", "retail"): [make_sim("1", 0)], ("tools", "retail"): [make_sim("1", 0)]}, k=1)
    (cfg["data"] / "released-base-retail-20261028-121251.json").write_text('{"simulations": []}', encoding="utf-8")
    assert set(_report(cfg)) == {"retail-1"}
    out = capsys.readouterr().out
    assert "no tools arm" not in out and "no base arm" not in out


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


def _seeded(make_sim, kinds: dict, seed=lambda task, trial: 7 * trial):
    """Saved runs {(task, trial): kind}, each with a seed (tools_ab seeds every run from its task and trial)."""
    return [make_sim(task, trial, kind) | {"seed": seed(task, trial)} for (task, trial), kind in kinds.items()]


def test_export_results_takes_a_tools_only_sets_base_arm_from_the_named_set(run_set, make_sim, monkeypatch, tmp_path):
    # banking-plan-v5-gated runs the tools arm only; its "alone" column is banking-plan-gated's base arm
    import export_results

    tools = run_set({("tools", "banking_knowledge"): _seeded(make_sim, {("a", 0): "pass", ("a", 1): "pass"})}, k=2,
                    name="v5")
    base = run_set({("base", "banking_knowledge"): _seeded(make_sim, {("a", 0): "fail", ("a", 1): "pass"}),
                    ("tools", "banking_knowledge"): _seeded(make_sim, {("a", 0): "fail", ("a", 1): "fail"})}, k=2,
                   name="plan")
    monkeypatch.setattr(tools_ab, "SETS", {"v5": {"which": "v5"}, "plan": {"which": "plan"}})
    monkeypatch.setattr(tools_ab, "for_backend", lambda c, modal: {"v5": tools, "plan": base}[c["which"]])
    monkeypatch.setattr(export_results, "SETS", [("v5", True, "V5", "tuned", "note")])
    monkeypatch.setattr(export_results, "BASE_FROM", {"v5": "plan"})
    monkeypatch.setattr(export_results, "OUT", tmp_path / "results.json")
    export_results.main()
    (s,) = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert [(r["alone"], r["tools"], r["change"]) for r in s["tasks"]] == [("1/2", "2/2", "improved")]
    assert s["runs"] == {"alone": [1, 2], "tools": [2, 2]}


def test_export_results_refuses_a_base_arm_with_other_seeds(run_set, make_sim):
    import export_results

    tools = run_set({("tools", "banking_knowledge"): _seeded(make_sim, {("a", 0): "pass"})}, k=1, name="v5")
    base = run_set({("base", "banking_knowledge"): _seeded(make_sim, {("a", 0): "pass"}, seed=lambda t, n: 99)}, k=1,
                   name="plan")
    with pytest.raises(SystemExit, match="differ in tasks, trials or seeds"):
        export_results.with_base({}, tools, base)


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


def test_export_results_rows_counts_filter_and_tie(run_set, make_sim, monkeypatch, tmp_path):
    # Kills the mutants that survived the 2026-10-28 cosmic-ray run on main (79 of 212): the tie boundary, the rate
    # arithmetic, the per-task overflow / infrastructure / Glint error counts the desk shows, the both-arms-overflowed
    # rule with k > 1, the task filter (tuned vs untuned sets), and a one-arm task that must not stop the loop.
    import export_results

    s = make_sim
    arms = {("base", "banking_knowledge"): [s("a", 0, "pass"), s("a", 1, "fail"), s("a2", 0, "pass"),
                                            s("b", 0, "context_overflow"), s("b", 1, "context_overflow"),
                                            s("c", 0, "context_overflow"), s("c", 1, "pass"),
                                            s("d", 0, "infrastructure_error"), s("d", 1, "pass"),
                                            s("e", 0, "infrastructure_error"), s("e", 1, "glint_error"),
                                            s("x", 0, "fail"), s("x", 1, "fail")],
            ("tools", "banking_knowledge"): [s("a", 0, "fail"), s("a", 1, "pass"),
                                             s("b", 0, "context_overflow"), s("b", 1, "context_overflow"),
                                             s("c", 0, "pass"), s("c", 1, "pass"),
                                             s("d", 0, "glint_error"), s("d", 1, "pass"),
                                             s("e", 0, "infrastructure_error"), s("e", 1, "glint_error"),
                                             s("x", 0, "pass"), s("x", 1, "pass")],
            ("base", "airline"): [s("z", 0, "pass")], ("tools", "airline"): [s("z", 0, "pass")]}
    cfg = run_set(arms, k=2)
    monkeypatch.setattr(tools_ab, "SETS", {"x": {}})
    monkeypatch.setattr(tools_ab, "for_backend", lambda c, modal: cfg)
    monkeypatch.setattr(export_results, "SETS", [("x", True, "X", "untuned", "note", lambda task: task != "x")])
    monkeypatch.setattr(export_results, "OUT", tmp_path / "results.json")
    export_results.main()
    (out,) = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    rows = [(r["task"].split("-")[-1], r["alone"], r["tools"], r["change"], r["overflows"], r["infra_errors"],
             r["glint_errors"]) for r in out["tasks"]]
    assert rows == [("z", "1/1", "1/1", "tied", 0, 0, 0),           # rows ordered by domain (airline first), then task
                    ("a", "1/2", "1/2", "tied", 0, 0, 0),           # equal nonzero rates: tied, not improved
                    ("b", "0/2", "0/2", "no data", 4, 0, 0),        # every run of both arms overflowed
                    ("c", "1/2", "2/2", "improved", 1, 0, 0),       # one overflow is a failed run, not "no data"
                    ("d", "1/1", "1/2", "worse", 0, 1, 1),          # infra excluded; a Glint error counts against tools
                    ("e", "0/1", "0/1", "tied", 0, 2, 2)]           # errors in both arms add up (not one arm's count)
    assert out["by_task"] == {"improved": 1, "tied": 3, "worse": 1, "no data": 1}  # a2 (one arm) and x (filtered) out
    assert out["runs"] == {"alone": [4, 9], "tools": [5, 10]}


def test_export_results_skips_a_missing_set_and_exports_the_next(run_set, make_sim, monkeypatch, tmp_path):
    import export_results

    cfg = run_set({("base", "retail"): [make_sim("a", 0)], ("tools", "retail"): [make_sim("a", 0)]}, k=1)
    monkeypatch.setattr(tools_ab, "SETS", {"gone": {}, "here": {}})
    monkeypatch.setattr(tools_ab, "for_backend", lambda c, modal: {"data": tmp_path / "missing"} if c is GONE else cfg)
    GONE = tools_ab.SETS["gone"]
    monkeypatch.setattr(export_results, "SETS", [("gone", True, "G", "untuned", "n"), ("here", True, "H", "untuned", "n")])
    monkeypatch.setattr(export_results, "OUT", tmp_path / "results.json")
    export_results.main()
    assert [x["set"] for x in json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))] == ["here"]


def test_export_results_sets_say_the_backend_they_load():
    # each set's Modal flag picks its data folder; its note tells the reader the backend. A flipped flag would publish
    # one backend's numbers under the other's note.
    import export_results

    for name, modal, _title, _role, note, *_ in export_results.SETS:
        assert ("Glimmer on Modal" in note) == modal and ("Glimmer on DeepInfra" in note) == (not modal), name


def test_the_pressure_set_adds_one_customer_instruction_and_changes_nothing_else(monkeypatch):
    tasks = {t["id"]: t for t in json.loads(tools_ab.BANKING_TASKS.read_text(encoding="utf-8"))}
    assert set(tools_ab.PRESSURE_TASKS) <= set(tools_ab.banking_unseen_tasks()["banking_knowledge"])
    assert len(tools_ab.PRESSURE_TASKS) == 20
    plain = tasks["task_093"]
    pushed = tools_ab.pressured(plain)
    assert pushed["user_scenario"]["instructions"] == plain["user_scenario"]["instructions"] + tools_ab.PRESSURE
    assert {k: v for k, v in pushed.items() if k != "user_scenario"} == {
        k: v for k, v in plain.items() if k != "user_scenario"}  # gold and initial state unchanged
    assert tasks["task_093"]["user_scenario"]["instructions"] == plain["user_scenario"]["instructions"]  # a copy

    seen = {}

    def fake_run_tasks(config, task_list, save_path, save_dir):
        seen["tasks"] = task_list

    monkeypatch.setattr(tools_ab, "run_tasks", fake_run_tasks)
    monkeypatch.setattr(tools_ab, "_save_dir", lambda cfg, arm, domain, stamp: tools_ab.MEASURE / "unused")
    cfg = tools_ab.SETS["banking-pressure-gated"]
    tools_ab._run(cfg, "banking_knowledge", ["task_093"], "base", "stamp", None)
    assert seen["tasks"][0].user_scenario.instructions.endswith(tools_ab.PRESSURE.strip()[-60:])
    tools_ab._run(tools_ab.SETS["banking-unseen-gated"], "banking_knowledge", ["task_093"], "base", "stamp", None)
    assert tools_ab.PRESSURE.strip()[-60:] not in seen["tasks"][0].user_scenario.instructions


def test_the_gated_test_split_set_runs_both_arms_on_the_untouched_split_with_gates():
    import tools_ab
    cfg = tools_ab.SETS["banking-test-gated"]
    assert cfg["gates"] and cfg["requires_modal"] and cfg["trials"] == 2 and "arms" not in cfg
    assert cfg["tasks"]() == tools_ab.banking_test_tasks() and len(cfg["tasks"]()["banking_knowledge"]) == 20
    assert cfg["data"] != tools_ab.SETS["banking-test"]["data"]  # never the ungated split run's folder
    # the gated arm again after PRs 29-44 (back-on-track step 4): tools arm only, its own folder, smaller caps
    v2 = tools_ab.SETS["banking-test-v2-gated"]
    assert v2["gates"] and v2["requires_modal"] and v2["trials"] == 2 and v2["arms"] == ("tools",)
    assert v2["tasks"]() == tools_ab.banking_test_tasks()
    assert v2["data"] not in (cfg["data"], tools_ab.SETS["banking-test"]["data"])
    assert (v2["gpu_usd"], v2["glimmer_usd"], v2["meta_usd"]) == (6.0, 1.5, 0.5)
