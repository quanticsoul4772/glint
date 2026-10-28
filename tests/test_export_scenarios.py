"""export_scenarios on stand-in runs of a real τ² banking task: the pair it picks and the counts it shows."""
import json

from loguru import logger

logger.remove()

import export_scenarios
import tools_ab


def _talk(make_sim, trial, kind):
    messages = [{"role": "user", "content": "I need a limit increase"},
                {"role": "assistant", "content": "Let me look.", "tool_calls": None}]
    return make_sim("task_050", trial, kind, messages=messages)


def test_replay_pairs_a_failed_alone_run_with_a_passed_tools_run_and_counts_like_the_results_tab(
        run_set, make_sim, monkeypatch, tmp_path):
    base = [_talk(make_sim, 0, "infrastructure_error"), _talk(make_sim, 1, "fail"), _talk(make_sim, 2, "pass")]
    tools = [_talk(make_sim, 0, "fail"), _talk(make_sim, 1, "pass"), _talk(make_sim, 2, "pass")]
    cfg = run_set({("base", "banking_knowledge"): base, ("tools", "banking_knowledge"): tools}, k=2,
                  name="tools_ab_banking_other_modal")
    monkeypatch.setattr(tools_ab, "for_backend", lambda c, modal: cfg)
    monkeypatch.setattr(export_scenarios, "DATA", cfg["data"].parent)
    monkeypatch.setattr(export_scenarios, "SOURCES", [("task_050", "banking-other", False)])
    monkeypatch.setattr(export_scenarios, "OUT", tmp_path / "scenarios.json")
    export_scenarios.main()
    (s,) = json.loads((tmp_path / "scenarios.json").read_text(encoding="utf-8"))
    # trial 0 of the base arm is an infrastructure error: never shown as Glimmer's failure; trial 2 is beyond k=2
    assert s["alone"]["trial"] == 1 and s["tools"]["trial"] == 1
    assert s["recorded"] == {"base": {"passed": 0, "runs": 1}, "tools": {"passed": 1, "runs": 2}}
    assert s["tuned"] is False and s["alone"]["events"][0] == {"kind": "customer", "text": "I need a limit increase"}


def test_a_task_without_a_pair_is_left_out_loudly(run_set, make_sim, monkeypatch, tmp_path, capsys):
    sims = [_talk(make_sim, 0, "pass")]
    cfg = run_set({("base", "banking_knowledge"): sims, ("tools", "banking_knowledge"): sims}, k=1,
                  name="tools_ab_banking_other_modal")
    monkeypatch.setattr(tools_ab, "for_backend", lambda c, modal: cfg)
    monkeypatch.setattr(export_scenarios, "DATA", cfg["data"].parent)
    monkeypatch.setattr(export_scenarios, "SOURCES", [("task_050", "banking-other", False)])
    monkeypatch.setattr(export_scenarios, "OUT", tmp_path / "scenarios.json")
    export_scenarios.main()
    assert "task_050 left out of the replay" in capsys.readouterr().out
    assert json.loads((tmp_path / "scenarios.json").read_text(encoding="utf-8")) == []
