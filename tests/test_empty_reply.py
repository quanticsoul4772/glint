"""An empty Glimmer reply (no content, no tool call) is a failed run, like a context overflow (decision 2026-10-28):
glint.runlog.outcome, tools_ab (analyze, the resume keeper, the release, the arm warning, the EMPTY REPLY line),
export_results, gate_labels, gate_catch and the RunEvents loud line and ALERT share. Nothing here calls a model."""
import json
import types

import litellm
import tools_ab
from loguru import logger
from test_generate import stand_in  # noqa: F401  (the fixture: tools_ab.generate with τ²'s batch replaced)

from glint.runlog import RunEvents, outcome

logger.remove()

# τ²'s own text for the run it ends (tau2.data_model.message.Message.validate), as saved in banking-gated 2026-10-28
EMPTY_ERROR = "AssistantMessage must have either content or tool_calls. Got AssistantMessage\nis None"


def empty_sim(task_id: str, trial: int) -> dict:
    return {"task_id": task_id, "trial": trial, "termination_reason": "infrastructure_error",
            "info": {"error": EMPTY_ERROR}, "reward_info": None, "messages": [], "duration": 1.0}


def test_outcome_names_an_empty_reply_and_keeps_other_infrastructure_errors_apart(make_sim):
    assert outcome(empty_sim("t", 0)) == "empty_reply"
    assert outcome(make_sim("t", 0, "infrastructure_error")) == "infrastructure_error"
    assert outcome(make_sim("t", 0, "context_overflow")) == "context_overflow"


def test_analyze_counts_an_empty_reply_as_a_failed_run(run_set, make_sim, capsys):
    cfg = run_set({("base", "retail"): [make_sim("1", 0), empty_sim("1", 1)],
                   ("tools", "retail"): [make_sim("1", 0), make_sim("1", 1)]}, k=2)
    tools_ab.analyze(cfg)
    base = json.loads(cfg["out"].read_text(encoding="utf-8"))["retail-1"]["base"]
    assert base["runs"] == 2 and base["passes"] == 1 and base["empty_replies"] == 1 and base["infra_errors"] == 0


def test_export_results_counts_empty_replies_and_a_task_of_only_model_failures_has_no_data(
        run_set, make_sim, monkeypatch, tmp_path):
    import export_results

    cfg = run_set({("base", "banking_knowledge"): [empty_sim("a", 0), make_sim("b", 0, "fail")],
                   ("tools", "banking_knowledge"): [make_sim("a", 0, "context_overflow"), empty_sim("b", 0)]}, k=1)
    monkeypatch.setattr(tools_ab, "SETS", {"x": {}})
    monkeypatch.setattr(tools_ab, "for_backend", lambda c, modal: cfg)
    monkeypatch.setattr(export_results, "SETS", [("x", True, "X", "untuned", "note")])
    monkeypatch.setattr(export_results, "OUT", tmp_path / "results.json")
    export_results.main()
    (s,) = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    rows = {r["task"].split("-")[-1]: r for r in s["tasks"]}
    assert rows["a"]["change"] == "no data" and rows["a"]["empty_replies"] == 1
    assert rows["a"]["no_data_reason"] == "every run overflowed the context or returned an empty reply"
    assert rows["b"]["change"] == "tied" and rows["b"]["tools"] == "0/1"  # counted as a failed run, not dropped


def test_gate_labels_and_wrong_writes_and_gate_catch_treat_it_as_failed():
    import gate_catch

    from glint.runlog import NOT_RESULTS

    assert "empty_reply" not in NOT_RESULTS  # gate_labels, wrong_writes and export_scenarios keep it as a run
    assert "empty_reply" in gate_catch.FAILED


def test_release_keeps_empty_replies_and_takes_out_only_infrastructure_errors(run_set, make_sim, monkeypatch,
                                                                              tmp_path):
    sims = [make_sim("task_1", 0), empty_sim("task_1", 1), make_sim("task_2", 0, "infrastructure_error")]
    cfg = run_set({("base", "banking_knowledge"): sims}, k=2) | {"arms": ("base",)}
    (cfg["data"] / "tasks.json").write_text(json.dumps({"banking_knowledge": ["task_1", "task_2"]}), encoding="utf-8")
    monkeypatch.setattr(tools_ab, "_save_dir", lambda c, arm, d, stamp: tmp_path / "ckpt" / arm)
    (tmp_path / "ckpt" / "base").mkdir(parents=True)
    checkpoint = tmp_path / "ckpt" / "base" / "results.json"
    checkpoint.write_text(json.dumps({"simulations": sims}), encoding="utf-8")
    assert tools_ab.release_infrastructure_errors(cfg, "s", "abc1234") == 1
    kept = json.loads(checkpoint.read_text(encoding="utf-8"))["simulations"]
    assert ("task_1", 1) in {(s["task_id"], s["trial"]) for s in kept}
    assert ("task_2", 0) not in {(s["task_id"], s["trial"]) for s in kept}


def test_resume_keeps_an_empty_reply_as_recorded(tmp_path, monkeypatch, capsys):
    from tau2.data_model.simulation import Results, TerminationReason
    from tau2.runner import batch, checkpoint

    shipped = Results.load(tools_ab.paths.tau2_data() / "tau2" / "results" / "final" /
                           "gpt-4.1-2025-04-14_retail_default_gpt-4.1-2025-04-14_4trials.json")
    tasks = shipped.tasks[:1]
    sims = [s for s in shipped.simulations if s.task_id == tasks[0].id]
    sims[0].termination_reason, sims[0].reward_info = TerminationReason.INFRASTRUCTURE_ERROR, None
    sims[0].info = {"error": EMPTY_ERROR}
    save = tmp_path / "results.json"
    save.write_text(Results(info=shipped.info, tasks=tasks, simulations=sims).model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(batch, "try_resume", checkpoint.try_resume)
    tools_ab.keep_failed_runs_on_resume()
    _, done, _ = batch.try_resume(save, Results(info=shipped.info, tasks=tasks, simulations=[]), tasks, 4, True)
    assert (sims[0].trial, sims[0].task_id, sims[0].seed) in done  # never rerun
    assert sims[0].id in {s.id for s in Results.load(save).simulations}
    assert "KEPT 1 failed runs (context overflow, empty reply or Glint error)" in capsys.readouterr().out


def test_the_run_line_is_loud_and_empty_replies_alert_on_their_share(tmp_path, capsys):
    results = tmp_path / "results.json"
    passed = [{"task_id": f"p{i}", "trial": 0, "termination_reason": "user_stop", "reward_info": {"reward": 1.0},
               "messages": []} for i in range(8)]
    results.write_text(json.dumps({"simulations": [empty_sim("t0", 0), empty_sim("t1", 0)] + passed}),
                       encoding="utf-8")
    RunEvents(results, tmp_path / "e.jsonl", label="tools banking", poll_s=60).poll()
    err = capsys.readouterr().err
    assert "t0 trial 0: EMPTY_REPLY (τ²: infrastructure_error" in err and "<-- AssistantMessage must have" in err
    assert "ALERT: 2 empty_reply runs of 10 finished (20%)" in err


def _empty_response(finish_reason="stop", usage=True):
    response = litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": ""},
                                               "finish_reason": finish_reason}])
    if usage:
        response.usage = litellm.Usage(prompt_tokens=31000, completion_tokens=812,
                                       completion_tokens_details={"reasoning_tokens": 812})
    else:  # a response without token counts
        return types.SimpleNamespace(choices=response.choices)
    return response


def test_empty_reply_line_names_the_run_its_finish_reason_and_tokens(monkeypatch):
    monkeypatch.setattr(tools_ab, "current_run", lambda: ("task_050", "3"))
    assert tools_ab.empty_reply_line(_empty_response()) == (
        "EMPTY REPLY: task_050 trial 3: finish_reason stop, prompt 31000 tokens, completion 812 (reasoning 812)")
    assert tools_ab.empty_reply_line(_empty_response("length", usage=False)) == (
        "EMPTY REPLY: task_050 trial 3: finish_reason length, no token counts")
    text = litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": " hi "}, "finish_reason": "stop"}])
    call = litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}])
    assert tools_ab.empty_reply_line(text) is None and tools_ab.empty_reply_line(call) is None


def test_generate_prints_the_empty_reply_line_and_warns_at_the_arm_end(stand_in, make_sim, monkeypatch,  # noqa: F811
                                                                       capsys):
    from tau2.utils import llm_utils

    cfg, outcomes = stand_in
    outcomes["base"], outcomes["tools"] = [make_sim("1", 0), empty_sim("1", 1)], [make_sim("1", 0), make_sim("1", 1)]
    monkeypatch.setattr(llm_utils, "completion", lambda **kw: _empty_response())

    def run(cfg_, domain, ids, arm, stamp, endpoint):
        if arm == "base":
            llm_utils.completion(model="openai/Muse-Glimmer-30B-modal", messages=[])
        return types.SimpleNamespace(
            simulations=[types.SimpleNamespace(model_dump=lambda mode=None, s=s: s) for s in outcomes[arm]],
            model_dump_json=lambda: json.dumps({"simulations": outcomes[arm]}))

    monkeypatch.setattr(tools_ab, "_run", run)
    tools_ab.generate(cfg)
    out = capsys.readouterr().out
    assert "EMPTY REPLY: unknown trial unknown: finish_reason stop, prompt 31000 tokens" in out
    assert "WARNING: base retail: 1 of 2 runs ended on an empty Glimmer reply" in out
    assert "WARNING: tools retail: 0" not in out



def test_run_context_names_the_attempt_by_its_simulation_id(tmp_path):
    # the gate log keyed runs by (task, trial) as text, events.jsonl by number and simulation id (2026-10-30 review)
    import tools_ab
    from tau2.utils import llm_utils

    token = tools_ab.CURRENT_RUN.set(("task_050", 3))
    log_token = llm_utils.llm_log_dir.set(tmp_path / "artifacts" / "task_task_050" / "sim_ab12" / "llm_debug")
    try:
        assert tools_ab.run_context() == {"task_id": "task_050", "trial": 3, "simulation_id": "ab12"}
    finally:
        llm_utils.llm_log_dir.reset(log_token)
        tools_ab.CURRENT_RUN.reset(token)
    assert tools_ab.run_context()["simulation_id"] is None  # outside a run
