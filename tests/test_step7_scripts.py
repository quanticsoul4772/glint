"""scripts/by_task.py and scripts/log_review.py on small fixtures."""
import json

import by_task
import log_review


def _sim(task, trial, kind):
    term = {"pass": "user_stop", "fail": "user_stop", "infra": "infrastructure_error"}[kind]
    reward = {"pass": 1.0, "fail": 0.0, "infra": None}[kind]
    return {"task_id": task, "trial": trial, "termination_reason": term,
            "reward_info": {"reward": reward} if reward is not None else None,
            "info": {"error": "HTTP 503"} if kind == "infra" else {}, "messages": [{}]}


def test_by_task_keeps_tuned_tasks_apart_and_counts_non_results_separately(tmp_path, capsys):
    a, b = tmp_path / "tools_ab_banking12_a_modal", tmp_path / "tools_ab_banking12_b_modal"
    for folder, sims in ((a, [_sim("task_031", 0, "pass"), _sim("task_031", 1, "fail"), _sim("task_039", 0, "pass")]),
                         (b, [_sim("task_031", 0, "pass"), _sim("task_031", 1, "infra")])):
        folder.mkdir()
        (folder / "tools-banking_knowledge.json").write_text(json.dumps({"simulations": sims}), encoding="utf-8")
    rows = by_task.table([a, b], "banking_knowledge")
    assert rows["task_031"] == {"banking12_a:tools": [1, 2, 0], "banking12_b:tools": [1, 1, 1]}
    assert by_task.main([str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "1/1 (1 not results)" in out and "UNTUNED" in out and "TUNED" in out
    untuned = next(line for line in out.splitlines() if line.startswith("UNTUNED"))
    tuned = next(line for line in out.splitlines() if line.startswith("TUNED"))
    # 031 is tuned by default since 2026-10-30
    assert untuned.split() == ["UNTUNED", "0/0", "0/0"] and tuned.split() == ["TUNED", "2/3", "1/1"]
    assert by_task.main([str(a), "--tuned", "task_031"]) == 0
    assert next(x for x in capsys.readouterr().out.splitlines() if x.startswith("TUNED")).split() == ["TUNED", "1/2"]
    assert by_task.main([]) == 2 and by_task.main([str(tmp_path)]) == 1


def test_by_task_prints_pass_k_wilson_intervals_and_the_sign_test_floor(tmp_path, capsys):
    a = tmp_path / "tools_ab_banking12_a_modal"
    a.mkdir()
    sims = [_sim("task_031", 0, "pass"), _sim("task_031", 1, "pass"), _sim("task_039", 0, "pass"),
            _sim("task_039", 1, "fail"), _sim("task_050", 0, "fail")]
    (a / "tools-banking_knowledge.json").write_text(json.dumps({"simulations": sims}), encoding="utf-8")
    assert by_task.main([str(a), "--tuned", "task_031,task_039"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert next(x for x in out if x.startswith("tuned pass^k")).split()[-2:] == ["1/2", "tasks"]
    assert next(x for x in out if x.startswith("untuned pass^k")).split()[-2:] == ["0/1", "tasks"]
    assert next(x for x in out if x.startswith("tuned 95% CI")).split()[-1] == "30%-95%"  # Wilson for 3/4
    assert "no result on 1 tasks can reach p < 0.05 by sign test (best p = 0.500)" in "\n".join(out)
    assert by_task.wilson(0, 0) == (0.0, 1.0)
    assert by_task.sign_test_floor(9) == 5 and by_task.sign_test_floor(4) is None


def test_log_review_reports_missed_error_lines_and_returned_noise(tmp_path, capsys):
    log = tmp_path / "run.log"
    lines = ["[serve 10:00:00] session start: GPU spent so far $0.000 of cap $8.00",
             "[runs 10:05:00 tools banking] task_031 trial 0: FAIL (τ²: user_stop, 400 s, 50 msgs; so far fail 1)",
             "[runs 10:06:00 tools banking] task_031 trial 1: CONTEXT_OVERFLOW (τ²: infrastructure_error, ...)",
             "[serve 10:07:00] something timed out quietly",
             "GATE OUTCOMES: tools banking: procedure.lookups: 2 denies; failed runs differ on missing writes",
             "Status: 2/48 complete. Avg reward: 0.0 (N=2). 12 running: task_031.2(60s)"]
    log.write_text("\n".join(lines), encoding="utf-8")
    assert log_review.main([str(log)]) == 0
    out = capsys.readouterr().out
    assert "error-like lines the monitor does not match: 1" in out and "something timed out quietly" in out
    assert "CONTEXT_OVERFLOW" not in out.split("does not match")[1].split("noise")[0]  # the monitor sees it
    assert "τ² Status lines (one per 10 minutes expected): 1" in out
    log.write_text(log.read_text(encoding="utf-8") + "\nProvider List: https://docs.litellm.ai/docs/providers\n",
                   encoding="utf-8")
    assert log_review.main([str(log)]) == 1
    assert "BACK LiteLLM 'Provider List' blocks: 1" in capsys.readouterr().out
    assert log_review.main([]) == 2
