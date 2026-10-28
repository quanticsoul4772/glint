"""glint.runlog on a stand-in τ² checkpoint, and the runner passing banking runs a checkpoint path."""
import json
import sys
import time
from pathlib import Path

from glint.runlog import RunEvents, outcome


def _sim(task, trial, reward=None, term="user_stop", error=None):
    return {"task_id": task, "trial": trial, "termination_reason": term, "reward_info": {"reward": reward},
            "info": {"error": error} if error else {}, "duration": 12.0, "messages": [{}]}


def test_outcome_separates_context_overflow_from_infrastructure():
    assert outcome(_sim("a", 0, 1.0)) == "pass" and outcome(_sim("a", 0, 0.0)) == "fail"
    overflow = _sim("a", 0, term="TerminationReason.INFRASTRUCTURE_ERROR",
                    error="litellm.ContextWindowExceededError: maximum context length is 131072 tokens")
    assert outcome(overflow) == "context_overflow"
    assert outcome(_sim("a", 0, term="infrastructure_error", error="APIConnectionError")) == "infrastructure_error"


def test_events_report_each_run_once_as_it_lands(tmp_path):
    results, events = tmp_path / "results.json", tmp_path / "events.jsonl"
    results.write_text(json.dumps({"simulations": [_sim("t1", 0, 1.0)]}), encoding="utf-8")  # resumed run
    with RunEvents(results, events, label="base banking", poll_s=0.05) as watch:
        results.write_text(json.dumps({"simulations": [_sim("t1", 0, 1.0), _sim("t2", 0, 0.0)]}), encoding="utf-8")
        deadline = time.time() + 5
        while len(watch.seen) < 2 and time.time() < deadline:
            time.sleep(0.05)
        results.write_text("{ half written", encoding="utf-8")  # caught mid-write: ignored, not a crash
        time.sleep(0.15)
        results.write_text(json.dumps({"simulations": [_sim("t1", 0, 1.0), _sim("t2", 0, 0.0),
                                                       _sim("t3", 1, term="infrastructure_error",
                                                            error="ContextWindowExceededError")]}), encoding="utf-8")
    rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    assert [(r["task_id"], r["outcome"]) for r in rows] == [("t2", "fail"), ("t3", "context_overflow")]
    assert watch.counts == {"fail": 1, "context_overflow": 1}


def test_stall_warning(tmp_path, capsys):
    results = tmp_path / "results.json"
    with RunEvents(results, tmp_path / "e.jsonl", label="x", poll_s=0.05, stall_s=0.1):
        time.sleep(0.4)
    assert "no run finished" in capsys.readouterr().err


def test_banking_runs_get_a_checkpoint_path(monkeypatch):
    # run_tasks persists nothing without save_path; banking runs had no checkpoint (2026-10-27)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import tools_ab

    seen = {}
    monkeypatch.setattr(tools_ab, "run_tasks", lambda config, tasks, **kw: seen.update(kw) or "ok")
    cfg = tools_ab.SETS["banking-other"]
    tools_ab._run(cfg, "banking_knowledge", ["task_031"], "base", "20261027-000000", None)
    assert seen["save_path"].name == "results.json"
    assert seen["save_dir"].name == "glint-tools-ab-banking-other-base-banking_knowledge-20261027-000000"
