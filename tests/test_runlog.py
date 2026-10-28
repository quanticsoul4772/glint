"""glint.runlog on a stand-in τ² checkpoint, and the runner passing banking runs a checkpoint path."""
import json
import sys
import time
from pathlib import Path

import pytest

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


def test_outcome_tells_glint_errors_cap_stops_and_missing_rewards_apart():
    from glint.atem import AtemParseError
    from glint.calllog import CallLogError
    from glint.meter import BudgetExceeded
    from glint.runlog import outcome

    def infra(err):
        return {"termination_reason": "infrastructure_error", "info": {"error": str(err)}}

    assert outcome(infra(BudgetExceeded("Meta API spend $1 reached cap $1"))) == "budget_stop"
    assert outcome(infra(AtemParseError("host calls [] != generated"))) == "glint_error"
    assert outcome(infra(CallLogError("no token ids"))) == "glint_error"
    assert outcome(infra("ContextWindowExceededError: too long")) == "context_overflow"
    assert outcome(infra("ConnectionError: reset")) == "infrastructure_error"
    assert outcome({"termination_reason": "user_stop", "reward_info": None}) == "eval_error"
    assert outcome({"termination_reason": "user_stop", "reward_info": {"reward": 1.0}}) == "pass"
    assert outcome({"termination_reason": "user_stop", "reward_info": {"reward": 0.0}}) == "fail"


def test_a_locked_checkpoint_is_retried_not_fatal(tmp_path, monkeypatch, capsys):
    # 2026-10-27: Windows refused the read while tau2 rewrote results.json and the watcher thread died
    import json
    from pathlib import Path

    from glint.runlog import RunEvents

    results = tmp_path / "results.json"
    results.write_text(json.dumps({"simulations": []}), encoding="utf-8")
    events = RunEvents(results, tmp_path / "events.jsonl", label="t", poll_s=60)
    real = Path.read_text

    def locked(self, *a, **k):
        if self == results:
            raise PermissionError(13, "Permission denied", str(results))
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", locked)
    for _ in range(3):
        assert events.poll() == []
    assert events.read_errors == 3 and "3 checkpoint reads in a row failed" in capsys.readouterr().err
    monkeypatch.setattr(Path, "read_text", real)
    results.write_text(json.dumps({"simulations": [{"task_id": "1", "trial": 0, "termination_reason": "user_stop",
                                                    "reward_info": {"reward": 1.0}, "messages": []}]}),
                       encoding="utf-8")
    assert len(events.poll()) == 1  # back to normal once the file reads again


def _watch(tmp_path, sims):
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"simulations": sims}), encoding="utf-8")
    return RunEvents(results, tmp_path / "events.jsonl", label="tools banking", poll_s=60)


def test_repeated_infrastructure_errors_raise_one_alert_then_one_per_doubling(tmp_path, capsys):
    drop = lambda task, trial: _sim(task, trial, term="infrastructure_error", error="APIError: Error code: 303")
    w = _watch(tmp_path, [drop("t1", 0), drop("t2", 0)])
    w.poll()
    assert "ALERT" not in capsys.readouterr().err  # 2 is below the threshold of 3
    (tmp_path / "results.json").write_text(json.dumps({"simulations": [drop("t1", 0), drop("t2", 0), drop("t2", 1)]}),
                                          encoding="utf-8")
    w.poll()
    err = capsys.readouterr().err
    assert "ALERT: 3 infrastructure_error runs of 3 finished (100%), tasks {'t1': 1, 't2': 2}" in err
    assert "latest: APIError: Error code: 303" in err
    (tmp_path / "results.json").write_text(json.dumps({"simulations": [
        drop("t1", 0), drop("t2", 0), drop("t2", 1), drop("t3", 0), _sim("t4", 0, 1.0)]}), encoding="utf-8")
    w.poll()
    assert "ALERT" not in capsys.readouterr().err  # 4 is not yet double 3
    (tmp_path / "results.json").write_text(json.dumps({"simulations": [
        drop("t1", 0), drop("t2", 0), drop("t2", 1), drop("t3", 0), _sim("t4", 0, 1.0), drop("t5", 0),
        drop("t6", 0)]}), encoding="utf-8")
    w.poll()
    assert "ALERT: 6 infrastructure_error runs" in capsys.readouterr().err


def test_context_overflows_alert_on_their_share_not_their_count(tmp_path, capsys):
    over = lambda task: _sim(task, 0, term="infrastructure_error", error="ContextWindowExceededError")
    w = _watch(tmp_path, [over("t0")] + [_sim(f"p{i}", 0, 0.0) for i in range(8)])
    w.poll()
    assert "ALERT" not in capsys.readouterr().err  # 9 finished: below the 10-run minimum
    (tmp_path / "results.json").write_text(json.dumps({"simulations": [over("t0")] + [
        _sim(f"p{i}", 0, 0.0) for i in range(8)] + [over("t1"), _sim("p9", 0, 0.0)]}), encoding="utf-8")
    w.poll()
    err = capsys.readouterr().err
    assert "ALERT: 2 context_overflow runs of 11 finished (18%)" in err and "'t0': 1, 't1': 1" in err


def test_the_end_of_arm_summary_lists_failures_by_task(tmp_path, capsys):
    over = _sim("t1", 2, term="infrastructure_error", error="ContextWindowExceededError")
    with _watch(tmp_path, []) as w:
        (tmp_path / "results.json").write_text(json.dumps({"simulations": [over, _sim("t2", 0, 1.0)]}),
                                              encoding="utf-8")
    err = capsys.readouterr().err
    assert "context_overflow by task: {'t1': 1}" in err and "pass by task" not in err
    assert w.tasks_by_outcome == {"context_overflow": {"t1": 1}, "pass": {"t2": 1}}


def test_a_plain_read_would_block_the_save(tmp_path):
    # the cause, pinned: if this stops failing, Windows or Python changed and read_shared may not be needed
    import os

    target, temp = tmp_path / "results.json", tmp_path / ".results_tmp.json"
    target.write_text("{}", encoding="utf-8")
    temp.write_text("{}", encoding="utf-8")
    if sys.platform != "win32":
        pytest.skip("Windows-only behavior")
    with open(target, encoding="utf-8"), pytest.raises(PermissionError):
        os.replace(temp, target)
