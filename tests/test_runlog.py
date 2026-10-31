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


# --- Mutation-testing gaps (cosmic-ray on main, 2026-10-28: 60 of 165 mutants survived) -----------------------------
def test_event_fields():
    from glint.runlog import event

    s = _sim("t9", 2, 1.0, term="TerminationReason.USER_STOP", error="x" * 400) | {"duration": 12.34,
                                                                                  "messages": [{}, {}, {}]}
    e = event(s, "lbl")
    assert (e["label"], e["task_id"], e["trial"], e["reward"], e["termination"], e["duration_s"], e["messages"]) == (
        "lbl", "t9", 2, 1.0, "USER_STOP", 12.3, 3)
    assert e["error"] == "x" * 300
    bare = event({"task_id": "t0", "trial": 0, "termination_reason": "user_stop", "reward_info": None}, "lbl")
    assert (bare["reward"], bare["error"], bare["duration_s"], bare["messages"]) == (None, None, 0.0, 0)


def test_counts_add_up_per_outcome_and_task(tmp_path, capsys):
    results = tmp_path / "results.json"
    watch = RunEvents(results, tmp_path / "e.jsonl", label="x")
    results.write_text(json.dumps({"simulations": [_sim("a", 0, 0.0), _sim("a", 1, 0.0), _sim("b", 0, 0.0),
                                                   _sim("b", 1, 1.0)]}), encoding="utf-8")
    watch.poll()
    assert watch.counts == {"fail": 3, "pass": 1}
    assert watch.tasks_by_outcome == {"fail": {"a": 2, "b": 1}, "pass": {"b": 1}}
    err = capsys.readouterr().err
    assert "b trial 1: PASS (τ²: user_stop, " in err and "; so far fail 3, pass 1)" in err
    assert "totals" not in err


def test_a_run_tau_kept_nothing_of_does_not_print_zero_seconds_and_messages(tmp_path, capsys):
    results = tmp_path / "results.json"
    watch = RunEvents(results, tmp_path / "e.jsonl", label="x")
    empty = _sim("a", 0, term="infrastructure_error", error="boom") | {"duration": 0.0, "messages": []}
    results.write_text(json.dumps({"simulations": [empty]}), encoding="utf-8")
    watch.poll()
    err = capsys.readouterr().err
    assert "a trial 0: INFRASTRUCTURE_ERROR (τ²: infrastructure_error, τ² kept no messages or time; so far " in err
    assert "0 s" not in err and "0 msgs" not in err


def test_stall_warning_only_after_the_stall_time(tmp_path, capsys):
    watch = RunEvents(tmp_path / "results.json", tmp_path / "e.jsonl", label="x", stall_s=3540.0)
    watch.poll()  # nothing new, but not stalled yet
    assert "no run finished" not in capsys.readouterr().err
    watch._last_new = time.time() - 4000
    watch.poll()
    assert "WARNING: no run finished in 59 min (0 so far)" in capsys.readouterr().err


def test_failed_reads_warn_at_3_and_10_in_a_row_and_reset_after_a_good_read(tmp_path, capsys):
    results = tmp_path / "results.json"
    watch = RunEvents(results, tmp_path / "e.jsonl", label="x")
    warnings = lambda: capsys.readouterr().err.count("checkpoint reads in a row failed")
    results.write_text("{ half", encoding="utf-8")
    for n in range(1, 11):
        watch.poll()
        if n == 2:
            assert warnings() == 0
        if n == 3:
            assert warnings() == 1
        if n == 9:
            assert warnings() == 0
    assert warnings() == 1 and watch.read_errors == 10  # the 10th
    results.write_text(json.dumps({"simulations": []}), encoding="utf-8")
    watch.poll()  # a good read resets the streak
    results.write_text("{ half", encoding="utf-8")
    watch.poll(), watch.poll()
    assert warnings() == 0  # two bad reads after the reset: not yet a streak of 3


def _write(results, sims):
    results.write_text(json.dumps({"simulations": sims}), encoding="utf-8")


def test_fail_fast_after_runs_in_a_row_with_the_same_cause(tmp_path, capsys):
    from glint import runlog

    results, reasons = tmp_path / "results.json", []
    watch = RunEvents(results, tmp_path / "e.jsonl", label="x", on_halt=reasons.append)
    n = runlog.HALT_SAME_CAUSE["infrastructure_error"]
    infra = lambda t, err="HTTP 303 to /x": _sim(t, 0, term="infrastructure_error", error=err)
    overflow = _sim("o", 0, term="infrastructure_error", error="litellm.ContextWindowExceededError: too long")
    # n - 1 in a row, then a different cause, a pass and an overflow each restart the count
    sims = [infra(f"a{i}") for i in range(n - 1)] + [infra("b", "HTTP 503")] + [infra(f"c{i}") for i in range(n - 1)]
    sims += [_sim("d", 0, 1.0)] + [infra(f"e{i}") for i in range(n - 1)] + [overflow] + [infra(f"f{i}") for i in range(n - 1)]
    _write(results, sims)
    watch.poll()
    assert reasons == [] and watch.halted is None
    sims.append(infra("g"))
    _write(results, sims)
    watch.poll()
    assert reasons == [f"{n} runs in a row ended infrastructure_error with the same cause: HTTP # to /x"]
    assert f"FAIL FAST: {reasons[0]}; stopping the arm" in capsys.readouterr().err
    _write(results, sims + [infra("h")])
    watch.poll()
    assert len(reasons) == 1  # once per arm


def test_glint_errors_stop_sooner_and_model_results_never_stop(tmp_path):
    from glint import runlog

    results, reasons = tmp_path / "results.json", []
    watch = RunEvents(results, tmp_path / "e.jsonl", label="x", on_halt=reasons.append)
    overflow = [_sim(f"o{i}", 0, term="infrastructure_error", error="ContextWindowExceededError") for i in range(20)]
    _write(results, overflow)
    watch.poll()
    assert reasons == []
    glint = [_sim(f"g{i}", 0, term="infrastructure_error", error="AtemParseError: no tool call parsed")
             for i in range(runlog.HALT_SAME_CAUSE["glint_error"])]
    assert all(runlog.outcome(s) == "glint_error" for s in glint)
    _write(results, overflow + glint)
    watch.poll()
    assert len(reasons) == 1 and "glint_error" in reasons[0]


def test_fail_fast_after_two_stall_windows_in_a_row(tmp_path, capsys):
    results, reasons = tmp_path / "results.json", []
    watch = RunEvents(results, tmp_path / "e.jsonl", label="x", stall_s=1200.0, on_halt=reasons.append)
    watch._last_new = time.time() - 1300
    watch.poll()
    assert reasons == [] and "WARNING: no run finished in 20 min" in capsys.readouterr().err
    _write(results, [_sim("a", 0, 0.0)])  # a finished run resets the count
    watch.poll()
    watch._last_new = time.time() - 1300
    watch.poll()
    assert reasons == []
    watch._last_new = time.time() - 1300
    watch.poll()
    assert reasons == ["no run finished in 40 min"]


def test_the_cause_key_ignores_quoted_model_text_and_numbers():
    from glint.runlog import cause_key

    a = "AtemParseError: invoke without a name: '<invoke>get_user 123</invoke>'"
    b = "AtemParseError: invoke without a name: '<invoke>find_order #W9</invoke>'"
    assert cause_key(a) == cause_key(b) == "AtemParseError: invoke without a name: '…'"
    assert cause_key("HTTP 303 to port 8000") == cause_key("HTTP 303 to port 8443") != cause_key("HTTP 503")
    assert cause_key(None) == ""


def test_a_resumed_arm_reports_the_reruns_of_runs_already_in_the_checkpoint(tmp_path):
    # τ²'s resume reruns infrastructure_error and budget_stop runs under the same (task, trial) and a new id
    results, reasons = tmp_path / "results.json", []
    stopped = [_sim(f"t{i}", 0, term="infrastructure_error", error="BudgetExceeded: stopped early (fail fast): x")
               | {"id": f"old-{i}"} for i in range(6)]
    _write(results, stopped)
    watch = RunEvents(results, tmp_path / "e.jsonl", label="x", on_halt=reasons.append)
    with watch:
        reruns = [_sim(f"t{i}", 0, term="infrastructure_error", error="HTTP 303 to /x") | {"id": f"new-{i}"}
                  for i in range(6)]
        _write(results, reruns)
        watch.poll()
    assert watch.counts == {"infrastructure_error": 6} and len(reasons) == 1
    assert len((tmp_path / "e.jsonl").read_text(encoding="utf-8").splitlines()) == 6


def test_a_pattern_seen_only_after_the_arm_finished_warns_and_keeps_the_arm(tmp_path, capsys):
    from glint import runlog

    results, reasons = tmp_path / "results.json", []
    watch = RunEvents(results, tmp_path / "e.jsonl", label="x", on_halt=reasons.append)
    with watch:
        tail = [_sim(f"t{i}", 0, term="infrastructure_error", error="HTTP 503")
                for i in range(runlog.HALT_SAME_CAUSE["infrastructure_error"])]
        _write(results, [_sim(f"p{i}", 0, 1.0) for i in range(20)] + tail)
    assert reasons == [] and watch.halted is None
    assert "fail-fast pattern at the arm's end" in capsys.readouterr().err


def test_a_long_run_that_keeps_calling_the_model_is_not_a_stall(tmp_path, capsys):
    # task_053 trial 3 ran 42+ minutes alone at an arm's tail, calling the model every few seconds (2026-10-29)
    results, reasons, last = tmp_path / "results.json", [], [time.time()]
    watch = RunEvents(results, tmp_path / "e.jsonl", label="x", stall_s=1200.0, on_halt=reasons.append,
                      activity=lambda: last[0])
    for _ in range(4):  # four stall windows with calls still being made
        watch._last_new = time.time() - 1300
        last[0] = time.time() - 5
        watch.poll()
    assert reasons == [] and watch._stalls == 0
    assert "model calls are still being made (last 5 s ago)" in capsys.readouterr().err
    for _ in range(2):  # then the calls stop too
        watch._last_new = time.time() - 1300
        last[0] = time.time() - 1500
        watch.poll()
    assert reasons == ["no run finished in 40 min and no model call in 25 min"]


def test_endings_long_runs_and_the_heartbeat_are_reported(tmp_path, capsys):
    # 2026-10-30 review: max_steps showed only as a fail, a 61-minute run (task_018) went unflagged, and a dead driver
    # left no line a watcher could miss
    results = tmp_path / "results.json"
    now = time.time()
    clock = {("t1", "0"): (now - 3600, 150, 90000, now), ("t2", "0"): (now - 3600, 9, 5000, now),
             ("t3", "0"): (now - 60, 3, 1000, now)}
    results.write_text(json.dumps({"simulations": []}), encoding="utf-8")
    with RunEvents(results, tmp_path / "e.jsonl", label="tools banking", poll_s=60, running=lambda: clock,
                   activity=lambda: now - 5, heartbeat_s=0.0) as watch:
        results.write_text(json.dumps({"simulations": [_sim("t2", 0, 0.0, term="max_steps")]}), encoding="utf-8")
        watch.poll()
        watch.poll()  # once per run
        watch._beat()
    err = capsys.readouterr().err
    assert err.count("WARNING: t1 trial 0 has run 60 min (150 model calls) without finishing") == 1
    assert "t2 trial 0 has run" not in err and "t3 trial 0 has run" not in err  # t2 finished, t3 is young
    assert "ALIVE: 1 finished (fail 1); 2 running, oldest 60 min; last model call 5 s ago" in err
    assert "done: {'fail': 1}; endings {'max_steps': 1}" in err


def test_a_run_tau_kept_nothing_of_gets_the_call_log_and_no_diff(tmp_path, capsys):
    # 2026-10-30 review: overflows recorded 0 s, 0 messages and peak 0, and their diffs of an empty conversation fed
    # the DIFFS line and the repeated-field warnings
    results, events = tmp_path / "results.json", tmp_path / "e.jsonl"
    now = time.time()
    overflow = {"task_id": "t9", "trial": 1, "termination_reason": "infrastructure_error", "reward_info": None,
                "info": {"error": "litellm.ContextWindowExceededError: maximum context length is 131072 tokens"},
                "duration": None, "messages": []}
    results.write_text(json.dumps({"simulations": [overflow]}), encoding="utf-8")
    watch = RunEvents(results, events, label="base banking", poll_s=60,
                      running=lambda: {("t9", "1"): (now - 600, 41, 121000, now - 30)})
    diffed = []
    watch.diff = lambda sim: diffed.append(sim) or {"text": "missing 7"}
    watch.poll()
    row = json.loads(events.read_text(encoding="utf-8"))
    assert row["outcome"] == "context_overflow" and diffed == [] and "diff" not in row
    assert row["from_calls"] == {"model_calls": 41, "seconds": 570.0, "peak_prompt_tokens": 121000}
    assert ("τ² kept no messages or time; call log: 41 model calls over 570 s, peak prompt 121000 tokens"
            in capsys.readouterr().err)
