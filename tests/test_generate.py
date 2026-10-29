"""tools_ab.generate with τ²'s batch replaced by a stand-in: arm files, warnings, the cap stop and the code guard.
Nothing here calls a model or Modal."""
import json
import types
from pathlib import Path

import pytest
from loguru import logger

logger.remove()

import tools_ab
from tau2.utils import llm_utils

from glint.meter import BudgetExceeded


class _Results:
    def __init__(self, sims):
        self.simulations = [types.SimpleNamespace(model_dump=lambda mode=None, s=s: s) for s in sims]
        self._sims = sims

    def model_dump_json(self):
        return json.dumps({"simulations": self._sims})


def _response(finish_reason: str):
    import litellm

    return litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": "x"},
                                           "finish_reason": finish_reason}])


@pytest.fixture
def stand_in(monkeypatch, tmp_path):
    """A set of 1 retail task at k=2 whose runs are `outcomes[arm]`; restores everything generate patches."""
    import litellm
    import tau2.evaluator.evaluator_nl_assertions as nl
    from tau2.runner import batch
    from tau2.utils.display import ConsoleDisplay

    for obj, attr in ((llm_utils, "completion"), (litellm, "callbacks"), (nl, "DEFAULT_LLM_NL_ASSERTIONS"),
                      (nl, "DEFAULT_LLM_NL_ASSERTIONS_ARGS"), (litellm, "suppress_debug_info"),
                      (ConsoleDisplay, "console"), (batch, "logger"), (batch, "run_with_retry")):
        monkeypatch.setattr(obj, attr, getattr(obj, attr))
    monkeypatch.setattr(tools_ab, "install", lambda domain, enforcer=None: None)
    monkeypatch.setattr(tools_ab, "_commit", lambda: "abc123")
    outcomes = {}

    def run(cfg, domain, ids, arm, stamp, endpoint):
        return _Results(outcomes[arm])

    monkeypatch.setattr(tools_ab, "_run", run)
    cfg = {"tasks": lambda: {"retail": ["1"]}, "trials": 2, "save": "gen-test", "glimmer_usd": 1.0, "meta_usd": 1.0,
           "gpu_usd": 1.0, "data": tmp_path / "set", "out": tmp_path / "report.json"}
    return cfg, outcomes


def test_generate_writes_both_arms_and_warns_about_what_is_not_a_clean_result(stand_in, make_sim, capsys):
    cfg, outcomes = stand_in
    outcomes["base"] = [make_sim("1", 0, "pass"), make_sim("1", 1, "infrastructure_error")]
    outcomes["tools"] = [make_sim("1", 0, "glint_error")]  # one run short of 1 task x k=2
    tools_ab.generate(cfg)
    assert (cfg["data"] / "base-retail.json").exists() and (cfg["data"] / "tools-retail.json").exists()
    assert (cfg["data"] / "commit.txt").read_text(encoding="utf-8") == "abc123"
    out = capsys.readouterr().out
    assert "base retail: 1 of 2 runs ended in an infrastructure error" in out
    assert "--rerun-infrastructure-errors (and --allow-mixed-code if the code changed)" in out
    assert "tools retail: 1 simulations, expected 2" in out
    assert "tools retail: 1 of 1 runs failed in Glint's own code" in out


def test_a_cap_stopped_arm_is_not_saved_as_finished(stand_in, make_sim):
    cfg, outcomes = stand_in
    outcomes["base"] = [make_sim("1", 0, "pass"), make_sim("1", 1, "budget_stop")]
    with pytest.raises(BudgetExceeded, match="stopped by a spend cap"):
        tools_ab.generate(cfg)
    assert not (cfg["data"] / "base-retail.json").exists()  # a resume reruns this arm


def test_a_resume_on_other_code_or_an_unknown_start_is_refused(stand_in, make_sim):
    cfg, _ = stand_in
    cfg["data"].mkdir(parents=True)
    (cfg["data"] / "stamp.txt").write_text("20261027-000000", encoding="utf-8")
    (cfg["data"] / "commit.txt").write_text("other", encoding="utf-8")
    with pytest.raises(SystemExit, match="started on commit other"):
        tools_ab.generate(cfg)
    (cfg["data"] / "commit.txt").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="is empty"):
        tools_ab.generate(cfg)


def test_a_finished_set_starts_nothing(stand_in, make_sim):
    cfg, _ = stand_in
    cfg["data"].mkdir(parents=True)
    for arm in ("base", "tools"):
        (cfg["data"] / f"{arm}-retail.json").write_text("{}", encoding="utf-8")
    tools_ab.generate(cfg)  # outcomes is empty: _run would raise if it were called


def test_commit_names_the_code_the_run_measures():
    commit = tools_ab._commit()  # the real repo: a short hash, "-dirty" when src has uncommitted changes
    assert commit and " " not in commit


def test_the_task_draws_reproduce_the_recorded_sets():
    # the held-out draws must not change: the recorded runs were made on exactly these tasks
    assert tools_ab.heldout_tasks() == {
        "retail": ["16", "19", "21", "57", "63", "66", "67", "92", "95", "110"],
        "airline": ["0", "11", "21", "23", "27", "33", "34", "41", "43", "46"]}
    assert tools_ab.heldout2_tasks() == {
        "retail": ["0", "6", "7", "14", "31", "43", "50", "83", "84", "99"],
        "airline": ["3", "9", "10", "15", "17", "20", "28", "38", "42", "49"]}
    assert tools_ab.banking_tasks() == {"banking_knowledge": [
        "task_031", "task_037", "task_038", "task_039", "task_040", "task_041", "task_050", "task_051", "task_052",
        "task_053", "task_054", "task_081"]}
    # the final check runs on the whole untouched test split, and on nothing the tools were built or tuned on
    test = tools_ab.banking_test_tasks()["banking_knowledge"]
    assert len(test) == 20 and not set(test) & set(tools_ab.banking_tasks()["banking_knowledge"])
    assert "task_080" in test and tools_ab.SETS["banking-test"]["tasks"] is tools_ab.banking_test_tasks


def test_the_modal_backend_keeps_its_data_apart():
    cfg = tools_ab.for_backend(tools_ab.SETS["banking-failures"], True)
    assert cfg["modal"] and cfg["data"].name == "tools_ab_banking_failures_modal"
    assert cfg["stamp_dir"].name == "tools_ab_banking_modal" and cfg["save"] == "tools-ab-banking-modal"
    assert tools_ab.for_backend(tools_ab.SETS["design"], False) is tools_ab.SETS["design"]


def test_the_agent_uses_the_endpoint_and_caps_banking_output(monkeypatch):
    monkeypatch.setenv("MODAL_PROXY_KEY", "k")
    monkeypatch.setenv("MODAL_PROXY_SECRET", "s")
    model, args = tools_ab._agent({}, "banking_knowledge", "https://x.modal.run")
    assert model.endswith("Muse-Glimmer-30B-modal") and args["max_tokens"] == 8192
    assert args["extra_headers"] == {"Modal-Key": "k", "Modal-Secret": "s"}
    model, args = tools_ab._agent({}, "retail", None)
    assert "max_tokens" not in args and "Muse-Glimmer" in model


def test_a_sets_reasoning_strength_reaches_glimmer_and_defaults_to_medium(monkeypatch):
    # decision 2026-10-28: comparisons stay at medium; banking-high-alone runs Glimmer alone at high
    monkeypatch.setenv("MODAL_PROXY_KEY", "k")
    monkeypatch.setenv("MODAL_PROXY_SECRET", "s")
    strength = lambda args: args["extra_body"]["chat_template_kwargs"]["reasoning_strength"]
    assert strength(tools_ab._agent({}, "banking_knowledge", "https://x.modal.run")[1]) == "medium"
    assert strength(tools_ab._agent({"strength": "high"}, "banking_knowledge", "https://x.modal.run")[1]) == "high"
    high = tools_ab.SETS["banking-high-alone"]
    assert high["strength"] == "high" and high["arms"] == ("base",) and high["tasks"] is tools_ab.banking_tasks
    assert high["trials"] == tools_ab.SETS["banking-gated"]["trials"]  # same tasks and k as the gated comparison
    with pytest.raises(SystemExit, match="Modal only"):
        tools_ab.for_backend(high, False)
    for name, cfg in tools_ab.SETS.items():  # every other set keeps medium, so earlier runs stay comparable
        if name != "banking-high-alone":
            assert cfg.get("strength", "medium") == "medium", name


GLIMMER_MODEL, SPARK_MODEL = "openai/meta-models/Muse-Glimmer-30B", "openai/muse-spark-1.3-contributor"


@pytest.mark.parametrize("spent_pool, refused, allowed", [("glimmer", GLIMMER_MODEL, SPARK_MODEL),
                                                          ("meta", SPARK_MODEL, GLIMMER_MODEL)])
def test_a_pool_past_its_cap_refuses_its_calls_and_only_its_calls(stand_in, make_sim, monkeypatch, tmp_path,
                                                                   spent_pool, refused, allowed):
    # generate installs two stacked guards on τ²'s completion, one per credit pool (Glimmer, Meta); a resumed run whose
    # ledger already holds a pool's cap must refuse that pool's next call before it is sent, and only that pool's
    from glint.meter import Meter

    cfg, outcomes = stand_in
    cfg["arms"] = ("base",)
    cfg["data"].mkdir(parents=True)
    (cfg["data"] / "stamp.txt").write_text("20261028-000000", encoding="utf-8")
    (cfg["data"] / "commit.txt").write_text("abc123", encoding="utf-8")
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"kind": "meta", "usd": 1.0, "phase": f"gen-test-{spent_pool}-20261028-000000"})
                      + "\n", encoding="utf-8")
    monkeypatch.setattr(tools_ab, "Meter", lambda caps, phase: Meter(caps, ledger, phase))
    sent = []
    monkeypatch.setattr(llm_utils, "completion", lambda **kw: sent.append(kw["model"]) or _response("stop"))
    outcomes["base"] = [make_sim("1", 0), make_sim("1", 1)]
    seen = {}

    def run(cfg_, domain, ids, arm, stamp, endpoint):  # the calls τ²'s batch would make, through the installed wrapper
        for model in (refused, allowed):
            try:
                llm_utils.completion(model=model, messages=[{"role": "user", "content": "hi"}])
                seen[model] = "sent"
            except BudgetExceeded as e:
                seen[model] = str(e)
        return _Results(outcomes[arm])

    monkeypatch.setattr(tools_ab, "_run", run)
    tools_ab.generate(cfg)
    assert seen[refused].startswith("BudgetExceeded: Meta API spend $1.0000 reached cap $1.0")
    assert seen[allowed] == "sent" and sent == [allowed]  # the refused call never reached the endpoint


def test_the_modal_path_deploys_waits_and_routes_glimmer_through_the_session(stand_in, make_sim, monkeypatch):
    import contextlib
    import threading

    cfg, outcomes = stand_in
    cfg["modal"] = True
    monkeypatch.setenv("MODAL_PROXY_KEY", "k")
    monkeypatch.setenv("MODAL_PROXY_SECRET", "s")
    halt, steps = threading.Event(), []

    @contextlib.contextmanager
    def session(*a, **k):
        steps.append("session")
        yield halt

    class Log:
        lines: tuple = ()

        def __init__(self, *a):
            steps.append("logs")

        def __enter__(self):
            return self

        def __exit__(self, *e):
            return False

        def failure(self):
            return None

    monkeypatch.setattr(tools_ab.glimmer_serve, "deploy", lambda *a: steps.append("deploy") or "https://x.modal.run")
    monkeypatch.setattr(tools_ab.glimmer_serve, "session", session)
    monkeypatch.setattr(tools_ab.glimmer_serve, "LogWatch", Log)
    monkeypatch.setattr(tools_ab.glimmer_serve, "wait_ready", lambda *a, **k: steps.append("ready"))
    outcomes["base"] = [make_sim("1", 0), make_sim("1", 1)]

    def halted(cfg_, domain, ids, arm, stamp, endpoint):  # the cap trips during the tools arm
        assert endpoint == "https://x.modal.run"
        if arm == "tools":
            halt.set()
        return _Results(outcomes["base"])

    monkeypatch.setattr(tools_ab, "_run", halted)
    with pytest.raises(BudgetExceeded, match="halted mid-arm"):
        tools_ab.generate(cfg)
    assert steps == ["deploy", "session", "logs", "ready"]
    assert (cfg["data"] / "base-retail.json").exists() and not (cfg["data"] / "tools-retail.json").exists()
    with pytest.raises(BudgetExceeded, match="session was halted"):  # Glimmer calls are refused once halted
        llm_utils.completion(model="openai/Muse-Glimmer-30B-modal", messages=[])


@pytest.fixture
def dropped_set(run_set, make_sim, monkeypatch, tmp_path):
    """A finished banking arm pair whose checkpoints hold two infrastructure errors (base) and none (tools)."""
    base = [make_sim("task_1", 0), make_sim("task_1", 1, "infrastructure_error"),
            make_sim("task_2", 0, "fail"), make_sim("task_2", 1, "infrastructure_error")]
    tools = [make_sim("task_1", 0), make_sim("task_1", 1), make_sim("task_2", 0), make_sim("task_2", 1, "fail")]
    cfg = run_set({("base", "banking_knowledge"): base, ("tools", "banking_knowledge"): tools}, k=2)
    data = cfg["data"]
    (data / "tasks.json").write_text(json.dumps({"banking_knowledge": ["task_1", "task_2"]}), encoding="utf-8")
    monkeypatch.setattr(tools_ab, "_save_dir", lambda c, arm, d, stamp: tmp_path / "ckpt" / arm)
    for arm, sims in (("base", base), ("tools", tools)):
        (tmp_path / "ckpt" / arm).mkdir(parents=True)
        (tmp_path / "ckpt" / arm / "results.json").write_text(json.dumps({"info": {"x": 1}, "simulations": sims}),
                                                               encoding="utf-8")
    events = [{"label": f"{arm} banking_knowledge", "task_id": s["task_id"], "trial": s["trial"],
               "outcome": "infrastructure_error" if s["termination_reason"] == "infrastructure_error" else "pass"}
              for arm, sims in (("base", base), ("tools", tools)) for s in sims]
    (data / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return cfg, tmp_path / "ckpt"


def test_release_takes_only_the_dropped_runs_out_and_keeps_them(dropped_set, capsys):
    cfg, ckpt = dropped_set
    data = cfg["data"]
    assert tools_ab.release_infrastructure_errors(cfg, "s", "abc1234") == 2
    kept = json.loads((ckpt / "base" / "results.json").read_text(encoding="utf-8"))
    assert [(s["task_id"], s["trial"]) for s in kept["simulations"]] == [("task_1", 0), ("task_2", 0)]
    assert kept["info"] == {"x": 1}  # τ²'s resume compares the saved config: it must be untouched
    side = json.loads(next(data.glob("released-base-banking_knowledge-*.json")).read_text(encoding="utf-8"))
    assert side["released_on_commit"] == "abc1234" and len(side["simulations"]) == 2  # nothing is lost
    assert not (data / "base-banking_knowledge.json").exists()  # the base arm runs again
    assert (data / "tools-banking_knowledge.json").exists()  # the tools arm had no drops and is not rerun
    assert len(json.loads((ckpt / "tools" / "results.json").read_text(encoding="utf-8"))["simulations"]) == 4
    events = [json.loads(x) for x in (data / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(events) == 6 and not any(e["outcome"] == "infrastructure_error" for e in events)
    assert "RELEASED 2 infrastructure-error runs of base banking_knowledge" in capsys.readouterr().out


def test_generate_refuses_a_rerun_with_nothing_to_rerun(run_set, monkeypatch, tmp_path):
    cfg = run_set({("base", "retail"): []}, k=1) | {"tasks": lambda: {"retail": ["0"]}}
    monkeypatch.setattr(tools_ab, "_commit", lambda: "abc1234")
    monkeypatch.setattr(tools_ab, "_save_dir", lambda c, arm, d, stamp: tmp_path / "none" / arm)
    monkeypatch.setattr("sys.argv", ["tools_ab.py", "x", "--rerun-infrastructure-errors"])
    from tau2.runner import batch

    monkeypatch.setattr(batch, "try_resume", batch.try_resume)  # restored after the test
    with pytest.raises(SystemExit, match="no infrastructure-error runs to rerun"):
        tools_ab.generate(cfg)
    assert batch.try_resume.glint_keeps_failures  # every run resumes through the keeper


def test_a_set_with_connections_gets_its_own_pool_and_others_keep_taus_client(stand_in, make_sim, monkeypatch,
                                                                                 tmp_path):
    # τ²'s pool has 10 connections for 12 simulations and their user simulator: requests queued silently. A set with
    # `connections` gets a pool that size, still following Modal's redirects and logging; a set without keeps τ²'s
    import litellm

    from glint import glimmer_serve

    monkeypatch.setattr(litellm, "client_session", litellm.client_session)  # restored after the test
    monkeypatch.setattr(litellm, "aclient_session", litellm.aclient_session)
    taus = litellm.client_session
    cfg, outcomes = stand_in
    outcomes["base"] = outcomes["tools"] = [make_sim("1", 0, "pass"), make_sim("1", 1, "pass")]
    tools_ab.generate(cfg)
    assert litellm.client_session is taus  # no key: τ²'s client object, untouched

    tools_ab.generate(cfg | {"connections": 24, "data": tmp_path / "bigger"})
    sync, async_ = litellm.client_session, litellm.aclient_session
    assert sync is not taus
    for client in (sync, async_):
        assert client._transport._pool._max_connections == 24 and client.follow_redirects
    assert sync.event_hooks["request"] == [glimmer_serve._stamp]
    assert sync.event_hooks["response"] == [glimmer_serve._report_unfollowed, glimmer_serve._report_pool_wait]


def test_no_existing_set_changes_its_pool():
    # existing sets keep τ²'s client so their runs stay comparable with the runs already made
    assert not [name for name, cfg in tools_ab.SETS.items() if "connections" in cfg]


def test_resume_reruns_only_infrastructure_errors_and_keeps_failed_runs(tmp_path, monkeypatch, capsys):
    # τ²'s own resume reruns every run whose termination reason is infrastructure_error, context overflows and Glint
    # errors included; the keeper puts those failed runs back, on disk and in the runs to skip
    from tau2.data_model.simulation import Results, TerminationReason
    from tau2.runner import batch, checkpoint

    shipped = Results.load(tools_ab.paths.tau2_data() / "tau2" / "results" / "final" /
                           "gpt-4.1-2025-04-14_retail_default_gpt-4.1-2025-04-14_4trials.json")
    tasks = shipped.tasks[:2]
    sims = [s for s in shipped.simulations if s.task_id in {t.id for t in tasks}]
    overflow, dropped = sims[0], sims[1]
    for s, error in ((overflow, "litellm.ContextWindowExceededError: too long"), (dropped, "APIError: 303")):
        s.termination_reason, s.reward_info = TerminationReason.INFRASTRUCTURE_ERROR, None
        s.info = {"error": error}
    save = tmp_path / "results.json"
    save.write_text(Results(info=shipped.info, tasks=tasks, simulations=sims).model_dump_json(), encoding="utf-8")
    fresh = Results(info=shipped.info, tasks=tasks, simulations=[])
    key = lambda s: (s.trial, s.task_id, s.seed)

    monkeypatch.setattr(batch, "try_resume", checkpoint.try_resume)
    tools_ab.keep_failed_runs_on_resume()
    tools_ab.keep_failed_runs_on_resume()  # installed once however often generate runs
    results, done, _ = batch.try_resume(save, fresh, tasks, 4, auto_resume=True)
    assert key(overflow) in done and key(dropped) not in done  # only the dropped run is rerun
    # τ² says "Resuming run from 7 runs. 2 runs remaining."; the line below counts the kept overflow back in
    resume_line = f"RESUME: {2 * 4 - 1} runs kept, rerunning 1: [({dropped.task_id!r}, {dropped.trial})]"
    assert overflow.id in {s.id for s in results.simulations} and dropped.id not in {s.id for s in results.simulations}
    on_disk = {s.id for s in Results.load(save).simulations}
    assert overflow.id in on_disk and dropped.id not in on_disk and len(on_disk) == len(sims) - 1
    out = capsys.readouterr().out
    assert "KEPT 1 failed runs" in out and resume_line in out


def test_without_the_keeper_taus_resume_drops_failed_runs(tmp_path):
    # the cause, pinned: if τ² changes this, the keeper may no longer be needed
    from tau2.data_model.simulation import Results, TerminationReason
    from tau2.runner import checkpoint

    shipped = Results.load(tools_ab.paths.tau2_data() / "tau2" / "results" / "final" /
                           "gpt-4.1-2025-04-14_retail_default_gpt-4.1-2025-04-14_4trials.json")
    tasks = shipped.tasks[:1]
    sims = [s for s in shipped.simulations if s.task_id == tasks[0].id]
    sims[0].termination_reason, sims[0].info = TerminationReason.INFRASTRUCTURE_ERROR, {"error": "ContextWindowExceededError"}
    save = tmp_path / "results.json"
    save.write_text(Results(info=shipped.info, tasks=tasks, simulations=sims).model_dump_json(), encoding="utf-8")
    _, done, _ = checkpoint.try_resume(save, Results(info=shipped.info, tasks=tasks, simulations=[]), tasks, 4, True)
    assert (sims[0].trial, sims[0].task_id, sims[0].seed) not in done


def test_a_release_records_its_commit_next_to_the_run(dropped_set):
    cfg, _ = dropped_set
    tools_ab.release_infrastructure_errors(cfg, "s", "abc1234")
    rec = json.loads((cfg["data"] / "reruns.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rec["commit"] == "abc1234" and rec["arm"] == "base" and rec["runs"] == [["task_1", 1], ["task_2", 1]]


def test_a_locked_checkpoint_save_is_retried_loudly_then_raised(monkeypatch, capsys):
    from tau2.runner import batch

    calls = {"n": 0}

    def flaky(sim):
        calls["n"] += 1
        if calls["n"] <= limit:
            raise PermissionError(5, "Access is denied", "results.json")
        return "saved"

    monkeypatch.setattr(batch, "create_checkpoint_fns", lambda save_path, lock: (flaky, flaky))
    monkeypatch.setattr(tools_ab, "SAVE_WAIT_S", 0.0)
    tools_ab.retry_locked_checkpoint_saves()
    tools_ab.retry_locked_checkpoint_saves()  # installed once
    save, _ = batch.create_checkpoint_fns("x", None)
    limit = 2
    assert save("sim") == "saved" and calls["n"] == 3  # two refusals, then the save lands
    assert capsys.readouterr().out.count("CHECKPOINT LOCKED") == 2
    calls["n"], limit = 0, 99
    with pytest.raises(PermissionError):
        save("sim")
    assert calls["n"] == tools_ab.SAVE_TRIES  # a lock that outlasts the retries still fails, as before


def test_generate_installs_the_save_retry(run_set, monkeypatch, tmp_path):
    from tau2.runner import batch

    cfg = run_set({("base", "retail"): []}, k=1) | {"tasks": lambda: {"retail": ["0"]}}
    monkeypatch.setattr(tools_ab, "_commit", lambda: "abc1234")
    monkeypatch.setattr(tools_ab, "_save_dir", lambda c, arm, d, stamp: tmp_path / "none" / arm)
    monkeypatch.setattr("sys.argv", ["tools_ab.py", "x", "--rerun-infrastructure-errors"])
    monkeypatch.setattr(batch, "create_checkpoint_fns", batch.create_checkpoint_fns)
    monkeypatch.setattr(batch, "try_resume", batch.try_resume)
    with pytest.raises(SystemExit):
        tools_ab.generate(cfg)
    assert batch.create_checkpoint_fns.glint_retries_saves


def test_a_truncated_glimmer_response_is_named_by_task_and_trial_and_counted_per_arm(stand_in, make_sim, monkeypatch,
                                                                                      capsys):
    # τ²'s "Output might be incomplete due to token limit!" names no task; Glint's line does
    from tau2.runner import batch

    cfg, outcomes = stand_in
    outcomes["base"] = outcomes["tools"] = [make_sim("1", 0), make_sim("1", 1)]
    replies = []
    monkeypatch.setattr(llm_utils, "completion", lambda **kw: _response(replies.pop(0)))

    def call():
        return llm_utils.completion(model="openai/Muse-Glimmer-30B-modal", messages=[], max_tokens=8192)

    def run(cfg_, domain, ids, arm, stamp, endpoint):
        if arm == "base":  # two simulations through τ²'s own run_with_retry, as the batch runner calls it
            for trial, reasons in ((1, ["length", "stop"]), (3, ["length"])):
                replies.extend(reasons)
                n = len(reasons)
                batch.run_with_retry(lambda n=n: [call() for _ in range(n)] and types.SimpleNamespace(),
                                     task=types.SimpleNamespace(id="task_022"), trial=trial, seed=1, max_retries=0,
                                     retry_delay=0.0, console_display=True)
            # outside a run, the task comes from τ²'s llm_log_dir, and what cannot be found is "unknown"
            token = llm_utils.llm_log_dir.set(tools_ab.Path("sims/artifacts/task_task_077/sim_ab/llm_debug"))
            replies.append("length")
            call()
            llm_utils.llm_log_dir.reset(token)
            replies.append("length")
            call()
        return _Results(outcomes[arm])

    monkeypatch.setattr(tools_ab, "_run", run)
    tools_ab.generate(cfg)
    out = capsys.readouterr().out
    assert [line for line in out.splitlines() if line.startswith("TRUNCATED")] == [
        "TRUNCATED: task_022 trial 1 (max_tokens 8192)", "TRUNCATED: task_022 trial 3 (max_tokens 8192)",
        "TRUNCATED: task_077 trial unknown (max_tokens 8192)", "TRUNCATED: unknown trial unknown (max_tokens 8192)"]
    assert "TRUNCATIONS: base retail: 4 Glimmer responses stopped at max_tokens" in out
    assert "TRUNCATIONS: tools retail: 0 Glimmer responses stopped at max_tokens" in out  # counted per arm


def test_each_arm_ends_with_its_own_pool_summary_even_with_no_requests(stand_in, make_sim, monkeypatch, capsys):
    # a missing POOL WAIT line cannot tell "no long wait" from "the hook never ran"; the summary can
    from glint import glimmer_serve

    cfg, outcomes = stand_in
    outcomes["base"] = outcomes["tools"] = [make_sim("1", 0), make_sim("1", 1)]

    def hop(sent, connected):
        return types.SimpleNamespace(request=types.SimpleNamespace(
            method="POST", url=types.SimpleNamespace(host="x", path="/v1"),
            extensions={"glint_sent": sent, "glint_marks": {"connected": connected, "answered": connected + 1}}))

    def run(cfg_, domain, ids, arm, stamp, endpoint):
        if arm == "base":
            for sent, connected in ((0.0, 0.2), (0.0, 45.0), (10.0, 12.5)):
                glimmer_serve._report_pool_wait(hop(sent, connected))
        return _Results(outcomes[arm])

    monkeypatch.setattr(tools_ab, "_run", run)
    tools_ab.generate(cfg)
    pools = [line for line in capsys.readouterr().out.splitlines() if line.startswith("POOL:")]
    assert pools == ["POOL: 3 requests, 1 waited over 30 s, longest 45.0 s",
                     "POOL: 0 requests, 0 waited over 30 s, longest 0.0 s"]


def test_a_knowing_resume_on_new_code_explains_taus_config_warning(stand_in, make_sim, monkeypatch, capsys):
    cfg, outcomes = stand_in
    outcomes["base"] = outcomes["tools"] = [make_sim("1", 0), make_sim("1", 1)]
    cfg["data"].mkdir(parents=True)
    (cfg["data"] / "stamp.txt").write_text("20261027-000000", encoding="utf-8")
    (cfg["data"] / "commit.txt").write_text("a2e9da7", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["tools_ab.py", "x", "--allow-mixed-code"])
    tools_ab.generate(cfg)
    assert ("MIXED CODE: this comparison started on a2e9da7, the code is now abc123; τ²'s \"Run config has changed\" "
            "warning below is expected") in capsys.readouterr().out


@pytest.fixture
def quiet(monkeypatch):
    """For quiet_run_log(): τ²'s console, batch logger and run_with_retry, and the loguru handlers, restored after."""
    import litellm
    from tau2.runner import batch
    from tau2.utils.display import ConsoleDisplay

    for obj, attr in ((litellm, "suppress_debug_info"), (ConsoleDisplay, "console"), (batch, "logger"),
                      (batch, "run_with_retry")):
        monkeypatch.setattr(obj, attr, getattr(obj, attr))
    yield
    logger.remove()


def test_the_run_log_leaves_out_taus_resume_skips_and_repeated_errors_but_keeps_status(quiet, monkeypatch, capsys):
    import litellm
    from rich.text import Text
    from tau2.runner import batch
    from tau2.utils.display import ConsoleDisplay

    seen = []
    monkeypatch.setattr(batch, "run_with_retry", lambda run_fn, **kw: seen.append(kw) or run_fn())
    tools_ab.quiet_run_log()
    tools_ab.quiet_run_log()  # installed once
    assert litellm.suppress_debug_info is True  # LiteLLM's "Provider List" / "Give Feedback" boilerplate
    console = ConsoleDisplay.console
    console.print(Text("Skipping task task_004, trial 1 because it has already been run.", style="bold yellow"))
    console.print(Text("  Task task_022 failed permanently after 1 attempts: litellm.ContextWindowExceededError:\n"
                       "more of the same error", style="bold red"))
    console.print(Text("Status: 73/80 complete. Avg reward: 0.16 (N=64). 7 running: task_022.1(1068s)", style="cyan"))
    console.print(Text("  Retry 1/2 for task task_022: boom", style="yellow"))
    console.print(Text("3/12 (trial 2/4). Running task task_039, trial 2", style="bold green"))
    out = capsys.readouterr().out
    assert "Skipping" not in out and "failed permanently" not in out and "Running task" not in out
    assert "Status: 73/80 complete" in out and "Retry 1/2 for task task_022" in out
    # one Status line every STATUS_EVERY_S; the first is kept
    clock = [1000.0]
    console._clock = lambda: clock[0]
    console._last_status = None
    for t, n in ((1000.0, 74), (1000.0 + tools_ab.STATUS_EVERY_S - 1, 75), (1000.0 + tools_ab.STATUS_EVERY_S, 76)):
        clock[0] = t
        console.print(Text(f"Status: {n}/80 complete. Avg reward: 0.16 (N=64). 6 running: task_022.1(1068s)"))
    out = capsys.readouterr().out
    assert "Status: 74/80" in out and "Status: 75/80" not in out and "Status: 76/80" in out
    # the evaluation table after each simulation is not printed; the run itself is untouched
    task = types.SimpleNamespace(id="t1")
    assert batch.run_with_retry(lambda: "sim", task=task, trial=2, console_display=True) == "sim"
    assert seen == [{"task": task, "trial": 2, "console_display": False}]


def test_only_taus_two_failed_run_errors_leave_the_run_log_and_the_simulation_log_keeps_them(quiet, monkeypatch,
                                                                                               capsys, tmp_path):
    # the records come from τ²'s real generate and run_with_retry; the handlers are added as run_tasks and
    # _TaskLogContext add theirs
    from tau2.runner import batch

    tools_ab.quiet_run_log()
    batch.logger.remove()
    batch.logger.add(lambda msg: print(msg), level="WARNING")  # run_tasks: the run log
    sim_log = tmp_path / "task.log"
    sim_handler = batch.logger.add(sim_log, level="DEBUG", filter=lambda record: True)  # one simulation's own log

    def overflow(**kw):
        raise RuntimeError("ContextWindowExceededError: too long")

    monkeypatch.setattr(llm_utils, "completion", overflow)

    def run_fn():
        return llm_utils.generate(model="openai/Muse-Glimmer-30B-modal", messages=[])

    failed = batch.run_with_retry(run_fn, task=types.SimpleNamespace(id="task_022"), trial=0, seed=1, max_retries=0,
                                  retry_delay=0.0)
    assert "ContextWindowExceededError" in failed.info["error"]  # the cause the [runs] line prints
    logger.error("an unrelated error")
    try:
        raise ValueError("ours")
    except ValueError:
        logger.exception("a Glint traceback")
    batch.logger.remove(sim_handler)
    out = capsys.readouterr().out
    assert "tau2.utils.llm_utils:generate" not in out and "tau2.runner.progress:run_with_retry" not in out
    assert "an unrelated error" in out and "a Glint traceback" in out and "ValueError: ours" in out
    kept = sim_log.read_text(encoding="utf-8")
    # a file handler gets both records: the filter is on the run log only. In a real run the simulation's task.log
    # keeps generate:417; run_with_retry logs after τ² removes that handler, and its cause is in info.error
    assert "generate:417" in kept and "run_with_retry:101" in kept


def test_the_run_log_drops_taus_cause_free_simulation_loop_warning():
    # the vendored τ² line the filter names: a τ² upgrade that moves it fails here instead of letting it back in
    source = (Path(tools_ab.__file__).resolve().parents[2] / "vendor" / "tau2-bench" / "src" / "tau2" /
              "orchestrator" / "orchestrator.py").read_text(encoding="utf-8").splitlines()
    assert source[286].strip() == "logger.warning(" and "emergency cleanup" in source[288]
    record = {"name": "tau2.orchestrator.orchestrator", "function": "run", "line": 287}
    assert ("tau2.orchestrator.orchestrator", "run", 287) in tools_ab.RUN_LOG_DROPPED
    assert not tools_ab.run_log_filter(record | {"level": types.SimpleNamespace(name="WARNING")})
