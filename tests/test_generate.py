"""tools_ab.generate with τ²'s batch replaced by a stand-in: arm files, warnings, the cap stop and the code guard.
Nothing here calls a model or Modal."""
import json
import types

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


@pytest.fixture
def stand_in(monkeypatch, tmp_path):
    """A set of 1 retail task at k=2 whose runs are `outcomes[arm]`; restores everything generate patches."""
    import litellm
    import tau2.evaluator.evaluator_nl_assertions as nl

    for obj, attr in ((llm_utils, "completion"), (litellm, "callbacks"), (nl, "DEFAULT_LLM_NL_ASSERTIONS"),
                      (nl, "DEFAULT_LLM_NL_ASSERTIONS_ARGS")):
        monkeypatch.setattr(obj, attr, getattr(obj, attr))
    monkeypatch.setattr(tools_ab, "install", lambda domain: None)
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
    monkeypatch.setattr(llm_utils, "completion", lambda **kw: sent.append(kw["model"]) or {"choices": [{}]})
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
