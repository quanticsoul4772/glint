"""Glint Bank Desk replay mode: the API serves the page and every scenario has two complete panes."""
import json

import pytest
from fastapi.testclient import TestClient

from glint.desk.app import app

client = TestClient(app)


def test_page_and_scenario_list():
    page = client.get("/")
    assert page.status_code == 200 and "Glint Bank Desk" in page.text
    listing = client.get("/api/scenarios").json()
    assert {s["id"] for s in listing} >= {"task_054", "task_038"}


def test_the_page_is_revalidated_so_a_deploy_reaches_returning_visitors():
    # regression: the page went out with Last-Modified 1970 and no Cache-Control, so browsers kept the old page
    assert client.get("/").headers["cache-control"] == "no-cache"


def test_hidden_elements_stay_hidden_over_class_display_rules():
    # regression 2026-10-28: .btn sets display, which overrode [hidden], so "Run live" showed with live mode off
    assert "[hidden] { display: none !important; }" in client.get("/").text


def test_every_scenario_pairs_a_failed_alone_run_with_a_passed_tools_run():
    for s in client.get("/api/scenarios").json():
        full = client.get(f"/api/scenarios/{s['id']}").json()
        assert full["alone"]["outcome"]["passed"] is False and full["tools"]["outcome"]["passed"] is True
        assert any(e["kind"] == "verdict" for e in full["tools"]["events"])  # the tools pane shows Glint verdicts
        assert not any(e["kind"] == "verdict" for e in full["alone"]["events"])
        marked = [e for e in full["alone"]["events"] if e.get("overruled")]
        assert len(marked) == full["alone"]["overruled_writes"]


def test_task054_shows_the_overruled_dispute_before_the_increase():
    full = client.get("/api/scenarios/task_054").json()
    marked = [e for e in full["alone"]["events"] if e.get("overruled")]
    assert any("credit limit increase first" in " ".join(e["overruled"]) for e in marked)


def test_unknown_scenario_is_404():
    assert client.get("/api/scenarios/nope").status_code == 404


class _FakeMessage:
    def __init__(self, d):
        self.d = d

    def model_dump(self):
        return self.d


class _FakeOrchestrator:
    def __init__(self, gate):
        self.trajectory = []
        self.gate = gate


def _fake_build(name):
    import threading
    import types

    orch = _FakeOrchestrator(threading.Event())

    def run(o):
        o.trajectory.append(_FakeMessage({"role": "user", "content": "hi"}))
        o.gate.wait(5)  # the test reads a snapshot while the pane is still running
        msgs = [m.model_dump() for m in o.trajectory] + [{"role": "assistant", "content": f"{name} done"}]
        reward = 1.0 if name == "tools" else 0.0
        return types.SimpleNamespace(model_dump=lambda: {"messages": msgs, "reward_info": {"reward": reward}})

    return orch, run


def test_live_session_streams_then_reports_outcomes():
    import time

    from glint.desk.live import LiveSession

    session = LiveSession("task_x", _fake_build, label=lambda sim: {0: {"tool": "x", "reasons": ["r"]}}).start()
    # wait for the first event with a deadline, not a fixed sleep: a fixed 0.2 s failed under load (2026-10-28)
    deadline = time.time() + 5
    while not session.snapshot()["alone"]["events"] and time.time() < deadline:
        time.sleep(0.01)
    snap = session.snapshot()
    assert snap["done"] is False and snap["alone"]["status"] == "running"
    assert snap["alone"]["events"] == [{"kind": "customer", "text": "hi"}]
    for pane in session.panes.values():
        pane.orchestrator.gate.set()
    # wait on the session's own end, not on the pane threads: the session starts them, and a join could reach a pane
    # whose thread was not started yet ("cannot join thread before it is started", about 1 run in 6, 2026-10-29)
    while not session.done and time.time() < deadline + 5:  # done is set after the panes end, not with them
        time.sleep(0.01)
    snap = session.snapshot()
    assert snap["done"] and snap["alone"]["outcome"]["passed"] is False and snap["tools"]["outcome"]["passed"] is True
    assert snap["alone"]["overruled_writes"] == 1 and "overruled_writes" not in snap["tools"]


def test_live_endpoints_refuse_when_not_configured(monkeypatch):
    for k in ("GLINT_GLIMMER_URL", "MODAL_PROXY_KEY", "MODAL_PROXY_SECRET", "MODEL_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert client.get("/api/live/status").json()["available"] is False
    r = client.post("/api/live/start", json={"task_id": "task_054"})
    assert r.status_code == 503 and "GLINT_GLIMMER_URL" in r.json()["detail"]


def test_live_endpoints_run_one_pair_at_a_time(monkeypatch):
    from glint.desk import app as desk_app
    from glint.desk.live import LiveSession

    for k in desk_app.LIVE_ENV:
        monkeypatch.setenv(k, "x")
    monkeypatch.setattr(desk_app, "_new_session", lambda task_id: LiveSession(task_id, _fake_build))
    monkeypatch.setattr(desk_app, "_sessions", {})
    first = client.post("/api/live/start", json={"task_id": "task_054"})
    assert first.status_code == 200
    assert client.post("/api/live/start", json={"task_id": "task_038"}).status_code == 409
    assert client.get(f"/api/live/{first.json()['id']}").json()["task_id"] == "task_054"
    for s in desk_app._sessions.values():
        for pane in s.panes.values():
            pane.orchestrator.gate.set()


def test_desk_budget_records_gpu_and_refuses_past_the_caps(tmp_path):
    import json
    import types

    import pytest

    from glint.desk.live import SCALEDOWN_S, BudgetExceeded, DeskBudget

    log, ledger = tmp_path / "sessions.jsonl", tmp_path / "ledger.jsonl"
    budget = DeskBudget(gpu_usd_per_day=0.40, pairs_per_day=2, ledger=ledger, log_path=log)
    budget.check()
    panes = {"alone": types.SimpleNamespace(error="boom", started=0.0, finished=60.0),
             "tools": types.SimpleNamespace(error=None, started=0.0, finished=70.0)}

    def session():
        return types.SimpleNamespace(id="s1", task_id="task_054", started=0.0, ended=180.0, wake_error=None,
                                     stop_reason=None, customer_fallback=None, panes=panes,
                                     take_uncharged=lambda until: until,
                                     outcomes=lambda: {"alone": None, "tools": True})

    budget.record(session())  # 180 s + the 300 s scale-down = 480 s at $2.94/h = $0.392
    row = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert row["gpu_seconds"] == 180 + SCALEDOWN_S and row["tools_outcome"] is True and row["alone_outcome"] is None
    assert row["alone_error"] == "boom" and row["tools_seconds"] == 70
    budget.check()  # $0.392 < $0.40, 1 pair < 2
    budget.record(session())
    with pytest.raises(BudgetExceeded):
        budget.check()


def test_a_failed_wake_ends_both_panes_with_the_error():
    import time

    from glint.desk.live import LiveSession

    done = []

    def wake():
        raise TimeoutError("not ready after 900 s")

    session = LiveSession("task_x", _fake_build, wake=wake, on_done=done.append).start()
    deadline = time.time() + 5
    while not session.done and time.time() < deadline:
        time.sleep(0.05)
    snap = session.snapshot()
    assert snap["done"] and done == [session]
    assert all("did not start" in snap[p]["error"] for p in ("alone", "tools"))


def test_live_start_is_refused_when_the_budget_is_spent(monkeypatch):
    from glint.desk import app as desk_app
    from glint.desk.live import BudgetExceeded

    for k in desk_app.LIVE_ENV:
        monkeypatch.setenv(k, "x")
    monkeypatch.setattr(desk_app, "_sessions", {})

    def spent():
        raise BudgetExceeded("GPU spend $3.00 reached cap $3")

    monkeypatch.setattr(desk_app.budget, "check", spent)
    r = client.post("/api/live/start", json={"task_id": "task_054"})
    assert r.status_code == 429 and "Replay mode still works" in r.json()["detail"]


def test_replay_discloses_the_pair_is_selected_with_real_pass_counts():
    # each replay pair is chosen (failed alone, passed with tools); the page must say so and give all recorded runs.
    # These hold for any correctly exported scenarios.json: a pair is only selected from such runs
    assert "Selected pair, not a typical run" in client.get("/").text
    for s in client.get("/api/scenarios").json():
        rec = client.get(f"/api/scenarios/{s['id']}").json()["recorded"]
        assert rec["base"]["passed"] < rec["base"]["runs"] and rec["tools"]["passed"] >= 1


def test_results_tab_counts_every_task_once():
    # holds for any results.json export_results writes (what it contains is tested in test_analysis)
    for s in client.get("/api/results").json():
        assert sum(s["by_task"].values()) == len(s["tasks"])
        assert all(r["change"] in ("improved", "tied", "worse", "no data") for r in s["tasks"])


@pytest.fixture
def desk_files(monkeypatch, tmp_path):
    """The desk reading its scenarios.json and results.json from a temporary folder; returns a writer for them."""
    from glint.desk import app as desk_app

    monkeypatch.setattr(desk_app, "HERE", tmp_path)
    desk_app.scenarios.cache_clear()

    def write(name, data):
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")

    yield write
    desk_app.scenarios.cache_clear()  # the next test reads the committed files again


def _scenario(sid, tuned=None):
    s = {"id": sid, "title": f"title {sid}", "alone": {"outcome": {"passed": False}, "events": []},
         "tools": {"outcome": {"passed": True}, "events": []},
         "recorded": {"base": {"runs": 4, "passed": 1}, "tools": {"runs": 4, "passed": 3}}}
    return s if tuned is None else s | {"tuned": tuned}


def test_the_scenario_list_and_pages_serve_what_scenarios_json_holds(desk_files):
    desk_files("scenarios.json", [_scenario("task_900", tuned=True), _scenario("task_901")])
    assert client.get("/api/scenarios").json() == [
        {"id": "task_900", "title": "title task_900", "alone_passed": False, "tools_passed": True, "tuned": True},
        {"id": "task_901", "title": "title task_901", "alone_passed": False, "tools_passed": True, "tuned": False}]
    assert client.get("/api/scenarios/task_901").json() == _scenario("task_901")
    assert client.get("/api/scenarios/task_054").status_code == 404  # only what the file holds


def test_the_results_tab_serves_what_results_json_holds(desk_files):
    sets = [{"set": "s", "role": "held-out", "by_task": {"tied": 1},
             "tasks": [{"task": "task_900", "change": "tied"}]}]
    desk_files("results.json", sets)
    assert client.get("/api/results").json() == sets
    desk_files("results.json", [])  # read on every request: a regenerated file is served without a restart
    assert client.get("/api/results").json() == []


def _fake_live_env(monkeypatch, tmp_path, pairs_per_day=5):
    """Live mode configured with placeholders and its budget on a temporary ledger and session log."""
    from glint.desk import app as desk_app
    from glint.desk import live

    for k in desk_app.LIVE_ENV:
        monkeypatch.setenv(k, "x")
    monkeypatch.setattr(desk_app, "_sessions", {})
    monkeypatch.setattr(live, "_active", None)
    budget = live.DeskBudget(gpu_usd_per_day=10, pairs_per_day=pairs_per_day, ledger=tmp_path / "ledger.jsonl",
                             log_path=tmp_path / "sessions.jsonl")
    monkeypatch.setattr(desk_app, "budget", budget)
    return budget


def _finish(session):
    import time

    for pane in session.panes.values():
        pane.orchestrator.gate.set()
    deadline = time.time() + 5
    while not session.done and time.time() < deadline:
        time.sleep(0.01)
    assert session.done


def test_live_start_logs_the_start_so_the_pair_counts_against_the_daily_limit(monkeypatch, tmp_path):
    from glint.desk import app as desk_app
    from glint.desk.live import LiveSession

    budget = _fake_live_env(monkeypatch, tmp_path, pairs_per_day=1)
    monkeypatch.setattr(desk_app, "_new_session", lambda task_id: LiveSession(task_id, _fake_build))
    first = client.post("/api/live/start", json={"task_id": "task_054"})
    assert first.status_code == 200
    rows = [json.loads(x) for x in (tmp_path / "sessions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r["event"], r["session"], r["task_id"]) for r in rows] == [("start", first.json()["id"], "task_054")]
    assert budget.pairs_today() == 1
    _finish(desk_app._sessions[first.json()["id"]])
    second = client.post("/api/live/start", json={"task_id": "task_038"})  # the first pair ended, the day's limit not
    assert second.status_code == 429 and "daily limit of 1 live runs" in second.json()["detail"]


def test_new_session_charges_the_gpu_ledger_while_running_and_records_the_pair_at_the_end(monkeypatch, tmp_path):
    import time

    from glint import glimmer_serve
    from glint.desk import app as desk_app
    from glint.desk import live

    _fake_live_env(monkeypatch, tmp_path)
    woke = []
    monkeypatch.setattr(live, "configure", lambda: {"url": "https://x.modal.run", "glimmer_args": {},
                                                    "glimmer_model": "openai/Muse-Glimmer-30B-modal",
                                                    "user_args": {}})
    monkeypatch.setattr(live, "tau2_builder", lambda *a, **k: _fake_build)
    monkeypatch.setattr(live, "tau2_labeler", lambda task: lambda sim: {})
    monkeypatch.setattr(glimmer_serve, "wait_ready", lambda url, *a, **k: woke.append(url))
    session = desk_app._new_session("task_054")
    session.started = time.time() - 100
    session.tick()  # what the meter loop does every CHARGE_EVERY_S
    ledger = tmp_path / "ledger.jsonl"
    rows = [json.loads(x) for x in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["kind"] == "gpu" and round(rows[0]["seconds"]) == 100
    assert session.id in rows[0]["tag"]
    session.start()
    _finish(session)
    assert woke == ["https://x.modal.run"]
    deadline = time.time() + 5  # on_done runs right after `done` turns true
    while not (tmp_path / "sessions.jsonl").exists() and time.time() < deadline:
        time.sleep(0.01)
    logged = json.loads((tmp_path / "sessions.jsonl").read_text(encoding="utf-8"))
    assert logged["event"] == "end" and logged["session"] == session.id and logged["tools_outcome"] is True
    rows = [json.loads(x) for x in ledger.read_text(encoding="utf-8").splitlines()]
    # the end charges the rest of the wall time plus the scale-down window, never the 100 s again
    assert len(rows) == 2 and live.SCALEDOWN_S <= rows[1]["seconds"] < live.SCALEDOWN_S + 10


def test_a_running_pair_is_charged_as_it_runs_and_stopped_at_the_cap(tmp_path):
    import json
    import time

    from glint.desk import live

    budget = live.DeskBudget(gpu_usd_per_day=0.1, pairs_per_day=5, ledger=tmp_path / "ledger.jsonl",
                             log_path=tmp_path / "sessions.jsonl")
    session = live.LiveSession("task_054", _fake_build, charge=budget.charge)
    session.started = time.time() - 200  # 200 s at $2.94/h = $0.163, past the $0.10 cap
    try:
        session.tick()
        assert session.stop_reason.startswith("the daily GPU cap is reached") and session.halt.is_set()
        assert round(session.charged_s) == 200
        session.ended = session.started + 210
        budget.record(session)  # charges only the rest: 10 s + the scale-down window
        rows = [json.loads(x) for x in (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
        assert round(sum(r["seconds"] for r in rows if r.get("kind") == "gpu")) == 210 + live.SCALEDOWN_S
        logged = json.loads((tmp_path / "sessions.jsonl").read_text(encoding="utf-8"))
        assert logged["stopped"] == session.stop_reason and logged["event"] == "end"
    finally:
        session.halt.clear()


def test_a_pair_past_its_time_limit_is_stopped():
    import time

    from glint.desk import live

    session = live.LiveSession("task_054", _fake_build, max_seconds=10)
    session.started = time.time() - 20
    try:
        session.tick()
        assert "minute limit" in session.stop_reason
        assert session.halt.is_set()
    finally:
        session.halt.clear()


def test_halt_refuses_glimmer_calls_but_not_the_customer():
    import pytest

    from glint.desk import live

    route = live.router(lambda **kw: "spark", lambda **kw: "glimmer")
    session = live.LiveSession("task_054", _fake_build)
    monkey = live._active
    live._active = session
    try:
        assert route(model="openai/Muse-Glimmer-30B-modal") == "glimmer"
        session.halt.set()
        with pytest.raises(live.BudgetExceeded):
            route(model="openai/Muse-Glimmer-30B-modal")
        assert route(model="openai/muse-spark-1.3-contributor") == "spark"
        # a finished pair's late stop cannot stop the next pair: each pair has its own halt
        live._active = live.LiveSession("task_054", _fake_build)
        assert route(model="openai/Muse-Glimmer-30B-modal") == "glimmer"
    finally:
        live._active = monkey


def test_a_finished_pane_is_labeled_once():
    import time

    from glint.desk import live

    calls = []
    session = live.LiveSession("task_054", _fake_build, label=lambda sim: calls.append(1) or {})
    for pane in session.panes.values():
        pane.orchestrator.gate.set()
    session.start()
    for pane in session.panes.values():
        while pane.finished is None:
            time.sleep(0.01)
    session.snapshot()
    session.snapshot()
    assert calls == [1]


def test_a_metering_failure_stops_the_pair():
    from glint.desk import live

    def broken(session, seconds):
        raise OSError("ledger volume unavailable")

    session = live.LiveSession("task_054", _fake_build, charge=broken)
    old = live.CHARGE_EVERY_S
    live.CHARGE_EVERY_S = 0.01
    try:
        session._meter_loop()  # returns after the first failed tick instead of dying quietly
    finally:
        live.CHARGE_EVERY_S = old
    assert session.stop_reason.startswith("GPU metering failed (OSError") and session.halt.is_set()


def test_tick_and_record_never_charge_the_same_seconds():
    from glint.desk import live

    session = live.LiveSession("task_054", _fake_build)
    assert session.take_uncharged(100) == 100
    assert session.take_uncharged(100) == 0  # a concurrent second taker gets nothing
    assert session.take_uncharged(90) == 0  # an older reading never charges backwards
    assert session.take_uncharged(130) == 30


def test_a_label_error_is_shown_not_raised_on_every_poll():
    import time

    from glint.desk import live

    def bad_label(sim):
        raise ValueError("no such task")

    session = live.LiveSession("task_054", _fake_build, label=bad_label)
    for pane in session.panes.values():
        pane.orchestrator.gate.set()
    session.start()
    while not session.done:
        time.sleep(0.01)
    snap = session.snapshot()
    assert "ValueError" in snap["alone"]["label_error"] and snap["alone"]["outcome"]["passed"] is False


def test_started_pairs_count_against_the_daily_limit_before_they_end(tmp_path):
    import types

    import pytest

    from glint.desk.live import BudgetExceeded, DeskBudget

    budget = DeskBudget(gpu_usd_per_day=10, pairs_per_day=1, ledger=tmp_path / "ledger.jsonl",
                        log_path=tmp_path / "sessions.jsonl")
    budget.check()
    budget.started(types.SimpleNamespace(id="s1", task_id="task_054"))  # its process may die before it ends
    with pytest.raises(BudgetExceeded):
        budget.check()


def test_a_failed_record_pauses_live_mode_and_is_kept_on_the_session(tmp_path):
    import time

    import pytest

    from glint.desk import live

    budget = live.DeskBudget(gpu_usd_per_day=10, pairs_per_day=5, ledger=tmp_path / "ledger.jsonl",
                             log_path=tmp_path / "sessions.jsonl")

    def broken(*a, **k):
        raise OSError("volume gone")

    budget._meter = broken  # the ledger cannot be opened
    session = live.LiveSession("task_054", _fake_build, on_done=budget.record)
    for pane in session.panes.values():
        pane.orchestrator.gate.set()
    session.start()
    deadline = time.time() + 5
    while session.done_error is None and time.time() < deadline:
        time.sleep(0.01)
    assert "OSError: volume gone" in session.done_error
    with pytest.raises(live.BudgetExceeded, match="live mode is paused"):
        budget.check()


def test_a_pane_that_raises_shows_the_error_and_the_other_pane_finishes():
    import time

    from glint.desk import live

    def build(name):
        orch, run = _fake_build(name)
        if name == "alone":
            def run(o):
                raise ConnectionError("Glimmer endpoint reset")
        return orch, run

    session = live.LiveSession("task_054", build)
    session.panes["tools"].orchestrator.gate.set()
    session.start()
    deadline = time.time() + 5
    while not session.done and time.time() < deadline:
        time.sleep(0.01)
    snap = session.snapshot()
    assert snap["alone"]["status"] == "error" and "ConnectionError: Glimmer endpoint reset" in snap["alone"]["error"]
    assert snap["tools"]["outcome"]["passed"] is True


def test_the_real_builder_gives_only_the_tools_pane_the_decision_tools():
    # τ²'s own build functions on a real banking task; nothing runs, no model is called
    from glint.desk import app as desk_app
    from glint.desk import live

    task = desk_app.banking_tasks()["task_054"]
    build = live.tau2_builder(task, {"api_base": "http://unused"}, "openai/Muse-Glimmer-30B-modal",
                              {"api_key": "unused"}, seed=1)
    tools = {name: {t.name for t in build(name)[0].environment.get_tools()} for name in ("alone", "tools")}
    assert "check_card_dispute" in tools["tools"] and "check_card_dispute" not in tools["alone"]
    assert tools["alone"] < tools["tools"]
    assert live.tau2_labeler(task)({"task_id": "task_054", "trial": 0, "messages": []}) == {}


def test_configure_routes_glimmer_and_caps_meta_once_per_process(monkeypatch):
    import litellm
    from tau2.utils import llm_utils

    from glint.desk import live

    for obj, attr in ((llm_utils, "completion"), (litellm, "callbacks")):
        monkeypatch.setattr(obj, attr, getattr(obj, attr))
    monkeypatch.setattr(live, "_configured", {})
    for k in ("GLINT_GLIMMER_URL", "MODAL_PROXY_KEY", "MODAL_PROXY_SECRET", "MODEL_API_KEY"):
        monkeypatch.setenv(k, "https://x.modal.run" if k == "GLINT_GLIMMER_URL" else "placeholder")
    first = live.configure()
    assert first["glimmer_model"].endswith("Muse-Glimmer-30B-modal")
    assert first["glimmer_args"]["api_base"] == "https://x.modal.run/v1"
    assert live.configure() is first  # once per process: a second call does not wrap the completion again
    monkeypatch.setattr(live, "_configured", {})
    monkeypatch.delenv("MODEL_API_KEY")
    import pytest

    with pytest.raises(KeyError, match="MODEL_API_KEY"):
        live.configure()


def test_configure_refuses_a_meta_call_over_the_cap_and_leaves_glimmer_uncapped(monkeypatch, tmp_path):
    # the customer (Muse Spark on Meta credit) is capped at META_CAP_USD per process; Glimmer is not Meta credit
    import litellm
    from tau2.utils import llm_utils

    from glint import meter as meter_mod
    from glint.desk import live

    monkeypatch.setattr(litellm, "callbacks", litellm.callbacks)
    sent = []
    monkeypatch.setattr(llm_utils, "completion", lambda **kw: sent.append(kw["model"]) or "ok")
    monkeypatch.setattr(live, "_configured", {})
    monkeypatch.setattr(live, "_active", None)
    meters = []

    class TmpMeter(meter_mod.Meter):  # the desk's meter, on a temporary ledger
        def __init__(self, caps, phase):
            super().__init__(caps, tmp_path / "ledger.jsonl", phase)
            meters.append(self)

    monkeypatch.setattr(meter_mod, "Meter", TmpMeter)
    for k in ("GLINT_GLIMMER_URL", "MODAL_PROXY_KEY", "MODAL_PROXY_SECRET", "MODEL_API_KEY"):
        monkeypatch.setenv(k, "https://x.modal.run" if k == "GLINT_GLIMMER_URL" else "placeholder")
    live.configure()
    meta, spark_open = meters
    assert spark_open.caps.meta_usd == live.SPARK_OPEN_CAP_USD and spark_open.phase.startswith("desk-live-open-")
    assert meta.caps.meta_usd == live.META_CAP_USD
    assert llm_utils.completion(model=live.SPARK, messages=[]) == "ok"
    meta.record_meta("muse-wonder-1.0", 250_000, 0, 0)  # $1.00: the cap is reached
    with pytest.raises(live.BudgetExceeded, match="reached cap"):
        llm_utils.completion(model=live.SPARK, messages=[])
    assert llm_utils.completion(model="openai/Muse-Glimmer-30B-modal", messages=[]) == "ok"
    assert sent == [live.SPARK, "openai/Muse-Glimmer-30B-modal"]  # the refused call was never sent


class _ApiError(Exception):
    """Stand-in for a LiteLLM API error: LiteLLM's exceptions carry the HTTP status as `status_code`."""

    def __init__(self, status, message="error"):
        super().__init__(message)
        self.status_code = status


def _customer_on_a_session(monkeypatch, meta_error):
    """live.customer with a fake Meta call that raises `meta_error` and a fake Spark Open call, on a running pair."""
    from glint.desk import live

    session = live.LiveSession("task_x", _fake_build)
    monkeypatch.setattr(live, "_active", session)
    monkeypatch.setenv(live.SPARK_OPEN_KEY_ENV, "placeholder")
    meta_sent, open_sent = [], []

    def meta_call(**kw):
        meta_sent.append(kw["model"])
        raise meta_error

    def open_call(**kw):
        open_sent.append(kw)
        return "spark-open reply"

    return session, live.customer(meta_call, open_call), meta_sent, open_sent


@pytest.mark.parametrize("error", [_ApiError(401), _ApiError(402), _ApiError(403), _ApiError(429),
                                   _ApiError(400, "Insufficient credit on this account")],
                         ids=["401", "402", "403", "429", "out-of-credit"])
def test_the_customer_switches_to_spark_open_when_meta_cannot_serve_it(monkeypatch, capsys, error):
    from glint.desk import live

    session, call, meta_sent, open_sent = _customer_on_a_session(monkeypatch, error)
    kwargs = {"model": live.SPARK, "messages": [], "api_base": "https://api.meta.ai/v1", "api_key": "meta"}
    assert call(**kwargs) == "spark-open reply"
    assert call(**kwargs) == "spark-open reply"  # the rest of the pair goes straight to Spark Open
    assert meta_sent == [live.SPARK]
    assert [(k["model"], k["api_base"], k["api_key"]) for k in open_sent] == \
        [(live.SPARK_OPEN, "https://inference.baseten.co/v1", "placeholder")] * 2
    assert session.customer_fallback and session.snapshot()["customer_fallback"] == session.customer_fallback
    assert "CUSTOMER SWITCHED to Spark Open" in capsys.readouterr().out  # never silent
    # a new pair tries Meta again
    monkeypatch.setattr(live, "_active", live.LiveSession("task_y", _fake_build))
    call(**kwargs)
    assert meta_sent == [live.SPARK, live.SPARK]


@pytest.mark.parametrize("error", [_ApiError(500), _ApiError(400, "bad request"), ValueError("parse"), None],
                         ids=["500", "400", "other", "glint-cap"])
def test_other_customer_errors_fail_as_before_without_a_switch(monkeypatch, error):
    from glint.desk import live

    error = error or live.BudgetExceeded("Meta API spend reached cap")  # Glint's own cap is not Meta failing
    session, call, meta_sent, open_sent = _customer_on_a_session(monkeypatch, error)
    with pytest.raises(type(error)):
        call(model=live.SPARK, messages=[])
    assert meta_sent == [live.SPARK] and open_sent == [] and session.customer_fallback is None


def test_a_switch_without_the_baseten_key_fails_loudly(monkeypatch):
    from glint.desk import live

    _, call, _, open_sent = _customer_on_a_session(monkeypatch, _ApiError(402))
    monkeypatch.delenv(live.SPARK_OPEN_KEY_ENV)
    with pytest.raises(RuntimeError, match="BASETEN_API_KEY is not set") as raised:
        call(model=live.SPARK, messages=[])
    assert isinstance(raised.value.__cause__, _ApiError) and open_sent == []


def test_a_switch_with_no_running_pair_still_serves_the_call(monkeypatch, capsys):
    from glint.desk import live

    _, call, _, open_sent = _customer_on_a_session(monkeypatch, _ApiError(429))
    monkeypatch.setattr(live, "_active", None)
    assert call(model=live.SPARK, messages=[]) == "spark-open reply" and len(open_sent) == 1
    assert "for this call" in capsys.readouterr().out


def _configured_with_fakes(monkeypatch, tmp_path, completion):
    import litellm
    from tau2.utils import llm_utils

    from glint import meter as meter_mod
    from glint.desk import live

    monkeypatch.setattr(litellm, "callbacks", litellm.callbacks)
    monkeypatch.setattr(llm_utils, "completion", completion)
    monkeypatch.setattr(live, "_configured", {})
    monkeypatch.setattr(live, "_active", live.LiveSession("task_x", _fake_build))
    meters = []

    class TmpMeter(meter_mod.Meter):  # the desk's meters, on a temporary ledger
        def __init__(self, caps, phase):
            super().__init__(caps, tmp_path / "ledger.jsonl", phase)
            meters.append(self)

    monkeypatch.setattr(meter_mod, "Meter", TmpMeter)
    for k in ("GLINT_GLIMMER_URL", "MODAL_PROXY_KEY", "MODAL_PROXY_SECRET", "MODEL_API_KEY", "BASETEN_API_KEY"):
        monkeypatch.setenv(k, "https://x.modal.run" if k == "GLINT_GLIMMER_URL" else "placeholder")
    live.configure()
    return llm_utils, meters


def _meta_out_of_credit(sent):
    from glint.desk import live

    def completion(**kw):
        sent.append(kw["model"])
        if kw["model"] == live.SPARK:
            raise _ApiError(402, "payment required")
        return "ok"

    return completion


def test_spark_open_is_capped_on_its_own(monkeypatch, tmp_path):
    from glint.desk import live

    sent = []
    llm_utils, (meta, spark_open) = _configured_with_fakes(monkeypatch, tmp_path, _meta_out_of_credit(sent))
    assert llm_utils.completion(model=live.SPARK, messages=[]) == "ok"
    spark_open.record_meta("muse-spark-open-1026", 0, 0, 236_000)  # $1.003 of output: the Spark Open cap is passed
    with pytest.raises(live.BudgetExceeded, match="reached cap"):
        llm_utils.completion(model=live.SPARK, messages=[])
    assert sent == [live.SPARK, live.SPARK_OPEN]  # the refused Spark Open call was never sent
    assert meta.meta_usd == 0.0


def test_spark_open_spend_is_charged_to_the_ledger_at_its_price(monkeypatch, tmp_path):
    import json
    import types

    import litellm

    from glint.desk import live

    sent = []
    _, (meta, spark_open) = _configured_with_fakes(monkeypatch, tmp_path, _meta_out_of_credit(sent))
    usage = types.SimpleNamespace(prompt_tokens=400_000, completion_tokens=100_000,
                                  prompt_tokens_details=types.SimpleNamespace(cached_tokens=100_000),
                                  completion_tokens_details=None)
    (logger,) = litellm.callbacks
    logger.log_success_event({"model": live.SPARK_OPEN}, types.SimpleNamespace(usage=usage), 0, 0)
    # 300k input at $1.25/M + 100k cached at $0.15/M + 100k output at $4.25/M (baseten.co/library/muse-spark-open)
    expected = 0.375 + 0.015 + 0.425
    assert spark_open.meta_usd == pytest.approx(expected) and meta.meta_usd == 0.0
    (row,) = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert row["phase"] == spark_open.phase and row["model"] == live.SPARK_OPEN
    assert row["usd"] == pytest.approx(expected)


def test_live_status_reports_a_missing_baseten_key(monkeypatch):
    for k in ("GLINT_GLIMMER_URL", "MODAL_PROXY_KEY", "MODAL_PROXY_SECRET", "MODEL_API_KEY"):
        monkeypatch.setenv(k, "x")
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    status = client.get("/api/live/status").json()
    assert status["available"] is True  # live mode runs on Meta alone
    assert status["fallback_available"] is False and status["fallback_missing"] == ["BASETEN_API_KEY"]
    monkeypatch.setenv("BASETEN_API_KEY", "x")
    status = client.get("/api/live/status").json()
    assert status["fallback_available"] is True and status["fallback_missing"] == []


def test_the_desk_bills_the_same_scale_down_window_glimmer_serve_sets():
    # the window went from 120 s to 300 s (2026-10-31); the desk's live budget must charge the same
    from glint import glimmer_serve
    from glint.desk import live as desk_live

    assert desk_live.SCALEDOWN_S == glimmer_serve.SCALEDOWN_S
