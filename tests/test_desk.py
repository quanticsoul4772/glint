"""Glint Bank Desk replay mode: the API serves the page and every scenario has two complete panes."""
from fastapi.testclient import TestClient

from glint.desk.app import app

client = TestClient(app)


def test_page_and_scenario_list():
    page = client.get("/")
    assert page.status_code == 200 and "Glint Bank Desk" in page.text
    listing = client.get("/api/scenarios").json()
    assert {s["id"] for s in listing} >= {"task_054", "task_038"}


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
    time.sleep(0.2)
    snap = session.snapshot()
    assert snap["done"] is False and snap["alone"]["status"] == "running"
    assert snap["alone"]["events"] == [{"kind": "customer", "text": "hi"}]
    for pane in session.panes.values():
        pane.orchestrator.gate.set()
    for pane in session.panes.values():
        pane.thread.join(5)
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

    from glint.desk.live import BudgetExceeded, DeskBudget

    log, ledger = tmp_path / "sessions.jsonl", tmp_path / "ledger.jsonl"
    budget = DeskBudget(gpu_usd_per_day=0.25, pairs_per_day=2, ledger=ledger, log_path=log)
    budget.check()
    panes = {"alone": types.SimpleNamespace(error="boom", started=0.0, finished=60.0),
             "tools": types.SimpleNamespace(error=None, started=0.0, finished=70.0)}

    def session():
        return types.SimpleNamespace(id="s1", task_id="task_054", started=0.0, ended=180.0, wake_error=None,
                                     stop_reason=None, panes=panes, take_uncharged=lambda until: until,
                                     outcomes=lambda: {"alone": None, "tools": True})

    budget.record(session())  # 180 s + 120 s scale-down = 300 s at $2.94/h = $0.245
    row = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert row["gpu_seconds"] == 300 and row["tools_outcome"] is True and row["alone_outcome"] is None
    assert row["alone_error"] == "boom" and row["tools_seconds"] == 70
    budget.check()  # $0.245 < $0.25, 1 pair < 2
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
    # each replay pair is chosen (failed alone, passed with tools); the page must say so and give all recorded runs
    assert "Selected pair, not a typical run" in client.get("/").text
    for s in client.get("/api/scenarios").json():
        rec = client.get(f"/api/scenarios/{s['id']}").json()["recorded"]
        assert rec["base"]["runs"] >= 2 and rec["tools"]["runs"] >= 2
        assert rec["base"]["passed"] < rec["base"]["runs"] and rec["tools"]["passed"] >= 1


def test_results_tab_reports_by_task_with_roles():
    sets = client.get("/api/results").json()
    roles = {s["role"] for s in sets}
    assert {"held-out", "untuned", "tuned"} <= roles
    for s in sets:
        assert sum(s["by_task"].values()) == len(s["tasks"])
        assert all(r["change"] in ("improved", "tied", "worse", "no data") for r in s["tasks"])
    # task 081 overflowed the context in every run: it carries no information, so it is not counted as tied
    untuned = next(s for s in sets if s["set"] == "banking-other")
    t081 = next(r for r in untuned["tasks"] if r["task"].endswith("081"))
    assert t081["change"] == "no data" and t081["no_data_reason"] == "every run overflowed the context"


def test_replay_marks_the_tuned_tasks():
    tuned = {s["id"]: s["tuned"] for s in client.get("/api/scenarios").json()}
    assert tuned == {"task_054": True, "task_038": True, "task_050": False, "task_052": False}


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
        budget.record(session)  # charges only the rest: 10 s + the 120 s scale-down
        rows = [json.loads(x) for x in (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
        assert round(sum(r["seconds"] for r in rows if r.get("kind") == "gpu")) == 330
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
