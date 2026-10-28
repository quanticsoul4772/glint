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
