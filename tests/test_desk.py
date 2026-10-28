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
