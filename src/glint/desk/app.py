"""Glint Bank Desk: the same banking customer request, handled by Glimmer alone and by Glimmer with the Glint
decision tools, side by side (research/3-project/demo-spec.md).

Replay mode: plays recorded runs from scenarios.json, so the page works with no GPU.
Live mode (glint.desk.live): runs a task now, both panes at once, on the Glimmer endpoint (GLINT_GLIMMER_URL);
one live pair at a time.
Run locally: python -m glint.desk.app  (then open http://127.0.0.1:8765)
"""
from __future__ import annotations

import json
import os
import threading
from functools import cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from glint.desk import live

HERE = Path(__file__).resolve().parent
app = FastAPI(title="Glint Bank Desk")


@cache
def scenarios() -> list[dict]:
    return json.loads((HERE / "scenarios.json").read_text(encoding="utf-8"))


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/scenarios")
def list_scenarios():
    return [{"id": s["id"], "title": s["title"], "alone_passed": s["alone"]["outcome"]["passed"],
             "tools_passed": s["tools"]["outcome"]["passed"]} for s in scenarios()]


@app.get("/api/scenarios/{scenario_id}")
def get_scenario(scenario_id: str):
    for s in scenarios():
        if s["id"] == scenario_id:
            return s
    raise HTTPException(404, f"no scenario {scenario_id}")


LIVE_ENV = ("GLINT_GLIMMER_URL", "MODAL_PROXY_KEY", "MODAL_PROXY_SECRET", "MODEL_API_KEY")
# Live mode wakes the Glimmer GPU ($2.94/h while up): a daily GPU cap and pair limit, kept in the meter ledger
budget = live.DeskBudget(gpu_usd_per_day=float(os.environ.get("GLINT_DESK_GPU_USD_PER_DAY", "3")),
                         pairs_per_day=int(os.environ.get("GLINT_DESK_PAIRS_PER_DAY", "12")),
                         log_path=os.environ.get("GLINT_DESK_LOG", "desk_sessions.jsonl"))
_sessions: dict[str, live.LiveSession] = {}
_lock = threading.Lock()


class StartLive(BaseModel):
    task_id: str


@cache
def banking_tasks() -> dict:
    from tau2.registry import registry

    return {t.id: t for t in registry.get_tasks_loader("banking_knowledge")()}


def _wake(settings: dict):
    from glint import glimmer_serve

    def wake():
        glimmer_serve.wait_ready(settings["url"], os.environ["MODAL_PROXY_KEY"], os.environ["MODAL_PROXY_SECRET"])

    return wake


def _new_session(task_id: str) -> live.LiveSession:
    settings = live.configure()
    task = banking_tasks()[task_id]
    build = live.tau2_builder(task, settings["glimmer_args"], settings["glimmer_model"], settings["user_args"],
                              seed=int.from_bytes(os.urandom(2), "big"))
    return live.LiveSession(task_id, build, live.tau2_labeler(task), wake=_wake(settings), on_done=budget.record)


@app.get("/api/live/status")
def live_status():
    missing = [k for k in LIVE_ENV if not os.environ.get(k)]
    running = [s.id for s in _sessions.values() if not s.done]
    return {"available": not missing, "missing": missing, "running": running,
            "pairs_today": budget.pairs_today(), "pairs_per_day": budget.pairs_per_day}


@app.post("/api/live/start")
def live_start(body: StartLive):
    missing = [k for k in LIVE_ENV if not os.environ.get(k)]
    if missing:
        raise HTTPException(503, f"live mode is not configured: missing {', '.join(missing)}")
    if body.task_id not in banking_tasks():
        raise HTTPException(404, f"no banking task {body.task_id}")
    with _lock:  # one live pair at a time: each holds two conversations on the one Glimmer endpoint
        if any(not s.done for s in _sessions.values()):
            raise HTTPException(409, "a live run is already in progress; watch it or wait for it to finish")
        try:
            budget.check()
        except live.BudgetExceeded as e:
            raise HTTPException(429, f"live mode is paused: {e}. Replay mode still works.") from e
        session = _new_session(body.task_id).start()
        _sessions[session.id] = session
    return {"id": session.id}


@app.get("/api/live/{session_id}")
def live_get(session_id: str):
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(404, f"no live run {session_id}")
    return session.snapshot()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765)
