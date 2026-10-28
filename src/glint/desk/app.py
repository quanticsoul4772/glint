"""Glint Bank Desk: the same banking customer request, handled by Glimmer alone and by Glimmer with the Glint
decision tools, side by side (research/3-project/demo-spec.md).

Replay mode (this module): plays recorded runs from scenarios.json, so the page works with no GPU.
Run locally: python -m glint.desk.app  (then open http://127.0.0.1:8765)
"""
from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765)
