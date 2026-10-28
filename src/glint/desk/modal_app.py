"""Glint Bank Desk on Modal: a public web page (replay and results), CPU only.

Deploy: MODAL_CONFIG_PATH=... modal deploy src/glint/desk/modal_app.py
Replay and results need only FastAPI and the bundled JSON (no τ², no keys, no GPU). Live mode stays off here: its
"Run live" button is hidden until live mode is deployed with τ² and the keys as Modal secrets (a separate step).
Cost: CPU only, scales to zero when idle.
One container at most: live mode's one-pair lock, its running sessions and the polls that read them live in that
process. The spend ledger and the pair log sit on a Volume (GLINT_HOME=/state), so a cold start cannot reset the daily
caps; Modal commits Volume writes every few seconds and on shutdown.
"""
from pathlib import Path

import modal

PACKAGE = Path(__file__).resolve().parents[1]  # src/glint

app = modal.App("glint-desk")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("fastapi==0.162.0", "pydantic>=2")
         .env({"GLINT_HOME": "/state"})  # the workspace (ledger, desk log): the Volume below
         .add_local_dir(str(PACKAGE), remote_path="/root/glint", ignore=["**/__pycache__/**"]))


state = modal.Volume.from_name("glint-desk-state", create_if_missing=True)


@app.function(image=image, cpu=0.25, memory=512, min_containers=0, max_containers=1, scaledown_window=300,
              timeout=120, volumes={"/state": state})
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def web():
    from glint.desk.app import app as desk

    return desk
