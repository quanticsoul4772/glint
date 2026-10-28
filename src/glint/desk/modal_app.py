"""Glint Bank Desk on Modal: a public web page (replay and results), CPU only.

Deploy: MODAL_CONFIG_PATH=... modal deploy src/glint/desk/modal_app.py
Replay and results need only FastAPI and the bundled JSON (no τ², no keys, no GPU). Live mode stays off here: its
"Run live" button is disabled until live mode is deployed with τ² and the keys as Modal secrets (a separate step).
Cost: CPU only, scales to zero when idle; at most 3 containers.
"""
from pathlib import Path

import modal

PACKAGE = Path(__file__).resolve().parents[1]  # src/glint

app = modal.App("glint-desk")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("fastapi==0.162.0", "pydantic>=2")
         .env({"GLINT_HOME": "/tmp/glint"})  # the workspace (ledger, desk log) inside the container
         .add_local_dir(str(PACKAGE), remote_path="/root/glint", ignore=["**/__pycache__/**"]))


@app.function(image=image, cpu=0.25, memory=512, min_containers=0, max_containers=3, scaledown_window=300,
              timeout=120)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def web():
    from glint.desk.app import app as desk

    return desk
