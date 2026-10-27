"""Glimmer generation on Modal: an OpenAI-compatible vLLM endpoint on one A100-80GB, bf16, inside Modal's free credit.

Research: research/3-project/modal-glimmer-generation.md. The weights are the ones the capture uses (Volume
`glint-glimmer`, pinned revision); vLLM's compile cache lives in Volume `glint-vllm-cache`. The endpoint scales to zero
after 2 idle minutes, never runs more than one container, and requires Modal proxy auth.

Deploy (no GPU runs until a request arrives):  MODAL_CONFIG_PATH=... modal deploy src/glint/glimmer_serve.py
Every GPU session goes through `session()`, which records container time in the meter and stops the app at the end.
"""
from __future__ import annotations

import json
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import modal

APP_NAME, MODEL_NAME = "glint-serve", "Muse-Glimmer-30B-modal"
REVISION = "cec94fcfd6512f2cfe708138c32533b346ea4f69"
GPU, GPU_USD_PER_HOUR = "A100-80GB", 2.94   # $2.50/h GPU + ~$0.44/h for 4 cores and 32 GiB (modal.com/pricing)
VLLM_IMAGE = "vllm/vllm-openai:v0.31.0"     # 2026-10-05; the Glimmer recipe needs vLLM >= 0.28
# Glimmer's stop tokens (generation_config.json); <|eom|> must never be a stop token
STOP_TOKEN_IDS = [200001, 200008]

app = modal.App(APP_NAME)
weights = modal.Volume.from_name("glint-glimmer")
vllm_cache = modal.Volume.from_name("glint-vllm-cache", create_if_missing=True)
image = modal.Image.from_registry(VLLM_IMAGE).entrypoint([])


@app.function(image=image, gpu=GPU, cpu=4, memory=32768, timeout=3600, max_containers=1, scaledown_window=120,
              volumes={"/vol": weights, "/root/.cache/vllm": vllm_cache})
@modal.concurrent(max_inputs=32)
@modal.web_server(8000, startup_timeout=900, requires_proxy_auth=True)
def serve():
    if not Path(f"/vol/weights/.complete-{REVISION}").exists():
        raise RuntimeError("Glimmer weights are missing on the glint-glimmer Volume")
    cmd = [
        "vllm", "serve", "/vol/weights", "--served-model-name", MODEL_NAME, "--host", "0.0.0.0", "--port", "8000",
        "--max-model-len", "65536", "--gpu-memory-utilization", "0.92", "--max-num-seqs", "16",
        "--enable-auto-tool-choice", "--tool-call-parser", "muse_glimmer", "--reasoning-parser", "muse_glimmer",
        "--generation-config", "auto", "--limit-mm-per-prompt", '{"image": 0}',
    ]
    subprocess.Popen(cmd)


def litellm_args(url: str, token_id: str, token_secret: str, strength: str = "medium") -> dict:
    """LiteLLM arguments for the endpoint (model `openai/Muse-Glimmer-30B-modal`). Logprobs need `top_logprobs` set;
    `return_token_ids` gives the exact ids, since per-token strings can split a UTF-8 character."""
    return {"api_base": url.rstrip("/") + "/v1", "api_key": "unused",
            "extra_headers": {"Modal-Key": token_id, "Modal-Secret": token_secret},
            "temperature": 1.0, "top_p": 0.95, "logprobs": True, "top_logprobs": 1,
            "extra_body": {"top_k": 64, "chat_template_kwargs": {"reasoning_strength": strength},
                           "return_token_ids": True, "stop_token_ids": STOP_TOKEN_IDS}}


@contextmanager
def session(meter, modal_cli: str, env: dict, tag: str):
    """Wrap a batch that uses the endpoint. Checks the GPU cap first; afterwards stops the app (so no container idles
    on) and records the elapsed wall time at the A100 rate, an upper bound on the billed container time."""
    meter.check_gpu()
    t0 = time.time()
    try:
        yield
    finally:
        subprocess.run([modal_cli, "app", "stop", APP_NAME], env=env, check=False, capture_output=True)
        meter.record_gpu(time.time() - t0, GPU_USD_PER_HOUR, GPU, tag=tag)
        listing = subprocess.run([modal_cli, "app", "list", "--json"], check=True, env=env, capture_output=True,
                                 text=True)
        running = [a for a in json.loads(listing.stdout)
                   if a.get("description") == APP_NAME and a.get("state") != "stopped" and str(a.get("tasks")) != "0"]
        if running:
            raise RuntimeError(f"{APP_NAME} still has running containers after stop: {running}")
