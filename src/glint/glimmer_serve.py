"""Glimmer generation on Modal: an OpenAI-compatible vLLM endpoint on one A100-80GB, bf16, inside Modal's free credit.

Research: research/3-project/modal-glimmer-generation.md. The weights are the ones the capture uses (Volume
`glint-glimmer`, pinned revision); vLLM's compile cache lives in Volume `glint-vllm-cache`. The endpoint scales to zero
after 2 idle minutes, never runs more than one container, and requires Modal proxy auth.

Deploy (no GPU runs until a request arrives):  MODAL_CONFIG_PATH=... modal deploy src/glint/glimmer_serve.py
Every GPU session goes through `session()`, which records container time in the meter and stops the app at the end.
Inside a session, call `wait_ready()` before any model request (research/3-project/modal-serving-diagnosis.md).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import modal

APP_NAME, MODEL_NAME = "glint-serve", "Muse-Glimmer-30B-modal"
REVISION = "cec94fcfd6512f2cfe708138c32533b346ea4f69"
GPU, GPU_USD_PER_HOUR = "A100-80GB", 2.94   # $2.50/h GPU + ~$0.44/h for 4 cores and 32 GiB (modal.com/pricing)
VLLM_VERSION = "0.31.0"                     # 2026-10-05; the Glimmer recipe needs vLLM >= 0.28
# Glimmer's stop tokens (generation_config.json); <|eom|> must never be a stop token
STOP_TOKEN_IDS = [200001, 200008]

app = modal.App(APP_NAME)
weights = modal.Volume.from_name("glint-glimmer")
vllm_cache = modal.Volume.from_name("glint-vllm-cache", create_if_missing=True)
# Modal cannot detect Python in the vllm/vllm-openai image, and debian_slim has no CUDA toolkit: torch.compile then
# fails with "Could not find nvcc" (2026-10-27). Modal's vLLM example uses a CUDA devel image with Python added.
image = (modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
         .entrypoint([])
         .pip_install(f"vllm=={VLLM_VERSION}"))


@app.function(image=image, gpu=GPU, cpu=4, memory=32768, timeout=3600, max_containers=1, scaledown_window=120,
              volumes={"/vol": weights, "/root/.cache/vllm": vllm_cache})
@modal.concurrent(max_inputs=32)
@modal.web_server(8000, startup_timeout=900, requires_proxy_auth=True)
def serve():
    if not Path(f"/vol/weights/.complete-{REVISION}").exists():
        raise RuntimeError("Glimmer weights are missing on the glint-glimmer Volume")
    cmd = [
        "vllm", "serve", "/vol/weights", "--served-model-name", MODEL_NAME, "--host", "0.0.0.0", "--port", "8000",
        "--max-model-len", "131072", "--gpu-memory-utilization", "0.92", "--max-num-seqs", "16",
        "--enable-auto-tool-choice", "--tool-call-parser", "muse_glimmer", "--reasoning-parser", "muse_glimmer",
        "--generation-config", "auto", "--limit-mm-per-prompt", '{"image": 0}',
    ]
    # The first boot (2026-10-27) was stopped mid-compile and left a partial torch.compile cache in the Volume root;
    # this subdirectory starts clean
    subprocess.Popen(cmd, env=os.environ | {"VLLM_CACHE_ROOT": "/root/.cache/vllm/v031"})


def litellm_args(url: str, token_id: str, token_secret: str, strength: str = "medium") -> dict:
    """LiteLLM arguments for the endpoint (model `openai/Muse-Glimmer-30B-modal`). Logprobs need `top_logprobs` set;
    `return_token_ids` gives the exact ids, since per-token strings can split a UTF-8 character."""
    return {"api_base": url.rstrip("/") + "/v1", "api_key": "unused",
            "extra_headers": {"Modal-Key": token_id, "Modal-Secret": token_secret},
            "temperature": 1.0, "top_p": 0.95, "logprobs": True, "top_logprobs": 1,
            "extra_body": {"top_k": 64, "chat_template_kwargs": {"reasoning_strength": strength},
                           "return_token_ids": True, "stop_token_ids": STOP_TOKEN_IDS}}


def deploy(modal_cli: str, env: dict) -> str:
    """`modal deploy` this module (builds the image on first deploy; starts no GPU) and return the endpoint URL."""
    say(f"deploying {APP_NAME} (builds the image on first deploy)")
    out = subprocess.run([modal_cli, "deploy", str(Path(__file__).resolve())], env=env, capture_output=True,
                         text=True, check=False)
    if out.returncode:
        say(f"FAILED: modal deploy exited {out.returncode}")
        print((out.stdout + out.stderr)[-4000:], file=sys.stderr, flush=True)
        raise RuntimeError(f"modal deploy exited {out.returncode}")
    url = re.search(r"https://\S+\.modal\.run", out.stdout).group(0)
    say(f"deployed: {url}")
    return url


def say(msg: str) -> None:
    """Every serving step reports here, to stderr at once, so a stuck or failing boot is visible while it happens."""
    print(f"[serve {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def wait_ready(url: str, token_id: str, token_secret: str, timeout_s: float = 900.0, poll_s: float = 10.0,
               failed=None, request_timeout_s: float = 30.0, log_lines=None) -> float:
    """Block until the endpoint's vLLM answers GET /health with 200; returns the seconds waited.

    Modal cuts every web request at 150 s and answers with a 303 redirect to a result URL. A request sent while the
    container boots (weights, torch.compile, CUDA graphs: minutes on the first boot) therefore comes back as a bare
    303, and the OpenAI client does not follow it. Send model requests only after this returns.

    Each poll gives up after `request_timeout_s` and redirects are not followed, so `timeout_s` holds (following
    Modal's redirects could chain 150 s waits). `failed`, if given, is called between polls and returns a reason when
    the container has crashed (see `LogWatch`), which raises at once instead of waiting out `timeout_s`. Every poll
    and its outcome is reported through `say`, with the count from `log_lines()` if given (0 while the container
    boots means the log watch is blind).
    """
    import httpx

    headers = {"Modal-Key": token_id, "Modal-Secret": token_secret}
    t0 = time.time()
    say(f"waiting for {url}/health (limit {timeout_s:.0f} s, poll every {poll_s:.0f} s)")
    with httpx.Client(headers=headers, follow_redirects=False, timeout=request_timeout_s) as client:
        while True:
            try:
                status = client.get(url.rstrip("/") + "/health").status_code
                outcome = f"HTTP {status}"
                if status == 200:
                    say(f"READY after {time.time() - t0:.0f} s")
                    return time.time() - t0
            except httpx.HTTPError as e:  # booting: refused or reset, or still queued past this poll's timeout
                outcome = type(e).__name__
            seen = f", container log lines {log_lines()}" if log_lines is not None else ""
            say(f"+{time.time() - t0:4.0f} s  /health -> {outcome}{seen}")
            if failed is not None and (reason := failed()):
                say(f"FAILED: {reason}")
                raise RuntimeError(f"{url} failed to start: {reason}")
            if time.time() - t0 > timeout_s:
                say(f"FAILED: not ready after {timeout_s:.0f} s")
                raise TimeoutError(f"{url} not ready after {timeout_s:.0f} s")
            time.sleep(poll_s)


FAILURE_MARKERS = ("Engine core initialization failed", "Runner failed with exception")
# vLLM boot stages and trouble, echoed live as they appear in the container log
STAGE_MARKERS = ("version ", "Resolved architecture", "Loading model", "Loading weights took", "Model loading took",
                 "torch.compile", "Compiling", "compile range", "Capturing CUDA graph", "Graph capturing finished",
                 "KV cache", "Maximum concurrency", "init engine", "Starting vLLM", "Application startup complete",
                 "ERROR", "Error:", "Traceback", "Killed", "OutOfMemory", "out of memory")


class LogWatch:
    """Streams `modal app logs` for the app in the background: `failure()` returns the first crash line seen (for
    `wait_ready(failed=...)`), `lines` holds the new log lines. Use as a context manager.

    `modal app logs` first replays the previous container's log (which may hold an old crash), so whatever arrives in
    the first `history_s` seconds is set aside in `history` and never counts as a failure. Start it before the first
    request wakes a container.

    `modal app logs` can exit (2026-10-27: right after a deploy it streamed nothing while the container booted and
    served). The stream is restarted whenever it ends, loudly; a restart replays earlier lines, so lines already seen
    are dropped."""

    def __init__(self, modal_cli: str, env: dict, app_name: str = APP_NAME, history_s: float = 10.0,
                 restart_s: float = 5.0):
        import threading

        self.history: list[str] = []
        self.lines: list[str] = []
        self.restarts = 0
        self._seen: set[str] = set()
        self._buf = self.history
        self._cmd, self._env, self._restart_s = [modal_cli, "app", "logs", app_name], env, restart_s
        self._closed = False
        self._proc = None
        threading.Thread(target=self._run, daemon=True).start()
        time.sleep(history_s)
        self._buf = self.lines
        say(f"watching {app_name} logs ({len(self.history)} replayed lines from the previous container ignored)")

    def _run(self):
        while not self._closed:
            self._proc = subprocess.Popen(self._cmd, env=self._env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          text=True, encoding="utf-8", errors="replace")
            for line in self._proc.stdout:
                self._take(line.rstrip("\n"))
            code = self._proc.wait()
            if self._closed:
                return
            self.restarts += 1
            say(f"WARNING: `modal app logs` ended (exit {code}) with {len(self.lines)} new lines so far; "
                f"restarting it (restart {self.restarts})")
            time.sleep(self._restart_s)

    def _take(self, line: str) -> None:
        if line in self._seen:  # replayed by a restart
            return
        self._seen.add(line)
        self._buf.append(line)
        if self._buf is self.lines and any(m in line for m in STAGE_MARKERS + FAILURE_MARKERS):
            say(f"[vllm] {line.strip()[:240]}")

    def failure(self) -> str | None:
        """The crash as its root cause: the last exception line before the first failure marker (for example
        "RuntimeError: Could not find nvcc ..."), else the marker line itself."""
        lines = list(self.lines)
        hit = next((i for i, line in enumerate(lines) if any(m in line for m in FAILURE_MARKERS)), None)
        if hit is None:
            return None
        causes = [line.strip() for line in lines[:hit] if re.match(r"\s*\w+(Error|Exception):", line)]
        return causes[-1] if causes else lines[hit].strip()

    def dump(self, n: int = 40) -> None:
        """Print the last `n` new log lines (call on failure)."""
        say(f"last {min(n, len(self.lines))} container log lines:")
        for line in self.lines[-n:]:
            print(f"    | {line[:240]}", file=sys.stderr, flush=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._closed = True
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait(timeout=10)


@contextmanager
def session(meter, modal_cli: str, env: dict, tag: str, check_s: float = 15.0):
    """Wrap a batch that uses the endpoint. Checks the GPU cap first; afterwards stops the app (so no container idles
    on) and records the elapsed wall time at the A100 rate, an upper bound on the billed container time.

    The cap also holds during the batch: a watcher stops the app as soon as the session's wall time at the A100 rate
    would take spend past the cap (requests then fail and the batch ends), and the session raises BudgetExceeded.
    Before this, the cap was only checked at the start, so a long run could pass it unnoticed (2026-10-27: a banking
    run reached $4.82 of a $5 cap with nothing able to stop it)."""
    import threading

    from glint.meter import BudgetExceeded

    meter.check_gpu()
    budget_s = (meter.caps.gpu_usd - meter.gpu_usd) / GPU_USD_PER_HOUR * 3600
    say(f"session start: GPU spent so far ${meter.gpu_usd:.3f} of cap ${meter.caps.gpu_usd:.2f}; "
        f"the app is stopped after {budget_s / 60:.0f} min")
    t0 = time.time()
    done, tripped = threading.Event(), threading.Event()

    def stop_app():
        # --yes: without a TTY the stop command otherwise waits for a confirmation that never comes
        out = subprocess.run([modal_cli, "app", "stop", APP_NAME, "--yes"], env=env, check=False, capture_output=True,
                             text=True)
        if out.returncode:
            say(f"WARNING: app stop exited {out.returncode}: {(out.stderr or out.stdout).strip()[:300]}")

    def watch():
        while not done.wait(check_s):
            if time.time() - t0 >= budget_s:
                tripped.set()
                say(f"GPU CAP REACHED: ${meter.gpu_usd + (time.time() - t0) * GPU_USD_PER_HOUR / 3600:.2f} of "
                    f"${meter.caps.gpu_usd:.2f}; stopping {APP_NAME} now")
                stop_app()
                return

    threading.Thread(target=watch, daemon=True).start()
    try:
        yield
    finally:
        done.set()
        say(f"stopping {APP_NAME} after {time.time() - t0:.0f} s")
        stop_app()
        usd = meter.record_gpu(time.time() - t0, GPU_USD_PER_HOUR, GPU, tag=tag)
        say(f"metered ${usd:.3f} ({time.time() - t0:.0f} s at ${GPU_USD_PER_HOUR}/h, upper bound)")
        running = _wait_stopped(modal_cli, env)
        if running:
            say(f"FAILED: {APP_NAME} still has running containers {STOP_WAIT_S:.0f} s after stop: {running}")
            raise RuntimeError(f"{APP_NAME} still has running containers after stop: {running}")
        say(f"{APP_NAME} stopped, 0 containers running")
        if tripped.is_set():
            raise BudgetExceeded(f"GPU cap ${meter.caps.gpu_usd:.2f} reached during the session; the app was stopped")


STOP_WAIT_S = 90.0


def _wait_stopped(modal_cli: str, env: dict) -> list[dict]:
    """`modal app stop` returns while the app is still "stopping..." with its container up (2026-10-27), so poll the
    listing until no deployment of the app has a container; returns what is still running after STOP_WAIT_S."""
    t0 = time.time()
    while True:
        listing = subprocess.run([modal_cli, "app", "list", "--json"], check=True, env=env, capture_output=True,
                                 text=True)
        running = [a for a in json.loads(listing.stdout)
                   if a.get("description") == APP_NAME and a.get("state") != "stopped" and str(a.get("tasks")) != "0"]
        if not running or time.time() - t0 > STOP_WAIT_S:
            return running
        say(f"{APP_NAME} is {running[0].get('state')} with {running[0].get('tasks')} container(s); checking again")
        time.sleep(5)
