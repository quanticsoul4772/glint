"""Glimmer generation on Modal: an OpenAI-compatible vLLM endpoint on one A100-80GB, bf16, inside Modal's free credit.

The weights are the ones the capture uses (Volume
`glint-glimmer`, pinned revision); vLLM's compile cache lives in Volume `glint-vllm-cache`. The endpoint scales to zero
after 2 idle minutes, never runs more than one container, and requires Modal proxy auth.

Deploy (no GPU runs until a request arrives):  MODAL_CONFIG_PATH=... modal deploy src/glint/glimmer_serve.py
Every GPU session goes through `session()`, which records container time in the meter and stops the app at the end.
Inside a session, call `wait_ready()` before any model request: a request
sent while vLLM still boots waits on the boot, and Modal answers a web request longer than 150 s with a
redirect the client must follow (smoke test, 2026-10-27).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
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


SCALEDOWN_S = 300  # desk/live.py bills each live pair with the same window (test_desk checks they agree)
# scaledown_window 300 s, not 120: on 2026-10-31 a lull of under 5 min with 12 runs open (all waiting on the simulated
# customer) dropped the container twice mid-run, each costing a ~2.5 min cold start. The session stops the app at the
# end of a run (session()), so the longer window adds no idle tail there; the meter bills session wall time anyway.
@app.function(image=image, gpu=GPU, cpu=4, memory=32768, timeout=3600, max_containers=1, scaledown_window=SCALEDOWN_S,
              volumes={"/vol": weights, "/root/.cache/vllm": vllm_cache})
@modal.concurrent(max_inputs=32)
@modal.web_server(8000, startup_timeout=900, requires_proxy_auth=True)
def serve():  # pragma: no cover  (runs only inside the Modal GPU container)
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


POOL_WAIT_WARN_S = 30.0  # a request that waited longer than this for a connection is reported
# httpcore trace events that mean the hop has a connection: a new one starts connecting, or a kept-alive one starts
# sending (whichever comes first)
_ON_CONNECTION = ("connection.connect_tcp.started", "http11.send_request_headers.started",
                  "http2.send_request_headers.started")
_ON_RESPONSE = ("http11.receive_response_headers.complete", "http2.receive_response_headers.complete")


def _stamp(request) -> None:
    """httpx request hook: when this hop was issued (response.elapsed is unreadable before the body is read), and a
    httpcore trace that records when the hop got a connection and when its response headers arrived. The trace only
    reads the clock, so it cannot fail the call."""
    marks: dict[str, float] = {}

    def trace(event: str, info: dict) -> None:
        if event in _ON_CONNECTION:
            marks.setdefault("connected", time.monotonic())
        elif event in _ON_RESPONSE:
            marks["answered"] = time.monotonic()

    request.extensions["glint_sent"] = time.monotonic()
    request.extensions["glint_marks"] = marks
    request.extensions["trace"] = trace


# What `_report_pool_wait` has seen since the last `reset_pool_stats()`: every response hop, the hops that waited over
# POOL_WAIT_WARN_S, and the longest wait. With no POOL WAIT line, these tell "no long waits" apart from a hook that
# never ran (0 requests).
_pool = {"requests": 0, "waited": 0, "longest_s": 0.0}
_pool_lock = threading.Lock()
# One POOL WAIT line per window: the first long wait is printed, later ones in the window are counted and named on the
# next printed line (2026-10-30 review: 18 lines in 17 min of the unseen run, half of the alerts)
POOL_WAIT_WINDOW_S = 300.0
_wait_window = {"start": float("-inf"), "more": 0, "longest_s": 0.0}


def pool_stats() -> dict:
    """The counts since the last reset, and the long waits of the current window not yet named on a POOL WAIT line
    (`unreported`, with the longest): the arm's POOL line prints them, so the last window is never lost (review,
    2026-10-31)."""
    with _pool_lock:
        return dict(_pool) | {"unreported": _wait_window["more"], "unreported_longest_s": _wait_window["longest_s"]}


def reset_pool_stats() -> None:
    with _pool_lock:
        _pool.update(requests=0, waited=0, longest_s=0.0)
        _wait_window.update(start=float("-inf"), more=0, longest_s=0.0)


def _report_pool_wait(response) -> None:
    """httpx response hook: a hop that waited over POOL_WAIT_WARN_S for a free connection in the shared pool is
    printed loudly with its queue wait and its server time, so queueing is told apart from a slow server
    (2026-10-28, banking-test: a call logged as 382 s was mostly queueing). Every hop is counted in `pool_stats()`.
    It reads only what `_stamp` stored, and a failure here is printed, never raised into the call (a hook reading
    `response.elapsed` once broke calls)."""
    try:
        req = response.request
        sent, marks = req.extensions.get("glint_sent"), req.extensions.get("glint_marks") or {}
        measured = sent is not None and "connected" in marks
        wait = marks["connected"] - sent if measured else 0.0
        with _pool_lock:
            _pool["requests"] += 1
            _pool["waited"] += wait > POOL_WAIT_WARN_S
            _pool["longest_s"] = max(_pool["longest_s"], wait)
        if wait <= POOL_WAIT_WARN_S:
            return
        now = time.monotonic()
        with _pool_lock:
            if now - _wait_window["start"] < POOL_WAIT_WINDOW_S:
                _wait_window["more"] += 1
                _wait_window["longest_s"] = max(_wait_window["longest_s"], wait)
                return
            more, longest = _wait_window["more"], _wait_window["longest_s"]
            _wait_window.update(start=now, more=0, longest_s=0.0)
        server = marks.get("answered", time.monotonic()) - marks["connected"]
        earlier = (f" (+{more} more over {POOL_WAIT_WARN_S:.0f} s in the window before, longest {longest:.1f} s)"
                   if more else "")
        print(f"[serve {time.strftime('%H:%M:%S')}] POOL WAIT {wait:.1f} s on {req.method} {req.url.host}"
              f"{req.url.path}: queued for a connection (pool full) before it was sent; server time {server:.1f} s"
              f"{earlier}", file=sys.stderr, flush=True)
    except Exception as e:  # noqa: BLE001  (a logging bug must not fail a model call; it is printed instead)
        print(f"[serve {time.strftime('%H:%M:%S')}] WARNING: the pool-wait log failed: {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)


def _report_unfollowed(response) -> None:
    """httpx response hook: a redirect with no Location (one httpx cannot follow) is printed loudly before the call
    fails."""
    if 300 <= response.status_code < 400 and not response.has_redirect_location:
        req = response.request
        sent = req.extensions.get("glint_sent")  # never raise into the call, even if the stamp hook did not run
        hop = f"{time.monotonic() - sent:.0f} s" if sent is not None else "unknown time"
        print(f"[serve {time.strftime('%H:%M:%S')}] UNFOLLOWED REDIRECT {response.status_code} on {req.method} "
              f"{req.url.host}{req.url.path} after {hop} this hop; "
              f"headers: {sorted(response.headers.keys())}", file=sys.stderr, flush=True)


def follow_modal_redirects() -> None:
    """Make LiteLLM's HTTP client follow Modal's redirects. Modal answers any web request still running after 150 s
    with a 303 to a result URL; τ² installs its own httpx client on import (tau2.utils.llm_utils, connection limits),
    and httpx clients do not follow redirects by default. So every Glimmer request over 150 s failed with
    "Error code: 303" (2026-10-28, banking-test: tasks 020, 022, 029). τ²'s client and its limits are kept; it now
    follows redirects, a redirect it still cannot follow is reported, and so is a long wait for a pooled connection
    (`_report_pool_wait`)."""
    import litellm
    import tau2.utils.llm_utils  # noqa: F401  (installs τ²'s client first, so it is the one changed here)

    for client in (litellm.client_session, litellm.aclient_session):
        if client is None:
            raise RuntimeError("LiteLLM has no shared HTTP client; τ² changed how it sets one up")
        client.follow_redirects = True
    hooks = litellm.client_session.event_hooks
    if _report_unfollowed not in hooks["response"]:
        litellm.client_session.event_hooks = {"request": [*hooks["request"], _stamp],
                                              "response": [*hooks["response"], _report_unfollowed, _report_pool_wait]}


def use_connections(connections: int) -> None:
    """Replace τ²'s shared LiteLLM clients (sync and async; 10 connections, 5 kept alive) with ones of `connections`
    connections, all kept alive, set up as `follow_modal_redirects` sets τ²'s. τ² runs up to `max_concurrency`
    simulations, plus their user-simulator calls, on the one client, so a smaller pool queues requests silently.
    Call before any model call. The event hooks are sync functions, so as before only the sync client (the one τ²'s
    `completion` uses) carries them."""
    import httpx
    import litellm
    import tau2.utils.llm_utils  # noqa: F401  (τ² installs its clients on import; replace them only after)

    limits = httpx.Limits(max_keepalive_connections=connections, max_connections=connections)
    litellm.client_session = httpx.Client(limits=limits)
    litellm.aclient_session = httpx.AsyncClient(limits=limits)
    follow_modal_redirects()
    say(f"LiteLLM's shared HTTP clients replaced: {connections} connections (τ²'s own: 10)")


def litellm_args(url: str, token_id: str, token_secret: str, strength: str = "medium") -> dict:
    """LiteLLM arguments for the endpoint (model `openai/Muse-Glimmer-30B-modal`). Logprobs need `top_logprobs` set;
    `return_token_ids` gives the exact ids, since per-token strings can split a UTF-8 character. Every caller of the
    endpoint gets Modal's redirects followed."""
    follow_modal_redirects()
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
    found = re.search(r"https://\S+\.modal\.run", out.stdout)
    if found is None:
        say("FAILED: modal deploy succeeded but printed no endpoint URL; its output:")
        print((out.stdout + out.stderr)[-4000:], file=sys.stderr, flush=True)
        raise RuntimeError("modal deploy printed no https://...modal.run URL (output above)")
    url = found.group(0)
    say(f"deployed: {url}")
    return url


def say(msg: str) -> None:
    """Every serving step reports here, to stderr at once, so a stuck or failing boot is visible while it happens."""
    print(f"[serve {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


HEALTH_SAY_EVERY_S = 60.0


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
    # a poll line when the outcome changes, else at most one a minute (a boot printed a ReadTimeout every 40 s)
    last_outcome, last_said = None, t0
    with httpx.Client(headers=headers, follow_redirects=False, timeout=request_timeout_s) as client:
        while True:
            try:
                status = client.get(url.rstrip("/") + "/health").status_code
                outcome = f"HTTP {status}"
                if status == 200:
                    say(f"READY after {time.time() - t0:.0f} s")
                    return time.time() - t0
                if status in (401, 403, 404):  # wrong proxy keys or URL: waiting cannot fix these
                    say(f"FAILED: /health -> HTTP {status}; "
                        + ("check MODAL_PROXY_KEY / MODAL_PROXY_SECRET" if status != 404 else "check the URL"))
                    raise RuntimeError(f"{url}/health answered HTTP {status}: "
                                       + ("the proxy keys were refused" if status != 404 else "no such endpoint"))
            except httpx.HTTPError as e:  # booting: refused or reset, or still queued past this poll's timeout
                outcome = type(e).__name__
            seen = f", container log lines {log_lines()}" if log_lines is not None else ""
            if outcome != last_outcome or time.time() - last_said >= HEALTH_SAY_EVERY_S:
                say(f"+{time.time() - t0:4.0f} s  /health -> {outcome}{seen}")
                last_outcome, last_said = outcome, time.time()
            if failed is not None and (reason := failed()):
                say(f"FAILED: {reason}")
                raise RuntimeError(f"{url} failed to start: {reason}")
            if time.time() - t0 > timeout_s:
                say(f"FAILED: not ready after {timeout_s:.0f} s")
                raise TimeoutError(f"{url} not ready after {timeout_s:.0f} s")
            time.sleep(poll_s)


FAILURE_MARKERS = ("Engine core initialization failed", "Runner failed with exception")
# vLLM trouble, echoed every time it appears in the container log
TROUBLE_MARKERS = FAILURE_MARKERS + ("ERROR", "Error:", "Traceback", "Killed", "OutOfMemory", "out of memory")
# vLLM boot stages, echoed once per container (about 34 echoed lines per boot before 2026-10-29, most of them repeats)
STAGE_MARKERS = ("version ", "Resolved architecture", "Loading model", "Loading weights took", "Model loading took",
                 "torch.compile", "Compiling", "compile range", "Capturing CUDA graph", "Graph capturing finished",
                 "KV cache", "Maximum concurrency", "init engine", "Starting vLLM", "Application startup complete")
# vLLM's periodic stats line (every 10 s while it serves: "Engine 000: Avg prompt throughput: ... GPU KV cache usage"),
# which "KV cache" above would echo: it is counted and kept in `lines`, not echoed, and summarized when the watch ends
PERIODIC_MARKERS = ("Avg prompt throughput", "Avg generation throughput", "SpecDecoding metrics")


class LogWatch:
    """Streams the app's container log (`modal app logs -f --timestamps --show-container-id`) in the background:
    `failure()` returns a crash by its root cause (for `wait_ready(failed=...)`), `lines` holds the log. Use as a
    context manager.

    Without `-f` the CLI fetches the last 100 entries and exits (modal 1.6.2, cli/app.py: "Use -f to live-stream"),
    which made an earlier version of this class a 5-second poller that could miss lines. With `-f` the stream only
    ends when the app stops or the CLI fails; it is then restarted, loudly. Every line carries its timestamp and
    container id, so the lines a restart repeats are dropped without hiding a repeated crash in a new container."""

    def __init__(self, modal_cli: str, env: dict, app_name: str = APP_NAME, restart_s: float = 5.0):
        import threading

        self.history: list[str] = []  # kept for callers; -f streams only new lines, nothing is replayed
        self.lines: list[str] = []
        self.restarts = 0
        self.suppressed, self.last_periodic = 0, None  # periodic stats lines not echoed (see PERIODIC_MARKERS)
        # boot stage lines echo once per (container, stage); repeats (vLLM's workers log each stage again) are counted
        self._echoed: set[tuple[str, str]] = set()
        self.stage_repeats = 0
        self._seen: set[str] = set()
        self._containers: list[tuple[str, str]] = []  # (container id, time of its latest log line), in start order
        self._buf = self.lines
        self._cmd = [modal_cli, "app", "logs", app_name, "-f", "--timestamps", "--show-container-id"]
        self._env, self._restart_s = env, restart_s
        self._closed = False
        self.error: str | None = None  # set when the watch itself fails (see _run)
        self._proc = None
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()
        say(f"watching {app_name} logs (streaming)")

    def _run(self):
        # A failure here (the CLI cannot start, a read error) would end the watch silently and leave wait_ready
        # waiting out its whole timeout: it is reported and returned by failure() instead
        try:
            self._stream()
        except Exception as e:  # noqa: BLE001  (surfaced through failure())
            self.error = f"the log watch failed: {type(e).__name__}: {e}"
            say(f"FAILED: {self.error}")

    def _stream(self):
        while True:
            with self._lock:
                if self._closed:
                    return
                self._proc = subprocess.Popen(self._cmd, env=self._env, stdout=subprocess.PIPE,
                                              stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in self._proc.stdout:
                self._take(line.rstrip("\n"))
            code = self._proc.wait()
            if self._closed:
                return
            self.restarts += 1
            say(f"WARNING: the log stream ended (exit {code}) after {len(self.lines)} lines; restarting it "
                f"(restart {self.restarts})")
            time.sleep(self._restart_s)

    def _take(self, line: str) -> None:
        if line in self._seen:  # repeated by a restart (same timestamp and container id)
            return
        self._seen.add(line)
        self._buf.append(line)
        self._container_change(line)
        if self._periodic(line):
            self.suppressed += 1
            self.last_periodic = line.strip()
            return
        if any(m in line for m in TROUBLE_MARKERS):  # every error, crash and out-of-memory line, every time
            say(f"[vllm] {line.strip()[:240]}")
            return
        stage = next((m for m in STAGE_MARKERS if m in line), None)
        if stage is None:
            return
        # once per container and stage: a restarted container (a new id) shows its boot again (a mid-run restart
        # was only visible as a second KV cache line, banking-p2-gated); vLLM's workers repeating a stage do not
        container = (re.search(r"\bta-\w+", line) or re.search(r"$", line)).group(0)
        if (container, stage) in self._echoed:
            self.stage_repeats += 1
            return
        self._echoed.add((container, stage))
        say(f"[vllm] {line.strip()[:240]}")

    def _container_change(self, line: str) -> None:
        """One CONTAINER STARTED line when a new container id first logs, naming the one before it, which is gone:
        on 2026-10-31 the container scaled down mid-run (scaledown_window) and a new one cold-started, and nothing
        in the run log said so until its boot stage lines were found by hand."""
        m = re.search(r"\bta-\w+", line)
        if m is None:
            return
        container = m.group(0)
        # the line's own time (`--timestamps` puts it first), not the time the watch read it (review, 2026-10-31)
        stamped = re.match(r"\s*\d{4}-\d\d-\d\dT(\d\d:\d\d:\d\d)", line)
        at = stamped.group(1) if stamped else time.strftime("%H:%M:%S")
        if self._containers and self._containers[-1][0] == container:
            self._containers[-1] = (container, at)
            return
        if any(c == container for c, _ in self._containers):
            return  # an older container's late line
        before = (f"; {self._containers[-1][0]} is gone (its last log line came at {self._containers[-1][1]})"
                  if self._containers else "")
        self._containers.append((container, at))
        say(f"CONTAINER STARTED: {container}, container {len(self._containers)} this session{before}")

    @staticmethod
    def _periodic(line: str) -> bool:
        """vLLM's periodic stats line, unless it also carries a failure, an error or a warning (those are echoed)."""
        return (any(m in line for m in PERIODIC_MARKERS)
                and not any(m in line for m in FAILURE_MARKERS + ("ERROR", "WARNING", "Error:", "Traceback")))

    def failure(self) -> str | None:
        """The crash as its root cause: the last exception line before the first failure marker (for example
        "RuntimeError: Could not find nvcc ..."), else the marker line itself. The exception may follow a log
        prefix such as "(EngineCore pid=25)". A failed log watch is reported here too: without it a crash is unseen."""
        if getattr(self, "error", None):
            return self.error
        lines = list(self.lines)
        hit = next((i for i, line in enumerate(lines) if any(m in line for m in FAILURE_MARKERS)), None)
        if hit is None:
            return None
        causes = [m.group(0).strip() for line in lines[:hit]
                  if (m := re.search(r"\b\w+(Error|Exception): .*", line))]
        return causes[-1] if causes else lines[hit].strip()

    def dump(self, n: int = 40) -> None:
        """Print the last `n` log lines (call on failure)."""
        say(f"last {min(n, len(self.lines))} container log lines:")
        for line in self.lines[-n:]:
            print(f"    | {line[:240]}", file=sys.stderr, flush=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        say(f"[vllm] {self.suppressed} periodic stats lines and {self.stage_repeats} repeated boot stage lines not "
            f"echoed; last stats: {self.last_periodic or 'none'}")
        with self._lock:
            self._closed = True
            proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


CLI_TIMEOUT_S = 120.0  # every modal CLI call is bounded; a hung call must not block the stop
STOP_WAIT_S = 90.0     # how long a stopping app may take to drain its container
STOP_TRIES = 5
RECORD_EVERY_S = 60.0  # GPU time reaches the ledger at least this often, so a killed process loses at most this
STOP_MARGIN_EXTRA_S = 30.0  # slack on top of the time to notice and drain, before the cap


def _cli(modal_cli: str, args: list[str], env: dict) -> subprocess.CompletedProcess | None:
    """Run one modal CLI command with a timeout; None (after a loud warning) if it could not run."""
    try:
        return subprocess.run([modal_cli, *args], env=env, check=False, capture_output=True, text=True,
                              timeout=CLI_TIMEOUT_S)
    except (subprocess.TimeoutExpired, OSError) as e:
        say(f"WARNING: `modal {' '.join(args)}` failed: {type(e).__name__}: {e}")
        return None


def _running(modal_cli: str, env: dict) -> list[dict] | None:
    """Deployments of the app that still have a container; None if the listing could not be read."""
    out = _cli(modal_cli, ["app", "list", "--json"], env)
    if out is None or out.returncode:
        say(f"WARNING: app list failed: {(out.stderr or out.stdout).strip()[:300] if out else 'no output'}")
        return None
    try:
        apps = json.loads(out.stdout)
    except json.JSONDecodeError:
        say(f"WARNING: app list returned non-JSON: {out.stdout[:200]!r}")
        return None
    return [a for a in apps
            if a.get("description") == APP_NAME and a.get("state") != "stopped" and str(a.get("tasks")) != "0"]


def _wait_stopped(modal_cli: str, env: dict) -> list[dict]:
    """`modal app stop` returns while the app is still "stopping..." with its container up (2026-10-27), so poll the
    listing until no deployment of the app has a container; returns what is still running after STOP_WAIT_S (an
    unreadable listing counts as still running)."""
    t0 = time.time()
    while True:
        running = _running(modal_cli, env)
        if running == [] or time.time() - t0 > STOP_WAIT_S:
            return running if running is not None else [{"description": APP_NAME, "state": "unknown"}]
        if running:
            say(f"{APP_NAME} is {running[0].get('state')} with {running[0].get('tasks')} container(s); checking again")
        time.sleep(5)


def stop_until_stopped(modal_cli: str, env: dict) -> list[dict]:
    """Stop the app and wait for its containers to drain, retrying the stop up to STOP_TRIES times; returns what is
    still running after the last try (empty when stopped)."""
    running: list[dict] = []
    for attempt in range(1, STOP_TRIES + 1):
        # --yes: without a TTY the stop command otherwise waits for a confirmation that never comes
        out = _cli(modal_cli, ["app", "stop", APP_NAME, "--yes"], env)
        if out is not None and out.returncode:
            say(f"WARNING: app stop exited {out.returncode}: {(out.stderr or out.stdout).strip()[:300]}")
        running = _wait_stopped(modal_cli, env)
        if not running:
            return []
        say(f"WARNING: {APP_NAME} still running after stop attempt {attempt}/{STOP_TRIES}: {running}")
    return running


@contextmanager
def session(meter, modal_cli: str, env: dict, tag: str, check_s: float = 15.0):
    """Wrap a batch that uses the endpoint. Checks the GPU cap first, and the cap holds during the batch; afterwards
    stops the app (so no container idles on) and records the wall time at the A100 rate, an upper bound on the
    billed container time.

    - Spend reaches the ledger every RECORD_EVERY_S, so a killed process loses at most that much, and a resumed run
      on the same phase sees the spend.
    - The watcher trips with a margin (the time to notice, stop and drain), stops the app with retries until no
      container is left, and sets the yielded `halt` event; the caller's completion wrapper refuses Glimmer calls once
      it is set, so the batch drains at once, and the session then raises BudgetExceeded.
    - Every modal CLI call has a timeout, and the drain time is metered.
    - A watcher error (a failed ledger write, a failed stop) is never silent: it is printed in full, the app is
      stopped, `halt` is set, and the session raises it.
    (2026-10-27 review: the cap was enforced by one stop attempt with no margin, and spend was only written at exit.
    Second review: the trip used `_thread.interrupt_main()`, which τ²'s batch runner turns into `os._exit(130)`,
    skipping this cleanup, the app stop and the BudgetExceeded; and a watcher exception killed the cap silently.)"""
    import threading
    import traceback

    from glint.meter import BudgetExceeded

    meter.check_gpu()
    margin_s = check_s + STOP_WAIT_S + STOP_MARGIN_EXTRA_S
    budget_s = (meter.caps.gpu_usd - meter.gpu_usd) / GPU_USD_PER_HOUR * 3600 - margin_s
    if budget_s <= 0:
        raise BudgetExceeded(f"GPU cap ${meter.caps.gpu_usd:.2f} leaves less than the {margin_s:.0f} s stop margin")
    say(f"session start: GPU spent so far ${meter.gpu_usd:.3f} of cap ${meter.caps.gpu_usd:.2f}; the app is stopped "
        f"after {budget_s / 60:.0f} min ({margin_s:.0f} s margin kept for stopping)")
    t0 = time.time()
    recorded_s = [0.0]
    record_lock = threading.Lock()
    done, tripped, halt = threading.Event(), threading.Event(), threading.Event()
    watch_error: list[BaseException] = []

    def record_until_now():
        with record_lock:
            elapsed = time.time() - t0
            if elapsed > recorded_s[0]:
                meter.record_gpu(elapsed - recorded_s[0], GPU_USD_PER_HOUR, GPU, tag=tag)
                recorded_s[0] = elapsed

    def watch_loop():
        last_record = time.time()
        while not done.wait(check_s):
            if time.time() - last_record >= RECORD_EVERY_S:
                record_until_now()
                last_record = time.time()
            if time.time() - t0 >= budget_s:
                tripped.set()
                halt.set()
                say(f"GPU CAP REACHED: ${meter.gpu_usd + (time.time() - t0 - recorded_s[0]) * GPU_USD_PER_HOUR / 3600:.2f}"
                    f" of ${meter.caps.gpu_usd:.2f}; stopping {APP_NAME} now and ending the batch")
                left = stop_until_stopped(modal_cli, env)
                record_until_now()
                if left:
                    say(f"FAILED: {APP_NAME} still running after {STOP_TRIES} stop attempts: {left}")
                return

    def watch():
        try:
            watch_loop()
        except BaseException as e:  # noqa: BLE001  (reported in full and re-raised by the session)
            watch_error.append(e)
            halt.set()
            say(f"FAILED: the GPU watcher died, so the cap and the spend writes are gone; stopping {APP_NAME} and "
                f"ending the batch:\n{traceback.format_exc()}")
            stop_until_stopped(modal_cli, env)

    threading.Thread(target=watch, daemon=True).start()
    try:
        yield halt
    finally:
        done.set()
        say(f"stopping {APP_NAME} after {time.time() - t0:.0f} s")
        left = stop_until_stopped(modal_cli, env)
        record_until_now()  # after the drain, so the drain time is metered
        say(f"metered ${meter.gpu_usd:.3f} in this phase so far ({time.time() - t0:.0f} s this session at "
            f"${GPU_USD_PER_HOUR}/h, upper bound)")
        if left:
            say(f"FAILED: {APP_NAME} still has running containers after {STOP_TRIES} stop attempts: {left}")
            raise RuntimeError(f"{APP_NAME} still has running containers after stop: {left}")
        say(f"{APP_NAME} stopped, 0 containers running")
    if watch_error:
        raise RuntimeError(f"the GPU watcher failed during the session: {watch_error[0]!r}") from watch_error[0]
    if tripped.is_set():
        raise BudgetExceeded(f"GPU cap ${meter.caps.gpu_usd:.2f} reached during the session; the app was stopped")
