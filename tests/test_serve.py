"""glint.glimmer_serve.wait_ready against a local stand-in for the Modal endpoint (no GPU, no network)."""
import http.server
import json
import threading

import pytest

from glint import glimmer_serve
from glint.glimmer_serve import wait_ready


def _server(statuses):
    """Answers /health with the given statuses in turn (then 200), and records the proxy headers it saw."""
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, self.headers["Modal-Key"], self.headers["Modal-Secret"]))
            self.send_response(statuses[len(seen) - 1] if len(seen) <= len(statuses) else 200)
            self.end_headers()

        def log_message(self, *args):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


def test_waits_through_boot_responses_until_health_is_200():
    srv, seen = _server([503, 503, 500])
    try:
        waited = wait_ready(f"http://127.0.0.1:{srv.server_port}/", "key", "secret", timeout_s=10, poll_s=0.05)
    finally:
        srv.shutdown()
    assert waited < 10 and len(seen) == 4
    assert all(s == ("/health", "key", "secret") for s in seen)


def test_raises_when_never_ready():
    with pytest.raises(TimeoutError):
        wait_ready("http://127.0.0.1:9", "key", "secret", timeout_s=0.3, poll_s=0.05)


def test_a_modal_303_counts_as_not_ready_and_is_not_followed():
    # Modal answers a request still queued after 150 s with 303 to a result URL; following it could chain 150 s waits
    srv, seen = _server([303, 303])
    try:
        wait_ready(f"http://127.0.0.1:{srv.server_port}", "key", "secret", timeout_s=10, poll_s=0.05)
    finally:
        srv.shutdown()
    assert [p for p, _, _ in seen] == ["/health"] * 3


def test_a_crash_in_the_log_stops_the_wait_at_once():
    import time

    from glint.glimmer_serve import LogWatch

    watch = LogWatch.__new__(LogWatch)
    watch.history = []
    # an exception line alone is not a crash: vLLM logs recoverable errors while it loads
    watch.lines = ["INFO Loading weights", "WARNING ValueError: tokenizer has no chat template; using the default"]
    assert watch.failure() is None
    watch.lines = ["INFO Loading weights"]
    watch.lines.append("RuntimeError: Engine core initialization failed. See root cause above.")
    t0 = time.time()
    with pytest.raises(RuntimeError, match="failed to start: RuntimeError: Engine core"):
        wait_ready("http://127.0.0.1:9", "key", "secret", timeout_s=60, poll_s=0.05, failed=watch.failure)
    assert time.time() - t0 < 5


def test_failure_reports_the_root_cause_not_the_wrapper():
    # The 2026-10-27 crash: the useful line is the nvcc error above vLLM's generic "initialization failed"
    from glint.glimmer_serve import LogWatch

    watch = LogWatch.__new__(LogWatch)
    watch.history, watch.lines = [], [
        "(EngineCore pid=25) INFO Loading model from scratch...",
        "RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist",
        "Traceback (most recent call last):",
        "RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}",
    ]
    assert watch.failure() == "RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist"


def test_session_waits_for_stopping_to_finish(monkeypatch):
    # modal app stop returns while the app is still "stopping..." with 1 container; that is not a leftover
    import json
    import subprocess
    import types

    from glint import glimmer_serve

    states = iter([("stopping...", "1"), ("stopping...", "1"), ("stopped", "0")])

    def fake_run(cmd, **kwargs):
        if cmd[1:3] == ["app", "list"]:
            state, tasks = next(states)
            return types.SimpleNamespace(returncode=0, stdout=json.dumps(
                [{"description": glimmer_serve.APP_NAME, "state": state, "tasks": tasks}]), stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(glimmer_serve.time, "sleep", lambda s: None)
    assert glimmer_serve._wait_stopped("modal", {}) == []


def test_logwatch_restarts_an_ended_stream_and_drops_replayed_lines():
    # `modal app logs` exited right after a deploy (2026-10-27) and the watch saw nothing; it must restart, and the
    # replay a restart brings must not duplicate lines
    import os
    import sys
    import threading
    import time

    from glint.glimmer_serve import LogWatch

    watch = LogWatch.__new__(LogWatch)
    watch.history, watch.lines, watch.restarts, watch._seen = [], [], 0, set()
    watch._buf, watch._closed, watch._proc, watch._restart_s = watch.lines, False, None, 0.05
    watch._lock = threading.Lock()
    watch._cmd = [sys.executable, "-c", "print('INFO Loading weights'); print('KV cache size: 950,814 tokens')"]
    watch._env = dict(os.environ)
    threading.Thread(target=watch._run, daemon=True).start()
    deadline = time.time() + 20
    while watch.restarts < 3 and time.time() < deadline:
        time.sleep(0.05)
    watch._closed = True
    assert watch.restarts >= 3
    assert watch.lines == ["INFO Loading weights", "KV cache size: 950,814 tokens"]


def test_session_stops_the_app_when_the_gpu_cap_is_reached_mid_batch(monkeypatch, tmp_path):
    # The cap was only checked at session start; a long batch could run past it (2026-10-27: $4.82 of $5)
    import json
    import subprocess
    import time
    import types

    from glint import glimmer_serve
    from glint.meter import BudgetExceeded, Caps, Meter

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd[1:3])
        out = json.dumps([{"description": glimmer_serve.APP_NAME, "state": "stopped", "tasks": "0"}])
        return types.SimpleNamespace(returncode=0, stdout=out if cmd[1:3] == ["app", "list"] else "", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(glimmer_serve, "STOP_WAIT_S", 0.5)
    monkeypatch.setattr(glimmer_serve, "STOP_MARGIN_EXTRA_S", 0.0)
    # a $0.01 cap at $2.94/h is about 12 s of session, minus the 0.7 s stop margin
    meter = Meter(Caps(meta_usd=0.0, gpu_usd=0.01), ledger=tmp_path / "ledger.jsonl", phase="guard-test")
    t0 = time.time()
    with (pytest.raises(BudgetExceeded, match="reached during the session"),
          glimmer_serve.session(meter, "modal", {}, tag="t", check_s=0.2) as halt):
        while not halt.is_set() and time.time() - t0 < 30:  # the batch, until the watcher halts it
            time.sleep(0.1)
    assert halt.is_set() and ["app", "stop"] in calls and 10 < time.time() - t0 < 30
    # the whole session, the stop and its drain included, is metered and stays within the cap
    assert 0.008 < meter.gpu_usd <= meter.caps.gpu_usd


def _fake_modal(calls, stop_codes=None):
    """A stand-in for subprocess.run on the modal CLI: `app stop` exits with the next code in stop_codes (0 after),
    `app list` shows the app stopped once a stop has succeeded."""
    import json
    import types

    from glint import glimmer_serve

    stop_codes = list(stop_codes or [])
    state = {"stopped": False}

    def run(cmd, **kwargs):
        calls.append(cmd[1:3])
        if cmd[1:3] == ["app", "stop"]:
            code = stop_codes.pop(0) if stop_codes else 0
            state["stopped"] = state["stopped"] or code == 0
            return types.SimpleNamespace(returncode=code, stdout="", stderr="network error" if code else "")
        if cmd[1:3] == ["app", "list"]:
            rows = [{"description": glimmer_serve.APP_NAME, "state": "stopped" if state["stopped"] else "deployed",
                     "tasks": "0" if state["stopped"] else "1"}]
            return types.SimpleNamespace(returncode=0, stdout=json.dumps(rows), stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    return run


def test_a_failed_stop_is_retried_until_no_container_is_left(monkeypatch):
    import subprocess

    from glint import glimmer_serve

    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_modal(calls, stop_codes=[1, 1]))
    monkeypatch.setattr(glimmer_serve, "STOP_WAIT_S", 0.2)
    monkeypatch.setattr(glimmer_serve.time, "sleep", lambda s: None)
    assert glimmer_serve.stop_until_stopped("modal", {}) == []
    assert calls.count(["app", "stop"]) == 3


def test_spend_reaches_the_ledger_during_the_session(monkeypatch, tmp_path):
    import json
    import subprocess
    import time

    from glint import glimmer_serve
    from glint.meter import Caps, Meter

    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_modal(calls))
    monkeypatch.setattr(glimmer_serve, "RECORD_EVERY_S", 0.3)
    ledger = tmp_path / "ledger.jsonl"
    meter = Meter(Caps(meta_usd=0.0, gpu_usd=5.0), ledger=ledger, phase="record-test")
    with glimmer_serve.session(meter, "modal", {}, tag="t", check_s=0.1):
        time.sleep(1.2)
        rows_mid = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(rows_mid) >= 2  # written while the batch ran, not only at the end
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert 1.2 <= sum(r["seconds"] for r in rows) < 5


def test_failure_finds_the_root_cause_after_a_log_prefix():
    from glint.glimmer_serve import LogWatch

    watch = LogWatch.__new__(LogWatch)
    watch.history, watch.lines = [], [
        "2026-10-27T18:09:30 ta-01 (EngineCore pid=25) RuntimeError: Could not find nvcc",
        "2026-10-27T18:09:31 ta-01 RuntimeError: Engine core initialization failed. See root cause above.",
    ]
    assert watch.failure() == "RuntimeError: Could not find nvcc"


def test_logwatch_streams_with_follow_and_keeps_a_repeated_crash_in_a_new_container(monkeypatch):
    # Without -f the CLI prints the last 100 entries and exits; without the timestamp and container id a restart's
    # replay cannot be told from the same crash in a new container. The stand-in CLI streams only when it is asked
    # to follow, and prefixes lines only when asked to.
    import subprocess
    import time

    from glint import glimmer_serve

    started = []
    crash = "RuntimeError: Engine core initialization failed. See root cause above."
    streams = [[("ta-01", "INFO Loading weights"), ("ta-01", crash)],
               [("ta-01", "INFO Loading weights"), ("ta-01", crash), ("ta-02", crash)]]  # a restart replays ta-01

    class FakeProc:
        def __init__(self, cmd, **kwargs):
            started.append(cmd)
            follow = "-f" in cmd
            rows = streams[min(len(started), len(streams)) - 1] if follow else streams[0][:1]
            stamp = "--timestamps" in cmd
            container = "--show-container-id" in cmd
            self.stdout = [f"{'2026-10-27T18:09:30 ' if stamp else ''}{c + ' ' if container else ''}{line}\n"
                           for c, line in rows]

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", FakeProc)
    watch = glimmer_serve.LogWatch("modal", {}, restart_s=0.01)
    deadline = time.time() + 5
    while len(started) < 3 and time.time() < deadline:
        time.sleep(0.01)
    watch._closed = True
    assert started[0][:4] == ["modal", "app", "logs", glimmer_serve.APP_NAME]
    crashes = [line for line in watch.lines if crash in line]
    assert len(crashes) == 2 and {c.split()[1] for c in crashes} == {"ta-01", "ta-02"}
    assert crash in watch.failure()


def test_wait_ready_fails_at_once_on_refused_keys(monkeypatch):
    import httpx

    from glint import glimmer_serve

    def handler(request):
        return httpx.Response(401)

    real = httpx.Client

    def client(**kwargs):
        return real(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    with pytest.raises(RuntimeError, match="proxy keys were refused"):
        glimmer_serve.wait_ready("https://x.modal.run", "k", "s", timeout_s=600, poll_s=0.01)


def test_a_log_watch_that_cannot_start_is_reported_by_failure():
    from glint.glimmer_serve import LogWatch

    watch = LogWatch("Z:/no/such/modal.exe", {})
    import time

    deadline = time.time() + 5
    while watch.failure() is None and time.time() < deadline:
        time.sleep(0.05)
    assert watch.failure().startswith("the log watch failed")
    watch._closed = True


def test_a_watcher_failure_halts_the_batch_stops_the_app_and_raises(monkeypatch, tmp_path):
    # a watcher exception used to kill the cap and the spend writes silently while the batch ran on
    import subprocess
    import time

    from glint import glimmer_serve
    from glint.meter import Caps, Meter

    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_modal(calls))
    monkeypatch.setattr(glimmer_serve, "RECORD_EVERY_S", 0.1)
    meter = Meter(Caps(meta_usd=0.0, gpu_usd=5.0), ledger=tmp_path / "ledger.jsonl", phase="watch-fail")

    def broken(*a, **k):
        raise OSError("ledger disk full")

    monkeypatch.setattr(meter, "record_gpu", broken)
    # the last spend write in the session's cleanup hits the same broken ledger, so that error is what surfaces
    with (pytest.raises(OSError, match="ledger disk full"),
          glimmer_serve.session(meter, "modal", {}, tag="t", check_s=0.05) as halt):
        t0 = time.time()
        while not halt.is_set() and time.time() - t0 < 10:
            time.sleep(0.05)
        assert halt.is_set() and ["app", "stop"] in calls


def _resumed_meter(tmp_path, prior_usd, cap_usd):
    """A meter for a resumed run: the ledger already holds `prior_usd` of GPU spend in its phase."""
    import json

    from glint.meter import Caps, Meter

    tmp_path.mkdir(parents=True, exist_ok=True)
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"kind": "gpu", "usd": prior_usd, "phase": "resume", "seconds": 0}) + "\n",
                      encoding="utf-8")
    return Meter(Caps(meta_usd=0.0, gpu_usd=cap_usd), ledger=ledger, phase="resume")


def test_a_resumed_session_gets_only_what_the_earlier_spend_left_of_the_cap(monkeypatch, tmp_path):
    import subprocess
    import time

    from glint import glimmer_serve
    from glint.meter import BudgetExceeded

    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_modal(calls))
    monkeypatch.setattr(glimmer_serve, "GPU_USD_PER_HOUR", 36.0)  # $0.01 per second, so the test takes seconds
    monkeypatch.setattr(glimmer_serve, "STOP_WAIT_S", 0.2)
    monkeypatch.setattr(glimmer_serve, "STOP_MARGIN_EXTRA_S", 0.0)
    # $0.05 cap with $0.03 already spent: about 2 s are left, minus the 0.25 s stop margin (a fresh run gets 4.75 s)
    meter = _resumed_meter(tmp_path, prior_usd=0.03, cap_usd=0.05)
    t0 = time.time()
    with (pytest.raises(BudgetExceeded, match="reached during the session"),
          glimmer_serve.session(meter, "modal", {}, tag="t", check_s=0.05) as halt):
        while not halt.is_set() and time.time() - t0 < 10:
            time.sleep(0.02)
    assert time.time() - t0 < 3.5
    assert 0.03 < meter.gpu_usd <= meter.caps.gpu_usd


def test_a_resumed_session_at_or_near_its_cap_is_refused_before_it_starts(monkeypatch, tmp_path):
    import subprocess

    from glint import glimmer_serve
    from glint.meter import BudgetExceeded

    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_modal(calls))
    # the earlier spend already reached the cap
    with (pytest.raises(BudgetExceeded, match="reached cap"),
          glimmer_serve.session(_resumed_meter(tmp_path / "a", 5.0, 5.0), "modal", {}, tag="t")):
        pytest.fail("the batch must not run")
    # $0.05 left of $5 is about 61 s at $2.94/h, less than the 135 s stop margin
    with (pytest.raises(BudgetExceeded, match="leaves less than the 135 s stop margin"),
          glimmer_serve.session(_resumed_meter(tmp_path / "b", 4.95, 5.0), "modal", {}, tag="t")):
        pytest.fail("the batch must not run")
    assert calls == []  # nothing was started or stopped


@pytest.fixture
def modal_mock():
    """A stand-in for Modal's web endpoint: each hop answers 303 to a result URL (with or without Location) until the
    call is `hops` deep, then the completion. Yields (base url, settings); restores litellm's client afterwards."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import litellm

    settings = {"hops": 2, "location": True, "seen": 0}
    body = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": "m", "choices": [
        {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def hop(self):
            settings["seen"] += 1
            if settings["seen"] <= settings["hops"]:
                self.send_response(303)
                if settings["location"]:
                    self.send_header("Location", "/v1/chat/completions?__modal_function_call_id=fc-1")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.hop()

        do_GET = hop

    import httpx
    import tau2.utils.llm_utils  # noqa: F401  (a real run imports τ², which installs its own client)

    # a fresh client set up as τ² sets its own (redirects not followed), so each test starts from a run's state
    saved = litellm.client_session, litellm.aclient_session
    litellm.client_session = httpx.Client(limits=httpx.Limits(max_keepalive_connections=5, max_connections=10))
    litellm.aclient_session = httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=5, max_connections=10))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_port}", settings
    finally:
        srv.shutdown()
        litellm.client_session.close()
        litellm.client_session, litellm.aclient_session = saved


def _complete(url):
    import litellm

    args = glimmer_serve.litellm_args(url, "key", "secret")
    return litellm.completion(model=f"openai/{glimmer_serve.MODEL_NAME}", messages=[{"role": "user", "content": "x"}],
                              api_base=args["api_base"], api_key=args["api_key"], extra_headers=args["extra_headers"],
                              num_retries=0)


def test_taus_client_does_not_follow_modal_redirects(modal_mock):
    # the cause of the banking-test failures: without the fix a request Modal redirects fails with a bare 303
    import litellm

    url, _ = modal_mock
    with pytest.raises(litellm.APIError, match="303"):
        litellm.completion(model=f"openai/{glimmer_serve.MODEL_NAME}", messages=[{"role": "user", "content": "x"}],
                           api_base=url + "/v1", api_key="unused", num_retries=0)


def test_modal_redirects_are_followed_on_taus_client(modal_mock, capsys):
    import litellm

    url, settings = modal_mock
    tau_client = litellm.client_session
    settings["hops"] = 4  # a request Modal redirected four times (about 600 s on Modal)
    assert _complete(url).choices[0].message.content == "hi"
    assert litellm.client_session is tau_client and tau_client.follow_redirects  # τ²'s client and limits are kept
    assert litellm.aclient_session.follow_redirects
    assert "UNFOLLOWED REDIRECT" not in capsys.readouterr().err
    _complete(url)  # a second caller adds no second report hook
    assert tau_client.event_hooks["response"].count(glimmer_serve._report_unfollowed) == 1


def test_a_redirect_that_cannot_be_followed_is_reported_and_still_fails(modal_mock, capsys):
    import litellm

    url, settings = modal_mock
    settings["location"] = False
    with pytest.raises(litellm.APIError, match="303"):
        _complete(url)
    err = capsys.readouterr().err
    assert "UNFOLLOWED REDIRECT 303 on POST 127.0.0.1/v1/chat/completions" in err and "content-length" in err


def test_a_missing_shared_client_fails_loudly(monkeypatch):
    import litellm
    import tau2.utils.llm_utils  # noqa: F401

    monkeypatch.setattr(litellm, "client_session", None)
    with pytest.raises(RuntimeError, match="no shared HTTP client"):
        glimmer_serve.follow_modal_redirects()
