"""glint.glimmer_serve.wait_ready against a local stand-in for the Modal endpoint (no GPU, no network)."""
import http.server
import threading

import pytest

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
    watch.history = ["RuntimeError: Engine core initialization failed. (previous container)"]
    watch.lines = ["INFO Loading weights"]
    assert watch.failure() is None  # an old crash replayed from history does not count
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
