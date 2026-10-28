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
    assert meter.gpu_usd > 0


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


def test_logwatch_streams_with_follow():
    # the command is built in __init__; read it without starting a process
    import inspect

    from glint.glimmer_serve import LogWatch

    src = inspect.getsource(LogWatch.__init__)
    assert '"-f"' in src and "--timestamps" in src and "--show-container-id" in src


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
