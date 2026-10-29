"""glimmer_serve's Modal CLI edges with a fake CLI: deploy, the listing, stop retries giving up, the session refusing
to end with a container left, a watcher error, and the log watch cleanup. Nothing here runs modal."""
import subprocess
import sys
import time
import types

import pytest

from glint import glimmer_serve
from glint.meter import Caps, Meter


def _proc(code=0, out="", err=""):
    return types.SimpleNamespace(returncode=code, stdout=out, stderr=err)


def test_litellm_args_point_at_the_endpoint_with_the_proxy_keys():
    args = glimmer_serve.litellm_args("https://x.modal.run/", "k", "s", "high")
    assert args["api_base"] == "https://x.modal.run/v1"
    assert args["extra_headers"] == {"Modal-Key": "k", "Modal-Secret": "s"}
    assert args["extra_body"]["chat_template_kwargs"] == {"reasoning_strength": "high"}
    assert args["extra_body"]["return_token_ids"] is True


def test_deploy_returns_the_url_or_fails_with_the_output(monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _proc(out="done: https://q--glint-serve-serve.modal.run"))
    assert glimmer_serve.deploy("modal", {}) == "https://q--glint-serve-serve.modal.run"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _proc(code=1, err="image build failed"))
    with pytest.raises(RuntimeError, match="modal deploy exited 1"):
        glimmer_serve.deploy("modal", {})
    assert "image build failed" in capsys.readouterr().err
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _proc(out="deployed, no link"))
    with pytest.raises(RuntimeError, match="printed no"):
        glimmer_serve.deploy("modal", {})
    assert "deployed, no link" in capsys.readouterr().err


def test_a_cli_call_that_cannot_run_is_reported_not_raised(monkeypatch, capsys):
    def hang(*a, **k):
        raise subprocess.TimeoutExpired("modal", 120)

    monkeypatch.setattr(subprocess, "run", hang)
    assert glimmer_serve._cli("modal", ["app", "list"], {}) is None
    assert "`modal app list` failed: TimeoutExpired" in capsys.readouterr().err


def test_an_unreadable_listing_is_not_mistaken_for_stopped(monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _proc(code=1, err="auth expired"))
    assert glimmer_serve._running("modal", {}) is None
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _proc(out="<html>"))
    assert glimmer_serve._running("modal", {}) is None
    err = capsys.readouterr().err
    assert "app list failed: auth expired" in err and "non-JSON" in err
    # an unreadable listing while waiting for the stop counts as still running
    monkeypatch.setattr(glimmer_serve, "STOP_WAIT_S", 0.0)
    assert glimmer_serve._wait_stopped("modal", {}) == [{"description": glimmer_serve.APP_NAME, "state": "unknown"}]


def test_stop_gives_up_after_its_tries_and_returns_what_is_still_running(monkeypatch):
    up = [{"description": glimmer_serve.APP_NAME, "state": "deployed", "tasks": "1"}]
    monkeypatch.setattr(glimmer_serve, "_cli", lambda *a, **k: _proc())
    monkeypatch.setattr(glimmer_serve, "_wait_stopped", lambda *a, **k: up)
    monkeypatch.setattr(glimmer_serve, "STOP_TRIES", 2)
    assert glimmer_serve.stop_until_stopped("modal", {}) == up


def test_a_session_that_cannot_stop_the_app_fails_loudly(monkeypatch, tmp_path):
    up = [{"description": glimmer_serve.APP_NAME, "state": "deployed", "tasks": "1"}]
    monkeypatch.setattr(glimmer_serve, "stop_until_stopped", lambda *a, **k: up)
    meter = Meter(Caps(meta_usd=0.0, gpu_usd=5.0), ledger=tmp_path / "ledger.jsonl", phase="left")
    with pytest.raises(RuntimeError, match="still has running containers"), \
            glimmer_serve.session(meter, "modal", {}, tag="t", check_s=0.05):
        time.sleep(0.05)  # longer than Windows' clock tick, so the session has metered time
    assert meter.gpu_usd > 0  # the session's time is metered even so


def test_a_watcher_error_is_raised_after_a_clean_stop(monkeypatch, tmp_path):
    monkeypatch.setattr(glimmer_serve, "stop_until_stopped", lambda *a, **k: [])
    monkeypatch.setattr(glimmer_serve, "RECORD_EVERY_S", 0.05)
    meter = Meter(Caps(meta_usd=0.0, gpu_usd=5.0), ledger=tmp_path / "ledger.jsonl", phase="watch")
    real, calls = meter.record_gpu, []

    def flaky(*a, **k):  # the watcher's write fails once; the session's last write works
        calls.append(1)
        if len(calls) == 1:
            raise OSError("ledger busy")
        return real(*a, **k)

    monkeypatch.setattr(meter, "record_gpu", flaky)
    with pytest.raises(RuntimeError, match="the GPU watcher failed"), \
            glimmer_serve.session(meter, "modal", {}, tag="t", check_s=0.02) as halt:
        deadline = time.time() + 5
        while not halt.is_set() and time.time() < deadline:
            time.sleep(0.02)


def test_a_cap_trip_reports_containers_it_could_not_stop(monkeypatch, tmp_path, capsys):
    up = [{"description": glimmer_serve.APP_NAME, "state": "deployed", "tasks": "1"}]
    stops = []
    monkeypatch.setattr(glimmer_serve, "stop_until_stopped", lambda *a, **k: stops.append(1) or (up if len(stops) == 1
                                                                                                  else []))
    monkeypatch.setattr(glimmer_serve, "STOP_WAIT_S", 0.0)
    monkeypatch.setattr(glimmer_serve, "STOP_MARGIN_EXTRA_S", 0.0)
    meter = Meter(Caps(meta_usd=0.0, gpu_usd=0.0002), ledger=tmp_path / "ledger.jsonl", phase="trip")
    from glint.meter import BudgetExceeded

    with pytest.raises(BudgetExceeded), glimmer_serve.session(meter, "modal", {}, tag="t", check_s=0.02) as halt:
        deadline = time.time() + 5
        while not halt.is_set() and time.time() < deadline:
            time.sleep(0.02)
    assert "still running after" in capsys.readouterr().err


def test_logwatch_dumps_its_lines_and_stops_its_stream(capsys):
    cmd = [sys.executable, "-c", "import time; print('KV cache size: 1 tokens', flush=True); time.sleep(30)"]
    watch = glimmer_serve.LogWatch.__new__(glimmer_serve.LogWatch)
    watch.history, watch.lines, watch.restarts, watch._seen, watch.error = [], [], 0, set(), None
    watch._buf, watch._closed, watch._proc, watch._restart_s = watch.lines, False, None, 0.05
    watch.suppressed, watch.last_periodic, watch._echoed, watch.stage_repeats = 0, None, set(), 0
    import threading

    watch._lock, watch._cmd, watch._env = threading.Lock(), cmd, None
    threading.Thread(target=watch._run, daemon=True).start()
    deadline = time.time() + 10
    while not watch.lines and time.time() < deadline:
        time.sleep(0.05)
    with watch:
        watch.dump(5)
    assert "| KV cache size: 1 tokens" in capsys.readouterr().err
    assert watch._proc.poll() is not None  # the stream process was ended on exit
