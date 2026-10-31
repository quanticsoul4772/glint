"""scripts/status.py on a stand-in run set."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_report_counts_progress_outcomes_and_errors(tmp_path, monkeypatch):
    import status

    data = tmp_path / "set"
    data.mkdir()
    (data / "tasks.json").write_text(json.dumps({"retail": ["1", "2"]}), encoding="utf-8")
    (data / "stamp.txt").write_text("20261027-000000", encoding="utf-8")
    rows = [{"ts": time.time(), "label": "base retail", "task_id": "1", "trial": 0, "outcome": "pass",
             "duration_s": 60.0, "error": None},
            {"ts": time.time(), "label": "base retail", "task_id": "2", "trial": 0, "outcome": "context_overflow",
             "duration_s": 60.0, "error": "ContextWindowExceededError: 131072"}]
    (data / "events.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"kind": "gpu", "usd": 0.5, "phase": "x-gpu-20261027-000000"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(status, "LEDGER", ledger)
    cfg = {"data": data, "trials": 2, "save": "x", "concurrency": 2}
    text = "\n".join(status.report("x", cfg))
    assert "2/4" in text and "'pass': 1" in text and "context_overflow: 2 trial 0" in text
    assert "gpu $0.500" in text and "0/4" in text  # the tools arm has not started


def test_report_flags_a_stalled_arm_a_finished_arm_and_every_non_result(tmp_path):
    import status

    data = tmp_path / "set"
    data.mkdir()
    (data / "tasks.json").write_text(json.dumps({"retail": ["1"]}), encoding="utf-8")
    (data / "tools-retail.json").write_text("{}", encoding="utf-8")
    old = time.time() - 3 * 3600
    rows = [{"ts": old, "label": "base retail", "task_id": "1", "trial": 0, "outcome": "glint_error",
             "duration_s": 60.0, "error": "AtemParseError: host calls"},
            {"ts": old, "label": "tools retail", "task_id": "1", "trial": 0, "outcome": "pass", "duration_s": 60.0,
             "error": None}]
    (data / "events.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    text = "\n".join(status.report("x", {"data": data, "trials": 2, "save": "x"}))
    assert "WARNING: no event for 180 min" in text and "[arm done]" in text
    assert "glint_error: 1 trial 0: AtemParseError" in text
    assert status.report("x", {"data": tmp_path / "missing", "trials": 1, "save": "x"}) == []


def test_modal_state_says_why_it_does_not_know(monkeypatch):
    import subprocess
    import types

    import status

    def run(out="", code=0, err=""):
        return lambda *a, **k: types.SimpleNamespace(returncode=code, stdout=out, stderr=err)

    monkeypatch.setattr(subprocess, "run", run(json.dumps([{"description": "glint-serve", "state": "deployed",
                                                             "tasks": "1"}])))
    assert status._modal_state() == "deployed (1 containers)"
    monkeypatch.setattr(subprocess, "run", run("[]"))
    assert status._modal_state() == "stopped (0 containers)"
    monkeypatch.setattr(subprocess, "run", run(code=1, err="token expired"))
    assert status._modal_state() == "UNKNOWN: modal app list exited 1: token expired"
    monkeypatch.setattr(subprocess, "run", run("Traceback"))
    assert status._modal_state().startswith("UNKNOWN: modal app list printed no JSON")

    def missing(*a, **k):
        raise FileNotFoundError("modal.exe")

    monkeypatch.setattr(subprocess, "run", missing)
    assert status._modal_state().startswith("UNKNOWN: modal app list could not run (FileNotFoundError")


def test_main_reports_the_named_sets_and_modal(monkeypatch, capsys):
    import status

    monkeypatch.setattr(sys, "argv", ["status.py", "design", "--modal"])
    monkeypatch.setattr(status, "_modal_state", lambda: "stopped (0 containers)")
    monkeypatch.setattr(status, "report", lambda name, cfg: [f"== {name}"])
    status.main()
    assert capsys.readouterr().out.splitlines() == ["== design", "== design --modal",
                                                    "Modal glint-serve: stopped (0 containers)"]


def test_a_modal_only_set_is_reported_on_modal_only(monkeypatch, capsys):
    # the gated sets refuse a non-Modal backend; status used to ask for it first and stop with an error
    import status

    monkeypatch.setattr(sys, "argv", ["status.py", "banking-gated"])
    monkeypatch.setattr(status, "report", lambda name, cfg: [f"== {name} {cfg['data'].name}"])
    status.main()
    assert capsys.readouterr().out.splitlines() == ["== banking-gated --modal tools_ab_banking12_gated_modal"]


def test_the_eta_leaves_out_runs_tau_kept_no_time_for(tmp_path, monkeypatch):
    # an overflow is recorded with 0 s: averaged in, it halved the ETA here (2026-10-30 review)
    import status

    data = tmp_path / "set"
    data.mkdir()
    (data / "tasks.json").write_text(json.dumps({"retail": ["1", "2"]}), encoding="utf-8")
    (data / "stamp.txt").write_text("20261027-000000", encoding="utf-8")
    rows = [{"ts": time.time(), "label": "base retail", "task_id": "1", "trial": 0, "outcome": "pass",
             "duration_s": 600.0, "error": None},
            {"ts": time.time(), "label": "base retail", "task_id": "2", "trial": 0, "outcome": "context_overflow",
             "duration_s": 0.0, "error": "ContextWindowExceededError: 131072"}]
    (data / "events.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(status, "LEDGER", tmp_path / "ledger.jsonl")
    text = "\n".join(status.report("x", {"data": data, "trials": 2, "save": "x", "concurrency": 2}))
    assert "~10 min left" in text  # 2 runs left x 600 s / 2 at a time
