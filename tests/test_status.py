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
