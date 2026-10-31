"""A clean checkout runs the replays the README cites: the split files and the recorded wrong-write runs are in the
repo, no script needs the workspace or a key at import, and an unknown argument to tools_ab starts nothing."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_the_split_files_are_in_the_repo_and_every_reader_uses_them():
    import gate_replay
    import tools_ab

    import splits

    assert splits.DIR == ROOT / "splits"
    assert len(json.loads(splits.BANKING.read_text(encoding="utf-8"))["test_untouched"]) == 20
    assert {"retail", "airline"} <= set(json.loads(splits.PARTITION.read_text(encoding="utf-8")))
    assert gate_replay.BANKING_SPLIT == tools_ab.BANKING_SPLIT == splits.BANKING


def test_the_recorded_wrong_write_runs_are_in_the_repo():
    import bank_replay

    assert bank_replay.RECS == ROOT / "data-public" / "wrong-write-runs.json"
    runs = json.loads(bank_replay.RECS.read_text(encoding="utf-8"))
    assert len(runs) == 32 and {r["task"] for r in runs} == set(bank_replay.EXPECTED)


def test_the_replays_and_tools_ab_import_with_an_empty_workspace_and_no_keys(tmp_path):
    """In a fresh interpreter with GLINT_HOME pointing at an empty folder (no splits, no .env): every import the
    replays and the comparison runner need succeeds, so nothing reads the workspace or a key at import."""
    env = {k: v for k, v in os.environ.items() if k not in ("MODEL_API_KEY", "DEEPINFRA_API_KEY")}
    env |= {"GLINT_HOME": str(tmp_path), "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT / "scripts")])}
    code = "import gate_replay, bank_replay, tools_ab, m2_pilot; print(m2_pilot.META['api_key'])"
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", code], env=env, capture_output=True, text=True,
                       encoding="utf-8", timeout=300, check=False)  # the assert below shows stderr on failure
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().splitlines()[-1] == "None"
    assert list(tmp_path.iterdir()) == []  # nothing was written into the workspace either


def test_require_keys_stops_a_run_without_its_keys(monkeypatch):
    import m2_pilot

    monkeypatch.setattr(m2_pilot, "ENV", {"MODEL_API_KEY": "x"})
    m2_pilot.require_keys("MODEL_API_KEY")  # present: no stop
    with pytest.raises(SystemExit, match="DEEPINFRA_API_KEY"):
        m2_pilot.require_keys()  # the default asks for both
    monkeypatch.setattr(m2_pilot, "ENV", {})
    with pytest.raises(SystemExit, match="MODEL_API_KEY"):
        m2_pilot.require_keys("MODEL_API_KEY")


def test_tools_ab_starts_nothing_on_help_or_an_unknown_argument(monkeypatch, capsys):
    import tools_ab

    def no_run(cfg):
        raise AssertionError("a run started")

    monkeypatch.setattr(tools_ab, "generate", no_run)
    assert tools_ab.main(["--help"]) == 0
    assert "Usage" in capsys.readouterr().out
    assert tools_ab.main(["banking-unseen", "--modl"]) == 2
    assert "unknown argument(s): banking-unseen --modl" in capsys.readouterr().err


def test_tools_ab_checks_the_keys_before_a_run(monkeypatch):
    import m2_pilot
    import tools_ab

    monkeypatch.setattr(tools_ab, "generate", lambda cfg: pytest.fail("a run started without its keys"))
    monkeypatch.setattr(m2_pilot, "ENV", {})
    with pytest.raises(SystemExit, match="MODEL_API_KEY"):
        tools_ab.main(["design", "--modal"])
    monkeypatch.setattr(m2_pilot, "ENV", {"MODEL_API_KEY": "x"})
    with pytest.raises(SystemExit, match="DEEPINFRA_API_KEY"):  # DeepInfra's key only off Modal
        tools_ab.main(["design"])
