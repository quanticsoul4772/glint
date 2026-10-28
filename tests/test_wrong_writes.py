"""wrong_writes: per set and arm, the writes Glint's checks overrule, split into executed and rejected."""
import json

import gate_labels
import pytest
import wrong_writes

from glint.decisions import RETAIL_WRITES


def _gold_calls():
    task = gate_labels.tasks("retail")["0"]
    return [{"id": str(i), "name": a.name, "arguments": a.arguments}
            for i, a in enumerate(task.evaluation_criteria.actions)]


@pytest.fixture
def arms(make_sim, run_set):
    """Retail task 0: base has a gold run and a run cancelling an order that does not exist (overruled, rejected by
    the environment) plus a run that is not a result; tools has only the gold run."""
    gold = _gold_calls()
    writes = sum(c["name"] in RETAIL_WRITES for c in gold)
    assert writes > 0  # the fixture means nothing without writes to judge
    bad = [{"id": "x", "name": "cancel_pending_order", "arguments": {"order_id": "#no-such-order", "reason": "x"}}]
    msg = lambda calls: [{"role": "assistant", "content": None, "tool_calls": calls}]
    cfg = run_set({
        ("base", "retail"): [make_sim("0", 0, "pass", messages=msg(gold)), make_sim("0", 1, "fail", messages=msg(bad)),
                             make_sim("0", 2, "infrastructure_error", messages=msg(bad))],
        ("tools", "retail"): [make_sim("0", 0, "pass", messages=msg(gold))],
    }, k=3)
    return cfg, writes


def test_counts_overruled_writes_per_arm(arms):
    cfg, writes = arms
    c = wrong_writes.count_set(cfg["data"], ["retail"])["retail"]
    base, tools = c["base"], c["tools"]
    assert base["runs"] == 2 and base["dropped"] == {"infrastructure_error": 1}  # a non-result is left out, counted
    assert base["writes"] == writes + 1 and base["overruled"] == 1
    assert base["overruled_executed"] == 0  # the environment rejected the cancel itself
    assert base["by_task"]["0"] == {"writes": writes + 1, "overruled": 1, "overruled_executed": 0}
    assert tools["runs"] == 1 and tools["writes"] == writes and tools.get("overruled", 0) == 0


def test_a_missing_arm_is_left_out(arms):
    cfg, _ = arms
    (cfg["data"] / "tools-retail.json").unlink()
    assert set(wrong_writes.count_set(cfg["data"], ["retail"])["retail"]) == {"base"}


def test_main_reports_sets_with_both_arms_and_names_the_rest(arms, monkeypatch, tmp_path, capsys):
    cfg, writes = arms
    (cfg["data"] / "tasks.json").write_text(json.dumps({"retail": ["0"]}), encoding="utf-8")
    half = tmp_path / "half"
    half.mkdir()
    (half / "tasks.json").write_text(json.dumps({"retail": ["0"]}), encoding="utf-8")
    (half / "base-retail.json").write_text(json.dumps({"simulations": []}), encoding="utf-8")
    sets = {"full": {"data": cfg["data"]}, "half": {"data": half}, "none": {"data": tmp_path / "missing"}}
    monkeypatch.setattr(wrong_writes.tools_ab, "SETS", sets)
    monkeypatch.setattr(wrong_writes.tools_ab, "for_backend", lambda c, modal: c if not modal else
                        {"data": tmp_path / "missing"})
    monkeypatch.setattr(wrong_writes, "OUT", tmp_path / "wrong-writes.json")
    monkeypatch.setattr("sys.argv", ["wrong_writes.py"])
    wrong_writes.main()
    out = capsys.readouterr().out
    assert "NOTE: half: no domain has both arm files; not counted" in out
    assert f"{writes + 1:4} writes,   1 overruled (0 executed)" in out
    report = json.loads((tmp_path / "wrong-writes.json").read_text(encoding="utf-8"))
    assert set(report) == {"full"}
    assert report["full"]["retail"]["base"]["overruled"] == 1
