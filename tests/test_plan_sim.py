"""scripts/plan_sim.py on run files built here from τ²'s train gold (task_039): no recorded run data is read."""
import json

import plan_sim
from tau2.data_model.tasks import Task

from glint import paths

TASKS = {t["id"]: t for t in json.loads((paths.tau2_data() / "tau2" / "domains" / "banking_knowledge" /
                                         "tasks.json").read_text(encoding="utf-8"))}


def _sim(task_id: str, trial: int, filings: list[dict]) -> dict:
    calls = [{"name": "call_discoverable_agent_tool", "arguments": {
        "agent_tool_name": plan_sim.DISPUTE, "arguments": json.dumps(f)}} for f in filings]
    return {"task_id": task_id, "trial": trial, "messages": [{"role": "assistant", "tool_calls": calls}]}


def test_plan_order_gives_golds_flags_whatever_order_the_run_filed(tmp_path, capsys):
    task = Task.model_validate(TASKS["task_039"])
    gold = [json.loads(a.arguments["arguments"]) for a in task.evaluation_criteria.actions
            if a.name == "call_discoverable_agent_tool" and a.arguments.get("agent_tool_name") == plan_sim.DISPUTE]
    flipped = [g | {"eligible_for_provisional_credit": not g["eligible_for_provisional_credit"]} for g in gold]
    sims = [_sim("task_039", 0, list(reversed(flipped)) + gold[:1]),  # wrong flags, reversed, one retried
            _sim("task_039", 1, gold[3:] + gold[:3]),
            _sim("task_039", 2, []), _sim("task_054", 0, gold)]  # no dispute; a task not asked for
    runs = tmp_path / "tools-banking_knowledge.json"
    runs.write_text(json.dumps({"simulations": sims}), encoding="utf-8")
    assert plan_sim.run_filings(sims[0])[gold[0]["transaction_id"]] == gold[0]  # the last filing of a transaction
    out = tmp_path / "rows.json"
    assert plan_sim.main([str(runs), "task_039", "--json", str(out)]) == 0
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert [(r["task"], r["trial"]) for r in rows] == [("task_039", 0), ("task_039", 1), ("task_039", 2)]
    for r in rows[:2]:
        assert (r["matching"], r["gold"], r["denied"], r["wrong"], r["missing"]) == (8, 8, [], [], [])
        assert r["plan_flags"] == "TTTFFFFF"
    assert rows[2] == {"task": "task_039", "trial": 2, "disputes": 0}
    text = capsys.readouterr().out
    assert "flags matching gold 8/8" in text and "no dispute filed" in text and "every flag as gold: 2 of 3" in text


def test_report_names_wrong_and_missing_flags():
    row = {"task": "task_040", "trial": 1, "disputes": 8, "gold": 8, "denied": ["t9"], "plan_flags": "TTFFFFFF",
           "matching": 6, "wrong": ["t1"], "missing": ["t2"]}
    line = ("task_040 t1: 8 disputes, plan flags TTFFFFFF, denied 1, flags matching gold 6/8, wrong ['t1'], missing "
            "['t2']")
    assert plan_sim.report([row]).splitlines() == [line, "runs with every flag as gold: 0 of 1"]
