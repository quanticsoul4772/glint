"""Would dispute planning have set gold's provisional credit flags in recorded runs? (PR #85, tasks 039-041.)

For each recorded run of a task with credit card disputes in gold, the run's own disputes (every transaction it filed,
in the order it first filed them, with the arguments of its last filing of each) are filed again in plan order: a
fresh environment at the task's initial state, gated by glint.gates, the user's verification logged, plan_disputes
called with the run's disputes, then each planned dispute filed with the plan's eligible_for_provisional_credit. The
final flag of each of gold's transactions is compared with gold's. Deterministic; no model, no network. Reads the
run file only.

Usage: python scripts/plan_sim.py RUNS.json [TASK_ID ...] [--json OUT]
       (default tasks: task_039 task_040 task_041; e.g. RUNS = .../tools_ab_banking12_p2_gated_modal/
       tools-banking_knowledge.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from loguru import logger
from tau2.data_model.message import ToolCall
from tau2.registry import registry

from glint.assist.banking import CARD_DIGITS_TOOL, PLAN_TOOL
from glint.gates import Enforcer, EventLog
from glint.gates.rules import DISPUTE

WRAPPER = "call_discoverable_agent_tool"
DEFAULT_TASKS = ("task_039", "task_040", "task_041")
VERIFY_FIELDS = ("name", "user_id", "address", "email", "phone_number", "date_of_birth")
NOW = "2025-11-14 03:40:00 EST"  # τ² banking's get_current_time


def gold_flags(task) -> dict[str, bool]:
    """Gold's eligible_for_provisional_credit by transaction."""
    return {d["transaction_id"]: d["eligible_for_provisional_credit"] for d in
            (json.loads(a.arguments["arguments"]) for a in task.evaluation_criteria.actions
             if a.name == WRAPPER and a.arguments.get("agent_tool_name") == DISPUTE)}


def gold_refunds(task) -> dict[str, float]:
    """Gold's partial_refund_amount by transaction (incorrect_amount disputes): the overcharge the customer states.
    The simulation measures the credit flags, so the plan gets the amount a customer gives when asked."""
    return {d["transaction_id"]: d["partial_refund_amount"] for d in
            (json.loads(a.arguments["arguments"]) for a in task.evaluation_criteria.actions
             if a.name == WRAPPER and a.arguments.get("agent_tool_name") == DISPUTE)
            if d.get("partial_refund_amount") is not None}


def run_filings(sim: dict) -> dict[str, dict]:
    """transaction id -> the run's last filing of it, in the order the run first filed each transaction."""
    filings: dict[str, dict] = {}
    for m in sim["messages"]:
        for tc in m.get("tool_calls") or []:
            a = tc.get("arguments") or {}
            if tc["name"] == WRAPPER and a.get("agent_tool_name") == DISPUTE:
                f = json.loads(a["arguments"]) if isinstance(a["arguments"], str) else a["arguments"]
                filings[f["transaction_id"]] = f
    return filings


def _call(env, name: str, /, **args):
    return env.get_response(ToolCall(id="sim", name=name, arguments=args, requestor="assistant"))


def simulate(task, filings: dict[str, dict]) -> dict:
    """File `filings` in plan order through a gated environment; compare the final flags with gold's."""
    env = registry.get_env_constructor("banking_knowledge")(retrieval_variant="bm25_grep", task=task)
    init = task.initial_state
    env.set_state(init.initialization_data if init else None, init.initialization_actions if init else None, [])
    Enforcer("banking_knowledge", EventLog(":memory:")).attach(env)
    user_id = next(iter(filings.values()))["user_id"]
    user = env.tools.db.users.data[user_id]
    _call(env, "log_verification", time_verified=NOW, **{k: user[k] for k in VERIFY_FIELDS})
    # the card-digits tool every dispute on a card without digits on file needs given first (procedure.card_digits)
    _call(env, "give_discoverable_user_tool", discoverable_tool_name=CARD_DIGITS_TOOL)
    refunds = gold_refunds(task)
    items = [{k: f[k] for k in ("transaction_id", "dispute_reason", "contacted_merchant")}
             | ({"partial_refund_amount": refunds[f["transaction_id"]]} if f["transaction_id"] in refunds else {})
             | ({"issue_noticed_date": f["issue_noticed_date"]} if f.get("issue_noticed_date") else {})  # the run's
             for f in filings.values()]
    plan = json.loads(_call(env, PLAN_TOOL, user_id=user_id, transactions=items).content)
    denied = []
    for e in plan["order"]:
        flag = str(e["eligible_for_provisional_credit"]) == "True"  # τ² shows tool-result bools as strings
        args = (filings[e["transaction_id"]] | {"eligible_for_provisional_credit": flag}
                | {k: e[k] for k in ("resolution_requested", "partial_refund_amount") if k in e})
        _call(env, "unlock_discoverable_agent_tool", agent_tool_name=DISPUTE)
        reply = _call(env, WRAPPER, agent_tool_name=DISPUTE, arguments=json.dumps(args))
        if "was NOT executed" in str(reply.content):
            denied.append(e["transaction_id"])
    gold = gold_flags(task)
    final = {r["transaction_id"]: r["eligible_for_provisional_credit"]
             for r in env.tools.db.transaction_disputes.data.values()
             if r.get("user_id") == user_id and r["transaction_id"] in gold}
    return {"disputes": len(filings), "gold": len(gold), "denied": denied,
            "plan_flags": "".join("T" if str(e["eligible_for_provisional_credit"]) == "True" else "F"
                                  for e in plan["order"]),
            "matching": sum(final.get(t) == gold[t] for t in gold),
            "wrong": sorted(t for t in gold if t in final and final[t] != gold[t]),
            "missing": sorted(t for t in gold if t not in final)}


def run(runs_file: Path, task_ids: tuple[str, ...]) -> list[dict]:
    tasks = {t.id: t for t in registry.get_tasks_loader("banking_knowledge")(None)}
    rows = []
    for sim in json.loads(runs_file.read_text(encoding="utf-8"))["simulations"]:
        if sim["task_id"] not in task_ids:
            continue
        filings = run_filings(sim)
        row = {"task": sim["task_id"], "trial": sim["trial"]}
        rows.append(row | (simulate(tasks[sim["task_id"]], filings) if filings else {"disputes": 0}))
    return sorted(rows, key=lambda r: (r["task"], r["trial"]))


def report(rows: list[dict]) -> str:
    lines = []
    for r in rows:
        if not r["disputes"]:
            lines.append(f"{r['task']} t{r['trial']}: no dispute filed")
            continue
        lines.append(f"{r['task']} t{r['trial']}: {r['disputes']} disputes, plan flags {r['plan_flags']}, denied "
                     f"{len(r['denied'])}, flags matching gold {r['matching']}/{r['gold']}"
                     + (f", wrong {r['wrong']}" if r["wrong"] else "") + (f", missing {r['missing']}"
                                                                          if r["missing"] else ""))
    full = sum(1 for r in rows if r["disputes"] and r["matching"] == r["gold"])
    lines.append(f"runs with every flag as gold: {full} of {len(rows)}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    logger.remove()
    end = argv.index("--json") if "--json" in argv else len(argv)
    rows = run(Path(argv[0]), tuple(argv[1:end]) or DEFAULT_TASKS)
    print(report(rows))
    if "--json" in argv:
        Path(argv[argv.index("--json") + 1]).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
