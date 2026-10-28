"""glint-bank replays through the MCP tool-call path (design V1 and V2, research/3-project/glint-bank-design.md §8).

- **gold** (V1, false refusals): for each of the 97 banking tasks, `load_scenario(task)` through an in-process MCP
  client, then every gold action: the assistant's through `call_tool` (the server's handler, so the session's gates),
  the user's straight into the session environment as `requestor="user"` (the server never takes customer calls). An
  ungated twin environment takes the same calls in step. A deny of a gold write the twin's τ² tool runs is a false
  refusal (target 0); a deny the twin's tool also rejects is counted apart. At the end the session's agent and user
  database hashes must equal the twin's.
- **wrong** (V2): the recorded wrong writes of Glimmer's banking12 runs (research/5-measurements/
  failure-review-2026-10-28/recs.json): task_054's over-maximum increase submit, task_051's approve with no prior deny,
  task_040/041's dispute with wrong arguments. Each run's writes are sent in order through the server on a fresh
  scenario until the first deny, which must carry the task's expected rule. The failure review judged writes against
  gold, and gold's provisional-credit flag depends on gold's filing order: a dispute a run filed in another order can
  be right on the live database (the gate allows it), and one whose flag matches gold can be wrong there (the gate
  denies it). task_051 runs that paid the card down before submitting get an approve verdict on the live database.
  Customer-side entries (a given user tool, the customer running it) are skipped: the server takes no customer calls,
  and no gate reads them.

Deterministic; no model, no network. Usage: python scripts/bank_replay.py [--recs PATH] [--json OUT]
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import anyio
from loguru import logger
from mcp.client.client import Client
from tau2.data_model.message import ToolCall

from glint import paths
from glint.bank.server import BankServer, build_environment, tasks
from glint.gates import rules

RECS = paths.RESULTS / "failure-review-2026-10-28" / "recs.json"
WRAPPER, UNLOCK = "call_discoverable_agent_tool", "unlock_discoverable_agent_tool"
VISIBLE = {"log_verification", "change_user_email", "transfer_to_human_agents"}  # agent writes called by name
# the recorded wrong writes the design names, and the rule each must be denied by
EXPECTED = {"task_054": (rules.SUBMIT, "banking.cli.submit.reduce_amount"),
            "task_051": (rules.APPROVE, "banking.cli.approve"),
            "task_040": (rules.DISPUTE, "banking.dispute"),
            "task_041": (rules.DISPUTE, "banking.dispute")}


def _text(result) -> str:
    return result.content[0].text if result.content else ""


def _denies(session, before: int) -> list[dict]:
    return [e["payload"] for e in session.events.events("gate.fired", after=before)
            if e["payload"]["decision"] == "deny"]


# ---- V1: gold actions ---------------------------------------------------------------------------------------------

async def gold(task_ids: list[str] | None = None, home: Path | None = None) -> dict:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        bank = BankServer(home=home or Path(tmp), sid="gold-replay")
        refusals, also_rejected, mismatches, errors = [], [], [], []
        writes = allowed = 0
        ids = task_ids or sorted(tasks())
        async with Client(bank.mcp_server()) as client:
            for task_id in ids:
                task = tasks()[task_id]
                loaded = await client.call_tool("load_scenario", {"task_id": task_id})
                if loaded.is_error:
                    raise RuntimeError(f"load_scenario({task_id}) failed: {_text(loaded)}")
                session, twin = bank.session(), build_environment(task)
                actions = task.evaluation_criteria.actions or [] if task.evaluation_criteria else []
                for k, action in enumerate(actions):
                    call = ToolCall(id=f"gold-{k}", name=action.name, arguments=action.arguments,
                                    requestor=action.requestor)
                    ungated = twin.get_response(call)
                    if action.requestor != "assistant":
                        with session.lock:
                            session.env.get_response(call)
                        continue
                    write = session.enforcer.write_call(session.env, call)
                    before = session.events.count()
                    reply = await client.call_tool(action.name, action.arguments)
                    if write is None:
                        if reply.is_error != bool(ungated.error):
                            errors.append({"task_id": task_id, "action": k, "tool": action.name, "text": _text(reply)})
                        continue
                    writes += 1
                    denied = _denies(session, before)
                    if denied:
                        row = {"task_id": task_id, "action": k, "tool": write[0], "rule_id": denied[0]["rule_id"],
                               "reason": denied[0]["reason"]}
                        (also_rejected if ungated.error else refusals).append(row)
                    else:
                        allowed += 1
                        if reply.is_error != bool(ungated.error):
                            errors.append({"task_id": task_id, "action": k, "tool": write[0], "text": _text(reply)})
                with session.lock:
                    got = (session.env.get_db_hash(), session.env.get_user_db_hash())
                want = (twin.get_db_hash(), twin.get_user_db_hash())
                if got != want:
                    mismatches.append({"task_id": task_id, "server": got, "twin": want})
        session.events.verify()
        session.events.close()
    return {"tasks": len(ids), "gold_writes": writes, "allowed": allowed, "false_refusals": refusals,
            "also_rejected": also_rejected, "hash_mismatches": mismatches, "result_mismatches": errors}


# ---- V2: recorded wrong writes ------------------------------------------------------------------------------------

def recorded_wrong(rec: dict, i: int) -> bool:
    """Whether write i of a recorded run is wrong by the failure review's gold-based label: an increase submit or a
    dispute it marked wrongarg, or (task_051) an approve with no deny of the request before it."""
    w = rec["writes"][i]
    if w["name"] == rules.APPROVE:
        return rec["task"] == "task_051" and not any(x["name"] == rules.DENY for x in rec["writes"][:i])
    return w["name"] in (rules.SUBMIT, rules.DISPUTE) and w["st"] == "wrongarg"


async def _send(client, name: str, args: dict):
    if name in VISIBLE:
        return await client.call_tool(name, args)
    await client.call_tool(UNLOCK, {"agent_tool_name": name})
    return await client.call_tool(WRAPPER, {"agent_tool_name": name, "arguments": json.dumps(args)})


async def wrong(recs: list[dict], home: Path | None = None) -> dict:
    """Each recorded run of the EXPECTED tasks, its writes sent in order through the server on a fresh scenario until
    the first deny. That deny must carry the task's expected rule, as an error result that leaves the database
    unchanged; whether gold's label also calls the write wrong is reported. A run with no deny is listed with its
    gold-labeled wrong writes."""
    stopped, not_stopped = [], []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        bank = BankServer(home=home or Path(tmp), sid="wrong-replay")
        async with Client(bank.mcp_server()) as client:
            for rec in recs:
                if rec.get("dom") != "banking_knowledge" or rec["task"] not in EXPECTED:
                    continue
                run = f"{rec['set']} {rec['arm']} {rec['task']} trial {rec['trial']}"
                await client.call_tool("load_scenario", {"task_id": rec["task"]})
                session, row = bank.session(), None
                for i, w in enumerate(rec["writes"]):
                    if w["name"].startswith(("give:", "U:")):
                        continue
                    before, db = session.events.count(), session.env.get_db_hash()
                    reply = await _send(client, w["name"], w["args"])
                    denied = _denies(session, before)
                    if denied:
                        expected = EXPECTED[rec["task"]][1]
                        row = {"run": run, "write": w["i"], "tool": w["name"], "recorded_wrong": recorded_wrong(rec, i),
                               "rule_id": denied[0]["rule_id"], "expected_rule": expected,
                               "is_error": reply.is_error, "db_unchanged": session.env.get_db_hash() == db,
                               "reason": denied[0]["reason"], "message": _text(reply)}
                        row["ok"] = row["rule_id"] == expected and row["is_error"] and row["db_unchanged"]
                        break
                if row is None:
                    not_stopped.append({"run": run, "recorded_wrong": [
                        f"i{w['i']} {w['name']}" for i, w in enumerate(rec["writes"]) if recorded_wrong(rec, i)]})
                else:
                    stopped.append(row)
        bank.session().events.close()
    return {"stopped": stopped, "not_stopped": not_stopped, "denied_as_expected": sum(r["ok"] for r in stopped)}


def main(argv: list[str]) -> int:
    logger.remove()
    recs_path = Path(argv[argv.index("--recs") + 1]) if "--recs" in argv else RECS
    g = anyio.run(gold)
    print(f"gold: {g['tasks']} tasks, {g['gold_writes']} gold writes through MCP, {g['allowed']} allowed, "
          f"{len(g['false_refusals'])} false refusals, {len(g['also_rejected'])} denies τ²'s tool also rejects, "
          f"{len(g['hash_mismatches'])} database hash mismatches with the ungated twin, "
          f"{len(g['result_mismatches'])} result mismatches")
    for f in g["false_refusals"]:
        print(f"  FALSE REFUSAL {f['task_id']} {f['tool']} [{f['rule_id']}]: {f['reason']}")
    for m in g["hash_mismatches"]:
        print(f"  HASH MISMATCH {m['task_id']}")
    for m in g["result_mismatches"]:
        print(f"  RESULT MISMATCH {m['task_id']} {m['tool']}: {m['text'][:160]}")
    w = anyio.run(wrong, json.loads(recs_path.read_text(encoding="utf-8")))
    print(f"wrong writes: {len(w['stopped'])} runs stopped by a deny, {w['denied_as_expected']} of them with the "
          f"expected rule (error result, database unchanged); {len(w['not_stopped'])} runs with no deny")
    for r in w["stopped"]:
        print(f"  {'ok  ' if r['ok'] else 'BAD '} {r['run']} i{r['write']} {r['tool']}: {r['rule_id']} "
              f"(expected {r['expected_rule']})"
              f"{'' if r['recorded_wrong'] else '; gold label ok (another filing order)'}"
              f" | {r['reason'][:120]}")
    for r in w["not_stopped"]:
        print(f"  no deny: {r['run']}; gold-labeled wrong writes: {r['recorded_wrong'] or 'none'}")
    if "--json" in argv:
        Path(argv[argv.index("--json") + 1]).write_text(json.dumps({"gold": g, "wrong": w}, indent=1),
                                                        encoding="utf-8")
    bad = (g["false_refusals"] or g["hash_mismatches"] or g["result_mismatches"]
           or w["denied_as_expected"] != len(w["stopped"]) or not w["stopped"])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
