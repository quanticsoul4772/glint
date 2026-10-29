"""glint-bank replays through the MCP tool-call path (design V1 and V2, research/3-project/glint-bank-design.md §8).

- **gold** (V1, false refusals): for each of the 77 banking TRAIN tasks (pilot + rest; the 20 test_untouched stay unread), `load_scenario(task)` through an in-process MCP
  client, then every gold action: the assistant's through `call_tool` (the server's handler, so the session's gates),
  the user's straight into the session environment as `requestor="user"` (the server never takes customer calls). An
  ungated twin environment takes the same calls in step. A deny of a gold write the twin's τ² tool runs is a false
  refusal (target 0); a deny the twin's tool also rejects is counted apart. At the end the session's agent and user
  database hashes must equal the twin's. Gold never calls plan_disputes, which the plan gate requires before a credit
  card dispute: the call an agent following the gate's Fix makes (glint.gates.replayplan, gold's own filings) goes
  through `call_tool` before the first filing; the twin has no such tool and skips it.
- **wrong** (V2): the recorded wrong writes of Glimmer's banking12 runs (research/5-measurements/
  failure-review-2026-10-28/recs.json): task_054's over-maximum increase submit, task_051's approve with no prior deny,
  task_040/041's dispute with wrong arguments. A run recorded before plan_disputes existed gets the plan made from its
  own filings, in its order (glint.gates.replayplan), before its first dispute; a plan-gate deny then means the run
  left that plan. Each run's writes are sent in order through the server on a fresh
  scenario until the first deny, which must carry the task's expected rule, or a gates-phase-2 procedure rule
  (verification, required lookups, transfer) on a step the run's full call list shows it skipped, or on a write the
  review marked wrong or extra. The failure review judged writes against
  gold, and gold's provisional-credit flag depends on gold's filing order: a dispute a run filed in another order can
  be right on the live database (the gate allows it), and one whose flag matches gold can be wrong there (the gate
  denies it). task_051 runs that paid the card down before submitting get an approve verdict on the live database.
  Customer-side entries (a given user tool, the customer running it) are skipped: the server takes no customer calls,
  and no gate reads them.

Deterministic; no model, no network. Usage: python scripts/bank_replay.py [--recs PATH] [--json OUT]
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

import anyio
import gate_replay  # scripts/ is on the path when this runs as a script or under the tests
from loguru import logger
from mcp.client.client import Client
from tau2.data_model.message import ToolCall

from glint import paths
from glint.assist.banking import PLAN_TOOL
from glint.bank.server import BankServer, build_environment, tasks
from glint.gates import rules
from glint.gates.replayplan import plan_arguments, with_plans

RECS = paths.RESULTS / "failure-review-2026-10-28" / "recs.json"
DOMAIN = "banking_knowledge"


def train_task_ids() -> list[str]:
    """The 77 banking train tasks (pilot + rest), as gate_replay replays them: the 20 test_untouched tasks of
    split-partition-banking.json stay unread (research/00-decisions.md, 2026-10-28)."""
    return sorted(t.id for t in gate_replay.tasks_of(DOMAIN))
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
        refusals, also_rejected, placeholder, placeholder_hashes, mismatches, errors = [], [], [], [], [], []
        writes = allowed = 0
        ids = task_ids or train_task_ids()
        async with Client(bank.mcp_server()) as client:
            for task_id in ids:
                task = tasks()[task_id]
                loaded = await client.call_tool("load_scenario", {"task_id": task_id})
                if loaded.is_error:
                    raise RuntimeError(f"load_scenario({task_id}) failed: {_text(loaded)}")
                session, twin = bank.session(), build_environment(task)
                actions = task.evaluation_criteria.actions or [] if task.evaluation_criteria else []
                calls = with_plans([ToolCall(id=f"gold-{k}", name=a.name, arguments=a.arguments,
                                             requestor=a.requestor) for k, a in enumerate(actions)])
                for call in calls:
                    k, action = call.id.removeprefix("gold-"), call
                    if call.name == PLAN_TOOL:  # Glint's tool: the ungated twin has none
                        reply = await client.call_tool(PLAN_TOOL, call.arguments)
                        if reply.is_error:
                            errors.append({"task_id": task_id, "action": k, "tool": PLAN_TOOL, "text": _text(reply)})
                        continue
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
                        if ungated.error:
                            also_rejected.append(row)
                        elif (DOMAIN, task_id) in gate_replay.PLACEHOLDER_GOLD:
                            placeholder.append(row | {"why": gate_replay.PLACEHOLDER_GOLD[(DOMAIN, task_id)]})
                        else:
                            refusals.append(row)
                    else:
                        allowed += 1
                        if reply.is_error != bool(ungated.error):
                            errors.append({"task_id": task_id, "action": k, "tool": write[0], "text": _text(reply)})
                with session.lock:
                    got = (session.env.get_db_hash(), session.env.get_user_db_hash())
                want = (twin.get_db_hash(), twin.get_user_db_hash())
                if got != want:
                    row = {"task_id": task_id, "server": got, "twin": want}
                    if any(p["task_id"] == task_id for p in placeholder):  # its gold writes were denied: expected
                        placeholder_hashes.append(row)
                    else:
                        mismatches.append(row)
        session.events.verify()
        session.events.close()
    return {"tasks": len(ids), "gold_writes": writes, "allowed": allowed, "false_refusals": refusals,
            "also_rejected": also_rejected, "placeholder_gold": placeholder,
            "placeholder_hash_differs": [r["task_id"] for r in placeholder_hashes], "hash_mismatches": mismatches,
            "result_mismatches": errors}


# ---- V2: recorded wrong writes ------------------------------------------------------------------------------------

PROCEDURE_RULES = ("procedure.", "banking.verification")  # gates phase 2: the steps before a write, and its record
PLAN_RULES = "procedure.dispute_plan"  # the run left the plan made from its own disputes (order or values)
LOOKUP = re.compile(r"get_[a-z_]+_\d{4}")


def skipped_in_recording(rec: dict, i: int, rule_id: str, reason: str) -> bool | None:
    """For a procedure deny of write i: whether the run's full call list confirms the step was skipped. The recorded
    writes keep only the hidden reads gold names, so the replay's own lookup record can lack a read the run did make;
    the call list (names, in order) does not. Judged on every call before the LAST call of that tool name, so a step
    done anywhere before it counts as done. None for a rule that judges the write's own arguments."""
    name = rec["writes"][i]["name"]
    calls = rec.get("calls") or []
    before = set(calls[:max((k for k, c in enumerate(calls) if c == name), default=len(calls))])
    if rule_id == "procedure.lookups":
        return not (set(LOOKUP.findall(reason.split(";")[0])) & before)
    if rule_id == "procedure.verified":
        return "log_verification" not in before
    return None


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
    the first deny. That deny must carry the task's expected rule, or a procedure rule the recording supports
    (`skipped_in_recording`, or a write recorded as wrong or extra), as an error result that leaves the database
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
                plans = plan_arguments([w["args"] for w in rec["writes"] if w["name"] == rules.DISPUTE])
                for i, w in enumerate(rec["writes"]):
                    if w["name"].startswith("give:"):  # the run gave the customer a tool: the card-digits gate reads it
                        await client.call_tool("give_discoverable_user_tool",
                                               {"discoverable_tool_name": w["name"].split(":", 1)[1]})
                        continue
                    if w["name"].startswith("U:"):  # the customer's own call
                        continue
                    user = str(w["args"].get("user_id", ""))
                    if w["name"] == rules.DISPUTE and user in plans:
                        await client.call_tool(PLAN_TOOL, plans.pop(user))
                    before, db = session.events.count(), session.env.get_db_hash()
                    reply = await _send(client, w["name"], w["args"])
                    denied = _denies(session, before)
                    if denied:
                        expected = EXPECTED[rec["task"]][1]
                        row = {"run": run, "write": w["i"], "tool": w["name"], "recorded_wrong": recorded_wrong(rec, i),
                               "rule_id": denied[0]["rule_id"], "expected_rule": expected,
                               "is_error": reply.is_error, "db_unchanged": session.env.get_db_hash() == db,
                               "reason": denied[0]["reason"], "message": _text(reply)}
                        row["skipped_in_recording"] = skipped_in_recording(rec, i, row["rule_id"], row["reason"])
                        reason_ok = row["rule_id"] == expected or row["rule_id"].startswith(PLAN_RULES) or (
                            row["rule_id"].startswith(PROCEDURE_RULES) and row["skipped_in_recording"] is not False
                            and (row["recorded_wrong"] or row["skipped_in_recording"] or w["st"] != "ok"))
                        row["ok"] = reason_ok and row["is_error"] and row["db_unchanged"]
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
          f"{len(g['placeholder_gold'])} denies of placeholder gold, "
          f"{len(g['hash_mismatches'])} database hash mismatches with the ungated twin, "
          f"{len(g['result_mismatches'])} result mismatches")
    for f in g["placeholder_gold"]:
        print(f"  placeholder gold {f['task_id']} {f['tool']} [{f['rule_id']}]: {f['why']}")
    if g["placeholder_hash_differs"]:
        print(f"  database differs from the twin only where placeholder gold was denied: {g['placeholder_hash_differs']}")
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
