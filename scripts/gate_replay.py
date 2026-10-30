"""False-refusal replay: τ²'s GOLD actions through gated environments. Every deny of a gold write is a false refusal.

For every TRAIN task of retail, airline and banking_knowledge (`tasks_of`; τ²'s test splits stay unread), a fresh
environment is set to the task's initial state and gated with glint.gates; each gold action
(`evaluation_criteria.actions`) is sent through `get_response` the way an agent's call would be. An ungated twin of
the environment takes the same calls in step: a denied gold write that τ²'s own tool also rejects on the twin (gold
data with an action the environment refuses, such as an exchange on a pending order) changes nothing either way and
is counted as `also_rejected`, not as a false refusal; a denied gold write of PLACEHOLDER_GOLD is listed apart. The
target is 0 false refusals. Gold never calls Glint's plan_disputes, which the banking plan gate requires before a
credit card dispute: the replay inserts the call an agent following the gate's Fix makes (glint.gates.replayplan:
the task's own gold filings, in gold's order) before the first filing, so the gate judges gold's order and flags. Deterministic; no model, no network. The gate log is in memory unless --log PATH is given.

Usage: python scripts/gate_replay.py [--log PATH] [--json OUT]
"""
from __future__ import annotations

import json
import sys
from collections import Counter

from loguru import logger
from tau2.data_model.message import ToolCall
from tau2.registry import registry
from tau2.runner.build import _derive_read_log_allowlist

from glint import paths
from glint.assist.banking import PLAN_TOOL
from glint.gates import Enforcer, EventLog
from glint.gates.replayplan import ADDRESS_ASK_SUFFIX, with_address_asks, with_plans
from glint.gates.rules import ADDRESS_ASK

DOMAINS = ("retail", "airline", "banking_knowledge")
BANKING_SPLIT = paths.RESULTS / "split-partition-banking.json"
# Gold data the gates deny by policy, listed and reported on its own, never silently: task_005's gold logs a
# verification whose every field is the placeholder '9K2X7M4P1N8Q3R5T6A' (no such user) and then changes the email of
# a user who was never verified (the user simulator is told to fail verification)
PLACEHOLDER_GOLD = {("banking_knowledge", "task_005"): "placeholder verification; email change of an unverified user"}


def tasks_of(domain: str) -> list:
    """The TRAIN tasks only: τ²'s test splits stay unread (research/00-decisions.md). Retail and airline use τ²'s
    train split; banking_knowledge has none, so its test tasks are the `test_untouched` 20 of
    split-partition-banking.json and the rest (pilot + rest, 77 tasks) are replayed."""
    loader = registry.get_tasks_loader(domain)
    if domain == "banking_knowledge":
        untouched = set(json.loads(BANKING_SPLIT.read_text(encoding="utf-8"))["test_untouched"])
        return [t for t in loader(None) if t.id not in untouched]
    return list(loader("train"))


def _env(domain: str, task):
    """The environment τ²'s runner builds for `task`; for banking that includes the read log, which records the
    hidden reads gold makes in the scored database (tau2.runner.build._build_env_kwargs)."""
    constructor = registry.get_env_constructor(domain)
    if domain == "banking_knowledge":
        return constructor(retrieval_variant="bm25_grep", task=task,
                           read_log_allowlist=_derive_read_log_allowlist(task))
    return constructor()


def replay(domain: str, tasks: list, events: EventLog) -> dict:
    """Replay each task's gold actions; returns counts and the false refusals."""
    current = {"task_id": None}
    enforcer = Enforcer(domain, events, context=lambda: dict(current))
    refusals, also_rejected, placeholder, writes, allowed, rules = [], [], [], 0, 0, Counter()
    for task in tasks:
        current["task_id"] = task.id
        env, twin = enforcer.attach(_env(domain, task)), _env(domain, task)
        init = task.initial_state
        for e in (env, twin):
            e.set_state(init.initialization_data if init else None, init.initialization_actions if init else None,
                        (init.message_history if init else None) or [])
        gold = [ToolCall(id=f"gold-{k}", name=a.name, arguments=a.arguments, requestor=a.requestor)
                for k, a in enumerate(task.evaluation_criteria.actions or [] if task.evaluation_criteria else [])]
        users = env.tools.db.users.data if domain == "banking_knowledge" else {}
        calls = (with_address_asks(with_plans(gold), lambda u: (users.get(u) or {}).get("address"))
                 if domain == "banking_knowledge" else gold)
        for call in calls:
            k = call.id.removeprefix("gold-")
            if call.name == PLAN_TOOL:  # Glint's tool: the ungated twin has none
                if env.get_response(call).error:
                    raise RuntimeError(f"{task.id}: plan_disputes failed on gold's disputes")
                continue
            if call.id.endswith(ADDRESS_ASK_SUFFIX):  # the order before asking: refused once, never sent to the twin
                before = events.count()
                env.get_response(call)
                fired = [e["payload"]["rule_id"] for e in events.events("gate.fired", after=before)
                         if e["payload"]["decision"] == "deny"]
                if fired[:1] != [ADDRESS_ASK]:
                    raise RuntimeError(f"{task.id}: the inserted order was not refused by {ADDRESS_ASK}: {fired}")
                continue
            write = enforcer.write_call(env, call)
            ungated = twin.get_response(call)
            if write is None:
                env.get_response(call)
                continue
            writes += 1
            before = events.count()
            reply = env.get_response(call)
            fired = events.events("gate.fired", after=before)
            denied = [e["payload"] for e in fired if e["payload"]["decision"] == "deny"]
            for e in fired:
                rules[e["payload"]["rule_id"]] += 1
            if denied:
                d = denied[0]
                row = {"domain": domain, "task_id": task.id, "action": k, "tool": write[0], "gate": d["gate"],
                       "rule_id": d["rule_id"], "reason": d["reason"]}
                if ungated.error:
                    also_rejected.append(row | {"environment": str(ungated.content)})
                elif (domain, task.id) in PLACEHOLDER_GOLD:
                    placeholder.append(row | {"why": PLACEHOLDER_GOLD[(domain, task.id)]})
                else:
                    refusals.append(row)
            else:
                allowed += 1
                if reply.error:
                    logger.warning(f"{domain} {task.id} gold {write[0]} was allowed but errored: {reply.content}")
    return {"domain": domain, "tasks": len(tasks), "gold_writes": writes, "allowed": allowed,
            "false_refusals": refusals, "also_rejected": also_rejected, "placeholder_gold": placeholder,
            "rules": dict(rules)}


def main(argv: list[str]) -> int:
    logger.remove()
    log_path = argv[argv.index("--log") + 1] if "--log" in argv else ":memory:"
    events = EventLog(log_path)
    reports = [replay(d, tasks_of(d), events) for d in DOMAINS]
    events.verify()
    total = sum(len(r["false_refusals"]) for r in reports)
    for r in reports:
        print(f"{r['domain']}: {r['tasks']} tasks, {r['gold_writes']} gold writes, {r['allowed']} allowed, "
              f"{len(r['false_refusals'])} false refusals, {len(r['also_rejected'])} denies of gold writes τ²'s tool "
              "also rejects")
        for rule, n in sorted(r["rules"].items()):
            print(f"  {rule}: {n}")
        for f in r["false_refusals"]:
            print(f"  FALSE REFUSAL {f['task_id']} {f['tool']} [{f['gate']} {f['rule_id']}]: {f['reason']}")
        for f in r["placeholder_gold"]:
            print(f"  placeholder gold {f['task_id']} {f['tool']} [{f['rule_id']}]: {f['why']}")
        for f in r["also_rejected"]:
            print(f"  also rejected by τ² {f['task_id']} {f['tool']} [{f['rule_id']}]: {f['reason']} | "
                  f"{f['environment']}")
    print(f"TOTAL false refusals on gold: {total}")
    if "--json" in argv:
        with open(argv[argv.index("--json") + 1], "w", encoding="utf-8") as fh:
            json.dump(reports, fh, indent=1)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
