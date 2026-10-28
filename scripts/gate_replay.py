"""False-refusal replay: τ²'s GOLD actions through gated environments. Every deny of a gold write is a false refusal.

For every task of retail and airline (train and test splits) and banking_knowledge (all tasks; τ² does not split
them), a fresh environment is set to the task's initial state and gated with glint.gates; each gold action
(`evaluation_criteria.actions`) is sent through `get_response` the way an agent's call would be. An ungated twin of
the environment takes the same calls in step: a denied gold write that τ²'s own tool also rejects on the twin (gold
data with an action the environment refuses, such as an exchange on a pending order) changes nothing either way and
is counted as `also_rejected`, not as a false refusal. The target is 0 false refusals. Deterministic; no model, no network. The gate log is in memory unless --log PATH is given.

Usage: python scripts/gate_replay.py [--log PATH] [--json OUT]
"""
from __future__ import annotations

import json
import sys
from collections import Counter

from loguru import logger
from tau2.data_model.message import ToolCall
from tau2.registry import registry

from glint.gates import Enforcer, EventLog

DOMAINS = ("retail", "airline", "banking_knowledge")


def tasks_of(domain: str) -> list:
    loader = registry.get_tasks_loader(domain)
    if domain == "banking_knowledge":
        return list(loader(None))
    return [t for split in ("train", "test") for t in loader(split)]


def _env(domain: str, task):
    constructor = registry.get_env_constructor(domain)
    if domain == "banking_knowledge":
        return constructor(retrieval_variant="bm25_grep", task=task)
    return constructor()


def replay(domain: str, tasks: list, events: EventLog) -> dict:
    """Replay each task's gold actions; returns counts and the false refusals."""
    current = {"task_id": None}
    enforcer = Enforcer(domain, events, context=lambda: dict(current))
    refusals, also_rejected, writes, allowed, rules = [], [], 0, 0, Counter()
    for task in tasks:
        current["task_id"] = task.id
        env, twin = enforcer.attach(_env(domain, task)), _env(domain, task)
        init = task.initial_state
        for e in (env, twin):
            e.set_state(init.initialization_data if init else None, init.initialization_actions if init else None,
                        (init.message_history if init else None) or [])
        for k, action in enumerate(task.evaluation_criteria.actions or [] if task.evaluation_criteria else []):
            call = ToolCall(id=f"gold-{k}", name=action.name, arguments=action.arguments,
                            requestor=action.requestor)
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
                else:
                    refusals.append(row)
            else:
                allowed += 1
                if reply.error:
                    logger.warning(f"{domain} {task.id} gold {write[0]} was allowed but errored: {reply.content}")
    return {"domain": domain, "tasks": len(tasks), "gold_writes": writes, "allowed": allowed,
            "false_refusals": refusals, "also_rejected": also_rejected, "rules": dict(rules)}


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
