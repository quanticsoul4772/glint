"""Catch rate of the write gates on RECORDED runs: would the gates have stopped the write that failed each run?

Input: the failure review's per-run write diffs (`recs.json`: every write of each run, in order, labeled ok / wrongarg
/ dup / extra against τ²'s expected actions) and its cause labels (`final.json`). For each run, a fresh environment
is set to the task's initial state (banking: with τ²'s read-log allowlist, as τ²'s runner builds it) and gated; the
run's writes are sent through `get_response` in order, the way the agent's calls were. An ungated twin takes the same
calls in step. The replay stops at the first deny.

- A FAILED run is `caught` when the first deny is its first wrong or extra gated write, `denied_earlier` when the
  gates deny an earlier write labeled ok (a procedure gate stopping the run before its mistake, or a false refusal:
  each is listed), `missed` when its first wrong or extra write goes through (a later deny comes too late), and
  `no_wrong_write` when it has no wrong gated write and none is denied (it failed by a missing action, an overflow,
  or a wrong read).
- A PASSING run with any write denied is a false refusal on real traffic, unless τ²'s own tool rejects that call too.

Deterministic; no model, no network.

Usage: python scripts/gate_catch.py RECS.json FINAL.json [--json OUT]
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from loguru import logger
from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import Task
from tau2.registry import registry
from tau2.runner.build import _derive_read_log_allowlist

from glint import paths
from glint.gates import Enforcer, EventLog

CAUSES = {"a": "wrong policy value", "b": "procedure step skipped or out of order",
          "c": "extra write", "d": "did not use a needed tool or info", "e": "overflow or max steps",
          "f": "ended or transferred early", "g": "wrong entity", "i": "wrong date argument"}
FAILED = ("fail", "context_overflow")


def load_tasks(domain: str) -> dict:
    path = paths.tau2_data() / "tau2" / "domains" / domain / "tasks.json"
    return {t["id"]: Task.model_validate(t) for t in json.loads(path.read_text(encoding="utf-8"))}


def environment(domain: str, task):
    constructor = registry.get_env_constructor(domain)
    if domain == "banking_knowledge":
        env = constructor(retrieval_variant="bm25_grep", task=task, read_log_allowlist=_derive_read_log_allowlist(task))
    else:
        env = constructor()
    init = task.initial_state
    env.set_state(init.initialization_data if init else None, init.initialization_actions if init else None, [])
    return env


def calls_for(env, write: dict, k: int) -> list[ToolCall]:
    """The tool call(s) that replay one recorded write: a hidden banking tool is unlocked, then called through the
    wrapper; user tools go through call_discoverable_user_tool with the user as requestor."""
    name, args, role = write["name"], write.get("args") or {}, write.get("role", "assistant")
    if name.startswith("give:"):
        return [ToolCall(id=f"w{k}", name="give_discoverable_user_tool", requestor="assistant",
                         arguments={"discoverable_tool_name": name[5:]})]
    if name.startswith("U:"):
        return [ToolCall(id=f"w{k}", name="call_discoverable_user_tool", requestor="user",
                         arguments={"discoverable_tool_name": name[2:], "arguments": json.dumps(args)})]
    tools = env.tools
    if hasattr(tools, "has_discoverable_tool") and tools.has_discoverable_tool(name):
        return [ToolCall(id=f"u{k}", name="unlock_discoverable_agent_tool", requestor="assistant",
                         arguments={"agent_tool_name": name}),
                ToolCall(id=f"w{k}", name="call_discoverable_agent_tool", requestor="assistant",
                         arguments={"agent_tool_name": name, "arguments": json.dumps(args)})]
    return [ToolCall(id=f"w{k}", name=name, arguments=args, requestor=role)]


def replay_run(rec: dict, task, events: EventLog, enforcer: Enforcer) -> dict:
    """Replay one run's writes; returns the first gated wrong write and the first deny (or None)."""
    env, twin = enforcer.attach(environment(rec["dom"], task)), environment(rec["dom"], task)
    first_bad, denied, gated = None, None, 0
    for k, write in enumerate(rec["writes"]):
        for call in calls_for(env, write, k):
            gate = enforcer.write_call(env, call)
            twin_reply = twin.get_response(call)
            before = events.count()
            env.get_response(call)
            if gate is None:
                continue
            gated += 1
            if first_bad is None and write["st"] != "ok":
                first_bad = {"index": k, "name": write["name"], "st": write["st"]}
            deny = next((e["payload"] for e in events.events("gate.fired", after=before)
                         if e["payload"]["decision"] == "deny"), None)
            if deny is not None:
                denied = {"index": k, "name": write["name"], "st": write["st"], "rule_id": deny["rule_id"],
                          "reason": deny["reason"], "tool_also_rejects": bool(twin_reply.error)}
                return {"first_bad": first_bad, "denied": denied, "gated_writes": gated}
    return {"first_bad": first_bad, "denied": None, "gated_writes": gated}


def classify(rec: dict, result: dict) -> str:
    bad, denied = result["first_bad"], result["denied"]
    if rec["outcome"] not in FAILED:
        return "false_refusal" if denied and not denied["tool_also_rejects"] else (
            "denied_tool_also_rejects" if denied else "clean")
    if denied and bad and denied["index"] == bad["index"]:
        return "caught"
    if denied and (bad is None or denied["index"] < bad["index"]):
        return "denied_earlier"
    return "missed" if bad else "no_wrong_write"  # a deny after the first wrong write came too late


def run(recs: list[dict], causes: dict, events: EventLog) -> list[dict]:
    rows, enforcers, tasks = [], {}, {}
    for rec in recs:
        dom = rec["dom"]
        if dom not in enforcers:
            enforcers[dom], tasks[dom] = Enforcer(dom, events), load_tasks(dom)
        result = replay_run(rec, tasks[dom][rec["task"]], events, enforcers[dom])
        key = (rec["set"], rec["arm"], rec["task"], rec["trial"])
        rows.append({k: rec[k] for k in ("set", "dom", "arm", "task", "trial", "outcome")}
                    | {"cause": causes.get(key)} | result | {"verdict": classify(rec, result)})
    return rows


def report(rows: list[dict]) -> str:
    failed = [r for r in rows if r["outcome"] in FAILED]
    passed = [r for r in rows if r["outcome"] not in FAILED]
    lines = [f"FAILED runs: {len(failed)}"]
    for label, key in (("by cause", "cause"), ("by domain", "dom")):
        table: dict[str, Counter] = defaultdict(Counter)
        for r in failed:
            table[str(r[key])][r["verdict"]] += 1
        lines.append(f"  {label}: caught / denied earlier / missed / no wrong write (of n)")
        for name in sorted(table):
            c = table[name]
            title = CAUSES.get(name, name) if key == "cause" else name
            lines.append(f"    {title}: {c['caught']} / {c['denied_earlier']} / {c['missed']} / "
                         f"{c['no_wrong_write']} (of {sum(c.values())})")
    total = Counter(r["verdict"] for r in failed)
    lines.append(f"  TOTAL: caught {total['caught']}, denied earlier {total['denied_earlier']}, missed "
                 f"{total['missed']}, no wrong write {total['no_wrong_write']}")
    rules = Counter(r["denied"]["rule_id"] for r in failed if r["denied"])
    lines.append("  denies by rule: " + ", ".join(f"{k} {v}" for k, v in rules.most_common()))
    for r in failed:
        if r["verdict"] == "denied_earlier":
            d = r["denied"]
            lines.append(f"    denied earlier {r['set']} {r['arm']} {r['task']} t{r['trial']} #{d['index']} "
                         f"{d['name']} ({d['st']}) [{d['rule_id']}]: {d['reason'][:160]}")
    refusals = [r for r in passed if r["verdict"] == "false_refusal"]
    lines.append(f"PASSING runs: {len(passed)}; false refusals {len(refusals)}; denies τ²'s tool also rejects "
                 f"{sum(r['verdict'] == 'denied_tool_also_rejects' for r in passed)}")
    for r in refusals:
        d = r["denied"]
        lines.append(f"  FALSE REFUSAL {r['set']} {r['arm']} {r['dom']} {r['task']} t{r['trial']} #{d['index']} "
                     f"{d['name']} [{d['rule_id']}]: {d['reason'][:200]}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    logger.remove()
    recs = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    final = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    causes = {(f["set"], f["arm"], f["task"], f["trial"]): f["cause"] for f in final}
    rows = run(recs, causes, EventLog(":memory:"))
    print(report(rows))
    if "--json" in argv:
        Path(argv[argv.index("--json") + 1]).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 1 if any(r["verdict"] == "false_refusal" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
