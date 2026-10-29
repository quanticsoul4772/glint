"""Catch rate of the write gates on RECORDED runs: would the gates have stopped the write that failed each run?

Inputs: the recorded runs themselves (τ² arm files `{base,tools}-{domain}.json` in each `--runs` folder, such as
tools_ab_banking12_modal and tools_ab_heldout2_modal; never the banking test split), the failure review's write labels
(`recs.json`: each run's writes by message index, labeled ok / wrongarg / dup / extra against τ²'s expected actions)
and its cause labels (`final.json`). A run is replayed only when its set has a `--runs` folder.

For each run, a fresh environment is set to the task's initial state and gated, and EVERY tool call of the run (reads
included, so the gates' own record of hidden lookups is rebuilt) is sent through `get_response` in order. An ungated
twin takes the same calls in step. A deny of a call τ²'s own tool also rejected changes nothing and the replay goes
on; any other deny stops it.

- A FAILED run is `caught` when the first deny is its first wrong or extra gated write, `denied_earlier` when the
  gates deny an earlier write labeled ok (a procedure gate stopping the run before its mistake, or a false refusal:
  each is listed), `missed` when its first wrong or extra write goes through (a later deny comes too late), and
  `no_wrong_write` when it has no wrong gated write and none is denied (it failed by a missing action, an overflow,
  or a wrong read).
- A PASSING run with any write denied is a false refusal on real traffic.

Deterministic; no model, no network.

Usage: python scripts/gate_catch.py RECS.json FINAL.json --runs DIR [DIR ...] [--json OUT]
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from loguru import logger
from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import Task
from tau2.environment.toolkit import ToolType
from tau2.registry import registry

from glint import paths
from glint.gates import Enforcer, EventLog

CAUSES = {"a": "wrong policy value", "b": "procedure step skipped or out of order",
          "c": "extra write", "d": "did not use a needed tool or info", "e": "overflow or max steps",
          "f": "ended or transferred early", "g": "wrong entity", "i": "wrong date argument"}
FAILED = ("fail", "context_overflow", "empty_reply")  # glint.runlog.outcome kinds that are failed runs
WRAPPER = "call_discoverable_agent_tool"


def load_tasks(domain: str) -> dict:
    path = paths.tau2_data() / "tau2" / "domains" / domain / "tasks.json"
    return {t["id"]: Task.model_validate(t) for t in json.loads(path.read_text(encoding="utf-8"))}


def environment(domain: str, task):
    constructor = registry.get_env_constructor(domain)
    env = constructor(retrieval_variant="bm25_grep", task=task) if domain == "banking_knowledge" else constructor()
    init = task.initial_state
    env.set_state(init.initialization_data if init else None, init.initialization_actions if init else None, [])
    return env


def set_name(folder: Path) -> str:
    """tools_ab_banking12_modal -> banking12 (the `set` of recs.json)."""
    return folder.name.removeprefix("tools_ab_").removesuffix("_modal")


def trajectory(sim: dict) -> list[dict]:
    """Every tool call of a saved simulation in order: message index, name, arguments, requestor."""
    calls = []
    for i, m in enumerate(sim["messages"]):
        for tc in m.get("tool_calls") or []:
            calls.append({"i": i, "name": tc["name"], "arguments": tc.get("arguments") or {}, "requestor": m["role"]})
    return calls


def label_of(call: dict, labels: dict) -> str | None:
    """The failure review's label of this call (ok / wrongarg / dup / extra), matched by message index and the
    (unwrapped) tool name; None for a call it did not label (a read, or a call the tool rejected)."""
    name = call["name"]
    if name == WRAPPER:
        name = str(call["arguments"].get("agent_tool_name", ""))
    return labels.get((call["i"], name))


def replay_run(dom: str, task, calls: list[dict], labels: dict, events: EventLog, enforcer: Enforcer) -> dict:
    env, twin = enforcer.attach(environment(dom, task)), environment(dom, task)
    first_bad, gated, also_rejected = None, 0, 0
    for k, c in enumerate(calls):
        call = ToolCall(id=f"r{k}", name=c["name"], arguments=c["arguments"], requestor=c["requestor"])
        if not env.tools.has_tool(call.name) and not (env.user_tools and env.user_tools.has_tool(call.name)):
            continue
        if call.name != WRAPPER and env.tools.has_tool(call.name) and env.tools.tool_type(call.name) == ToolType.READ:
            continue  # a plain read changes nothing the gates judge
        gate = enforcer.write_call(env, call)
        twin_reply = twin.get_response(call)
        before = events.count()
        env.get_response(call)
        if gate is None:
            continue
        gated += 1
        st = label_of(c, labels)
        if first_bad is None and st in ("wrongarg", "dup", "extra"):
            first_bad = {"index": k, "name": gate[0], "st": st}
        deny = next((e["payload"] for e in events.events("gate.fired", after=before)
                     if e["payload"]["decision"] == "deny"), None)
        if deny is None:
            continue
        if twin_reply.error:
            also_rejected += 1  # τ²'s tool rejected it too: the deny changes nothing
            continue
        denied = {"index": k, "name": gate[0], "st": st, "rule_id": deny["rule_id"], "reason": deny["reason"]}
        return {"first_bad": first_bad, "denied": denied, "gated_writes": gated, "also_rejected": also_rejected}
    return {"first_bad": first_bad, "denied": None, "gated_writes": gated, "also_rejected": also_rejected}


def classify(rec: dict, result: dict) -> str:
    bad, denied = result["first_bad"], result["denied"]
    if rec["outcome"] not in FAILED:
        return "false_refusal" if denied else "clean"
    if denied and bad and denied["index"] == bad["index"]:
        return "caught"
    if denied and (bad is None or denied["index"] < bad["index"]):
        return "denied_earlier"
    return "missed" if bad else "no_wrong_write"  # a deny after the first wrong write came too late


def run(recs: list[dict], causes: dict, folders: list[Path], events: EventLog) -> list[dict]:
    index = {(r["set"], r["dom"], r["arm"], r["task"], r["trial"]): r for r in recs}
    rows, enforcers, tasks = [], {}, {}
    for folder in folders:
        tag = set_name(folder)
        for arm_file in sorted(folder.glob("*-*.json")):
            arm, dom = arm_file.stem.split("-", 1)
            if dom not in enforcers:
                enforcers[dom], tasks[dom] = Enforcer(dom, events), load_tasks(dom)
            for sim in json.loads(arm_file.read_text(encoding="utf-8"))["simulations"]:
                rec = index.get((tag, dom, arm, sim["task_id"], sim["trial"]))
                if rec is None:
                    continue
                labels = {(w["i"], w["name"]): w["st"] for w in rec["writes"]}
                result = replay_run(dom, tasks[dom][rec["task"]], trajectory(sim), labels, events, enforcers[dom])
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
    lines.append(f"PASSING runs: {len(passed)}; false refusals {len(refusals)}")
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
    end = argv.index("--json") if "--json" in argv else len(argv)
    folders = [Path(p) for p in argv[argv.index("--runs") + 1:end]]
    rows = run(recs, causes, folders, EventLog(":memory:"))
    print(report(rows))
    if "--json" in argv:
        Path(argv[argv.index("--json") + 1]).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 1 if any(r["verdict"] == "false_refusal" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
