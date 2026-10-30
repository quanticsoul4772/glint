"""Why a run failed: the agent's writes against τ²'s expected actions, and what followed each gate deny.

Analysis only. This module reads gold (each task's expected actions), so nothing under glint.gates may import it: a
gate that read gold would grade the agent against the answer (tests/test_inspect_imports.py checks).

τ² details this module handles (each broke an earlier throwaway script):
- a task is a `Task.model_validate` of its tasks.json entry, and an environment is put at the task's initial state
  with `env.set_state(init.initialization_data, init.initialization_actions, [])`;
- a hidden banking write, gold or the agent's, is a `call_discoverable_agent_tool` call whose `arguments.arguments` is
  a JSON string;
- an `unlock_discoverable_agent_tool` action names a hidden tool (agent_tool_name) without arguments: not a write;
- a banking transaction's amount and date are `transaction_amount` and `transaction_date`.

Helpers: task_env, gold_writes, run_writes, diff_run (with compact and diffs_line for the run log), arm_diffs and
run_denies. CLI: `python -m glint.inspect diff DATA_FOLDER [--task T] [--arm tools]`.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from functools import cache
from pathlib import Path

from glint import paths
from glint.gates import summary as gate_summary
from glint.gates import write_tools
from glint.runlog import FAILED_RUNS, outcome

WRAPPER = gate_summary.WRAPPER
CONTEXT = ("transaction_amount", "transaction_date", "merchant_name")  # shown beside a write on a banking transaction


@cache
def _tasks(domain: str) -> dict:
    from tau2.data_model.tasks import Task

    raw = json.loads((paths.tau2_data() / "tau2" / "domains" / domain / "tasks.json").read_text(encoding="utf-8"))
    return {str(t["id"]): Task.model_validate(t) for t in raw}


def _new_env(domain: str):
    if domain == "banking_knowledge":  # the retrieval variant tools_ab runs with
        from tau2.domains.banking_knowledge.environment import get_environment

        return get_environment(retrieval_variant="bm25_grep")
    from tau2.registry import registry

    return registry.get_env_constructor(domain)()


@cache
def _writes(domain: str) -> frozenset[str]:
    return frozenset(write_tools(_new_env(domain).tools))


def task_env(domain: str, task_id: str):
    """A fresh environment of `domain` at the initial state of task `task_id`."""
    task = _tasks(domain)[str(task_id)]
    env, init = _new_env(domain), task.initial_state
    env.set_state(init.initialization_data if init else None, init.initialization_actions if init else None, [])
    return env


def _unwrap(name: str, args: dict) -> tuple[str, dict | None]:
    if name != WRAPPER:
        return name, args
    inner = args.get("arguments") or {}
    if isinstance(inner, str):
        try:
            inner = json.loads(inner)
        except json.JSONDecodeError:
            return str(args.get("agent_tool_name", "")), None
    return str(args.get("agent_tool_name", "")), inner if isinstance(inner, dict) else None


def gold_writes(domain: str, task_id: str) -> list[dict]:
    """τ²'s expected agent writes for the task in order, hidden banking writes unwrapped: {tool, args, action_id}.
    Reads, unlocks and the user's own actions are left out."""
    task, writes = _tasks(domain)[str(task_id)], _writes(domain)
    out = []
    for a in (task.evaluation_criteria.actions if task.evaluation_criteria else None) or []:
        if str(getattr(a.requestor, "value", a.requestor)) != "assistant":
            continue
        tool, args = _unwrap(a.name, a.arguments or {})
        if tool in writes:
            out.append({"tool": tool, "args": args or {}, "action_id": a.action_id})
    return out


def run_writes(sim: dict, domain: str) -> dict:
    """The agent's writes in one run, in order, hidden banking writes unwrapped: {"executed": [...], "rejected": [...]}.
    A rejected write is one a gate denied (`by` "gate", with its rule_id) or the environment refused (`by` "env"); it
    changed nothing, so it is not compared with gold."""
    writes, out = _writes(domain), {"executed": [], "rejected": []}
    for c in gate_summary.run_calls(sim):
        if c["tool"] not in writes:
            continue
        w = {"index": c["index"], "tool": c["tool"], "args": c["args"]}
        if c["deny"] is not None:
            out["rejected"].append(w | {"by": "gate", "rule_id": c["deny"]["rule_id"], "reason": c["deny"]["reason"]})
        elif c["error"] or c["args"] is None:
            out["rejected"].append(w | {"by": "env", "reason": c["content"][:200]})
        else:
            out["executed"].append(w)
    return out


def _differing(gold: dict, run: dict) -> list[str]:
    """The argument names whose values differ; an absent argument counts as None (τ² fills defaults the same way)."""
    return sorted(k for k in gold.keys() | run.keys() if gate_summary.norm(gold.get(k)) != gate_summary.norm(run.get(k)))


def _target(args: dict) -> dict:
    return {k: v for k, v in args.items() if k.endswith("_id")}


def pair_writes(gold: list[dict], run: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """(missing, extra, changed). Identical writes pair first; then each remaining gold write pairs with the run write
    of the same tool on the same record (equal `*_id` arguments) that differs in the fewest fields."""
    gold_left, run_left, changed = list(gold), list(run), []
    for g in list(gold_left):
        r = next((r for r in run_left if r["tool"] == g["tool"] and not _differing(g["args"], r["args"])), None)
        if r is not None:
            gold_left.remove(g)
            run_left.remove(r)
    for g in list(gold_left):
        same = [r for r in run_left if r["tool"] == g["tool"] and gate_summary.same_target(g["args"], r["args"])]
        if not same:
            continue
        r = min(same, key=lambda r: len(_differing(g["args"], r["args"])))
        gold_left.remove(g)
        run_left.remove(r)
        changed.append({"tool": g["tool"], "target": _target(g["args"]), "action_id": g.get("action_id"),
                        "index": r["index"], "fields": [{"field": k, "gold": g["args"].get(k), "run": r["args"].get(k)}
                                                        for k in _differing(g["args"], r["args"])]})
    return gold_left, run_left, changed


def reward_breakdown(info: dict | None) -> dict | None:
    """τ²'s own grading of the run (reward_info): reward, basis, db_check, and each check list as met of total with
    the unmet ones named."""
    if info is None:
        return None
    out = {"reward": info.get("reward"), "basis": info.get("reward_basis"), "breakdown": info.get("reward_breakdown"),
           "db_match": (info.get("db_check") or {}).get("db_match")}
    actions = info.get("action_checks")
    if actions:
        out["actions"] = {"met": sum(bool(a.get("action_match")) for a in actions), "total": len(actions),
                          "unmet": [f"{a['action'].get('action_id')} "
                                    f"{_unwrap(a['action']['name'], a['action'].get('arguments') or {})[0]}"
                                    for a in actions if not a.get("action_match")]}
    for key, text in (("env_assertions", "env_assertion"), ("nl_assertions", "nl_assertion"),
                      ("communicate_checks", "info")):
        checks = info.get(key)
        if checks:
            out[key] = {"met": sum(bool(c.get("met")) for c in checks), "total": len(checks),
                        "unmet": [str(c.get(text))[:120] for c in checks if not c.get("met")]}
    note = (info.get("info") or {}).get("note")
    if note:
        out["note"] = note
    return out


def _context(env, args: dict) -> dict:
    table = getattr(env.tools.db, "credit_card_transaction_history", None) if env is not None else None
    txn = table.data.get(args.get("transaction_id")) if table is not None and args.get("transaction_id") else None
    return {k: txn.get(k) for k in CONTEXT if k in txn} if txn else {}


def diff_run(domain: str, sim: dict) -> dict:
    """One run against τ²'s expected writes: missing gold writes, extra executed writes, per-write field differences,
    the writes a gate or the environment rejected, and τ²'s own reward breakdown. A write on a banking transaction
    carries the transaction's amount and date (`context`)."""
    task_id = str(sim["task_id"])
    runw = run_writes(sim, domain)
    missing, extra, changed = pair_writes(gold_writes(domain, task_id), runw["executed"])
    if any((w.get("args") or w.get("target") or {}).get("transaction_id") for w in missing + extra + changed):
        env = task_env(domain, task_id)
        for w in missing + extra:
            w["context"] = _context(env, w["args"])
        for w in changed:
            w["context"] = _context(env, w["target"])
    return {"task_id": task_id, "trial": sim.get("trial"), "outcome": outcome(sim), "missing": missing,
            "extra": extra, "changed": changed, "rejected": runw["rejected"],
            "reward": reward_breakdown(sim.get("reward_info"))}


def compact(diff: dict) -> dict:
    """A diff_run result in a few numbers and one line, for the run log's event record:
    `diff: 5 field (eligible_for_provisional_credit x4, resolution_requested x1); missing 0, extra 0`."""
    fields = Counter(f["field"] for w in diff["changed"] for f in w["fields"])
    tools = Counter(w["tool"] for w in diff["changed"] + diff["missing"] + diff["extra"])
    text = f"diff: {sum(fields.values())} field"
    if fields:
        text += f" ({', '.join(f'{k} x{n}' for k, n in fields.most_common())})"
    for kind in ("missing", "extra"):
        names = Counter(w["tool"] for w in diff[kind])
        text += f"{';' if kind == 'missing' else ','} {kind} {len(diff[kind])}"
        if names:
            text += f" ({', '.join(f'{k} x{n}' for k, n in names.most_common())})"
    if diff["rejected"]:
        text += f"; rejected {len(diff['rejected'])}"
    return {"fields": dict(fields), "tools": dict(tools), "missing": len(diff["missing"]), "extra": len(diff["extra"]),
            "rejected": len(diff["rejected"]), "text": text}


def diffs_line(compacts: list[dict], top: int = 5) -> str:
    """The DIFFS line at an arm's end: the fields and tools that differ most often across its failed runs."""
    fields, tools = Counter(), Counter()
    for c in compacts:
        fields.update(c["fields"])
        tools.update(c["tools"])
    return (f"DIFFS: {len(compacts)} failed runs diffed against τ²'s expected writes; top fields "
            f"{dict(fields.most_common(top))}; top tools {dict(tools.most_common(top))}")


HIDDEN_READ = re.compile(r"get_[a-z_0-9]+_\d{4}$")  # banking's hidden lookups (get_user_dispute_history_7291, ...)
SEARCH_TOOLS = ("KB_search", "grep", "KB_read")
DISPUTE, HISTORY = "file_credit_card_transaction_dispute_4829", "get_user_dispute_history_7291"


def run_facts(sim: dict) -> dict:
    """What a run did beside its writes, from its messages: the hidden lookups that ran (in order, once each), whether
    the dispute-history read came before the first dispute filing that ran, plan_disputes calls, dispute filings that
    ran and those refused (by a gate or τ²), the knowledge-base search results (count and characters) and how the run
    ended (τ²'s termination reason and info.error). A refused filing changed nothing, so it is not "the first filing":
    a run the history-read gate refused, that then read the history and filed, read it before its first filing. Each
    answered a question the run diff could not (v6 diagnosis, 2026-10-30)."""
    calls = gate_summary.run_calls(sim)
    reads = list(dict.fromkeys(c["tool"] for c in calls if c["tool"] and HIDDEN_READ.match(c["tool"])
                               and not c["error"]))
    filed = [c["index"] for c in calls if c["tool"] == DISPUTE and not c["error"]]
    first_filing = filed[0] if filed else None
    history = [c["index"] for c in calls if c["tool"] == HISTORY and not c["error"]]
    searches = [len(c["content"]) for c in calls if c["tool"] in SEARCH_TOOLS]
    return {"reads": reads, "plans": sum(c["tool"] == "plan_disputes" for c in calls), "filings": len(filed),
            "filings_refused": sum(c["tool"] == DISPUTE and c["error"] for c in calls),
            "history_before_first_filing": None if first_filing is None else any(i < first_filing for i in history),
            "searches": len(searches), "search_chars": sum(searches), "largest_search": max(searches, default=0),
            "termination": sim.get("termination_reason"), "error": str((sim.get("info") or {}).get("error") or "")}


def facts_line(f: dict) -> str:
    history = ("" if f["history_before_first_filing"] is None else
               f" (dispute history before the first filing: {'yes' if f['history_before_first_filing'] else 'NO'})")
    ended = f"ended: {f['termination']}" + (f" ({f['error'][:160]})" if f["error"] else "")
    return (f"  facts: lookups {', '.join(f['reads']) or 'none'}{history}; plan_disputes {f['plans']}, filings ran "
            f"{f['filings']} (refused {f['filings_refused']}); searches {f['searches']} ({f['search_chars']:,} chars, largest {f['largest_search']:,}); "
            f"{ended}")


def arm_diffs(domain: str, sims: list[dict]) -> dict[tuple[str, int], dict]:
    """(task_id, trial) -> compact diff of every failed run in an arm (glint.gates.summary.rule_outcomes takes it). A
    run of a task τ² does not have gets no diff; rule_outcomes counts it as a failed run not diffed."""
    known = _tasks(domain)
    return {(str(s["task_id"]), int(s.get("trial") or 0)): compact(diff_run(domain, s))
            for s in sims if outcome(s) in FAILED_RUNS and str(s["task_id"]) in known}


def run_denies(gate_db: str | Path, task: str, trial: int, sim: dict | None = None) -> list[dict]:
    """Each live gate deny of one run: rule_id, reason and fix, the next write the agent made, and whether it followed
    the fix (glint.gates.summary.deny_followups). The run is `sim`, or else read from the arm file beside the log
    (`gates-<domain>.sqlite` -> `tools-<domain>.json`). A deny in the log that the saved run does not show (an
    earlier attempt of a resumed run) is listed with saved_run False and no follow-up."""
    gate_db = Path(gate_db)
    events = gate_summary.read_log(gate_db)
    if sim is None:
        arm = gate_db.parent / f"tools-{gate_db.stem.removeprefix('gates-')}.json"
        sims = json.loads(arm.read_text(encoding="utf-8"))["simulations"]
        sim = next((s for s in sims if str(s["task_id"]) == str(task) and int(s["trial"]) == int(trial)), None)
        if sim is None:
            raise KeyError(f"{arm} has no run of {task} trial {trial}")
    logged = [e["payload"] for e in events if e["type"] == "gate.fired" and not e["payload"].get("replay")
              and e["payload"]["decision"] != "allow" and str(e["payload"].get("task_id")) == str(task)
              and str(e["payload"].get("trial")) == str(trial)]
    out = []
    for f in gate_summary.deny_followups(sim, gate_summary.gated_tools(events)):
        match = next((p for p in logged if p["rule_id"] == f["rule_id"] and p["reason"] == f["reason"]), None)
        if match is not None:
            logged.remove(match)
        out.append(f | {"logged": match is not None, "saved_run": True})
    out += [{"rule_id": p["rule_id"], "reason": p["reason"], "fix": None, "tool": p["tool"], "args": p.get("args"),
             "next_write": None, "fix_followed": None, "how": "not in the saved run", "logged": True,
             "saved_run": False} for p in logged]
    return out
