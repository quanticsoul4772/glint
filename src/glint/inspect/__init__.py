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
from glint.gates.rules import GATED_GENERIC
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


# Rules that hold a write until a step comes first (verification, a lookup, the plan, the increase decision, the
# expedite question, the card-digits tool): gold takes those steps before its writes, so refusing a write identical to
# gold's on one of them holds it for the step and is not a false refusal. Every other rule judges the write's values.
PRECONDITION_RULES = ("procedure.verified", "procedure.lookups", "procedure.dispute_plan.none",
                      "procedure.dispute_plan.ask", "procedure.dispute_plan.order",
                      "procedure.dispute_plan.increase_first", "procedure.dispute_plan.not_planned",
                      "procedure.increase_first", "procedure.increase_undecided", "procedure.transfer.increase_undecided",
                      "procedure.card_digits.give", "banking.replacement.increase_first",
                      "banking.replacement.expedite_ask", "banking.replacement.address_ask")
# Rules that judge the write against the database at that moment (a policy check, a card's status): the gold
# replays show each allows gold's write in gold's state, so refusing the same arguments in another state (051's
# approve before the payment, 081's close of a still-frozen card) is the state differing, not a false refusal.
STATE_RULES = ("banking.cli.", "debit.", "banking.replacement", "banking.dispute", "banking.update_rewards",
               "banking.checking_credit", "banking.close", "banking.account_flag", "banking.statement_credit",
               "banking.log_closure", "procedure.transfer.reason")
# The plan gate holds a filing to the plan's values: a gold-identical filing refused there means the plan itself
# differed from gold (v6 041 t1: the disputes were planned in batches), a refusal the plan caused.
PLAN_RULES = ("procedure.dispute_plan.values",)
# Rules that refuse a value's type: "True" for true looks like gold's write once normalized, but τ² compares the stored
# record exactly, so the write does differ from gold (banking.dispute.types, 2026-10-30)
TYPE_RULES = ("banking.dispute.types",)
# A refused call of a gated tool that is not a write (rules.GATED_GENERIC: transfer_to_human_agents): gold's writes
# cannot judge it, so it is counted apart, never judged.
KINDS = ("live_false_refusals", "plan_differs_from_gold", "state_differs", "held_for_precondition",
         "differs_from_gold", "gated_tool")


def live_refusals(domain: str, sim: dict) -> dict:
    """Each gate refusal in a live run, classified against τ²'s gold for the task. A write different from every gold
    write: differs_from_gold. A write identical to a gold write refused on: an order rule, held_for_precondition (gold
    takes that step first); a state rule, state_differs (the same arguments are wrong in this state); the plan's
    values, plan_differs_from_gold (the plan was wrong); any other rule, which judges the write's values against the
    record, live_false_refusals. A refused gated tool that is not a write: gated_tool, unjudged. Train tasks only
    (gold is read)."""
    gold = gold_writes(domain, str(sim["task_id"]))
    out = {"refused": 0} | {kind: [] for kind in KINDS}
    for w in run_writes(sim, domain)["rejected"]:
        if w["by"] != "gate" or w["args"] is None:
            continue
        out["refused"] += 1
        row = {"rule_id": w["rule_id"], "tool": w["tool"], "index": w["index"]}
        rule = str(w["rule_id"])
        if rule in TYPE_RULES or not any(g["tool"] == w["tool"] and not _differing(g["args"], w["args"])
                                         for g in gold):
            kind = "differs_from_gold"
        elif rule.startswith(PRECONDITION_RULES):
            kind = "held_for_precondition"
        elif rule.startswith(PLAN_RULES):
            kind = "plan_differs_from_gold"
        elif rule.startswith(STATE_RULES):
            kind = "state_differs"
        else:
            kind = "live_false_refusals"
        out[kind].append(row)
    for c in gate_summary.run_calls(sim):
        if c["tool"] in GATED_GENERIC.get(domain, ()) and c["deny"] is not None:
            out["refused"] += 1
            out["gated_tool"].append({"rule_id": c["deny"]["rule_id"], "tool": c["tool"], "index": c["index"]})
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
    # the first wrong values, gold against filed: the run log named only the field, so v6's wrong noticed dates
    # (gold 11/14/2025, filed 10/30/2026) were visible only after the arm ended (2026-10-30)
    values = [{"tool": w["tool"], "field": f["field"], "gold": f["gold"], "run": f["run"]}
              for w in diff["changed"] for f in w["fields"]][:2]
    if values:
        text += "; first: " + "; ".join(f"{v['field']} gold {json.dumps(v['gold'], default=str)[:40]} run "
                                       f"{json.dumps(v['run'], default=str)[:40]}" for v in values)
    return {"fields": dict(fields), "tools": dict(tools), "missing": len(diff["missing"]), "extra": len(diff["extra"]),
            "rejected": len(diff["rejected"]), "values": values, "text": text}


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
ORDER, SUBMIT = "order_replacement_credit_card_7291", "submit_credit_limit_increase_request_7392"
APPROVE, DENY_CLI = "approve_credit_limit_increase_5847", "deny_credit_limit_increase_5848"


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
    order, submit, decision = ([c["index"] for c in calls if c["tool"] in tools and not c["error"]]
                               for tools in ((ORDER,), (SUBMIT,), (APPROVE, DENY_CLI)))
    # the step orders the v6 regressions came from: in 054 the replacement was ordered, and the dispute filed, before
    # the credit limit increase (a pending replacement or dispute makes the increase fail)
    order_before_submit = bool(order and submit and order[0] < submit[0])
    filing_before_decision = bool(filed and submit and filed[0] < (decision[0] if decision else float("inf")))
    prompts = [int((m.get("usage") or {}).get("prompt_tokens") or 0) for m in sim.get("messages") or []]
    return {"reads": reads, "plans": sum(c["tool"] == "plan_disputes" for c in calls), "filings": len(filed),
            "filings_refused": sum(c["tool"] == DISPUTE and c["error"] for c in calls),
            "history_before_first_filing": None if first_filing is None else any(i < first_filing for i in history),
            "searches": len(searches), "search_chars": sum(searches), "largest_search": max(searches, default=0),
            "order_before_submit": order_before_submit, "filing_before_decision": filing_before_decision,
            "peak_prompt_tokens": max(prompts, default=0), "claims_not_done": claims_not_done(sim, calls),
            **exposure_and_verdict(calls), **check_facts(calls),
            "termination": sim.get("termination_reason"), "error": str((sim.get("info") or {}).get("error") or "")}


# Claims of a finished write in the agent's own words, each with the write that must have run by then. On τ²-bench a
# false success (completion asserted, the record shows otherwise) is 45-48% of failures in single-control domains
# (arXiv 2606.09863); v6 039 t0/t1 and v7 031 t0/t1 said a dispute was filed and filed none, v7 051 t3 said the
# increase was denied and filed no deny (2026-10-30).
CLAIMS = (
    ("dispute filed", (DISPUTE,), re.compile(
        r"\bdisputes?\b[^\n]{0,120}?\b(?:has|have)\s+been\s+(?:successfully\s+)?(?:filed|submitted)"
        r"|\bdisputes?\b[^\n]{0,120}?\b(?:filed|submitted)\s+successfully", re.IGNORECASE)),
    ("replacement ordered", (ORDER,), re.compile(
        r"\b(?:replacement|new card)\b[^\n]{0,80}?\b(?:has|have)\s+been\s+(?:successfully\s+)?(?:ordered|placed)",
        re.IGNORECASE)),
    ("increase decided", (APPROVE, DENY_CLI), re.compile(
        r"\b(?:increase|request)\b[^\n]{0,80}?\b(?:has|have)\s+been\s+(?:approved|denied)"
        r"|\b(?:increase|request)\b[^\n]{0,40}?\b(?:is|was)\s+(?:approved|denied)\b", re.IGNORECASE)),
)
# the procedure document of each area a run can touch (tool names that put a run in the area)
KEY_DOCS = (
    ("increase", (SUBMIT, APPROVE, DENY_CLI, "check_credit_limit_increase"),
     "doc_credit_cards_credit_card_account_logistics_007"),
    ("dispute", (DISPUTE, "plan_disputes"), "doc_credit_cards_credit_cards_(general)_014"),
    ("replacement", (ORDER,), "doc_credit_cards_credit_card_replacements_001"),
)
DOC_ID = re.compile(r"ID: (doc_[\w().\-]+)")


def _verdict(content: str) -> str | None:
    """The `verdict` of a check_credit_limit_increase reply (a JSON object, possibly followed by the open-items
    footer)."""
    try:
        obj, _ = json.JSONDecoder().raw_decode(content.lstrip())
    except (ValueError, TypeError):
        return None
    return obj.get("verdict") if isinstance(obj, dict) else None


def claims_not_done(sim: dict, calls: list[dict]) -> list[str]:
    """The agent's claims of a write that never ran in the whole run (one entry per kind): a false success. A claim
    made a turn before the write it announces is not counted (passing runs do that; tried on v6 and v7, 2026-10-30)."""
    ran = {c["tool"] for c in calls if c["tool"] and not c["error"]}
    texts = [str(m["content"]) for m in sim.get("messages") or [] if m.get("role") == "assistant" and m.get("content")]
    return [name for name, tools, rx in CLAIMS if not ran & set(tools) and any(rx.search(t) for t in texts)]


def exposure_and_verdict(calls: list[dict]) -> dict:
    """The knowledge-base documents a run was shown, the key procedure documents of its areas it never saw, the last
    credit limit increase verdict and whether a matching decision was filed, and its gate refusals by rule."""
    seen = list(dict.fromkeys(d for c in calls if c["tool"] in SEARCH_TOOLS for d in DOC_ID.findall(c["content"])))
    tools = {c["tool"] for c in calls if c["tool"]}
    missing = [area for area, area_tools, doc in KEY_DOCS if tools & set(area_tools) and doc not in seen]
    verdicts = [v for c in calls if c["tool"] == "check_credit_limit_increase" and not c["error"]
                and (v := _verdict(c["content"]))]
    filed = {APPROVE: "approve", DENY_CLI: "deny"}
    decisions = [filed[c["tool"]] for c in calls if c["tool"] in filed and not c["error"]]
    last = verdicts[-1] if verdicts else None
    not_filed = (f"{last} verdict, {'no decision filed' if not decisions else 'filed ' + decisions[-1]}"
                 if last in ("approve", "deny") and (not decisions or decisions[-1] != last) else None)
    return {"docs_seen": seen, "key_docs_not_seen": missing, "increase_verdict": last,
            "verdict_not_filed": not_filed, "gate_refusals": [c["deny"]["rule_id"] for c in calls if c["deny"]]}


def _reply_json(content: str) -> dict:
    """A decision tool's JSON answer, without the open-items footer; {} when it is not JSON."""
    try:
        v = json.loads(content.split("\n\n[open]")[0])
    except json.JSONDecodeError:
        return {}
    return v if isinstance(v, dict) else {}


def check_facts(calls: list[dict]) -> dict:
    """What the decision tools told the run and whether it acted on it (2026-10-30 review: v9's "the check said
    ready and the filing tool was never unlocked", 7 runs, was counted by hand):
    - dispute_checks_ready: check_card_dispute answers with "ready" true; dispute_ready_not_unlocked: at least one,
      and the filing tool was never unlocked or run;
    - verified_not_logged: check_verification said verified and no log_verification ran after it;
    - open_items_at_end: the [open] lines on the last reply that carried any (what was still open when it ended)."""
    ready = [c["index"] for c in calls if c["tool"] == "check_card_dispute" and not c["error"]
             and str(_reply_json(c["content"]).get("ready")) == "True"]
    unlocked = any((c["tool"] == "unlock_discoverable_agent_tool" and c["named"] == DISPUTE) or c["tool"] == DISPUTE
                   for c in calls)
    verified = [c["index"] for c in calls if c["tool"] == "check_verification" and not c["error"]
                and str(_reply_json(c["content"]).get("verified")) == "True"]
    logged = [c["index"] for c in calls if c["tool"] == "log_verification" and not c["error"]]
    footers = [c["content"] for c in calls if "\n[open] " in c["content"]]
    open_at_end = [line for line in footers[-1].splitlines() if line.startswith("[open] ")] if footers else []
    return {"dispute_checks_ready": len(ready), "dispute_ready_not_unlocked": bool(ready) and not unlocked,
            "verified_not_logged": bool(verified) and not any(i > verified[0] for i in logged),
            "open_items_at_end": len(open_at_end)}


def facts_brief(f: dict) -> str:
    """run_facts in one short clause for the run log's per-run line; the step-order flags in capitals."""
    history = {None: "-", True: "yes", False: "NO"}[f["history_before_first_filing"]]
    flags = [name for name, on in (("ORDER BEFORE INCREASE SUBMIT", f["order_before_submit"]),
                                   ("FILED BEFORE INCREASE DECIDED", f["filing_before_decision"])) if on]
    flags += [f"CLAIMED NOT DONE: {c}" for c in f.get("claims_not_done", [])]
    if f.get("verdict_not_filed"):
        flags.append(f"VERDICT NOT FILED: {f['verdict_not_filed']}")
    if f.get("dispute_ready_not_unlocked"):
        flags.append(f"READY NEVER FILED: {f['dispute_checks_ready']} ready checks, filing tool never unlocked")
    if f.get("verified_not_logged"):
        flags.append("VERIFIED NOT LOGGED: check_verification said verified, no log_verification after it")
    if f.get("open_items_at_end"):
        flags.append(f"OPEN AT END: {f['open_items_at_end']} item(s)")
    refusals = Counter(f.get("gate_refusals", []))
    tail = (", refusals " + ", ".join(f"{r} x{n}" for r, n in refusals.most_common())) if refusals else ""
    # information, not a flag: passing runs often skip the procedure document (050 follows the decision tool's steps)
    if f.get("key_docs_not_seen"):
        tail += ", key doc not seen: " + "/".join(f["key_docs_not_seen"])
    return (f"facts: plans {f['plans']}, filings {f['filings']} (+{f['filings_refused']} refused), history first "
            f"{history}, searches {f['searches']} ({f['search_chars'] // 1000}k chars), peak prompt "
            f"{f['peak_prompt_tokens'] // 1000}k tokens{tail}" + "".join(f", {x}" for x in flags))


SEQUENCE_TOOLS = ("plan_disputes", DISPUTE, ORDER, SUBMIT, APPROVE, DENY_CLI, HISTORY)


def run_sequence(sim: dict) -> list[str]:
    """The order of a run's dispute, replacement and credit limit increase steps, one line each with the message index
    of its reply: plan_disputes (the transactions planned), filings (transaction and credit flag), the replacement
    order, the increase submit and decision, the dispute-history read; a refused call (gate or τ²) is marked. The
    questions of the v6 diagnosis (what came first; were disputes planned in batches) read off these lines."""
    out = []
    for c in gate_summary.run_calls(sim):
        if c["tool"] not in SEQUENCE_TOOLS:
            continue
        args = c["args"] or {}
        if c["tool"] == "plan_disputes":
            what = f"{len(args.get('transactions') or [])} txns: " + ", ".join(
                str(t.get("transaction_id")) for t in args.get("transactions") or [] if isinstance(t, dict))
        elif c["tool"] == DISPUTE:
            what = f"{args.get('transaction_id')} credit={args.get('eligible_for_provisional_credit')}"
        else:
            what = ", ".join(f"{k}={args[k]}" for k in ("credit_card_account_id", "user_id", "denial_reason")
                             if args.get(k))
        refused = (" REFUSED " + (f"({c['deny']['rule_id']})" if c["deny"] else f"({c['content'][:80]})")
                   if c["error"] else "")
        out.append(f"[{c['index']}] {c['tool']} {what}{refused}")
    return out


def run_messages(sim: dict, pattern: str, width: int = 240) -> list[str]:
    """Each message of a run matching a regex (case-insensitive) in its text or its tool calls, as
    "[index role] ...text around the first match..."; tool results included. Conversation text only: never the run's
    metadata."""
    rx, out = re.compile(pattern, re.IGNORECASE), []
    for i, m in enumerate(sim.get("messages") or []):
        calls = " ".join(f"{c.get('name')}({json.dumps(c.get('arguments'), ensure_ascii=False)})"
                         for c in m.get("tool_calls") or [])
        text = f"{m.get('content') or ''} {calls}".strip()
        hit = rx.search(text)
        if hit:
            out.append(f"[{i} {m.get('role')}] ...{text[max(0, hit.start() - width):hit.end() + width]!r}")
    return out


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
