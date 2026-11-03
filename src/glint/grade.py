"""Offline requested-vs-done grader over a saved simulation (fix-research 2026-11-02, decision-table row 16; reports
3 K5 "reconcile intended against sent" and 8 E1 "flag forbidden actions, not only required ones").

Measurement only: it never runs inside a conversation, never decides a write and never reads τ²'s gold (the expected
actions), so it is a start on a grader independent of τ²'s answer key (memo M7). It reads only the run's own tool calls
and replies:
- **requested**: the customer's requests as the agent recorded them with record_requests (the replies that say
  `recorded`, accumulated as glint.gates.enforce does). Only the gated tools arm has the tool; a run without a record
  is graded `no_record`, and every write it made on a request tool is `unrecorded`, not judged.
- **done**: the judged writes that ran (no gate refusal, no tool error), hidden banking writes unwrapped: the
  request tools, the credit card filing and the retention-offer writes (JUDGED).
- **required**: follow-ups the run's own records make due, each with its source:
  - a recorded close_credit_card request needs its closure reason logged (credit_card_account_logistics_003 step 3)
    and, unless the customer accepted the retention offer, the closure (step 6);
  - an accepted retention offer needs the offer applied to that card (step 5 names the offer, not the tool; which
    writes count is a Glint convention: RETENTION_APPLY);
  - every transaction in the latest dispute plan (plan_disputes) for a user needs its credit card filing (Glint
    convention: the plan is Glint's tool; the filing requirements are credit_cards_(general)_014/_015).

The findings per run: requested_not_done (a request with no write), done_not_requested (a write on a request tool no
request covers: the extra-write class), required_not_done (a follow-up the records made due and no write met).
"""
from __future__ import annotations

import json

from glint.assist import as_flag
from glint.assist.banking import FILING_TOOL, PLAN_TOOL, REQUEST_ACTIONS, REQUEST_TOOLS, REQUESTS_TOOL, RETENTION_TOOL
from glint.gates.summary import norm, run_calls

CLOSE_CARD, LOG_REASON = "close_credit_card_account_7834", "log_credit_card_closure_reason_4521"
# writes that apply an accepted retention offer to the card (Glint convention: logistics_003 step 5 offers bonus points
# or a statement credit and names no tool; the statement credit tool and the account flag are the writes that can)
RETENTION_APPLY = ("apply_statement_credit_8472", "apply_credit_card_account_flag_6147")


def _reply(call: dict) -> dict:
    """A decision tool's JSON reply, without an open-items footer; {} when it is not a JSON object."""
    try:
        v = json.loads(call["content"].split("\n\n[open]")[0])
    except (TypeError, json.JSONDecodeError):
        return {}
    return v if isinstance(v, dict) else {}


def requests_of(calls: list[dict]) -> list[dict] | None:
    """The recorded requests ({action, target_id}), each (action, target) once, in recording order; None when the
    run never recorded any (no record_requests reply that says recorded)."""
    out: list[dict] | None = None
    for c in calls:
        if c["tool"] != REQUESTS_TOOL or c["error"]:
            continue
        v = _reply(c)
        if not as_flag(v.get("recorded")):
            continue
        out = out if out is not None else []
        for r in v.get("requests") or []:
            if not isinstance(r, dict):
                continue
            item = {"action": str(r.get("action", "")), "target_id": str(r.get("target_id") or "")}
            if item not in out:
                out.append(item)
    return out


JUDGED = frozenset(REQUEST_TOOLS) | {FILING_TOOL, *RETENTION_APPLY}  # the writes this grader judges


def done_writes(calls: list[dict]) -> list[dict]:
    """The judged writes (JUDGED) that ran: no gate refusal, no tool error, readable arguments. {index, tool, args}."""
    return [{"index": c["index"], "tool": c["tool"], "args": c["args"]} for c in calls
            if c["tool"] in JUDGED and c["args"] is not None and not c["error"] and c["deny"] is None]


def _on(write: dict, tools: tuple[str, ...], arg: str, target: str) -> bool:
    return write["tool"] in tools and (not target or norm(write["args"].get(arg)) == norm(target))


def covered(write: dict, requests: list[dict]) -> bool:
    """A request covers the write: same action, and the request names no target or the write's (the request gate's
    test, glint.gates.banking.requested)."""
    action, arg = REQUEST_TOOLS[write["tool"]]
    return any(r["action"] == action and (not r["target_id"] or norm(r["target_id"]) == norm(write["args"].get(arg)))
               for r in requests)


def retention_answers(calls: list[dict]) -> dict[str, dict]:
    """card -> the latest recorded answer to the retention offer {accepted, offer, index}."""
    out = {}
    for c in calls:
        if c["tool"] == RETENTION_TOOL and not c["error"]:
            v = _reply(c)
            if as_flag(v.get("recorded")) and v.get("credit_card_account_id"):
                out[str(v["credit_card_account_id"])] = {"accepted": as_flag(v.get("accepted")),
                                                         "offer": str(v.get("offer", "")), "index": c["index"]}
    return out


def planned_disputes(calls: list[dict]) -> dict[str, list[str]]:
    """user -> the transactions of the latest dispute plan the agent got for that user, in the plan's order."""
    out = {}
    for c in calls:
        if c["tool"] == PLAN_TOOL and not c["error"]:
            v = _reply(c)
            if v.get("user_id") and isinstance(v.get("order"), list):
                out[str(v["user_id"])] = [str(e.get("transaction_id")) for e in v["order"] if isinstance(e, dict)]
    return out


def grade(sim: dict) -> dict:
    """One run's requested-vs-done findings (module docstring)."""
    calls = run_calls(sim)
    requests, done = requests_of(calls), done_writes(calls)
    answers = retention_answers(calls)
    out = {"task_id": sim.get("task_id"), "trial": sim.get("trial"), "no_record": requests is None,
           "requests": requests or [], "done": len(done), "requested_not_done": [], "done_not_requested": [],
           "unrecorded": [], "required_not_done": []}
    for w in done:
        if w["tool"] in REQUEST_TOOLS:
            if requests is None:
                out["unrecorded"].append(w)
            elif not covered(w, requests):
                out["done_not_requested"].append(w)
    for r in requests or []:
        tools, arg = REQUEST_ACTIONS[r["action"]]
        if r["action"] == "close_credit_card":
            accepted = (answers.get(r["target_id"]) or {}).get("accepted")
            tools = (LOG_REASON,) if accepted else tools
            for tool in tools:
                if not any(_on(w, (tool,), arg, r["target_id"]) for w in done):
                    out["requested_not_done"].append(r | {"tool": tool})
            continue
        if not any(_on(w, tools, arg, r["target_id"]) for w in done):
            out["requested_not_done"].append(r | {"tool": "/".join(tools)})
    for card, a in sorted(answers.items()):
        if a["accepted"] and not any(w["index"] > a["index"] and _on(w, RETENTION_APPLY, "credit_card_account_id",
                                                                       card) for w in done):
            out["required_not_done"].append({"rule": "retention offer accepted and not applied "
                                             "(credit_card_account_logistics_003 step 5)", "target_id": card})
    for user, txns in sorted(planned_disputes(calls).items()):
        for txn in txns:
            if not any(_on(w, (FILING_TOOL,), "transaction_id", txn) for w in done):
                out["required_not_done"].append({"rule": f"dispute planned for {user} and not filed (Glint "
                                                 "convention: plan_disputes)", "target_id": txn})
    return out


def totals(grades: list[dict]) -> dict:
    """Counts over runs: runs, runs with a record, and each finding's total and runs with at least one."""
    out = {"runs": len(grades), "runs_with_record": sum(not g["no_record"] for g in grades),
           "requests": sum(len(g["requests"]) for g in grades), "done": sum(g["done"] for g in grades)}
    for key in ("requested_not_done", "done_not_requested", "unrecorded", "required_not_done"):
        out[key] = sum(len(g[key]) for g in grades)
        out[f"runs_{key}"] = sum(bool(g[key]) for g in grades)
    return out
