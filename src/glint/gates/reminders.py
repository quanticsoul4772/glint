"""Reminders and a transfer flag on the enforcer (fix-research appendix #60, #68, #69; PR #40), registered as
enforcer options and hooks (glint.gates.extend). The reminders are on by default since the owner's decision of
2026-11-03 (every fix on by default).

- `reminders=("declare",)` (#60, the default; `reminders=()` is off): the open items also list the pending
  obligations of the agreed blind clause list (glint.declare) whose activities the trace of replayed calls can show.
- `step_warn=N` (#69; default 60, `step_warn=None` is off): once N calls τ²'s evaluator replays have run, the open
  items end with STEP_WARN_NOTE. 60 was chosen by a parallax score at the owner's request (60 scored 78, 80: 66,
  40: 62).
- A transfer.flagged event (#68) when a transfer ran while every recorded request names an action this toolkit has a
  tool for. Flag only, never a refusal.

Each only adds [open] lines or an event; none refuses a write. The trace and the replayed-call count are built from
calls τ²'s evaluator replays, so every reply replays unchanged.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from tau2.data_model.message import ToolCall, ToolMessage
from tau2.environment.toolkit import ToolType

from glint import declare
from glint.assist.banking import REQUEST_ACTIONS
from glint.gates import extend
from glint.gates.rules import parse_hidden_arguments

log = logging.getLogger("glint.gates")
WRAPPER = declare.WRAPPER
REMINDERS = ("declare",)  # the reminders (Enforcer(reminders=...)); each only adds [open] lines, never refuses
CLAUSES = declare.CLAUSE_DIR / "reconciled.json"  # package data: an installed Glint has it too
# appendix #69 (reader 3 K5; B2 §1): a step-limit warning listing the open items. Glint convention: the gates see tool
# calls only, never τ²'s step count, so the trigger counts the calls τ²'s evaluator replays (a lower bound on steps)
STEP_WARN_NOTE = ("[open] this conversation is near its step limit: finish the open items above (or tell the customer "
                  "what is left) instead of searching again")
# appendix #68 (B2 §1): a transfer while the request record names only actions this toolkit has tools for. Flag only,
# never a refusal: tasks 004, 014 and 5 training tasks expect transfers (B2), and coverage of the customer's real request
# needs the agent-kept record (M12). Glint convention.
TOOLS_COVER = "banking.transfer.tools_cover"
TRANSFER_TOOL = "transfer_to_human_agents"


def declare_reminder_clauses(path: Path = CLAUSES) -> list[dict]:
    """The agreed, checkable response and existence clauses of the reconciled blind clause list: the obligations the
    "declare" reminder can show (glint.declare.pending). Clauses τ²'s training gold violates
    (gold_violations.json next to the list, written by scripts/declare_grade.py gold --out) are left out: fitted to
    τ² gold, so the reminder never asks for a call training gold does not make."""
    data = json.loads(path.read_text(encoding="utf-8"))
    skip_file = path.parent / "gold_violations.json"
    skip = set(json.loads(skip_file.read_text(encoding="utf-8"))) if skip_file.exists() else set()
    return [c for c in map(declare.validate, data["agreed"]) if c["id"] not in skip
            and c["checkable"] and c["template"] in ("response", "responded_existence", "existence")]


def _replayed(env, name: str) -> bool:
    """τ²'s evaluator replays this tool (environment._is_mutating_tool), so a count of such calls replays unchanged."""
    return bool(env._has_tool(name) and env._is_mutating_tool(name))


def _trace(env, call: ToolCall, reply: ToolMessage, replayed: bool) -> None:
    """Add an assistant call that τ²'s evaluator replays and that ran to env.tools.glint_trace (hidden tools
    unwrapped); a trace of replayed calls only rebuilds identically in replay."""
    if call.requestor != "assistant" or not replayed or reply.error or str(reply.content).startswith("Error"):
        return
    name, args = call.name, call.arguments or {}
    if name == WRAPPER:
        name, args = str(args.get("agent_tool_name", "")), parse_hidden_arguments(args.get("arguments", "{}"))
    if args is not None:
        env.tools.glint_trace.append({"tool": name, "args": args})


def _observable(toolkit, tool: str) -> bool:
    """A call to `tool` reaches glint_trace: a hidden tool, or a visible one τ²'s evaluator replays."""
    if toolkit.has_discoverable_tool(tool):
        return True
    return toolkit.has_tool(tool) and toolkit.tool_type(tool) != ToolType.READ


# ---- registrations (glint.gates.extend) ----------------------------------------------------------------------------

def _setup_reminders(enforcer, reminders) -> None:
    if set(reminders) - set(REMINDERS):
        raise ValueError(f"unknown reminders {sorted(set(reminders) - set(REMINDERS))}; any of {REMINDERS}")
    enforcer.declare_clauses = declare_reminder_clauses() if "declare" in reminders else []


extend.option("reminders", ("declare",), "open-items reminders (REMINDERS); \"declare\" lists pending blind-clause "
              "obligations; on by default, () is off", order=40, owner=__name__, setup=_setup_reminders,
              manifest=lambda enforcer, reminders: {"reminders": sorted(reminders)})
extend.option("step_warn", 60, "the step-limit warning (STEP_WARN_NOTE) once that many replayed calls ran; 60 by "
              "default, None is off", order=41, owner=__name__,
              manifest=lambda enforcer, step_warn: {"step_warn": step_warn})


@extend.hook("attach", order=40)
def _init_records(enforcer, env) -> None:
    env.tools.glint_trace = []  # replayed assistant calls that ran, hidden tools unwrapped (glint.declare)
    env.tools.glint_replayed = 0  # calls τ²'s evaluator replays (mutating tools), any requestor


@extend.hook("before_call", order=5)
def _count_replayed(call: extend.Call) -> None:
    call.scratch["replayed"] = replayed = _replayed(call.env, call.message.name)
    call.tools.glint_replayed += replayed


@extend.hook("after_read", order=5)
def _trace_read(call: extend.Call, reply: ToolMessage) -> None:
    _trace(call.env, call.message, reply, call.scratch["replayed"])


@extend.hook("after_write", order=50)
def _trace_write(call: extend.Call, reply: ToolMessage) -> None:
    _trace(call.env, call.message, reply, call.scratch["replayed"])


@extend.hook("after_write", order=60)
def _flag_transfer(call: extend.Call, reply: ToolMessage) -> None:
    """A transfer.flagged event when a transfer ran while every recorded request names an action this toolkit
    has a tool for (TOOLS_COVER). Never decides anything; a failure to log is logged, not raised."""
    enforcer, tools = call.enforcer, call.tools
    requests = getattr(tools, "glint_requests", None)
    if call.write[0] != TRANSFER_TOOL or not requests:
        return
    if not all(any(tools.has_tool(t) for t in REQUEST_ACTIONS.get(r.get("action"), ((),))[0]) for r in requests):
        return
    try:
        enforcer.events.emit("transfer.flagged", {
            "domain": enforcer.domain, "rule_id": TOOLS_COVER, "replay": call.replay,
            "requests": [[r.get("action"), r.get("target_id")] for r in requests],
            "source": "B2 §1, Glint convention: flag only (gold expects transfers in 004, 014, 5 training tasks)",
            **enforcer.context()})
    except Exception:
        log.exception("the transfer.flagged event was not logged")


@extend.hook("open_items", order=10)
def _extra_items(call: extend.Call) -> list[str]:
    """The reminders' [open] lines: pending Declare obligations whose activities glint_trace can show,
    then the step-limit warning. Deterministic over what the replay rebuilds."""
    enforcer, tools = call.enforcer, call.tools
    clauses = [c for c in enforcer.declare_clauses
               if _observable(tools, c["a"]) and (c["b"] is None or _observable(tools, c["b"]))]
    items = [declare.reminder(p) for p in declare.pending(clauses, tools.glint_trace)]
    if enforcer.step_warn is not None and tools.glint_replayed >= enforcer.step_warn:
        items.append(STEP_WARN_NOTE)
    return items
