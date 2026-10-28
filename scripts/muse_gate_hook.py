"""Muse Code PreToolUse hook: the same deterministic write gates, applied to a Muse tool or MCP call.

Muse Code runs this command before every tool call (`.muse/hooks.json`) and passes `hook_event_name`, `tool_name` and
`tool_input` as JSON on stdin (research/5-measurements/M0-results.md, M6a). The hook maps the call to the τ² tool it
names (an MCP name such as `mcp__tau2__cancel_pending_order` ends in the tool name; a banking
`call_discoverable_agent_tool` names its hidden tool), builds the domain's environment on the given database, and runs
glint.gates on it.

- A denied write prints `{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny",
  "permissionDecisionReason": <the gate's message: reason, purpose, fix, verdict>}}`, which blocks the call.
- An allowed write, or a call that is not a gated write, prints nothing: Muse rejects a bare `allow`.
- A hook that cannot decide (a bad payload, a database that does not load) denies: the gates fail closed.

Usage: python scripts/muse_gate_hook.py --domain DOMAIN [--db PATH] [--log PATH]  (payload on stdin)
"""
from __future__ import annotations

import json
import sys

from loguru import logger
from tau2.data_model.message import ToolCall
from tau2.registry import registry

from glint.gates import Enforcer, EventLog

WRAPPER = "call_discoverable_agent_tool"


def deny_output(reason: str) -> dict:
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                   "permissionDecisionReason": reason}}


def tool_name(name: str) -> str:
    """The τ² tool an MCP or Muse tool name stands for: `mcp__server__tool` -> `tool`."""
    return name.rsplit("__", 1)[-1] if name.startswith("mcp__") else name


def environment(domain: str, db_path: str | None):
    constructor = registry.get_env_constructor(domain)
    kwargs = {"retrieval_variant": "bm25_grep"} if domain == "banking_knowledge" else {}
    env = constructor(**kwargs)
    if db_path:
        env = constructor(db=type(env.tools.db).load(db_path), **kwargs)
    return env


def decide(payload: dict, domain: str, db_path: str | None = None, log_path: str = ":memory:") -> dict | None:
    """The hook's output for one PreToolUse payload: a deny, or None to let the call run."""
    if payload.get("hook_event_name", "PreToolUse") != "PreToolUse":
        return None
    name, args = tool_name(str(payload.get("tool_name", ""))), payload.get("tool_input") or {}
    if not isinstance(args, dict):
        return deny_output(f"Glint gate: tool_input of {name} is not a JSON object; the call is blocked.")
    if name == WRAPPER and isinstance(args.get("arguments"), dict):
        args = args | {"arguments": json.dumps(args["arguments"])}
    env = environment(domain, db_path)
    enforcer = Enforcer(domain, EventLog(log_path))
    enforcer.attach(env)
    write = enforcer.write_call(env, ToolCall(id="muse", name=name, arguments=args, requestor="assistant"))
    if write is None:
        return None
    result = enforcer.check(*write, env.tools.db, toolkit=env.tools)
    return None if result.allowed else deny_output(result.message())


def main(argv: list[str], stdin=sys.stdin, stdout=sys.stdout) -> int:
    logger.remove()
    domain = argv[argv.index("--domain") + 1]
    db_path = argv[argv.index("--db") + 1] if "--db" in argv else None
    log_path = argv[argv.index("--log") + 1] if "--log" in argv else ":memory:"
    try:
        out = decide(json.load(stdin), domain, db_path, log_path)
    except Exception as exc:  # noqa: BLE001  (fails closed: a hook that cannot decide blocks the call, saying why)
        out = deny_output(f"Glint gate could not decide ({type(exc).__name__}: {exc}); the call is blocked.")
    if out is not None:
        stdout.write(json.dumps(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
