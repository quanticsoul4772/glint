"""The interception point: every assistant write through one environment's `get_response` is checked first.

`Enforcer(domain, log)` builds the domain's gates, runs the boot check (REQUIRED_GATES) and logs the manifest.
`enforcer.attach(env)` checks that every write tool of that environment's toolkit has a gate, then wraps THAT
instance's `get_response`. Only `requestor == "assistant"` calls to WRITE tools are gated; a banking
`call_discoverable_agent_tool` is unwrapped and its hidden tool gated. On a deny the write does not run and the agent
gets an error message carrying the reason, purpose, fix and verdict; arguments are never rewritten. τ²'s gold actions
go through `make_tool_call`, not `get_response`, so they stay ungated. A few GENERIC tools that end or change the
conversation are gated like writes (rules.GATED_GENERIC: banking's transfer_to_human_agents).

The wrapper also keeps the environment's own record of the hidden READ tools the agent called successfully
(`env.tools.glint_reads`); the procedure gates judge required lookups on it. Hidden reads go through
call_discoverable_agent_tool, a WRITE that τ²'s evaluator replays, so the replay rebuilds the same record and every
gate decision replays unchanged. In replay a gate runs without a wall-clock timeout (core.run_gate).

The gates run in the server that serves the tools (decision 2026-10-28). A Muse Code PreToolUse hook would return
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
"permissionDecisionReason": GateResult.message()} } to block a call (research/5-measurements/M0-results.md, M6a).
"""
from __future__ import annotations

import logging
from collections.abc import Callable

from tau2.data_model.message import ToolCall, ToolMessage
from tau2.environment.toolkit import ToolType

from glint.gates.core import GateResult, assert_required, deny, run_gate
from glint.gates.eventlog import EventLog
from glint.gates.rules import GATED_GENERIC, REQUIRED_GATES, build_registry, manifest, parse_hidden_arguments

log = logging.getLogger("glint.gates")
WRAPPER = "call_discoverable_agent_tool"


def write_tools(toolkit) -> set[str]:
    """Every WRITE tool the toolkit can run, hidden (discoverable) ones included; the discoverable-call wrapper is not
    one itself, its hidden tool is."""
    return {n for n in toolkit.tools if toolkit.tool_type(n) == ToolType.WRITE and n != WRAPPER}


class Enforcer:
    def __init__(self, domain: str, events: EventLog, context: Callable[[], dict] | None = None):
        self.domain, self.events, self.context = domain, events, context or (dict)
        self.registry = build_registry(domain)
        assert_required(self.registry, REQUIRED_GATES[domain], f"gates for {domain}")
        self.events.emit("gates.manifest", manifest(domain, self.registry))

    def write_call(self, env, call: ToolCall) -> tuple[str, dict | None] | None:
        """(write tool, its arguments) when `call` is an assistant write, else None. Arguments are None when a hidden
        tool's arguments do not parse."""
        if call.requestor != "assistant" or env.tools is None:
            return None
        name, args = call.name, call.arguments or {}
        if name == WRAPPER:
            name, args = str(args.get("agent_tool_name", "")), parse_hidden_arguments(args.get("arguments", "{}"))
        if not env.tools.has_tool(name) or name == WRAPPER:
            return None
        if env.tools.tool_type(name) != ToolType.WRITE and name not in GATED_GENERIC.get(self.domain, ()):
            return None
        return name, args

    def check(self, tool: str, args: dict | None, db, replay: bool = False, toolkit=None) -> GateResult:
        """Run every gate on the write; the first deny wins. Each gate's decision is logged as gate.fired. A gate with
        `toolkit=True` gets `toolkit` (the environment's toolkit) instead of `db`."""
        if args is None:
            result = deny("unparsed_arguments", "glint.gates.enforce", "the hidden tool's arguments are not a JSON "
                          "object", "Pass `arguments` as a JSON object string.")
            self._log(tool, args, result, 0.0, replay, "unparsed")
            return result
        final = None
        for gate in self.registry.for_tool(tool):
            if gate.toolkit and toolkit is None:
                result, ms = deny("gate_error", gate.name, f"gate {gate.name!r} needs the environment's toolkit and "
                                  "was given none", "Call Enforcer.check with toolkit=env.tools."), 0.0
            else:
                result, ms = run_gate(gate, toolkit if gate.toolkit else db, args, wait=replay)
            if not self._log(tool, args, result, ms, replay, gate.name):
                result = deny("audit_failed", gate.name, f"gate {gate.name!r} fired but its event could not be "
                              "logged; denying rather than allowing a write with no audit trail",
                              "Check the gate log file.")
            if not result.allowed:
                return result
            final = final or result
        if final is None:  # attach() refuses an environment with an ungated write; this is a second line
            raise RuntimeError(f"write {tool} has no gate")
        return final

    def _log(self, tool, args, result: GateResult, ms: float, replay: bool, gate: str) -> bool:
        try:
            self.events.emit("gate.fired", {
                "domain": self.domain, "gate": gate, "tool": tool, "args": args, "decision": result.decision,
                "rule_id": result.rule_id, "source": result.source, "reason": result.reason,
                "duration_ms": round(ms, 3), "replay": replay, **self.context()})
            return True
        except Exception:
            log.exception("gate %s: the gate.fired event was not logged", gate)
            return False

    @staticmethod
    def _record_read(env, call: ToolCall, reply: ToolMessage) -> None:
        """Note a hidden READ tool the agent called through the wrapper and that ran (its reply is not an error)."""
        if call.requestor != "assistant" or call.name != WRAPPER or reply.error:
            return
        name = str((call.arguments or {}).get("agent_tool_name", ""))
        if str(reply.content).startswith("Error") or not env.tools.has_tool(name):
            return
        if env.tools.tool_type(name) == ToolType.READ:
            env.tools.glint_reads.add(name)

    def attach(self, env):
        """Gate this environment instance's writes. Raises when a write tool of its toolkit has no gate."""
        if getattr(env, "glint_enforcer", None) is self:
            return env
        if getattr(env, "glint_enforcer", None) is not None:
            raise RuntimeError("this environment is already gated by another enforcer")
        assert_required(self.registry, write_tools(env.tools), f"gates for {self.domain} on {type(env.tools).__name__}")
        original_response, original_set_state = env.get_response, env.set_state
        state = {"replay": False}

        env.tools.glint_reads = set()

        def get_response(message: ToolCall) -> ToolMessage:
            write = self.write_call(env, message)
            if write is None:
                reply = original_response(message)
                self._record_read(env, message, reply)
                return reply
            result = self.check(*write, env.tools.db, replay=state["replay"], toolkit=env.tools)
            if result.allowed:
                return original_response(message)
            return ToolMessage(id=message.id, content=result.message(), requestor=message.requestor,
                               role="tool", error=True)

        def set_state(*args, **kwargs):  # τ²'s evaluator replays the trajectory here: those events say replay=true
            state["replay"] = True
            try:
                return original_set_state(*args, **kwargs)
            finally:
                state["replay"] = False

        env.get_response, env.set_state, env.glint_enforcer = get_response, set_state, self
        return env
