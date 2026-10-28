"""The interception point: every assistant write through one environment's `get_response` is checked first.

`Enforcer(domain, log)` builds the domain's gates, runs the boot check (REQUIRED_GATES) and logs the manifest.
`enforcer.attach(env)` checks that every write tool of that environment's toolkit has a gate, then wraps THAT
instance's `get_response`. Only `requestor == "assistant"` calls to WRITE tools are gated; a banking
`call_discoverable_agent_tool` is unwrapped and its hidden tool gated. On a deny the write does not run and the agent
gets an error message carrying the reason, purpose, fix and verdict; arguments are never rewritten. τ²'s gold actions
go through `make_tool_call`, not `get_response`, so they stay ungated.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

from tau2.data_model.message import ToolCall, ToolMessage
from tau2.environment.toolkit import ToolType

from glint.gates.core import GateResult, assert_required, deny, run_gate
from glint.gates.eventlog import EventLog
from glint.gates.rules import REQUIRED_GATES, build_registry, manifest, parse_hidden_arguments

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
        if not env.tools.has_tool(name) or env.tools.tool_type(name) != ToolType.WRITE or name == WRAPPER:
            return None
        return name, args

    def check(self, tool: str, args: dict | None, db, replay: bool = False) -> GateResult:
        """Run every gate on the write; the first deny wins. Each gate's decision is logged as gate.fired."""
        if args is None:
            result = deny("unparsed_arguments", "glint.gates.enforce", "the hidden tool's arguments are not a JSON "
                          "object", "Pass `arguments` as a JSON object string.")
            self._log(tool, args, result, 0.0, replay, "unparsed")
            return result
        final = None
        for gate in self.registry.for_tool(tool):
            result, ms = run_gate(gate, db, args)
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

    def attach(self, env):
        """Gate this environment instance's writes. Raises when a write tool of its toolkit has no gate."""
        if getattr(env, "glint_enforcer", None) is self:
            return env
        if getattr(env, "glint_enforcer", None) is not None:
            raise RuntimeError("this environment is already gated by another enforcer")
        assert_required(self.registry, write_tools(env.tools), f"gates for {self.domain} on {type(env.tools).__name__}")
        original_response, original_set_state = env.get_response, env.set_state
        state = {"replay": False}

        def get_response(message: ToolCall) -> ToolMessage:
            write = self.write_call(env, message)
            if write is None:
                return original_response(message)
            result = self.check(*write, env.tools.db, replay=state["replay"])
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
