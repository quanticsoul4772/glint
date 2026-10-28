"""Gate results, the gate registry, the fail-closed runner and the boot check.

Ported from agent-harness-v2 (`enforcement/runner.py`: HookResult, the registry's purpose check, the fail-closed
wrapper and REQUIRED_GATES) and pgharness (`intents/runner.py`: gates judge a proposed action; a gate that raises
blocks it). A gate is a plain function of the live database and the write's arguments; it must be stateless and
deterministic, because τ²'s evaluator replays the agent's writes through the same gated environment.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Literal

log = logging.getLogger("glint.gates")

Decision = Literal["allow", "deny"]
# A gate reads a few database rows; 60 s is far beyond any real evaluation. A live timeout is the one way a live
# decision can differ from τ²'s replay (which waits without a clock, see run_gate), so it is logged as an ERROR.
DEFAULT_TIMEOUT_S = 60.0
NO_RULE = "no_rule_yet"  # rule_id of the allow-and-log gate on a write no rule covers yet


@dataclass(frozen=True)
class GateResult:
    """One gate's verdict on one write. A deny must carry `reason` and `fix`; `purpose` is filled from the gate."""
    decision: Decision
    reason: str = ""
    purpose: str = ""
    fix: str = ""
    rule_id: str = ""
    source: str = ""
    details: dict = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"

    def message(self) -> str:
        """The tool result the agent sees on a deny: deterministic, so τ²'s replay of the call matches it."""
        verdict = json.dumps(self.details, sort_keys=True, default=str)
        return (f"Error: this write was blocked by a policy gate (rule {self.rule_id}, check {self.source}) and was NOT "
                f"executed."
                f"\nReason: {self.reason}\nPurpose: {self.purpose}\nFix: {self.fix}\nVerdict: {verdict}")


def allow(rule_id: str, source: str, details: dict | None = None, reason: str = "") -> GateResult:
    return GateResult("allow", reason=reason, rule_id=rule_id, source=source, details=details or {})


def deny(rule_id: str, source: str, reason: str, fix: str, details: dict | None = None) -> GateResult:
    return GateResult("deny", reason=reason, fix=fix, rule_id=rule_id, source=source, details=details or {})


GateFn = Callable[[Any, dict], GateResult]


@dataclass(frozen=True)
class Gate:
    """One gate. `fn(db, args)` judges the write; with `toolkit=True` it gets the environment's toolkit instead of its
    database (a procedure gate reads `toolkit.db` and which hidden reads that environment records)."""
    name: str
    tools: tuple[str, ...]
    purpose: str
    fn: GateFn
    timeout_s: float = DEFAULT_TIMEOUT_S
    toolkit: bool = False


class GateRegistry:
    def __init__(self):
        self._gates: dict[str, Gate] = {}

    def register(self, gate: Gate) -> Gate:
        if not gate.purpose or not gate.purpose.strip():
            raise ValueError(f"gate {gate.name!r} has no purpose: every gate states why it exists when it fires")
        if not gate.tools:
            raise ValueError(f"gate {gate.name!r} names no tool")
        if gate.name in self._gates:
            raise ValueError(f"gate {gate.name!r} is already registered")
        self._gates[gate.name] = gate
        return gate

    def for_tool(self, tool: str) -> list[Gate]:
        return [g for g in self._gates.values() if tool in g.tools]

    def covered(self) -> set[str]:
        return {t for g in self._gates.values() for t in g.tools}

    def gates(self) -> list[Gate]:
        return list(self._gates.values())


def assert_required(registry: GateRegistry, required: set[str], where: str) -> None:
    """Boot check: raise when a write tool has no gate. A missing gate means that write is silently open."""
    missing = sorted(set(required) - registry.covered())
    if missing:
        raise RuntimeError(f"{where}: write tools without a gate: {missing}. Enforcement is aborted because an "
                           "ungated write is silently open; register a gate (or a no-rule-yet gate) for each.")


_POOL = ThreadPoolExecutor(max_workers=16, thread_name_prefix="glint-gate")


def run_gate(gate: Gate, db, args: dict, wait: bool = False) -> tuple[GateResult, float]:
    """Run one gate, failing CLOSED: an exception, a timeout or a malformed result becomes a deny whose reason names
    the error. Returns the result and the duration in milliseconds. With `wait=True` (τ²'s evaluator replay) there is
    no wall-clock timeout, so the replayed decision depends only on the database and the arguments."""
    start = time.perf_counter()
    future = _POOL.submit(gate.fn, db, args)
    try:
        result = future.result(timeout=None if wait else gate.timeout_s)
    except FutureTimeout:
        log.error("gate %s timed out after %ss: failing closed; τ²'s replay of this call waits and may decide "
                  "differently", gate.name, gate.timeout_s)
        result = deny("gate_timeout", gate.name, f"gate {gate.name!r} timed out after {gate.timeout_s}s",
                      "Retry later or report it; a gate that cannot decide blocks the write by design.")
    except Exception as exc:
        log.exception("gate %s raised: failing closed", gate.name)
        result = deny("gate_error", gate.name, f"gate {gate.name!r} raised {type(exc).__name__}: {exc}",
                      "Check the write's arguments; a gate that cannot decide blocks the write by design.")
    if not isinstance(result, GateResult):
        result = deny("gate_error", gate.name, f"gate {gate.name!r} returned {type(result).__name__}, not a "
                      "GateResult", "Fix the gate; a gate that cannot decide blocks the write by design.")
    elif result.decision not in ("allow", "deny"):
        result = deny("gate_error", gate.name, f"gate {gate.name!r} returned decision {result.decision!r}",
                      "Fix the gate; a gate that cannot decide blocks the write by design.")
    elif not result.allowed and not (result.reason and result.fix):
        result = deny(result.rule_id or "gate_error", gate.name, result.reason or f"gate {gate.name!r} denied "
                      "without a reason", result.fix or "Fix the gate: every deny names its fix.", result.details)
    result = GateResult(result.decision, result.reason, gate.purpose, result.fix, result.rule_id,
                        result.source or gate.name, result.details)
    return result, (time.perf_counter() - start) * 1000
