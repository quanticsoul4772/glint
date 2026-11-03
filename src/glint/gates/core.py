"""Gate results, the gate registry, the fail-closed runner and the boot check.

All code was written during the event; no earlier code was copied. The design follows the author's earlier private
projects (agent-harness-v2 and pgharness): a result type with a stated purpose, a registry that refuses a gate
without one, a fail-closed runner, a boot check that every write has a gate, and gates that judge a proposed action
before it runs. A gate is a plain function of the live database and the write's arguments; it must be stateless and
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
# How a deny reads to the agent: "plain" (what the check found, the facts it used, why it matters, what to do next;
# since PR 87) or "terse" (PR 86's layout: Reason, Purpose, Fix and Verdict, with the facts joined into the Reason line
# as PR 86's reasons carried their specifics inline, and no consequence line). "terse" exists for the measurement that
# separates dispute planning from the plain-language messages (decision 2026-10-29).
WORDINGS = ("plain", "terse")
# A gate reads a few database rows; 60 s is far beyond any real evaluation. A live timeout is the one way a live
# decision can differ from τ²'s replay (which waits without a clock, see run_gate), so it is logged as an ERROR.
DEFAULT_TIMEOUT_S = 60.0
NO_RULE = "no_rule_yet"  # rule_id of the allow-and-log gate on a write no rule covers yet
# the start of every deny the agent sees (GateResult.message); glint.gates.summary counts denied writes by it
DENY_PREFIX = "Error: this write was blocked by a policy gate"


@dataclass(frozen=True)
class GateResult:
    """One gate's verdict on one write. A deny must carry `reason`, `fix`, `consequence` and `facts`; `purpose` is
    filled from the gate.

    The agent reads a deny in plain language: what the check found (`reason`), the facts it used (`facts`: the
    database values and arguments the decision rests on), what that means for the rest of the customer's request
    (`consequence`) and the next step (`fix`). `fix` stays the immediate step, because glint.gates.summary judges
    from its wording whether the agent followed it."""
    decision: Decision
    reason: str = ""
    purpose: str = ""
    fix: str = ""
    rule_id: str = ""
    source: str = ""
    details: dict = field(default_factory=dict)
    consequence: str = ""
    facts: tuple[str, ...] = ()
    # an allow's reminder to the agent: appended to the tool's reply (glint.gates.enforce); empty on most allows
    note: str = ""

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"

    def message(self, wording: str = "plain") -> str:
        """The tool result the agent sees on a deny: deterministic, so τ²'s replay of the call matches it."""
        verdict = json.dumps(self.details, sort_keys=True, default=str)
        if wording == "terse":
            return (f"{DENY_PREFIX} (rule {self.rule_id}, check {self.source}) and was NOT executed."
                    f"\nReason: {'; '.join((self.reason, *self.facts))}\nPurpose: {self.purpose}\nFix: {self.fix}"
                    f"\nVerdict: {verdict}")
        if wording != "plain":
            raise ValueError(f"unknown deny wording {wording!r}; one of {WORDINGS}")
        facts = "".join(f"\n- {f}" for f in self.facts)
        return (f"{DENY_PREFIX} and was NOT executed. Nothing changed.\nRule: {self.rule_id} ({self.source})"
                f"\nWhat the check found: {self.reason}\nFacts it used:{facts}\nWhy it matters: {self.consequence}"
                f"\nWhat to do next: {self.fix}\nVerdict: {verdict}")


def allow(rule_id: str, source: str, details: dict | None = None, reason: str = "", note: str = "") -> GateResult:
    return GateResult("allow", reason=reason, rule_id=rule_id, source=source, details=details or {}, note=note)


def deny(rule_id: str, source: str, reason: str, fix: str, details: dict | None = None, *, consequence: str,
         facts: list[str] | tuple[str, ...]) -> GateResult:
    return GateResult("deny", reason=reason, fix=fix, rule_id=rule_id, source=source, details=details or {},
                      consequence=consequence, facts=tuple(facts))


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


CANNOT_DECIDE = ("A check that cannot decide blocks the write by design, so a write it cannot judge never runs "
                 "unchecked. The rest of the request is not affected.")


def _cannot_decide(rule_id: str, gate: Gate, reason: str, fact: str, fix: str) -> GateResult:
    return deny(rule_id, gate.name, reason, fix, consequence=CANNOT_DECIDE, facts=[fact])


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
        result = _cannot_decide("gate_timeout", gate, f"the check {gate.name!r} did not finish within "
                                f"{gate.timeout_s:g} s", f"timeout: {gate.timeout_s:g} s",
                                "Retry this write once; if it is blocked again, tell the customer it cannot be done "
                                "right now.")
    except Exception as exc:
        log.exception("gate %s raised: failing closed", gate.name)
        result = _cannot_decide("gate_error", gate, f"the check {gate.name!r} failed with an error",
                                f"error: {type(exc).__name__}: {exc}", "Check the write's arguments against the tool's "
                                "description, then retry with corrected arguments.")
    if not isinstance(result, GateResult):
        result = _cannot_decide("gate_error", gate, f"the check {gate.name!r} returned no verdict",
                                f"returned: {type(result).__name__}", "Fix the gate.")
    elif result.decision not in ("allow", "deny"):
        result = _cannot_decide("gate_error", gate, f"the check {gate.name!r} returned an unknown decision",
                                f"decision: {result.decision!r}", "Fix the gate.")
    elif not result.allowed and not (result.reason and result.fix and result.consequence and result.facts):
        missing = [k for k in ("reason", "fix", "consequence", "facts") if not getattr(result, k)]
        result = GateResult("deny", result.reason or f"gate {gate.name!r} denied without a reason",
                            fix=result.fix or "Fix the gate: every deny names its fix.", rule_id=result.rule_id
                            or "gate_error", details=result.details,
                            consequence=result.consequence or CANNOT_DECIDE,
                            facts=result.facts or (f"the deny is missing {missing}",))
    result = GateResult(result.decision, result.reason, gate.purpose, result.fix, result.rule_id,
                        result.source or gate.name, result.details, result.consequence, result.facts, result.note)
    return result, (time.perf_counter() - start) * 1000
