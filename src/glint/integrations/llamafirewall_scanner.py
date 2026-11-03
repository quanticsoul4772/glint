"""Glint's gate decision as a LlamaFirewall custom scanner (design 2026-11-02 §9; agenda Q14).

LlamaFirewall 1.0.3 runs scanners per message role. `make_scanner(enforcer, env)` builds a `Scanner` subclass for
`Role.ASSISTANT` messages and registers it under a name with `register_llamafirewall_scanner`; LlamaFirewall then
creates it with no arguments (`create_scanner` calls `scanner_class()`), so the class is bound to one Enforcer and one
gated environment when it is made. Its `scan` reads the message's `tool_calls`, asks the same Enforcer the gated
environment uses about each write, and returns a ScanResult:

- a refusal -> BLOCK, with the refusal message the agent would get (rule, what the check found, the facts it used,
  why it matters, what to do next);
- a hold -> HUMAN_IN_THE_LOOP_REQUIRED, with the same message. A hold is one of the two ask rules, which refuse an
  order once so the agent asks the customer before it is sent again (rules.EXPEDITE_ASK, rules.ADDRESS_ASK;
  policy credit_card_replacements_001). The mapping of those two rules to human review is a Glint convention;
- a read, or a write every gate allows -> ALLOW.

The first BLOCK in a message wins; otherwise a hold wins over ALLOW. Every call is judged on the database as it is
when the message is scanned, since none of the message's calls has run yet. A tool call whose arguments do not parse
is a BLOCK (fail closed, as Enforcer.check refuses unparsed hidden-tool arguments).

The environment must be gated by the same Enforcer (`enforcer.attach(env)`): the gates read the records the wrapper
keeps on `env.tools` (verified users, lookups, plans). The scanner only reads; it records nothing, so a hold repeats
until the call runs through the environment. A scan logs no gate.fired and no write.attested event
(`Enforcer.check(record=False)`): a call the scanner allows is checked again when it runs through the environment's
wrapper, which logs its decisions and attests it once.

llamafirewall is an optional dependency (`pip install glint[llamafirewall]`): it is imported only inside
`make_scanner`. The mapping is plain functions so it is tested without it.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass

from tau2.data_model.message import ToolCall

from glint.gates.core import GateResult
from glint.gates.rules import ADDRESS_ASK, EXPEDITE_ASK

ALLOW, BLOCK, HOLD = "ALLOW", "BLOCK", "HUMAN_IN_THE_LOOP_REQUIRED"  # llamafirewall.ScanDecision member names
HOLD_RULES = frozenset({EXPEDITE_ASK, ADDRESS_ASK})  # Glint convention: an ask rule waits on the customer's answer
DEFAULT_NAME = "glint_gates"


@dataclass(frozen=True)
class Verdict:
    """One scan's outcome in LlamaFirewall's terms: a ScanDecision member name, the reason and the score."""
    decision: str
    reason: str
    score: float


def parse_tool_call(raw: dict, index: int = 0) -> ToolCall | None:
    """A tool call from an AssistantMessage's `tool_calls` entry, as an assistant ToolCall; None when its arguments
    do not parse to an object. LlamaFirewall does not fix the entry's shape, so both the OpenAI shape
    ({"id", "function": {"name", "arguments": JSON string}}) and a flat one ({"name", "arguments": dict}) are read."""
    body = raw.get("function") if isinstance(raw.get("function"), dict) else raw
    args = body.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if not isinstance(args, dict):
        return None
    return ToolCall(id=str(raw.get("id") or f"llamafirewall-{index}"), name=str(body.get("name", "")),
                    arguments=args, requestor="assistant")


def to_verdict(result: GateResult, wording: str = "plain") -> Verdict:
    """A gate result in LlamaFirewall's terms: a refusal blocks, a hold asks for human review, an allow allows."""
    if result.allowed:
        return Verdict(ALLOW, f"allowed by Glint rule {result.rule_id} ({result.source})", 0.0)
    return Verdict(HOLD if result.rule_id in HOLD_RULES else BLOCK, result.message(wording), 1.0)


def judge_call(enforcer, env, call: ToolCall) -> Verdict:
    """The Enforcer's decision on one tool call, without running it. A call that is not a gated write is allowed."""
    write = enforcer.write_call(env, call)
    if write is None:
        return Verdict(ALLOW, f"{call.name} is not a gated write", 0.0)
    # record=False: no gate.fired and no write.attested; the call is logged and attested once, when it runs through
    # the environment's wrapper (defect 6, 2026-11-03)
    return to_verdict(enforcer.check(*write, env.tools.db, toolkit=env.tools, record=False), enforcer.wording)


def judge_tool_calls(enforcer, env, tool_calls: Iterable[dict] | None) -> Verdict:
    """One message's verdict: the first BLOCK, else the first hold, else ALLOW."""
    hold = None
    for index, raw in enumerate(tool_calls or ()):
        call = parse_tool_call(raw, index)
        if call is None:
            return Verdict(BLOCK, f"tool call {index}: its arguments are not a JSON object, so no check could judge "
                           "it", 1.0)
        verdict = judge_call(enforcer, env, call)
        if verdict.decision == BLOCK:
            return verdict
        hold = hold or (verdict if verdict.decision == HOLD else None)
    return hold or Verdict(ALLOW, "no gated write was refused", 0.0)


def make_scanner(enforcer, env, name: str = DEFAULT_NAME):
    """A LlamaFirewall Scanner class bound to `enforcer` and the environment it gates, registered under `name`.
    Use it as `LlamaFirewall({Role.ASSISTANT: [name]})`. Needs the llamafirewall extra."""
    if getattr(env, "glint_enforcer", None) is not enforcer:
        raise ValueError("the environment must be gated by this enforcer (enforcer.attach(env)) before scanning")
    from llamafirewall import Role, ScanDecision, Scanner, ScanResult, register_llamafirewall_scanner

    @register_llamafirewall_scanner(name)
    class GlintGateScanner(Scanner):
        def __init__(self) -> None:
            super().__init__(name)

        async def scan(self, message, past_trace=None):
            if message.role != Role.ASSISTANT:
                verdict = Verdict(ALLOW, "Glint gates judge assistant tool calls only", 0.0)
            else:
                verdict = judge_tool_calls(enforcer, env, message.tool_calls)
            return ScanResult(decision=ScanDecision[verdict.decision], reason=verdict.reason, score=verdict.score)

    return GlintGateScanner
