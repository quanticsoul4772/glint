"""The field-value LLM stages, built with a pluggable model call and OFF by default (appendix rows #44 and #45 of
research/fix-research-2026-11-02.md, "Field values"; "LLM verifiers" is on its "Not before Nov 4" list). The
request-gate verifiers (#10 PolicyGuard, #11 AlignmentCheck, #26 cascade) are glint.gates.conversation's, on the
rest-refusals-defaults branch.

Neither is wired into the gates: a gate is a deterministic function of the database and the write
(glint.gates.core), and an LLM stage is neither deterministic nor measurable offline (memo M12). Each takes `model`
(prompt -> completion text); with no model it returns "off" (or no rules) and calls nothing. Nothing here calls a paid
service; tests use fakes.

- #45 `provenance_guard`: is the checking credit derivable from the account's transactions under
  bank_accounts_(general)_017's net-correction rule (ProvenanceGuard, reader 2 F)? The deterministic form of the same
  rule is glint.gates.fees.
- #44 `compile_rule`, `accept_drafted_rule`, `draft_rules`: rules a model drafts as data are compiled to deterministic
  predicates; a draft is accepted automatically only when it refuses no gold write and adds a refusal (Progent,
  reader 2 E).
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

ModelCall = Callable[[str], str]
STATUSES = ("off", "pass", "fail", "unsure")


@dataclass(frozen=True)
class Verdict:
    status: str
    reason: str = ""
    details: dict = field(default_factory=dict)


OFF = Verdict("off", "no model given: the verifier is off")


def _json_object(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return v if isinstance(v, dict) else None


PROVENANCE_PROMPT = ("Bank policy (bank_accounts_(general)_017): a checking credit equals the net correction across "
                     "every fee that differs from the documented fee schedule, plus rebates due and not applied; do "
                     "not round or estimate.\nFee schedule:\n{schedule}\nTransactions:\n{rows}\nThe agent credited "
                     "{amount}. Answer with JSON only: {{\"expected\": number, \"why\": \"...\"}}.")


def provenance_guard(rows: Iterable[dict], amount: float, schedule: str, model: ModelCall | None = None) -> Verdict:
    """#45: the model derives the credit from the rows; pass when its expected amount equals the credit to the cent,
    fail when it differs, unsure when it gives none."""
    if model is None:
        return OFF
    prompt = PROVENANCE_PROMPT.format(schedule=schedule, amount=amount,
                                      rows="\n".join(json.dumps(r, sort_keys=True) for r in rows))
    v = _json_object(model(prompt))
    try:
        expected = float(v["expected"]) if v else None
    except (KeyError, TypeError, ValueError):
        expected = None
    if expected is None:
        return Verdict("unsure", "the model gave no expected amount")
    status = "pass" if abs(expected - float(amount)) < 0.005 else "fail"
    return Verdict(status, str(v.get("why", "")), {"expected": expected, "amount": float(amount)})


# ---- drafted rules (#44) -------------------------------------------------------------------------------------------

OPS: dict[str, Callable] = {"eq": lambda x, v: x == v, "ne": lambda x, v: x != v, "in": lambda x, v: x in v,
                            "not_in": lambda x, v: x not in v, "le": lambda x, v: float(x) <= float(v),
                            "ge": lambda x, v: float(x) >= float(v), "lt": lambda x, v: float(x) < float(v),
                            "gt": lambda x, v: float(x) > float(v)}


@dataclass(frozen=True)
class DraftedRule:
    """A rule as data: on `tool`, the argument `arg` must satisfy `op value`, else the write is refused."""
    tool: str
    arg: str
    op: str
    value: object
    source: str = ""

    def allows(self, tool: str, args: dict) -> bool:
        if tool != self.tool or self.arg not in args:
            return True
        try:
            return bool(OPS[self.op](args[self.arg], self.value))
        except (TypeError, ValueError):
            return False


def compile_rule(spec: dict) -> DraftedRule | None:
    """A drafted rule from its JSON form; None when it names an unknown operator or misses a field."""
    if not isinstance(spec, dict) or spec.get("op") not in OPS or not spec.get("tool") or not spec.get("arg") \
            or "value" not in spec:
        return None
    return DraftedRule(str(spec["tool"]), str(spec["arg"]), spec["op"], spec["value"], str(spec.get("source", "")))


def accept_drafted_rule(rule: DraftedRule, gold: Iterable[tuple[str, dict]],
                        current: Callable[[str, dict], bool], probes: Iterable[tuple[str, dict]] = ()) -> dict:
    """A drafted rule only ever runs after the current gates (allowed = current and rule), so it is a narrowing by
    construction. It is accepted automatically only when it allows every gold write (0 false refusals, in-sample) and
    adds at least one refusal on the probe writes (a write the current gates allow and the rule refuses); otherwise it
    goes to a person. Returns {"accepted", "gold_refused", "adds"}."""
    gold_refused = [(t, a) for t, a in gold if not rule.allows(t, a)]
    adds = [(t, a) for t, a in probes if current(t, a) and not rule.allows(t, a)]
    return {"accepted": not gold_refused and bool(adds), "gold_refused": gold_refused, "adds": adds}


DRAFT_PROMPT = ("Policy text:\n{text}\n\nWrite the value rules this text states for the tool {tool}, as a JSON array "
                "of objects {{\"tool\", \"arg\", \"op\", \"value\", \"source\"}} with op one of {ops}. Quote the "
                "sentence each rule comes from in source. Answer with the JSON array only.")


def draft_rules(policy_text: str, tool: str, model: ModelCall | None = None) -> list[DraftedRule]:
    """Rules a model drafts from policy text, compiled; [] when no model is given (off)."""
    if model is None:
        return []
    raw = model(DRAFT_PROMPT.format(text=policy_text, tool=tool, ops=", ".join(OPS)))
    m = re.search(r"\[.*\]", raw or "", re.DOTALL)
    try:
        specs = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        specs = []
    return [r for r in map(compile_rule, specs if isinstance(specs, list) else []) if r is not None]
