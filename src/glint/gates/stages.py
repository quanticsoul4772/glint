"""Stages around the write gates: conversation checks, model verifiers, a human approver.

On by default since the owner's decision of 2026-11-03 (every fix on by default): the enforcer option `stages`
defaults to AUTO, which builds `default_stages(resolve_model())`, so each configured stage runs. MODEL is the config
entry ("module:function" naming a function prompt -> reply text); it is empty by default and the repo ships no
function that calls a paid service, so with no model configured AUTO is `Stages()`, which enables nothing: the
enforcer behaves exactly as without stages and no write changes. `stages=None` is the off switch. Each stage comes
from the research note fix-research-2026-11-02, appendix "Request gate, holds and defaults", where it failed the
Nov 4 gate because it needs conversation text, a model, or a person (glint.gates.conversation has the checks):

- #9  `extractor`: before a request-gated write with no request record, build the record from the conversation.
- #8  `address_check`: before a replacement order, skip the address hold when the conversation shows the customer
      confirmed the address on file.
- #16 `ask_rules`: a refusal under one of these rules becomes a question for the customer; the same call sent again
      after the customer agreed runs, attested `asked_confirmed`.
- #10 `checklist`, #11 `alignment`, #26 `cascade`: after every gate allowed the write, a model reads the conversation;
      mode "flag" logs and attests `conversation_flagged`, mode "deny" refuses.
- #12 `approver` on `approval_tools`: a person (or any callable) approves the write; `ConversationAllowlist` is the
      per-conversation allowlist form.

Replay. τ²'s evaluator replays the agent's calls through a fresh gated environment (glint.gates.enforce), and a replayed
decision must equal the live one. A stage never calls its model or approver in replay: each live answer is kept,
keyed by the run context, the stage, the call and its occurrence, and the replay reads it back. A call the live run
never put to the stage adds nothing in replay.

A stage that fails (a model reply that cannot be read, an exception) is logged as `stage.error`. A deny-mode stage
then refuses (a check that cannot decide blocks, as core.run_gate); a flag-mode or record-building stage adds nothing.
"""
from __future__ import annotations

import importlib
import json
import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from glint.assist.banking import REQUEST_TOOLS
from glint.gates import conversation as conv
from glint.gates import extend
from glint.gates.core import CANNOT_DECIDE, GateResult, deny
from glint.gates.rules import REPLACE

log = logging.getLogger("glint.gates")
# config entry: "module:function" of the model call the model-based stages use; empty (default): no model configured,
# so no stage runs and nothing is called
MODEL = ""
AUTO = "auto"  # the `stages` option's default: default_stages(resolve_model())
# (deny_below, allow_at) of the default cascade (#26): the values tests/test_stages.py uses. No labeled benign set exists
# to calibrate them (conversation.calibrate), so they are unvalidated (Glint convention)
CASCADE_THRESHOLDS = (0.2, 0.8)
MODES = ("flag", "deny")
# the request gate's refusals; the suggested ask_rules for research #16 (18 false refusals by gold, research row 0)
REQUEST_RULES = ("banking.request.no_record", "banking.request.not_requested", "banking.request.other_target")
SOURCE = "glint.gates.stages (research fix-research-2026-11-02 appendix #{n})"
Approver = Callable[[str, dict, dict], bool]  # (tool, arguments, run context) -> approved


class ConversationAllowlist:
    """Research #12: writes approved ahead for one conversation, by (tool, target id); target "" approves the tool on
    any target. Keyed by the run context's task_id and trial (glint's run_context)."""

    def __init__(self):
        self._granted: dict[tuple, set[tuple[str, str]]] = {}

    @staticmethod
    def _conversation(context: dict) -> tuple:
        return context.get("task_id"), context.get("trial"), context.get("simulation_id")

    def grant(self, context: dict, tool: str, target: str = "") -> None:
        self._granted.setdefault(self._conversation(context), set()).add((tool, target))

    def __call__(self, tool: str, args: dict, context: dict) -> bool:
        granted = self._granted.get(self._conversation(context), set())
        targets = {str(v) for k, v in args.items() if k.endswith("_id") and k != "user_id" and v}
        return (tool, "") in granted or any((tool, t) in granted for t in targets)


@dataclass(frozen=True)
class Stages:
    transcript: Callable[[], list[conv.Turn]] | None = None  # e.g. a conversation.TranscriptTap
    extractor: conv.ModelCall | None = None
    address_check: bool = False
    address_model: conv.ModelCall | None = None
    ask_rules: tuple[str, ...] = ()
    confirm_model: conv.ModelCall | None = None
    checklist: conv.ModelCall | None = None
    checklist_mode: str = "flag"
    alignment: conv.ModelCall | None = None
    alignment_mode: str = "flag"
    cascade: conv.Cascade | None = None
    approver: Approver | None = None
    approval_tools: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("checklist_mode", "alignment_mode"):
            if getattr(self, name) not in MODES:
                raise ValueError(f"{name} {getattr(self, name)!r}: one of {MODES}")
        reads = [n for n in ("extractor", "checklist", "alignment", "cascade") if getattr(self, n) is not None]
        reads += ["address_check"] * self.address_check + ["ask_rules"] * bool(self.ask_rules)
        if reads and self.transcript is None:
            raise ValueError(f"{', '.join(reads)} read the conversation: give a transcript source")
        if (self.approver is None) != (not self.approval_tools):
            raise ValueError("approver and approval_tools go together")

    def describe(self) -> dict:
        """What is on, for the gate manifest (no model or key is named)."""
        return {"extractor": self.extractor is not None, "address_check": self.address_check,
                "ask_rules": list(self.ask_rules),
                "checklist": self.checklist_mode if self.checklist is not None else None,
                "alignment": self.alignment_mode if self.alignment is not None else None,
                "cascade": None if self.cascade is None else [self.cascade.deny_below, self.cascade.allow_at],
                "approval_tools": list(self.approval_tools)}

    @property
    def enabled(self) -> bool:
        return any(self.describe()[k] for k in self.describe())


def resolve_model(spec: str | None = None) -> conv.ModelCall | None:
    """The function a "module:function" spec names (MODEL when `spec` is None), or None for an empty spec: no model
    configured."""
    spec = MODEL if spec is None else spec
    if not spec:
        return None
    module, _, name = spec.partition(":")
    if not module or not name:
        raise ValueError(f"a stage model is named as module:function, got {spec!r}")
    fn = getattr(importlib.import_module(module), name)
    if not callable(fn):
        raise TypeError(f"{spec} is not callable")
    return fn


def default_stages(model: conv.ModelCall | None, transcript: Callable[[], list[conv.Turn]] | None = None) -> Stages:
    """Every stage a model can run: none without a model, or without a transcript source, since each one reads the
    conversation (Stages(): nothing runs); with both, #9, #8, #16, #10 and #11 in flag mode, and #26. The approver
    (#12) needs a person and is not included."""
    if model is None or transcript is None:
        return Stages()
    return Stages(transcript=transcript, extractor=model, address_check=True, address_model=model,
                  ask_rules=REQUEST_RULES, confirm_model=model, checklist=model, alignment=model,
                  cascade=conv.Cascade(conv.model_score(model), *CASCADE_THRESHOLDS))


_MISSING = object()


class StageRunner:
    """Runs the configured stages for one Enforcer; `events` and `context` are the enforcer's."""

    def __init__(self, stages: Stages, events, context: Callable[[], dict]):
        self.stages, self.events, self.context = stages, events, context
        self._kept: dict[tuple, object] = {}

    @staticmethod
    def init(toolkit) -> None:
        """The per-environment records the stages keep (attach() calls this on every gated environment)."""
        toolkit.glint_stage_seen = Counter()  # (stage, tool, arguments) -> calls seen, for the replay keys
        toolkit.glint_stage_asked = set()  # (tool, arguments, rule) asked once (#16, #26)
        toolkit.glint_stage_trace = []  # (tool, arguments) of every write the stages saw allowed (#11)

    def _emit(self, kind: str, payload: dict) -> None:
        try:
            self.events.emit(kind, payload | self.context())
        except Exception:
            log.exception("the %s event was not logged", kind)

    def _once(self, toolkit, stage: str, tool: str, args: dict, replay: bool, compute: Callable[[], object]):
        """`compute()` live, kept under the call's key; in replay the kept value (_MISSING when the live run never
        asked). An exception is logged as stage.error and returned, so the caller decides fail-closed or no-op."""
        call = (stage, tool, json.dumps(args, sort_keys=True, default=str))
        toolkit.glint_stage_seen[call] += 1
        key = (json.dumps(self.context(), sort_keys=True, default=str), *call, toolkit.glint_stage_seen[call])
        if replay:
            return self._kept.get(key, _MISSING)
        try:
            value = compute()
        except Exception as exc:
            log.exception("stage %s failed on %s", stage, tool)
            self._emit("stage.error", {"stage": stage, "tool": tool, "error": f"{type(exc).__name__}: {exc}"[:300]})
            value = exc
        self._kept[key] = value
        return value

    def _conversation(self) -> list[conv.Turn]:
        return self.stages.transcript()

    # ---- before the gates ------------------------------------------------------------------------------------------

    def prime(self, toolkit, tool: str, args: dict, replay: bool) -> None:
        """#9 and #8: fill what the gates read from the conversation before they run."""
        s = self.stages
        if s.extractor is not None and tool in REQUEST_TOOLS and getattr(toolkit, "glint_requests", None) is None:
            found = self._once(toolkit, "extractor", tool, args, replay,
                               lambda: conv.extract_requests(self._conversation(), s.extractor))
            if isinstance(found, list) and found:
                toolkit.glint_requests = found
                self._emit("requests.extracted", {"tool": tool, "replay": replay,
                                                  "requests": [[r["action"], r["target_id"]] for r in found]})
        if s.address_check and tool == REPLACE:
            card = str(args.get("credit_card_account_id", ""))
            on_file = (toolkit.db.users.data.get(str(args.get("user_id", ""))) or {}).get("address")
            if on_file and card not in toolkit.glint_address_asked:
                confirmed = self._once(toolkit, "address_check", tool, args, replay,
                                       lambda: conv.address_confirmed(self._conversation(), on_file, s.address_model))
                if confirmed is True:
                    toolkit.glint_address_asked.add(card)
                    self._emit("address.confirmed_in_conversation", {"credit_card_account_id": card,
                                                                     "replay": replay})

    # ---- on a refusal ----------------------------------------------------------------------------------------------

    def on_deny(self, toolkit, tool: str, args: dict, result: GateResult, replay: bool) -> GateResult | None:
        """#16: a refusal under an ask rule. The first time, the refusal asks the customer instead; on the same call
        again, None (the gate passes) when the customer agreed, else the refusal."""
        if result.rule_id not in self.stages.ask_rules:
            return result
        return self._ask(toolkit, tool, args, result.rule_id, result.reason, replay)

    def _ask(self, toolkit, tool: str, args: dict, rule: str, reason: str, replay: bool) -> GateResult | None:
        key = (tool, json.dumps(args, sort_keys=True, default=str), rule)
        question = conv.ask_question(tool, args, reason)
        if key not in toolkit.glint_stage_asked:
            toolkit.glint_stage_asked.add(key)
            return deny(f"{rule}.ask", SOURCE.format(n=16), f"before this write, the customer must confirm it: {reason}",
                        f"Ask the customer: \"{question}\" If they agree, send this same call again; if not, do not "
                        "make it.", {"tool": tool, "asked_rule": rule}, consequence="Nothing was changed. The "
                        "customer decides: the same call after they agree goes through.",
                        facts=[f"rule: {rule}", f"what the check found: {reason}"])
        confirmed = self._once(toolkit, "confirm", tool, args, replay,
                               lambda: conv.customer_confirmed(self._conversation(), question, self.stages.confirm_model))
        if confirmed is True:
            self._emit("stage.ask_confirmed", {"tool": tool, "rule": rule, "replay": replay})
            return None
        return deny(f"{rule}.not_confirmed", SOURCE.format(n=16), "the customer's last reply does not agree to this "
                    "write", f"Ask the customer: \"{question}\" Send the call again only after they say yes.",
                    {"tool": tool, "asked_rule": rule}, consequence="Nothing was changed.",
                    facts=[f"rule: {rule}", "the customer's reply to the last question: not a clear yes"])

    # ---- after every gate allowed ----------------------------------------------------------------------------------

    def review(self, toolkit, tool: str, args: dict, replay: bool) -> tuple[GateResult | None, list[str]]:
        """#12, #10, #11, #26 on an allowed write: (a refusal or None, extra attestation classes)."""
        s, flags = self.stages, []
        if s.approver is not None and tool in s.approval_tools:
            ok = self._once(toolkit, "approval", tool, args, replay, lambda: s.approver(tool, args, self.context()))
            if ok is not _MISSING and ok is not True:
                return deny("stage.approval.refused", SOURCE.format(n=12), "the approver did not approve this write",
                            "Tell the customer this change needs approval and was not approved; do not repeat it.",
                            {"tool": tool}, consequence="Nothing was changed.",
                            facts=[f"approver answer: {ok!r}"]), []
            flags.append("approved")
        if s.checklist is not None and tool in conv.CHECKLISTS:
            v = self._once(toolkit, "checklist", tool, args, replay,
                           lambda: conv.checklist_verify(self._conversation(), tool, args, s.checklist))
            refusal = self._verdict(10, "checklist", s.checklist_mode, v, tool, flags,
                                    bad=lambda v: v["unmet"], unsure=lambda v: v["unsure"],
                                    what=lambda v: [i.question for i in conv.CHECKLISTS[tool] if i.id in v["unmet"]])
            if refusal:
                return refusal, []
        if s.alignment is not None:
            done = list(toolkit.glint_stage_trace)
            v = self._once(toolkit, "alignment", tool, args, replay,
                           lambda: conv.alignment_check(self._conversation(), done, tool, args, s.alignment))
            refusal = self._verdict(11, "alignment", s.alignment_mode, v, tool, flags,
                                    bad=lambda v: not v["aligned"], unsure=lambda v: False,
                                    what=lambda v: [v["reason"] or "the write does not serve the customer's request"])
            if refusal:
                return refusal, []
        if s.cascade is not None:
            v = self._once(toolkit, "cascade", tool, args, replay,
                           lambda: s.cascade.decide(self._conversation(), tool, args))
            refusal = self._cascade(toolkit, tool, args, v, replay, flags)
            if refusal is not None:
                return refusal, []
        toolkit.glint_stage_trace.append((tool, args))
        return None, sorted(set(flags))

    def _verdict(self, n: int, stage: str, mode: str, v, tool: str, flags: list, *, bad, unsure, what):
        if v is _MISSING:
            return None
        if isinstance(v, Exception):
            if mode == "deny":
                return deny(f"stage.{stage}.cannot_decide", SOURCE.format(n=n), f"the {stage} verifier could not "
                            "judge this write", "Retry once; if it is refused again, tell the customer it cannot be "
                            "done right now.", {"tool": tool}, consequence=CANNOT_DECIDE,
                            facts=[f"error: {v}"[:300]])
            flags.append("conversation_flagged")
            return None
        if bad(v):
            self._emit("stage.flagged", {"stage": stage, "tool": tool, "mode": mode, "verdict": v})
            if mode == "deny":
                return deny(f"stage.{stage}", SOURCE.format(n=n), f"the {stage} verifier found the conversation "
                            "does not support this write yet", "Do this first: " + "; ".join(what(v)),
                            {"tool": tool, "verdict": v}, consequence="Nothing was changed. The write can run once "
                            "the conversation covers it.", facts=what(v))
            flags.append("conversation_flagged")
        elif unsure(v):
            self._emit("stage.flagged", {"stage": stage, "tool": tool, "mode": mode, "verdict": v})
            flags.append("conversation_flagged")
        return None

    def _cascade(self, toolkit, tool: str, args: dict, v, replay: bool, flags: list) -> GateResult | None:
        if v is _MISSING:
            return None
        if isinstance(v, Exception):
            return deny("stage.cascade.cannot_decide", SOURCE.format(n=26), "the verifier could not judge this write",
                        "Retry once; if it is refused again, tell the customer it cannot be done right now.",
                        {"tool": tool}, consequence=CANNOT_DECIDE, facts=[f"error: {v}"[:300]])
        decision, p = v
        self._emit("stage.cascade", {"tool": tool, "decision": decision, "p": p, "replay": replay})
        if decision == "allow":
            return None
        if decision == "deny":
            return deny("stage.cascade", SOURCE.format(n=26), "the verifier scored this write below the deny "
                        "threshold", "Do not make this write; tell the customer it could not be confirmed as "
                        "following the policy.", {"tool": tool, "p": p}, consequence="Nothing was changed.",
                        facts=[f"verifier score: {p:.3f} < {self.stages.cascade.deny_below}"])
        refusal = self._ask(toolkit, tool, args, "stage.cascade.unsure", f"the verifier is unsure (score {p:.3f})",
                            replay)
        if refusal is None:
            flags.append("asked_confirmed")
        return refusal


# The stages' enforcer option and records (glint.gates.extend).

def _setup(enforcer, stages: Stages | str | None) -> None:
    if isinstance(stages, str):
        if stages != AUTO:
            raise ValueError(f"stages {stages!r}: a Stages(), {AUTO!r} or None")
        stages = default_stages(resolve_model())
    enforcer.stages = (StageRunner(stages, enforcer.events, enforcer.context)
                       if stages is not None and stages.enabled else None)


extend.option("stages", AUTO, "a Stages() of conversation checks, model verifiers and an approver; \"auto\" "
              "(default) runs each configured stage, None is off",
              order=20, owner=__name__, setup=_setup,
              manifest=lambda enforcer, stages: {"stages": stages.describe()} if enforcer.stages else {})


@extend.hook("attach", order=20)
def _init_records(enforcer, env) -> None:
    StageRunner.init(env.tools)
