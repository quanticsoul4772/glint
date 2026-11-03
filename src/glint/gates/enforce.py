"""The interception point: every assistant write through one environment's `get_response` is checked first.

`Enforcer(domain, log)` builds the domain's gates, runs the boot check (REQUIRED_GATES) and logs the manifest.
`enforcer.attach(env)` checks that every write tool of that environment's toolkit has a gate, then wraps THAT
instance's `get_response`. Only `requestor == "assistant"` calls to WRITE tools are gated; a banking
`call_discoverable_agent_tool` is unwrapped and its hidden tool gated. On a deny the write does not run and the agent
gets an error message carrying the rule, what the check found, the facts it used, why it matters, the next step
and the verdict; arguments are never rewritten. τ²'s gold actions
go through `make_tool_call`, not `get_response`, so they stay ungated. A few GENERIC tools that end or change the
conversation are gated like writes (rules.GATED_GENERIC: banking's transfer_to_human_agents).

The wrapper also keeps the environment's own record of the hidden READ tools the agent called successfully
(`env.tools.glint_reads`); the procedure gates judge required lookups on it. Hidden reads go through
call_discoverable_agent_tool, a WRITE that τ²'s evaluator replays, so the replay rebuilds the same record and every
gate decision replays unchanged. In replay a gate runs without a wall-clock timeout (core.run_gate).

It also keeps who has a credit limit increase submitted and not decided (`env.tools.glint_increases`: user ->
the request ids every submit that ran created for them; an approve or deny that ran clears the requests on its card,
whichever user it names), which the increase
gates judge on (glint.gates.bankdb.undecided_increases): the database cannot show a second decision. Submits and
decisions go through call_discoverable_agent_tool, so the replay rebuilds the same record.

On banking it also keeps the dispute plans (`env.tools.glint_plans`: user id -> the latest plan_disputes result the
agent got), which the plan gate judges each dispute on. τ²'s evaluator replays only mutating calls and skips READ
tools, so a plan recorded from a READ call would be missing in replay and the gate would decide differently there.
plan_disputes is therefore typed GENERIC with mutates_state=True (like τ²'s own give_discoverable_user_tool): τ²
replays it, it never writes the database (the database hash is unchanged), and its result depends only on the
database and its arguments, so the strict replay compare matches and the replay rebuilds the same plans. attach()
puts plan_disputes on every gated banking toolkit, with or without the other decision tools: the gate needs it.

On banking every reply the agent gets that is not an error (reads and allowed writes alike) ends with the open items
of the customer's request that the recorded plans, the increase record and the database show (`_open_items`):
disputes planned and not filed, replacements the customer chose and not ordered, increase verdicts
(check_credit_limit_increase approve or deny, `env.tools.glint_verdicts`) with no decision filed, increases submitted
and not decided. plan_disputes' own reply carries none (the plan lists them). In banking-plan-v6/v7/v8-gated, 2, 6
and 6 runs got an increase verdict and filed no matching decision, every one a failure (052 in all eight of v7/v8);
check_credit_limit_increase is GENERIC with mutates_state=True so the replay rebuilds the verdict record. In banking-plan-v5-gated, 031 trial 1, 037 trial 3 and
038 trial 3 each made the plan and filed none of its disputes before the customer ended the conversation. The items are a function of what the replay rebuilds, so a write's
reply, footer included, replays unchanged.

Since 2026-11-03 (fix research 2026-11-02; glint.gates.aids) it also keeps the writes that ran (`glint_ran`, for the
duplicate-write deny) and the shipping address recorded with record_address_confirmation (`glint_address_confirmed`). An account read's reply carries the one-credit-per-checking-
account fact (bank_accounts_(general)_017) ahead of the open items.

The gates run in the server that serves the tools (decision 2026-10-28). A Muse Code PreToolUse hook would return
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
"permissionDecisionReason": GateResult.message()} } to block a call (a PreToolUse deny
blocking the call was checked in Muse Code on 2026-10-26).
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import replace

from tau2.data_model.message import ToolCall, ToolMessage
from tau2.environment.toolkit import ToolType

from glint.assist import as_flag
from glint.assist.banking import (
    ADDRESS_TOOL,
    CLOSURE_REASON_TOOL,
    DISPUTE_ANSWERS_TOOL,
    ORDER_TOOL,
    PLAN_TOOL,
    REQUESTS_TOOL,
    RETENTION_TOOL,
    STATEMENT_CREDIT_TOOL,
    TODAY,
    _date,
    _tier,
    closure_reasons,
    replacements_today,
    waiver_call,
    waiver_due,
)
from glint.assist.tau2_tools import add_plan_tool, drop_request_tool
from glint.gates import aids, debit
from glint.gates.bankdb import increase_request_ids, undecided_increases
from glint.gates.banking import RETENTION_CREDIT, UNRECORDED, closure_blockers
from glint.gates.core import CANNOT_DECIDE, WORDINGS, GateResult, assert_required, deny, run_gate
from glint.gates.eventlog import EventLog
from glint.gates.procedure import TRANSFER_UNCODED
from glint.gates.rules import (
    ADDRESS_ASK,
    APPROVE,
    DENY,
    DISPUTE,
    EXPEDITE_ASK,
    GATED_GENERIC,
    REQUIRED_GATES,
    SUBMIT,
    build_registry,
    manifest,
    parse_hidden_arguments,
)

log = logging.getLogger("glint.gates")
WRAPPER = "call_discoverable_agent_tool"
VERDICT_TOOL = "check_credit_limit_increase"
GLINT_RECORDS = (PLAN_TOOL, RETENTION_TOOL, REQUESTS_TOOL, DISPUTE_ANSWERS_TOOL, ADDRESS_TOOL)  # replies: no open items
CHECK_TOOLS = ("check_card_dispute", "check_verification", VERDICT_TOOL)  # decision tools logged as check.recorded


def _open_items(tools, request_gate: bool = True) -> list[str]:
    """The customer's open items as the recorded plans, the increase record and the database show them: disputes
    planned and not filed, cards the customer chose to cancel and replace with no replacement ordered today, increases
    submitted and not decided, and, with the request gate on, the request record not made yet. Deterministic."""
    db, items = tools.db, []
    for user, plan in sorted(getattr(tools, "glint_plans", {}).items()):
        filed = {r.get("transaction_id") for r in db.transaction_disputes.data.values() if r.get("user_id") == user}
        unfiled = [str(e["transaction_id"]) for e in plan.get("order", []) if e["transaction_id"] not in filed]
        if unfiled:
            # banking-plan-v9-gated: in 7 failed runs this item was on every check_card_dispute reply, and the agent
            # told the customer disputes were filed without ever unlocking the filing tool
            items.append(f"[open] disputes planned for {user} and not filed yet, in the plan's order: "
                         f"{', '.join(unfiled)}. A dispute is filed only when {DISPUTE} (unlock it, then call it) "
                         "returns a dispute id.")
        for c in plan.get("card_actions", []):
            card = str(c.get("credit_card_account_id", ""))
            if c.get("card_action") == "cancel_and_reissue" and not replacements_today(db, card):
                items.append(f"[open] {card}: the customer chose to cancel it and get a new one; {ORDER_TOOL} not "
                             "called yet")
    for card, v in sorted(getattr(tools, "glint_verdicts", {}).items()):
        decision = (f"{APPROVE} with new_credit_limit {v['new_credit_limit']}" if v["verdict"] == "approve"
                    else f"{DENY} with denial_reason {v['denial_reason']}")
        items.append(f"[open] check_credit_limit_increase said {v['verdict']} for {card}, and no decision is filed yet: "
                     f"{SUBMIT} (if not submitted), the required lookups, then {decision}")
    pending = undecided_increases(tools)
    if pending:
        items.append(f"[open] credit limit increase submitted and not decided yet: {', '.join(pending)} (the "
                     "required lookups, then approve or deny)")
    items += _closure_items(tools)
    verified = [r.get("user_id") for r in db.verification_history.data.values()
                if str(r.get("time_verified", "")).startswith(TODAY.isoformat())]
    if request_gate and verified and getattr(tools, "glint_requests", "absent") is None:
        # banking-test-gated (2026-11-02): the request gate refused 17 writes τ² expects before the agent recorded
        # them; this reminder comes before the first account change instead of after a refusal
        items.append(f"[open] the customer's requests are not recorded yet: before any account change, call "
                     f"{REQUESTS_TOOL}(user_id={verified[-1]!r}, requests=[{{action, target_id, customer_words}}]) "
                     "with every account action they asked for; closes, payments and filings are refused until "
                     "recorded, and a closure reason log or card order made without a record is flagged for review")
    return items + debit.answer_items(tools)


def _closure_items(tools) -> list[str]:
    """Open steps of the credit card closure procedure (credit_card_account_logistics_003), per account still open:
    a closure the request record names with no reason logged (step 3; skipped when a reason within the past year
    skips retention, or when the account cannot close yet, step 1); the one-year annual fee waiver due and not
    offered (step 4: annual_fee reason, a customer for 2+ years, tenure read as glint.assist.banking.customer_since,
    an ambiguity tagged there); an accepted retention credit not applied (step 5, the card tier's amount). The agent
    ignored shown open items in 7 v9 runs and in banking-test-gated 047 t1, so this reminds and never refuses."""
    db, items = tools.db, []
    accounts = db.credit_card_accounts.data
    retention = getattr(tools, "glint_retention", {}) or {}
    asked = [str(r.get("target_id") or "") for r in getattr(tools, "glint_requests", None) or []
             if r.get("action") == "close_credit_card"]
    logged_today = {str(r.get("credit_card_account_id")) for r in db.credit_card_closure_reasons.data.values()
                    if _date(r.get("logged_at")) == TODAY}
    cards = sorted({c for c in asked if c} | set(retention) | logged_today)
    for card in cards:
        acc = accounts.get(card)
        if acc is None or str(acc.get("status", "")).upper() == "CLOSED":
            continue
        user = str(acc.get("user_id", ""))
        today, earlier = closure_reasons(db, card)
        answer = retention.get(card)
        if card in asked and not today and not earlier and not closure_blockers(db, card):
            items.append(f"[open] {card}: the customer asked to close it and no closure reason is logged: ask why and "
                         f"log it with {CLOSURE_REASON_TOOL} before any retention offer or closure")
        if (since := waiver_due(db, card)) is not None and not (answer and answer.get("fee_waiver_offered")):
            items.append(f"[open] {card}: closing over the annual fee, a customer since {since:%m/%d/%Y} (2+ years): "
                         "offer to waive the annual fee for one year before the retention offer; if they accept, "
                         f"{waiver_call(card, user)}")
        if answer and answer.get("accepted") and "point" not in str(answer.get("offer", "")).lower():
            credited = any(r.get("credit_card_account_id") == card and r.get("credit_reason") == "retention_offer"
                           and _date(r.get("transaction_date")) == TODAY
                           for r in db.credit_card_transaction_history.data.values())
            amount = RETENTION_CREDIT.get(_tier(str(acc.get("card_type", ""))))
            if not credited:
                items.append(f"[open] {card}: the customer accepted the retention offer ({answer.get('offer')}) and it "
                             f"is not applied yet: {STATEMENT_CREDIT_TOOL}(user_id={user!r}, credit_card_account_id="
                             f"{card!r}, amount={amount if amount is not None else 'the card tier amount'}, "
                             "reason='retention_offer'); the account stays open")
    return items


def write_tools(toolkit) -> set[str]:
    """Every WRITE tool the toolkit can run, hidden (discoverable) ones included; the discoverable-call wrapper is not
    one itself, its hidden tool is."""
    return {n for n in toolkit.tools if toolkit.tool_type(n) == ToolType.WRITE and n != WRAPPER}


REPEAT_NOTE = ("\n\nThis exact call was already refused for the same reason, so sending it again will not work. Ask "
               "the customer how they want to proceed, or transfer to a human agent (transfer_to_human_agents) if it "
               "cannot be resolved.")


def attestation_classes(passed: list[str], flagged: list[str] | tuple[str, ...] = (),
                        uncoded: list[str] | tuple[str, ...] = ()) -> list[str]:
    """What stood behind an allowed write, from the names of the gates that allowed it: 'rule_checked' (a policy rule
    on the write's values or state), 'customer_requested' (the request record covers it), or 'uncovered' (only the
    procedure gates and a no_rule_yet allow). `flagged` names the request gates that allowed the write without a
    covering request (bank_rules.UNRECORDED): those add 'request_unrecorded', never 'customer_requested'. `uncoded`
    names the gates that allowed a transfer with no doc-042 reason code (procedure.TRANSFER_UNCODED): those add
    'transfer_reason_uncoded', never 'rule_checked'."""
    classes = []
    if any(not g.startswith(("no_rule_", "procedure_", "banking_request_")) and g not in uncoded for g in passed):
        classes.append("rule_checked")
    if any(g.startswith("banking_request_") and g not in flagged for g in passed):
        classes.append("customer_requested")
    return ((classes or ["uncovered"]) + (["request_unrecorded"] if flagged else [])
            + (["transfer_reason_uncoded"] if uncoded else []))


class Enforcer:
    def __init__(self, domain: str, events: EventLog, context: Callable[[], dict] | None = None,
                 wording: str = "plain", request_gate: bool = True):
        """`request_gate` False is research row 6 (fix-research-2026-11-02), a Glint convention: no banking request
        gates, no request-record reminder among the open items, and record_requests taken off every attached
        toolkit. On by default."""
        if wording not in WORDINGS:
            raise ValueError(f"unknown deny wording {wording!r}; one of {WORDINGS}")
        self.domain, self.events, self.context, self.wording = domain, events, context or (dict), wording
        self.request_gate = request_gate
        self.registry = build_registry(domain, request_gate)
        assert_required(self.registry, REQUIRED_GATES[domain], f"gates for {domain}")
        self.events.emit("gates.manifest", manifest(domain, self.registry) | {"wording": wording,
                                                                                 "request_gate": request_gate})

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
        `toolkit=True` gets `toolkit` (the environment's toolkit) instead of `db`. The returned allow carries every
        passing gate's note, which get_response appends to the tool's reply."""
        if args is None:
            result = deny("unparsed_arguments", "glint.gates.enforce", "the hidden tool's arguments could not be "
                          "read, so no check could judge the write", "Pass `arguments` as a JSON object string.",
                          consequence="The tool was not called. Retrying with readable arguments is enough; the rest "
                          "of the request is not affected.", facts=["`arguments` is not a JSON object"])
            self._log(tool, args, result, 0.0, replay, "unparsed")
            return result
        final, passed, flagged, uncoded, notes = None, [], [], [], []
        for gate in self.registry.for_tool(tool):
            if gate.toolkit and toolkit is None:
                result, ms = deny("gate_error", gate.name, f"the check {gate.name!r} needs the environment's "
                                  "toolkit and was given none", "Call Enforcer.check with toolkit=env.tools.",
                                  consequence=CANNOT_DECIDE, facts=["toolkit: none"]), 0.0
            else:
                result, ms = run_gate(gate, toolkit if gate.toolkit else db, args, wait=replay)
            if not self._log(tool, args, result, ms, replay, gate.name):
                result = deny("audit_failed", gate.name, f"the check {gate.name!r} ran but its decision could not "
                              "be logged, and no write runs without a logged decision", "Check the gate log file.",
                              consequence=CANNOT_DECIDE, facts=["the gate.fired event was not written"])
            if not result.allowed:
                return result
            passed.append(gate.name)
            if result.rule_id == UNRECORDED:
                flagged.append(gate.name)
            if result.rule_id == TRANSFER_UNCODED:
                uncoded.append(gate.name)
            if result.note:
                notes.append(result.note)
            final = final or result
        if final is None:  # attach() refuses an environment with an ungated write; this is a second line
            raise RuntimeError(f"write {tool} has no gate")
        self._attest(tool, passed, replay, flagged, uncoded)
        return replace(final, note="\n\n".join(notes))

    def _attest(self, tool: str, passed: list[str], replay: bool, flagged: list[str] | None = None,
                uncoded: list[str] | None = None) -> None:
        """A write.attested event for each allowed write: which kind of check stood behind it (attestation_classes).
        Never decides anything; a failure to log is logged, not raised."""
        try:
            self.events.emit("write.attested", {"domain": self.domain, "tool": tool,
                                                "classes": attestation_classes(passed, flagged or (), uncoded or ()),
                                                "gates": passed,
                                                "replay": replay, **self.context()})
        except Exception:
            log.exception("the write.attested event was not logged")

    def _log(self, tool, args, result: GateResult, ms: float, replay: bool, gate: str) -> bool:
        try:
            self.events.emit("gate.fired", {
                "domain": self.domain, "gate": gate, "tool": tool, "args": args, "decision": result.decision,
                "rule_id": result.rule_id, "source": result.source, "reason": result.reason,
                # since PR 87 a deny's specifics (which value, which lookup) are in its facts, not its reason
                "facts": list(result.facts), "consequence": result.consequence,
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

    @staticmethod
    def _record_increase(env, tool: str, args: dict, reply: ToolMessage, before: set[str] | None) -> None:
        """Note the request ids a credit limit increase submit that ran created, added under its user (a second
        submit, on another card, does not replace the first); an approve or deny that ran clears the requests on its
        card, whichever user it names (τ²'s deny does not check that the card is that user's)."""
        if reply.error or str(reply.content).startswith("Error"):
            return
        record = env.tools.glint_increases
        if tool == SUBMIT and before is not None:
            user = str(args.get("user_id", ""))
            record[user] = record.get(user, []) + sorted(increase_request_ids(env.tools.db) - before)
        elif tool in (APPROVE, DENY):
            card, rows = args.get("credit_card_account_id"), env.tools.db.credit_limit_increase_requests.data
            env.tools.glint_verdicts.pop(str(card), None)
            env.tools.glint_increase_decided.add(str(args.get("user_id", "")))
            for user in list(record):
                record[user] = [r for r in record[user] if (rows.get(r) or {}).get("credit_card_account_id") != card]
                if not record[user]:
                    del record[user]

    @staticmethod
    def _record_verdict(env, call: ToolCall, reply: ToolMessage) -> None:
        """Keep the latest approve or deny verdict check_credit_limit_increase gave for a card, until an approve or
        deny runs on it (_record_increase). Any other verdict (reduce_amount, ask_amount, error) clears the card."""
        if call.requestor != "assistant" or call.name != VERDICT_TOOL or reply.error:
            return
        try:
            v = json.loads(reply.content)
        except (TypeError, json.JSONDecodeError):
            return
        card = str((call.arguments or {}).get("credit_card_account_id", ""))
        if isinstance(v, dict) and v.get("verdict") in ("approve", "deny"):
            env.tools.glint_verdicts[card] = {k: v.get(k) for k in ("verdict", "new_credit_limit", "denial_reason")}
        else:
            env.tools.glint_verdicts.pop(card, None)

    def _record_plan(self, env, call: ToolCall, reply: ToolMessage, replay: bool) -> None:
        """Keep the dispute plan the agent got from plan_disputes; a later plan for the same user replaces it."""
        if call.requestor != "assistant" or call.name != PLAN_TOOL or reply.error:
            return
        try:
            plan = json.loads(reply.content)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(plan, dict) or not plan.get("user_id") or not isinstance(plan.get("order"), list):
            return
        env.tools.glint_plans[plan["user_id"]] = plan
        try:
            self.events.emit("dispute_plan.recorded", {
                "domain": self.domain, "user_id": plan["user_id"], "replay": replay,
                "order": [[e["transaction_id"], e["eligible_for_provisional_credit"]] for e in plan["order"]],
                **self.context()})
        except Exception:
            log.exception("the dispute_plan.recorded event was not logged")

    def _record_requests(self, env, call: ToolCall, reply: ToolMessage, replay: bool) -> None:
        """Keep the customer's requests as the agent recorded them with record_requests (the latest replaces)."""
        if call.requestor != "assistant" or call.name != REQUESTS_TOOL or reply.error:
            return
        try:
            record = json.loads(reply.content)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(record, dict) or not as_flag(record.get("recorded")):
            return
        merged = list(env.tools.glint_requests or [])  # each call adds: agents record one request per call
        for r in record.get("requests") or []:
            if isinstance(r, dict) and all((r.get("action"), r.get("target_id")) != (m.get("action"),
                                                                                   m.get("target_id")) for m in merged):
                merged.append(r)
        env.tools.glint_requests = merged
        try:
            self.events.emit("requests.recorded", {
                "domain": self.domain, "user_id": record.get("user_id"), "replay": replay,
                "requests": [[r.get("action"), r.get("target_id")] for r in env.tools.glint_requests],
                **self.context()})
        except Exception:
            log.exception("the requests.recorded event was not logged")

    def _record_retention(self, env, call: ToolCall, reply: ToolMessage, replay: bool) -> None:
        """Keep the customer's recorded answer to the retention offer, per credit card account (the latest wins)."""
        if call.requestor != "assistant" or call.name != RETENTION_TOOL or reply.error:
            return
        try:
            answer = json.loads(reply.content)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(answer, dict) or not as_flag(answer.get("recorded")):
            return
        answer["accepted"] = as_flag(answer.get("accepted"))  # τ² shows bools as "True"/"False"
        env.tools.glint_retention[answer["credit_card_account_id"]] = answer
        try:
            self.events.emit("retention_answer.recorded", {
                "domain": self.domain, "credit_card_account_id": answer["credit_card_account_id"],
                "accepted": answer["accepted"], "customer_words": answer["customer_words"][:200], "replay": replay,
                **self.context()})
        except Exception:
            log.exception("the retention_answer.recorded event was not logged")

    def _record_address(self, env, call: ToolCall, reply: ToolMessage, replay: bool) -> None:
        """Keep the shipping address the customer confirmed for a card's replacement (record_address_confirmation; the
        latest wins), which satisfies the address_ask gate (credit_card_replacements_001)."""
        if call.requestor != "assistant" or call.name != ADDRESS_TOOL or reply.error:
            return
        try:
            record = json.loads(reply.content)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(record, dict) or not as_flag(record.get("recorded")):
            return
        env.tools.glint_address_confirmed[record["credit_card_account_id"]] = record["shipping_address"]
        try:
            self.events.emit("address.recorded", {
                "domain": self.domain, "credit_card_account_id": record["credit_card_account_id"],
                "address_on_file": record.get("address_on_file"), "replay": replay, **self.context()})
        except Exception:
            log.exception("the address.recorded event was not logged")
    def _record_dispute_answers(self, env, call: ToolCall, reply: ToolMessage, replay: bool) -> None:
        """Keep the customer's recorded answers to the two debit dispute questions, per card (the latest wins)."""
        if call.requestor != "assistant" or call.name != DISPUTE_ANSWERS_TOOL or reply.error:
            return
        try:
            answer = json.loads(reply.content)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(answer, dict) or not as_flag(answer.get("recorded")):
            return
        answer["card_in_possession"] = as_flag(answer.get("card_in_possession"))  # τ² shows bools as "True"/"False"
        env.tools.glint_dispute_answers[answer["card_id"]] = answer
        try:
            self.events.emit("dispute_answers.recorded", {
                "domain": self.domain, "card_id": answer["card_id"], "card_in_possession": answer["card_in_possession"],
                "pin_compromised": answer["pin_compromised"], "customer_words": answer["customer_words"][:200],
                "replay": replay, **self.context()})
        except Exception:
            log.exception("the dispute_answers.recorded event was not logged")

    def _record_check(self, call: ToolCall, reply: ToolMessage, replay: bool) -> None:
        """A check.recorded event for each decision-tool answer the agent got (check_card_dispute,
        check_verification, check_credit_limit_increase): what it said, so a diagnosis counts "the check said ready
        and nothing was filed" from the gate log instead of by hand (2026-10-30 review: v9's 7 such runs were counted
        by reading messages). Never decides anything; a failure to log is logged, not raised."""
        if call.requestor != "assistant" or call.name not in CHECK_TOOLS:
            return
        try:
            v = json.loads(str(reply.content).split("\n\n[open]")[0]) if not reply.error else {}
        except json.JSONDecodeError:
            v = {}
        v = v if isinstance(v, dict) else {}
        a = call.arguments or {}
        filing = v.get("filing") if isinstance(v.get("filing"), dict) else None
        try:
            self.events.emit("check.recorded", {
                "domain": self.domain, "tool": call.name, "replay": replay, "error": bool(reply.error),
                **{k: a[k] for k in ("user_id", "credit_card_account_id", "transaction_id") if a.get(k)},
                "ready": v.get("ready"), "verdict": v.get("verdict"), "verified": v.get("verified"),
                "filing": None if filing is None else ("arguments" if "arguments_json" in filing else "next_step"),
                **self.context()})
        except Exception:
            log.exception("the check.recorded event was not logged")

    def attach(self, env):
        """Gate this environment instance's writes. Raises when a write tool of its toolkit has no gate."""
        if getattr(env, "glint_enforcer", None) is self:
            return env
        if getattr(env, "glint_enforcer", None) is not None:
            raise RuntimeError("this environment is already gated by another enforcer")
        assert_required(self.registry, write_tools(env.tools), f"gates for {self.domain} on {type(env.tools).__name__}")
        original_response, original_set_state = env.get_response, env.set_state
        state = {"replay": False}

        if self.domain == "banking_knowledge":
            add_plan_tool(env)
            if not self.request_gate:
                drop_request_tool(env)
        env.tools.glint_reads, env.tools.glint_plans, env.tools.glint_increases = set(), {}, {}
        env.tools.glint_denied = set()  # (tool, arguments, rule) of every refusal, for the repeat escalation
        env.tools.glint_requests = None  # the latest record_requests list (None: not recorded yet)
        env.tools.glint_retention = {}  # credit card account -> the recorded answer to the retention offer
        env.tools.glint_dispute_answers = {}  # debit card -> the recorded answers to the two dispute questions
        env.tools.glint_verdicts = {}  # card -> the increase verdict check_credit_limit_increase gave, not filed yet
        env.tools.glint_increase_decided = set()  # users with an increase approved or denied in this conversation
        env.tools.glint_expedite_asked = set()  # cards whose replacement the expedite_ask gate refused once
        env.tools.glint_address_asked = set()  # cards whose replacement the address_ask gate refused once
        env.tools.glint_address_confirmed = {}  # card -> the shipping address recorded with record_address_confirmation
        env.tools.glint_ran = set()  # (tool, arguments) of every write that ran, for the duplicate-write deny (N2)

        def with_open_items(message: ToolCall, reply: ToolMessage) -> ToolMessage:
            if (self.domain != "banking_knowledge" or message.requestor != "assistant" or message.name in GLINT_RECORDS
                    or reply.error or str(reply.content).startswith("Error")):
                return reply
            items = _open_items(env.tools, self.request_gate)
            if message.name == WRAPPER:  # an account read carries the one-credit-per-account fact (row 14, doc 017)
                args = message.arguments or {}
                hidden = parse_hidden_arguments(args.get("arguments", "{}")) or {}
                note = aids.credit_note(env.tools.db, str(args.get("agent_tool_name", "")), hidden)
                items = [note, *items] if note else items
            return reply.model_copy(update={"content": f"{reply.content}\n\n" + "\n".join(items)}) if items else reply

        def get_response(message: ToolCall) -> ToolMessage:
            write = self.write_call(env, message)
            if write is None:
                reply = original_response(message)
                self._record_read(env, message, reply)
                self._record_plan(env, message, reply, state["replay"])
                self._record_retention(env, message, reply, state["replay"])
                self._record_dispute_answers(env, message, reply, state["replay"])
                self._record_requests(env, message, reply, state["replay"])
                self._record_address(env, message, reply, state["replay"])
                self._record_verdict(env, message, reply)
                self._record_check(message, reply, state["replay"])
                return with_open_items(message, reply)
            result = self.check(*write, env.tools.db, replay=state["replay"], toolkit=env.tools)
            if result.allowed:
                before = increase_request_ids(env.tools.db) if write[0] == SUBMIT else None
                reply = original_response(message)
                if result.note:  # an allow's reminder (procedure.TRANSFER_UNCODED): deterministic, so it replays
                    reply = reply.model_copy(update={"content": f"{reply.content}\n\n{result.note}"})
                self._record_increase(env, *write, reply, before)
                if not reply.error and not str(reply.content).startswith("Error"):
                    env.tools.glint_ran.add(aids.write_key(*write))
                return with_open_items(message, reply)
            asked = {EXPEDITE_ASK: env.tools.glint_expedite_asked, ADDRESS_ASK: env.tools.glint_address_asked}
            if result.rule_id in asked:  # asked once per card: the repeat order is the customer's choice
                asked[result.rule_id].add(str((write[1] or {}).get("credit_card_account_id", "")))
            key = (write[0], json.dumps(write[1], sort_keys=True, default=str), result.rule_id)
            repeat = key in env.tools.glint_denied  # the same call refused for the same rule before: escalate
            env.tools.glint_denied.add(key)
            content = result.message(self.wording) + (REPEAT_NOTE if repeat else "")
            return ToolMessage(id=message.id, content=content, requestor=message.requestor, role="tool", error=True)

        def set_state(*args, **kwargs):  # τ²'s evaluator replays the trajectory here: those events say replay=true
            state["replay"] = True
            try:
                return original_set_state(*args, **kwargs)
            finally:
                state["replay"] = False

        env.get_response, env.set_state, env.glint_enforcer = get_response, set_state, self
        return env
