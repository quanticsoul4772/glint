"""Bank Desk events: a τ² conversation as a timeline the page shows (shared by replay export and live mode)."""
from __future__ import annotations

import json
import re

from glint.decisions import GLINT_TOOLS

# glint.gates.core.DENY_PREFIX, kept as a copy: importing glint.gates loads τ², which the desk's own image (Modal, and
# the GitHub Pages build) does not have; tests/test_desk_gates.py checks the two are equal
DENY_PREFIX = "Error: this write was blocked by a policy gate"

# τ² banking's write tools (glint.gates.enforce.write_tools on the domain's toolkit, without its example tool), kept
# as a list because the desk's own image has no τ²; tests/test_desk_gates.py checks it against the toolkit
WRITES = {
    "activate_debit_card_8291", "activate_debit_card_8292", "activate_debit_card_8293",
    "apply_checking_account_credit_5829", "apply_credit_card_account_flag_6147", "apply_savings_account_credit_6831",
    "apply_statement_credit_8472", "approve_credit_limit_increase_5847", "change_debit_card_pin_6285",
    "change_user_email", "clear_debit_card_fraud_alert_4892", "close_bank_account_7392",
    "close_credit_card_account_7834", "close_debit_card_4721", "deny_credit_limit_increase_5848",
    "file_credit_card_transaction_dispute_4829", "file_debit_card_transaction_dispute_6281", "freeze_debit_card_3892",
    "log_credit_card_closure_reason_4521", "log_verification", "open_bank_account_4821", "order_debit_card_5739",
    "order_replacement_credit_card_7291", "pay_credit_card_from_checking_9182",
    "request_temporary_debit_card_limit_increase_8374", "reset_debit_card_pin_6284",
    "set_debit_card_recurring_block_7382", "submit_credit_limit_increase_request_7392",
    "submit_interest_discrepancy_report_7294", "transfer_funds_between_bank_accounts_7291",
    "unfreeze_debit_card_3893", "update_transaction_rewards_3847"}


def _short(value, n=400) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= n else text[:n] + " …"


_DENY_LINES = {"What the check found": "found", "Why it matters": "why", "What to do next": "next"}


def parse_deny(text: str) -> dict:
    """A gate's deny (glint.gates.core GateResult.message, plain wording) as card fields: the rule and its source,
    what the check found, the facts it used, why it matters and the next step. A deny in another layout keeps its
    whole text in `found`."""
    fields = {"rule": "", "source": "", "found": "", "facts": [], "why": "", "next": ""}
    in_facts = False
    for line in text.splitlines():
        if m := re.match(r"Rule: (\S+) \((.*)\)$", line):
            fields["rule"], fields["source"] = m.group(1), m.group(2)
            continue
        if line.startswith("Facts it used:"):
            in_facts = True
            continue
        head, _, rest = line.partition(": ")
        if head in _DENY_LINES:
            fields[_DENY_LINES[head]], in_facts = rest, False
        elif line.startswith("Verdict: "):
            in_facts = False
        elif in_facts and line.startswith("- "):
            fields["facts"].append(line[2:])
    if not fields["found"]:  # another layout: its text without the Verdict JSON (the gate's raw details)
        fields["found"] = "\n".join(line for line in text.removeprefix(DENY_PREFIX).strip().splitlines()
                                    if not line.startswith("Verdict:"))
    return fields


def timeline(messages: list[dict], overrules: dict[int, dict] | None = None) -> list[dict]:
    """Events in conversation order from τ² messages (dicts). `overrules`: assistant message index -> decision
    (tool, reasons) for writes a Glint tool would have overruled. A call a policy gate denied becomes a "gate" event
    in place of its action or lookup: the write did not run."""
    overrules = overrules or {}
    results = {m.get("id"): m.get("content") for m in messages if m.get("role") == "tool"}
    events = []
    for i, m in enumerate(messages):
        if m.get("role") == "user" and m.get("content"):
            events.append({"kind": "customer", "text": m["content"]})
        if m.get("role") != "assistant":
            continue
        if m.get("content"):
            events.append({"kind": "agent", "text": m["content"]})
        for tc in m.get("tool_calls") or []:
            name, args = tc["name"], tc.get("arguments") or {}
            result = results.get(tc.get("id"))
            if name == "unlock_discoverable_agent_tool":
                continue
            if name in GLINT_TOOLS:
                events.append({"kind": "verdict", "tool": name, "args": args,
                               "result": None if result is None else _short(result, 1500)})
                continue
            if name == "call_discoverable_agent_tool":
                inner = args.get("arguments")
                try:
                    name, args = args.get("agent_tool_name", ""), json.loads(inner) if isinstance(inner, str) else inner
                except json.JSONDecodeError:
                    name, args = args.get("agent_tool_name", ""), {"arguments": inner}
            if str(result or "").startswith(DENY_PREFIX):  # parsed whole: a deny runs 1-2 KB, past _short's cut
                events.append({"kind": "gate", "tool": name, "args": args, **parse_deny(str(result))})
                continue
            event = {"kind": "action" if name in WRITES else "lookup", "tool": name, "args": args,
                     "result": None if result is None else _short(result),
                     "error": str(result or "").startswith("Error")}
            if i in overrules and overrules[i]["tool"] == name:
                event["overruled"] = overrules[i]["reasons"]
            events.append(event)
    return events


def outcome(sim: dict) -> dict:
    r = sim.get("reward_info") or {}
    return {"passed": r.get("reward") == 1.0, "database_matches_gold": (r.get("db_check") or {}).get("db_match")}


def customer_summary(instructions: str) -> str:
    """The customer's situation and goal from the simulator brief, without markdown (the brief also holds persona
    details and hidden facts the agent must ask for, which the page should not show)."""
    import re

    parts = []
    for label in ("Your situation", "Your goal"):
        # the section runs to the next blank line or the next **Label:** heading
        m = re.search(rf"\*\*{label}:\*\*\s*(.+?)(?=\n\s*\n|\*\*[A-Z][^*]+:\*\*|$)", instructions, re.DOTALL)
        if m:
            parts.append(" ".join(m.group(1).replace("**", "").split()))
    text = " ".join(parts) or " ".join(instructions.replace("**", "").split())
    return _short(text, 500)
