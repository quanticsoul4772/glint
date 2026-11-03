"""What the write gates did in one arm: one GATES line at the arm's end, an ALERT while it runs, and the same summary
for a finished set's data folder (scripts/gate_summary.py).

The decisions are in the arm's gate log (`gates-<domain>.sqlite`, glint.gates.eventlog); what each run saw is in τ²'s
saved simulation. A denied write comes back to the agent as an error tool message that starts with core.DENY_PREFIX,
and τ² counts every error tool message toward its error limit (tau2.orchestrator.Orchestrator, max_errors, default
10): a run that reaches it ends with termination_reason "too_many_errors". So a gate that keeps denying can end a run
early; the summary says how many error-limit endings had gate denies among their errors.

Only live decisions count: τ²'s evaluator replays every run's writes through the same gates (payload replay=true).
The log is one file per data folder, so a resumed or released-and-rerun arm counts the decisions of every attempt.

Deny outcomes (`rule_outcomes`): per rule, how many denies the agent got, how many it followed (`fix_followed`: the
next relevant call matches the deny's fix), and how the runs with that deny ended. The failed runs' differences from
τ²'s expected actions come in as `diffs`, computed by the caller with glint.inspect: this package never reads gold
(the expected actions), so a gate cannot grade itself against the answer (tests/test_inspect_imports.py checks).
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

from glint.gates.core import DENY_PREFIX
from glint.gates.rules import parse_hidden_arguments
from glint.runlog import FAILED_RUNS, outcome

ERROR_LIMIT = "too_many_errors"  # τ²'s TerminationReason.TOO_MANY_ERRORS
UNPARSED = "unparsed"  # the gate name Enforcer.check logs for a hidden write whose arguments do not parse
WRAPPER = "call_discoverable_agent_tool"  # τ²'s call of a hidden banking tool: `arguments` is a JSON string
NAMING_ARGS = ("agent_tool_name", "discoverable_tool_name")  # unlock/give calls name a hidden tool, no arguments


def ended_on_error_limit(sim: dict) -> bool:
    """The run ended on τ²'s error limit (enum text in memory, value on disk)."""
    return ERROR_LIMIT in str(sim.get("termination_reason")).lower()


def run_errors(sim: dict) -> tuple[int, int]:
    """(error tool results, of which gate denies) in one saved simulation: the errors τ² counts toward its limit."""
    errors = denies = 0
    for m in sim.get("messages") or []:
        if m.get("role") == "tool" and m.get("error"):
            errors += 1
            denies += str(m.get("content") or "").startswith(DENY_PREFIX)
    return errors, denies


def read_log(path: str | Path) -> list[dict]:
    """A gate log's events, opened read-only (a finished set's folder is never written)."""
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT type, payload FROM events ORDER BY rowid").fetchall()
    finally:
        conn.close()
    return [{"type": t, "payload": json.loads(p)} for t, p in rows]


def decisions(events: list[dict]) -> tuple[int, Counter]:
    """(writes checked, denies by rule_id) from a gate log's live decisions. A write runs its tool's gates in manifest
    order until one denies, so each write logs exactly one event from its tool's first gate."""
    first: dict[str, str] = {}
    writes, denies = 0, Counter()
    for e in events:
        p = e["payload"]
        if e["type"] == "gates.manifest":  # one per Enforcer; the latest one describes the events after it
            first = {}
            for gate in p["gates"]:
                for tool in gate["tools"]:
                    first.setdefault(tool, gate["name"])
            continue
        if e["type"] != "gate.fired" or p.get("replay"):
            continue
        if not first:
            raise ValueError("a gate.fired event comes before any gates.manifest: this is not a log Enforcer wrote")
        writes += p["gate"] == UNPARSED or first.get(p["tool"]) == p["gate"]
        if p["decision"] != "allow":
            denies[p["rule_id"]] += 1
    return writes, denies


OTHER_CHECKS = "other"  # a deny from no registered gate (unparsed arguments, a stage, a gate since renamed)


def denies_by_checks(events: list[dict], checks: dict[str, str]) -> dict[str, Counter]:
    """A gate log's live denies by what the denying gate rests on (`checks`: gate name -> core.CHECKS kind, from
    today's registry, since older logs' manifests do not carry it), each kind's denies counted by (gate, rule_id). A
    deny whose gate `checks` does not name counts under OTHER_CHECKS."""
    out: dict[str, Counter] = {}
    for e in events:
        p = e["payload"]
        if e["type"] != "gate.fired" or p.get("replay") or p["decision"] == "allow":
            continue
        out.setdefault(checks.get(p["gate"], OTHER_CHECKS), Counter())[(p["gate"], p["rule_id"])] += 1
    return out


def denied_runs(events: list[dict]) -> set[tuple[str, str]]:
    """(task_id, trial) of every run with a live deny in the gate log. A run that overflowed the context keeps no
    messages in τ²'s save, so its denies are only here (banking-plan-v5-gated: 17 runs counted from the messages,
    19 in the log, 2026-10-29)."""
    return {(str(e["payload"].get("task_id")), str(e["payload"].get("trial"))) for e in events
            if e["type"] == "gate.fired" and not e["payload"].get("replay") and e["payload"]["decision"] != "allow"}


HELD_THEN_ALLOWED = "held_then_allowed"


def _run_of(payload: dict) -> tuple[str, str, str]:
    """The run an event belongs to: τ²'s simulation id, task and trial (the Enforcer's context)."""
    return (str(payload.get("simulation_id")), str(payload.get("task_id")), str(payload.get("trial")))


def held_then_allowed(events: list[dict]) -> list[dict]:
    """The "held, then allowed" attestation class (fix-research 2026-11-02 N1; design 2026-11-02 §3; Glint
    convention): each live write the gates allowed after an earlier live refusal of the same tool on the same record
    (every `*_id` argument both calls carry is equal, same_target) in the same run. Computed from the gate log alone:
    a write.attested event carries no arguments, so the allowed write's arguments are those of the latest live
    gate.fired event for that tool in that run (a write's gates run one after another and log before its attestation).
    Returns one row per such write: task_id, trial, simulation_id, tool, args and the earlier refusals' rule_ids."""
    last_args: dict[tuple, dict] = {}
    refused: dict[tuple, list[tuple[dict | None, str]]] = {}
    out = []
    for e in events:
        p = e["payload"]
        if e["type"] not in ("gate.fired", "write.attested") or p.get("replay"):
            continue
        run, tool = _run_of(p), p.get("tool")
        if e["type"] == "gate.fired":
            last_args[run + (tool,)] = p.get("args")
            if p["decision"] != "allow":
                refused.setdefault(run + (tool,), []).append((p.get("args"), p.get("rule_id")))
            continue
        args = last_args.get(run + (tool,))
        rules = [rule for denied, rule in refused.get(run + (tool,), [])
                 if denied is not None and args is not None and same_target(denied, args)]
        if rules:
            out.append({"simulation_id": run[0], "task_id": run[1], "trial": run[2], "tool": tool, "args": args,
                        "classes": list(p.get("classes") or []) + [HELD_THEN_ALLOWED], "refused_by": rules})
    return out


def summarize(events: list[dict], sims: list[dict]) -> dict:
    """The arm's gate summary from its gate log's events and its saved simulations."""
    writes, denies = decisions(events)
    logged = denied_runs(events)
    attested = Counter("+".join(e["payload"]["classes"]) for e in events
                       if e["type"] == "write.attested" and not e["payload"].get("replay"))
    s = {"writes": writes, "denies": sum(denies.values()), "denies_by_rule": dict(sorted(denies.items())),
         "attested": dict(sorted(attested.items())), HELD_THEN_ALLOWED: len(held_then_allowed(events)),
         "runs": len(sims), "runs_with_deny": 0, "error_limit_runs": 0, "error_limit_runs_with_deny": 0,
         "error_limit_errors": 0, "error_limit_deny_errors": 0}
    for sim in sims:
        errors, denied = run_errors(sim)
        s["runs_with_deny"] += denied > 0 or (str(sim.get("task_id")), str(sim.get("trial"))) in logged
        if ended_on_error_limit(sim):
            s["error_limit_runs"] += 1
            s["error_limit_runs_with_deny"] += denied > 0
            s["error_limit_errors"] += errors
            s["error_limit_deny_errors"] += denied
    return s


def line(label: str, s: dict) -> str:
    out = (f"GATES: {label}: {s['writes']} writes checked, {s['denies']} denied {s['denies_by_rule'] or ''}".rstrip()
           + f"; {s['runs_with_deny']} of {s['runs']} runs had at least one deny; {s['error_limit_runs']} runs ended "
           f"on τ²'s error limit ({ERROR_LIMIT})")
    if s["error_limit_runs"]:
        out += (f", {s['error_limit_runs_with_deny']} of them with gate denies; {s['error_limit_deny_errors']} of "
                f"their {s['error_limit_errors']} errors were gate denies")
    if s.get("attested"):
        out += "; allowed writes by what checked them: " + ", ".join(f"{k} {n}" for k, n in s["attested"].items())
    if s.get(HELD_THEN_ALLOWED):
        out += f"; held, then allowed {s[HELD_THEN_ALLOWED]}"
    return out


class ErrorLimitAlerts:
    """Called with each run as it finishes (glint.runlog.RunEvents `on_run`): an ALERT line when the runs that ended on
    τ²'s error limit with gate denies among their errors reach `first`, and again at each doubling."""

    def __init__(self, first: int = 2):
        self.count, self._next = 0, first
        self.tasks: Counter = Counter()

    def __call__(self, sim: dict) -> str | None:
        if not ended_on_error_limit(sim):
            return None
        errors, denied = run_errors(sim)
        if not denied:
            return None
        self.count += 1
        self.tasks[str(sim.get("task_id"))] += 1
        if self.count < self._next:
            return None
        self._next = 2 * self.count
        return (f"ALERT: {self.count} runs ended on τ²'s error limit ({ERROR_LIMIT}) with gate denies among their "
                f"errors, tasks {dict(sorted(self.tasks.items()))}; latest {sim.get('task_id')} trial "
                f"{sim.get('trial')}: {denied} of {errors} errors were gate denies")


def gated_tools(events: list[dict]) -> set[str]:
    """Every tool any gates.manifest in the log gates: all of the domain's writes, and the gated generic tools."""
    tools = set()
    for e in events:
        if e["type"] == "gates.manifest":
            tools.update(e["payload"].get("gated_generic") or [])
            for gate in e["payload"]["gates"]:
                tools.update(gate["tools"])
    return tools


def parse_deny(content: str) -> dict:
    """rule_id, reason, fix and verdict from the deny message the agent saw (glint.gates.core.GateResult.message).
    Reads both formats: the plain-language one ("Rule: X (source)", "What the check found:", "What to do next:") and
    the one runs recorded before 2026-10-29 carry ("(rule X, check ...)", "Reason:", "Fix:")."""
    rule = re.search(r"^Rule: (\S+)", content, re.MULTILINE) or re.search(r"\(rule (\S+), check", content)
    lines = dict(line.split(": ", 1) for line in content.splitlines()[1:] if ": " in line)
    return {"rule_id": rule.group(1) if rule else None,
            "reason": lines.get("What the check found", lines.get("Reason", "")),
            "fix": lines.get("What to do next", lines.get("Fix", "")),
            "verdict": json.loads(lines["Verdict"]) if "Verdict" in lines else {}}


def run_calls(sim: dict) -> list[dict]:
    """The agent's tool calls in order, each with its reply: `tool` and `args` with a hidden banking tool unwrapped
    (`args` None when its arguments do not parse), `named` (the hidden tool an unlock or give call names), `error`
    (the environment or a gate refused it) and `deny` (the gate's rule_id, reason, fix and verdict). A deny reply
    with no call before it is kept, with tool None."""
    out, pending = [], []
    for i, m in enumerate(sim.get("messages") or []):
        if m.get("role") in ("assistant", "user"):
            pending = [(m["role"], tc) for tc in m.get("tool_calls") or []]
            continue
        if m.get("role") != "tool":
            continue
        role, tc = pending.pop(0) if pending else ("assistant", None)
        if role != "assistant":
            continue
        content = str(m.get("content") or "")
        name, args = (tc["name"], tc.get("arguments") or {}) if tc else (None, None)
        named = next((args[k] for k in NAMING_ARGS if args and args.get(k)), None)
        if name == WRAPPER:
            name, args = str(args.get("agent_tool_name", "")), parse_hidden_arguments(args.get("arguments", "{}"))
        out.append({"index": i, "tool": name, "args": args, "named": named, "content": content,
                    "error": bool(m.get("error")) or content.startswith("Error"),
                    "deny": parse_deny(content) if content.startswith(DENY_PREFIX) else None})
    return out


def norm(value):
    """A value as τ² compares it: a number as a float, a "True"/"False" string as the bool (τ² shows every bool to the
    model as text, so a model can send it back as a string)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return value


def same_target(a: dict | None, b: dict | None) -> bool:
    """Two calls act on the same record: every `*_id` argument both carry is equal."""
    a, b = a or {}, b or {}
    return all(norm(a[k]) == norm(b[k]) for k in a.keys() & b.keys() if k.endswith("_id"))


def _number(text: str):
    return float(text) if re.fullmatch(r"-?\d+(?:\.\d+)?", text) else text


def expected_values(deny: dict, args: dict | None) -> dict:
    """The values a deny's fix asks for: `with field value[, field value]` pairs in the fix ("Approve it with
    new_credit_limit 5000.", "Apply amount 75."), and the verdict's values for the denied call's own arguments that
    the fix names ("File the dispute with the card_action and eligible_for_provisional_credit values in the
    verdict.")."""
    out = {}
    clause = re.search(r"\b(?:with|Apply) (.+?)\.?$", deny["fix"])
    for part in re.split(r", | and ", clause.group(1)) if clause else []:
        pair = re.fullmatch(r"([a-z]+(?:_[a-z0-9]+)*) '?([^' ]+?)'?", part.strip())
        # a name ending in four digits is a hidden tool ("Unfreeze it with unfreeze_debit_card_3893 first"), not a field
        if pair and ("_" in pair.group(1) or pair.group(1) in ("amount", "reason")) \
                and not re.search(r"_\d{4}$", pair.group(1)):
            out[pair.group(1)] = _number(pair.group(2))
    for key, value in (deny.get("verdict") or {}).items():
        if (key in (args or {}) and re.search(rf"\b{re.escape(key)}\b", deny["fix"]) and value is not None
                and not isinstance(value, (dict, list))):
            out.setdefault(key, value)
    return out


def fix_followed(call: dict, later: list[dict], writes: set[str]) -> tuple[bool, str]:
    """Did the agent follow the deny's fix? Judged on the calls after the deny, by the kind of fix:
    - "Do not ...": the denied tool is not called again on the same record;
    - values to use (expected_values): the next call on the same record of the denied tool, or of a write that carries
      one of those fields, carries them all;
    - a tool to call first or instead ("Unlock and call X ..., then retry", "Unfreeze it with X first", "Activate it
      with X"): X is called (or unlocked or given) before the denied tool is retried on the same record;
    - anything else: the denied tool is retried on the same record and the retry is not refused.
    Returns (followed, how)."""
    deny, tool, args = call["deny"], call["tool"], call["args"]
    retry = next((c for c in later if c["tool"] == tool and same_target(args, c["args"])), None)
    if deny["fix"].startswith("Do not"):
        return retry is None, "not retried" if retry is None else f"retried {tool}"
    expected = expected_values(deny, args)
    if expected:
        nxt = next((c for c in later if same_target(args, c["args"]) and (c["tool"] == tool or (
            c["tool"] in writes and expected.keys() & (c["args"] or {}).keys()))), None)
        if nxt is None:
            return False, "no later call on the same record"
        wrong = sorted(k for k, v in expected.items() if norm((nxt["args"] or {}).get(k)) != norm(v))
        return not wrong, f"{nxt['tool']} " + (f"still differs on {wrong}" if wrong else "with the fix values")
    named = [t for t in re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", deny["fix"])
             if t != tool and (t in writes or re.search(r"_\d{4}$", t))]
    if named and re.search(r"\bfirst\b|then retry|^Activate it with|^Give the user", deny["fix"]):
        before = later[:later.index(retry)] if retry is not None else later
        hit = next((c for c in before if c["tool"] in named or c["named"] in named), None)
        return hit is not None, f"called {hit['tool']}" if hit else f"did not call {named}"
    if retry is None:
        return False, "not retried"
    return not retry["error"], f"retry of {tool} {'refused' if retry['error'] else 'executed'}"


def deny_followups(sim: dict, writes: set[str]) -> list[dict]:
    """Every gate deny in one run, in order: its rule_id, reason and fix, the denied call, the next write the agent
    made, and whether it followed the fix (fix_followed)."""
    calls = run_calls(sim)
    out = []
    for k, call in enumerate(calls):
        if call["deny"] is None:
            continue
        later = calls[k + 1:]
        nxt = next((c for c in later if c["tool"] in writes), None)
        followed, how = fix_followed(call, later, writes)
        out.append({"rule_id": call["deny"]["rule_id"], "reason": call["deny"]["reason"], "fix": call["deny"]["fix"],
                    "tool": call["tool"], "args": call["args"], "index": call["index"],
                    "next_write": None if nxt is None else {"tool": nxt["tool"], "args": nxt["args"],
                                                           "refused": nxt["error"]},
                    "fix_followed": followed, "how": how})
    return out


def rule_outcomes(events: list[dict], sims: list[dict], diffs: dict | None = None) -> dict:
    """Per rule: denies, fix_followed, and the runs with that deny by how they ended (passed, failed, not a result).
    `diffs` maps (task_id, trial) to a failed run's compact diff (glint.inspect.compact), from which the failed runs'
    differing fields are counted."""
    writes, table = gated_tools(events), {}
    for sim in sims:
        followups = deny_followups(sim, writes)
        kind = outcome(sim)
        ended = "passed" if kind == "pass" else "failed" if kind in FAILED_RUNS else "not_result"
        diff = (diffs or {}).get((str(sim.get("task_id")), int(sim.get("trial") or 0)))
        for rule in sorted({f["rule_id"] for f in followups}, key=str):
            row = table.setdefault(rule, {"denies": 0, "fix_followed": 0, "passed": 0, "failed": 0, "not_result": 0,
                                          "failed_fields": Counter(), "failed_missing": 0, "failed_extra": 0,
                                          "failed_not_diffed": 0})
            mine = [f for f in followups if f["rule_id"] == rule]
            row["denies"] += len(mine)
            row["fix_followed"] += sum(f["fix_followed"] for f in mine)
            row[ended] += 1
            row["failed_not_diffed"] += ended == "failed" and diffs is not None and diff is None
            if ended == "failed" and diff:
                row["failed_fields"].update(diff["fields"])
                row["failed_missing"] += diff["missing"]
                row["failed_extra"] += diff["extra"]
    return dict(sorted(table.items(), key=lambda kv: str(kv[0])))


def outcome_lines(label: str, table: dict, top: int = 3) -> list[str]:
    """One GATE OUTCOMES line per rule (rule_outcomes)."""
    if not table:
        return [f"GATE OUTCOMES: {label}: no denies in the saved runs"]
    lines = []
    for rule, r in table.items():
        line = (f"GATE OUTCOMES: {label}: {rule}: {r['denies']} denies, fix followed {r['fix_followed']}; runs with "
                f"this deny: {r['passed']} passed, {r['failed']} failed")
        if r["not_result"]:
            line += f", {r['not_result']} not results"
        diffs = [f"{k} x{n}" for k, n in r["failed_fields"].most_common(top)]
        diffs += [f"missing {r['failed_missing']}"] * bool(r["failed_missing"])
        diffs += [f"extra {r['failed_extra']}"] * bool(r["failed_extra"])
        if r["failed"] and diffs:
            line += f" (failed runs differ on: {', '.join(diffs)})"
        if r["failed_not_diffed"]:
            line += f"; {r['failed_not_diffed']} failed runs not diffed (task not in τ²'s tasks)"
        lines.append(line)
    return lines
