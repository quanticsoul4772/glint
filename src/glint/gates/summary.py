"""What the write gates did in one arm: one GATES line at the arm's end, an ALERT while it runs, and the same summary
for a finished set's data folder (scripts/gate_summary.py).

The decisions are in the arm's gate log (`gates-<domain>.sqlite`, glint.gates.eventlog); what each run saw is in τ²'s
saved simulation. A denied write comes back to the agent as an error tool message that starts with core.DENY_PREFIX,
and τ² counts every error tool message toward its error limit (tau2.orchestrator.Orchestrator, max_errors, default
10): a run that reaches it ends with termination_reason "too_many_errors". So a gate that keeps denying can end a run
early; the summary says how many error-limit endings had gate denies among their errors.

Only live decisions count: τ²'s evaluator replays every run's writes through the same gates (payload replay=true).
The log is one file per data folder, so a resumed or released-and-rerun arm counts the decisions of every attempt.
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

from glint.gates.core import DENY_PREFIX

ERROR_LIMIT = "too_many_errors"  # τ²'s TerminationReason.TOO_MANY_ERRORS
UNPARSED = "unparsed"  # the gate name Enforcer.check logs for a hidden write whose arguments do not parse


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


def summarize(events: list[dict], sims: list[dict]) -> dict:
    """The arm's gate summary from its gate log's events and its saved simulations."""
    writes, denies = decisions(events)
    s = {"writes": writes, "denies": sum(denies.values()), "denies_by_rule": dict(sorted(denies.items())),
         "runs": len(sims), "runs_with_deny": 0, "error_limit_runs": 0, "error_limit_runs_with_deny": 0,
         "error_limit_errors": 0, "error_limit_deny_errors": 0}
    for sim in sims:
        errors, denied = run_errors(sim)
        s["runs_with_deny"] += denied > 0
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
