"""Request-gate recount: every banking.request.* decision in a gate log, as recorded and as the current rules decide.

The gold replays (gate_replay, bank_replay) insert a request record built from the conversation's own writes, so they
cannot see what the request gate refuses. This script reads a gate log (`gates-<domain>.sqlite`, opened read-only) and,
for each live decision of a request gate (gate `banking_request_<tool>`), runs the current `requested(tool)` gate on
the same write with the request record the agent had kept at that moment (the latest `requests.recorded` event of the
same simulation) and the task's initial database (τ²'s environment for the task, set to its initial state). It prints
the recorded decisions per rule, the current rules' decisions per rule, and each change.

Static: the same calls in the same order. It does not see writes another gate refused before the request gate ran,
or what the agent would have done differently after a decision that changes. Database lookups use the task's initial
state; the request gate reads only which checking account a debit card belongs to, which no write changes. Counts
from τ²'s banking test split are tuned (the split's failures shaped these rules). Free: no model, no network.

Usage: python scripts/request_recount.py GATE_LOG [--json OUT]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from glint.gates import banking as bank_rules
from glint.gates.summary import read_log

GATE_PREFIX = "banking_request_"
DOMAIN = "banking_knowledge"


def _sim(p: dict) -> str:
    return str(p.get("simulation_id") or (p.get("task_id"), p.get("trial")))


def request_decisions(events: list[dict]) -> list[dict]:
    """The live request-gate decisions in log order, each with the request record its simulation had at that moment
    (None: nothing recorded yet)."""
    records: dict[str, list[dict]] = {}
    out = []
    for e in events:
        p = e["payload"]
        if p.get("replay"):
            continue
        if e["type"] == "requests.recorded":
            records[_sim(p)] = [{"action": a, "target_id": t or ""} for a, t in p.get("requests") or []]
        elif e["type"] == "gate.fired" and str(p.get("gate", "")).startswith(GATE_PREFIX):
            out.append({"task_id": p.get("task_id"), "trial": p.get("trial"), "tool": p["tool"],
                        "args": p.get("args") or {}, "old": p["rule_id"], "old_decision": p["decision"],
                        "record": records.get(_sim(p))})
    return out


def task_db(task_id: str, _cache: dict = {}):  # noqa: B006 - one database per task for the whole recount
    """τ²'s banking database for `task_id`, set to the task's initial state (the environment τ²'s runner builds)."""
    if task_id not in _cache:
        from tau2.registry import registry
        from tau2.runner.build import _derive_read_log_allowlist
        task = next(t for t in registry.get_tasks_loader(DOMAIN)(None) if t.id == task_id)
        env = registry.get_env_constructor(DOMAIN)(retrieval_variant="bm25_grep", task=task,
                                                   read_log_allowlist=_derive_read_log_allowlist(task))
        init = task.initial_state
        env.set_state(init.initialization_data if init else None, init.initialization_actions if init else None,
                      (init.message_history if init else None) or [])
        _cache[task_id] = env.tools.db
    return _cache[task_id]


def recount(decisions: list[dict], db_of=task_db) -> list[dict]:
    """Each decision with `new`: the rule_id the current request gate gives the same write and record."""
    out = []
    for d in decisions:
        toolkit = SimpleNamespace(db=db_of(d["task_id"]), glint_requests=d["record"])
        result = bank_rules.requested(d["tool"])(toolkit, d["args"])
        out.append(d | {"new": result.rule_id, "new_decision": result.decision})
    return out


def tally(rows: list[dict]) -> dict:
    """Counts per rule, recorded and current, and the changes (old rule -> new rule, with the tool)."""
    return {"decisions": len(rows),
            "old": dict(sorted(Counter(r["old"] for r in rows).items())),
            "new": dict(sorted(Counter(r["new"] for r in rows).items())),
            "old_denies": sum(r["old_decision"] != "allow" for r in rows),
            "new_denies": sum(r["new_decision"] != "allow" for r in rows),
            "changes": dict(sorted(Counter(f"{r['old']} -> {r['new']} ({r['tool']})" for r in rows
                                           if r["old"] != r["new"]).items()))}


def lines(t: dict) -> list[str]:
    out = [(f"REQUEST RECOUNT (static; tuned when the log is from the banking test split): {t['decisions']} live "
           f"request-gate decisions; refusals recorded {t['old_denies']}, under the current rules {t['new_denies']}")]
    out += [f"  recorded  {rule}: {n}" for rule, n in t["old"].items()]
    out += [f"  current   {rule}: {n}" for rule, n in t["new"].items()]
    out += [f"  change    {change}: {n}" for change, n in t["changes"].items()]
    return out


def main(argv: list[str], db_of=task_db) -> int:
    if not argv or len(argv) not in (1, 3) or (len(argv) == 3 and argv[1] != "--json"):
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    t = tally(recount(request_decisions(read_log(argv[0])), db_of))
    for line in lines(t):
        print(line)
    if len(argv) == 3:
        Path(argv[2]).write_text(json.dumps(t, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
