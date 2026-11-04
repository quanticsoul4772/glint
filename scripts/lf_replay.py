"""Recorded Glimmer tool calls through a real LlamaFirewall 1.0.3, against the live gate decisions (judge round 2, B9).

Replays every recorded run of the Glint arm (`tools-banking_knowledge.json`) in the four gated banking sets the README
counts (v9, the unseen set from its lean copy, v10, the pressure set) on CPU, with no model and no network:

- a fresh τ² banking environment at the task's initial state, with the tools arm's decision tools
  (glint.assist.tau2_tools.assist_environment, as tools_ab installs them) and gated by a current-code Enforcer (stages
  off, so no model verifier can run), and LlamaFirewall built with only Glint's scanner: `make_scanner(enforcer, env)` and
  `LlamaFirewall({Role.ASSISTANT: ["glint_gates"]})`;
- every tool call of the run is sent in its recorded order (reads too, so the gates' records of lookups, verification
  and plans are rebuilt as they were live). Before each gated assistant call runs, LlamaFirewall scans an
  AssistantMessage carrying that one recorded call (OpenAI shape, the message's own text), so each call is judged on
  the database as it was just before it ran live;
- the database follows the RECORDED run, not the replayed decision: a call the live gate allowed runs (through the
  gated environment; when the current gates refuse it, through τ²'s own tool instead, with the refusal's records
  undone), and a call the live gate refused does not run.

The live decision is read from the recorded tool reply: a reply starting with the gate's refusal prefix is a deny
(a hold when its rule is one of the two ask rules, glint.integrations.llamafirewall_scanner.HOLD_RULES), anything
else an allow. Reported, per set and in all: the LlamaFirewall decision against the live one (BLOCK vs deny,
HUMAN_IN_THE_LOOP_REQUIRED vs hold, ALLOW vs allow), every disagreement with both rules, and, as a check of the
scanner path itself, the LlamaFirewall decision against the gated environment's own decision on the same call (the
wrapper, which logs and runs it). Live decisions came from the gate code each set ran with (`commit.txt`), so a
disagreement with the live gate can come from a rule changed since; the disagreement list names the rules.

It needs llamafirewall==1.0.3 in a SEPARATE virtual environment, never the run folder's .venv (docs/llamafirewall.md:
it pulls in torch, transformers, codeshield and semgrep, and semgrep pins mcp 1.29.0, so reinstall mcp==2.3.0
afterwards), with Glint run from the checkout's src. Not run in CI. From the repo root:
  PYTHONPATH=src LF_VENV/Scripts/python -X utf8 scripts/lf_replay.py [--data DIR] [--json OUT] [--no-request-gate]
DIR defaults to D:/Projects/meta/measure/data. `--no-request-gate` builds the Enforcer with request_gate=False: none of
the four sets had the request gate or its record_requests tool live, so its refusals are rule changes, not scanner
differences. Exit 1 when LlamaFirewall and the gated environment disagree on any call.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from loguru import logger
from tau2.data_model.message import ToolCall

from glint.assist.tau2_tools import assist_environment
from glint.gates import Enforcer, EventLog
from glint.gates.core import DENY_PREFIX
from glint.gates.rules import ADDRESS_ASK, EXPEDITE_ASK, GATED_GENERIC
from glint.gates.summary import parse_deny
from glint.integrations.llamafirewall_scanner import ALLOW, BLOCK, HOLD, HOLD_RULES, make_scanner
from glint.runlog import NOT_RESULTS, outcome

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_catch import environment, load_tasks

DOMAIN = "banking_knowledge"
DATA = Path("D:/Projects/meta/measure/data")
SETS = (("v9", "tools_ab_banking12_plan_v9_gated_modal"), ("unseen", "tools_ab_banking_unseen_gated_lean_modal"),
        ("v10", "tools_ab_banking_unseen_v10_gated_modal"), ("pressure", "tools_ab_banking_pressure_gated_modal"))
ARM_FILE = f"tools-{DOMAIN}.json"
DECISIONS = (BLOCK, HOLD, ALLOW)
LIVE_NAME = {BLOCK: "deny", HOLD: "hold", ALLOW: "allow"}
ALLOW_RULE = re.compile(r"allowed by Glint rule (\S+)")


def decision_of(content: str | None) -> tuple[str, str | None]:
    """(decision, rule) of a tool reply: a gate refusal is BLOCK, or HOLD under an ask rule; anything else ALLOW."""
    text = str(content or "")
    if not text.startswith(DENY_PREFIX):
        return ALLOW, None
    rule = parse_deny(text)["rule_id"]
    return (HOLD if rule in HOLD_RULES else BLOCK), rule


def lf_rule(decision: str, reason: str) -> str | None:
    if decision == ALLOW:
        found = ALLOW_RULE.search(reason)
        return found.group(1) if found else None
    return parse_deny(reason)["rule_id"]


def recorded_calls(sim: dict) -> list[dict]:
    """Every tool call of a saved run in order, with the text of its message and its recorded reply (None when the
    run ended before the reply). Replies follow their message's calls in order, as glint.gates.summary.run_calls
    pairs them."""
    out, pending = [], []
    for i, m in enumerate(sim.get("messages") or []):
        if m.get("role") in ("assistant", "user"):
            pending = []
            for tc in m.get("tool_calls") or []:
                call = {"i": i, "tc": tc, "text": str(m.get("content") or ""), "reply": None}
                out.append(call)
                pending.append(call)
            continue
        if m.get("role") == "tool" and pending:
            pending.pop(0)["reply"] = m
    return out


def is_error(reply) -> bool:
    if reply is None:
        return False
    content = str(getattr(reply, "content", None) if not isinstance(reply, dict) else reply.get("content") or "")
    error = getattr(reply, "error", None) if not isinstance(reply, dict) else reply.get("error")
    return str(error) == "True" or error is True or content.startswith("Error")


def records(tools) -> dict:
    return {"glint_denied": set(tools.glint_denied), "glint_expedite_asked": set(tools.glint_expedite_asked),
            "glint_address_asked": set(tools.glint_address_asked)}


def restore(tools, saved: dict) -> None:
    for name, value in saved.items():
        setattr(tools, name, value)


def note_live_refusal(tools, write: tuple[str, dict | None], rule: str | None) -> None:
    """The records the gated environment keeps for a refusal (glint.gates.enforce.Enforcer.attach), for a call the
    live gate refused and the replay does not send through the environment."""
    asked = {EXPEDITE_ASK: tools.glint_expedite_asked, ADDRESS_ASK: tools.glint_address_asked}
    if rule in asked:
        asked[rule].add(str((write[1] or {}).get("credit_card_account_id", "")))
    tools.glint_denied.add((write[0], json.dumps(write[1], sort_keys=True, default=str), rule))


def replay_run(tag: str, sim: dict, task, enforcer: Enforcer, writes: frozenset[str]) -> dict:
    from llamafirewall import AssistantMessage, LlamaFirewall, Role

    env = environment(DOMAIN, task)
    raw = env.get_response  # τ²'s own tools, for a call the live gate allowed and the current gates refuse
    assist_environment(env, DOMAIN, enforcer)  # the tools arm's decision tools (check_*), gated, as tools_ab ran it
    make_scanner(enforcer, env)  # registers "glint_gates", bound to this run's environment
    firewall = LlamaFirewall({Role.ASSISTANT: ["glint_gates"]})
    rows, stats = [], Counter()
    for c in recorded_calls(sim):
        tc = c["tc"]
        call = ToolCall(id=str(tc.get("id") or f"r{c['i']}"), name=tc["name"], arguments=tc.get("arguments") or {},
                        requestor=tc.get("requestor") or "assistant")
        known = env.tools.has_tool(call.name) or bool(env.user_tools and env.user_tools.has_tool(call.name))
        if not known:
            stats[f"calls_unknown_tool:{call.name}"] += 1
            continue
        write = enforcer.write_call(env, call)
        if write is None:
            reply = env.get_response(call)
            stats["other_calls"] += 1
            stats["other_error_status_differs"] += c["reply"] is not None and is_error(reply) != is_error(c["reply"])
            continue
        if c["reply"] is None:
            stats["gated_no_recorded_reply"] += 1
            continue
        live, live_rule = decision_of(c["reply"].get("content"))
        message = AssistantMessage(content=c["text"], tool_calls=[{
            "id": call.id, "type": "function",
            "function": {"name": call.name, "arguments": json.dumps(call.arguments)}}])
        result = firewall.scan(message)
        lf = result.decision.name
        wrapper = None
        if live == ALLOW:  # it ran live, so it runs here
            saved = records(env.tools)
            reply = env.get_response(call)
            wrapper = decision_of(reply.content)[0]
            if wrapper != ALLOW:
                restore(env.tools, saved)
                reply = raw(call)
                stats["forced_runs"] += 1
            stats["write_error_status_differs"] += is_error(reply) != is_error(c["reply"])
        elif lf != ALLOW:  # refused live and by the scanner: the gated environment refuses it and keeps its records
            reply = env.get_response(call)
            wrapper = decision_of(reply.content)[0]
            stats["diverged_ran_here_not_live"] += wrapper == ALLOW
        else:  # refused live, allowed now: not run, and the refusal's records kept as live
            note_live_refusal(env.tools, write, live_rule)
        kind = "write" if write[0] in writes else "generic" if write[0] in GATED_GENERIC[DOMAIN] else "other"
        rows.append({"set": tag, "task": sim["task_id"], "trial": sim.get("trial"), "message": c["i"],
                     "tool": write[0], "kind": kind, "live": live, "live_rule": live_rule, "lf": lf,
                     "lf_rule": lf_rule(lf, result.reason), "score": result.score, "wrapper": wrapper})
    return {"rows": rows, "stats": stats}


def run(data: Path, request_gate: bool = True) -> tuple[list[dict], Counter, dict]:
    from glint import inspect

    tasks, writes = load_tasks(DOMAIN), inspect._writes(DOMAIN)
    enforcer = Enforcer(DOMAIN, EventLog(":memory:"), stages=None, request_gate=request_gate)
    rows, stats, runs = [], Counter(), {}
    for tag, folder in SETS:
        sims = json.loads((data / folder / ARM_FILE).read_text(encoding="utf-8"))["simulations"]
        counted = [s for s in sims if outcome(s) not in NOT_RESULTS]
        runs[tag] = (len(counted), len(sims))
        for sim in counted:
            out = replay_run(tag, sim, tasks[sim["task_id"]], enforcer, writes)
            rows += out["rows"]
            stats += out["stats"]
    return rows, stats, runs


def matrix(rows: list[dict]) -> list[str]:
    table = Counter((r["live"], r["lf"]) for r in rows)
    lines = ["    live \\ LlamaFirewall   " + "  ".join(f"{d:>26}" for d in DECISIONS)]
    for live in DECISIONS:
        lines.append(f"    {LIVE_NAME[live]:<22}" + "  ".join(f"{table[(live, d)]:>26}" for d in DECISIONS))
    return lines


def report(rows: list[dict], stats: Counter, runs: dict) -> str:
    lines = [f"LlamaFirewall {version('llamafirewall')}, only the glint_gates scanner; CPU, no model scanner"]
    for tag, _ in SETS + (("all", None),):
        part = rows if tag == "all" else [r for r in rows if r["set"] == tag]
        kinds = Counter(r["kind"] for r in part)
        agree = sum(r["live"] == r["lf"] for r in part)
        head = f"runs {runs[tag][0]} of {runs[tag][1]} saved" if tag != "all" else \
            f"runs {sum(v[0] for v in runs.values())}"
        lines.append(f"\n[{tag}] {head}; gated calls scanned {len(part)} (write {kinds['write']}, transfer or "
                     f"give-tool {kinds['generic']}, other {kinds['other']}); agree with live {agree} of {len(part)} "
                     f"(writes {sum(r['live'] == r['lf'] for r in part if r['kind'] == 'write')} of {kinds['write']})")
        lines += matrix(part)
    checked = [r for r in rows if r["wrapper"] is not None]
    same = sum(r["wrapper"] == r["lf"] for r in checked)
    lines.append(f"\nLlamaFirewall vs the gated environment's own decision on the same call: {same} of {len(checked)} "
                 "agree (calls refused live and allowed by the scanner were not sent through the environment)")
    lines.append("replay: " + ", ".join(f"{k} {v}" for k, v in sorted(stats.items())))
    lines.append("state check, replies whose error status differs from the recorded reply: "
                 f"{stats['other_error_status_differs']} of {stats['other_calls']} other calls, "
                 f"{stats['write_error_status_differs']} of the live-allowed gated calls; "
                 f"gated calls run here but refused live: {stats['diverged_ran_here_not_live']}")
    off = [r for r in rows if r["live"] != r["lf"]]
    lines.append(f"\nDisagreements with the live gate: {len(off)}")
    for (live, lf), n in Counter((r["live"], r["lf"]) for r in off).most_common():
        lines.append(f"  live {LIVE_NAME[live]} -> LlamaFirewall {lf}: {n}")
    for r in off:
        lines.append(f"  {r['set']} {r['task']} t{r['trial']} msg {r['message']} {r['tool']}: live {LIVE_NAME[r['live']]}"
                     f" [{r['live_rule']}] -> {r['lf']} [{r['lf_rule']}]")
    for r in checked:
        if r["wrapper"] != r["lf"]:
            lines.append(f"  SCANNER vs ENVIRONMENT {r['set']} {r['task']} t{r['trial']} msg {r['message']} "
                         f"{r['tool']}: {r['lf']} vs {r['wrapper']}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    logger.remove()
    data = Path(argv[argv.index("--data") + 1]) if "--data" in argv else DATA
    rows, stats, runs = run(data, request_gate="--no-request-gate" not in argv)
    print(f"request gate (research row 6, added after these runs): {'off' if '--no-request-gate' in argv else 'on'}")
    print(report(rows, stats, runs))
    if "--json" in argv:
        Path(argv[argv.index("--json") + 1]).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 1 if any(r["wrapper"] is not None and r["wrapper"] != r["lf"] for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
