"""A second simulator (appendix #80; reader 4 S4; agenda T11b): the harness for checking the customer simulator with
another customer model. "Changing only the LLM that plays the user moves tau-bench retail pass rates by up to about
9 points" (reader 4, report 15.6).

Two forms, both off by default:
- **Offline cross-check** (this module): for each customer turn of a saved conversation, a second customer model
  gets the task's user instructions and the conversation before that turn, and writes the turn it would send. Each
  pair is compared without a model: both end the conversation or not (τ²'s stop tokens), and the word overlap
  (Jaccard over lower-case words). Turns where the recorded customer stops and the second would not (or the other
  way) are the simulator-error candidates for a hand audit (class 8). The model call is pluggable
  (glint.modelcall); with none it raises ModelOff and calls nothing.
- **Live rerun**: scripts/tools_ab.py takes set keys `llm_user` and `llm_args_user` (default: Muse Spark on Meta's
  API, as every run so far), so a set can rerun a task subset under another customer model and compare failure counts
  by task. That is a paid run: it needs the owner's yes and caps; no such set is defined here.
"""
from __future__ import annotations

import re

from glint.modelcall import ModelCall, require

STOP_TOKENS = ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###")  # τ²'s user simulator stop markers
WORDS = re.compile(r"[a-z0-9']+")


def stops(text: str) -> bool:
    return any(t in str(text or "") for t in STOP_TOKENS)


def jaccard(a: str, b: str) -> float:
    x, y = set(WORDS.findall(str(a).lower())), set(WORDS.findall(str(b).lower()))
    return round(len(x & y) / len(x | y), 4) if x | y else 1.0


def _render(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            who = "Customer" if m["role"] == "user" else "Agent"
            lines.append(f"{who}: {m['content']}")
    return "\n".join(lines)


def prompt(instructions: str, before: list[dict]) -> str:
    return ("You play a bank customer in a support conversation. Your instructions:\n"
            f"{instructions}\n\nThe conversation so far:\n{_render(before)}\n\n"
            "Write only your next message as the customer. When your instructions say the conversation is over, "
            "write ###STOP###.")


def cross_check(sim: dict, instructions: str, model: ModelCall | None) -> list[dict]:
    """One row per recorded customer text turn after the first: {index, recorded_stops, second_stops, agree_stop,
    overlap}. The first turn is left out (it opens the conversation, with nothing before it to condition on)."""
    call = require(model, "second simulator")
    messages = sim.get("messages") or []
    rows = []
    for i, m in enumerate(messages):
        if m.get("role") != "user" or not m.get("content") or not any(x.get("role") == "assistant"
                                                                      for x in messages[:i]):
            continue
        second = str(call(prompt(instructions, messages[:i])))
        rec_stop, sec_stop = stops(m["content"]), stops(second)
        rows.append({"index": i, "recorded_stops": rec_stop, "second_stops": sec_stop,
                     "agree_stop": rec_stop == sec_stop, "overlap": jaccard(m["content"], second)})
    return rows


def summary(rows_by_run: dict[str, list[dict]]) -> dict:
    """Totals over runs: turns compared, stop disagreements (with the runs they are in) and the mean overlap."""
    turns = [r for rows in rows_by_run.values() for r in rows]
    runs = sorted(k for k, rows in rows_by_run.items() if any(not r["agree_stop"] for r in rows))
    return {"runs": len(rows_by_run), "turns": len(turns),
            "stop_disagreements": sum(1 for r in turns if not r["agree_stop"]),
            "runs_with_stop_disagreement": runs,
            "mean_overlap": round(sum(r["overlap"] for r in turns) / len(turns), 4) if turns else None}


def instructions_of(task) -> str:
    """The user instructions of a τ² task (object or dict), as the simulator gets them."""
    scenario = task["user_scenario"] if isinstance(task, dict) else task.user_scenario
    ins = scenario["instructions"] if isinstance(scenario, dict) else scenario.instructions
    return ins if isinstance(ins, str) else str(ins)
