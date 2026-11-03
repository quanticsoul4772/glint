"""Single-fact extraction from the conversation (agenda T7, appendix #9; design §1 B, §8): the harness for a request
record or customer facts built by an extractor instead of by the agent.

Off by default: every extraction goes through a pluggable model call (glint.modelcall); with none it raises
ModelOff and calls nothing. No conversation text reaches Glint's gates (M12), so this is measurement only: it never
decides a write. Its accuracy is untested (fix-research "Lead results": premise held, accuracy untested); `score`
measures it against hand labels.

One call asks one closed question about one fact (single-fact extraction): the model sees the customer's turns,
numbered, and answers with one value from the fact's closed set, the turn it relies on and a verbatim quote from that
turn. An answer outside the set, or a quote the cited turn does not contain, is recorded as `invalid`, never
trusted. Facts (FACTS), each with its source:
- card_in_possession, pin_compromised: the two debit dispute questions (bank_accounts_(general)_031; the closed
  values follow tools.py's file_debit_card_transaction_dispute_6281 arguments);
- lost_or_stolen_reported: whether the customer said the card was lost or stolen (doc 031, the liability tiers);
- asked:<action>: whether the customer asked for that account action, one per glint.assist.banking.REQUEST_ACTIONS
  action (the request gate's actions; Glint convention).
"""
from __future__ import annotations

import json
from collections import Counter

from glint.assist.banking import REQUEST_ACTIONS
from glint.modelcall import ModelCall, parse_object, require

YES_NO = ("yes", "no", "not_said")
FACTS: dict[str, dict] = {
    "card_in_possession": {"question": "Does the customer say the debit card is still in their possession?",
                           "answers": YES_NO, "source": "bank_accounts_(general)_031"},
    "pin_compromised": {"question": "Does the customer say their PIN was compromised (shared, seen or stolen)?",
                        "answers": ("yes", "no", "unknown", "not_said"), "source": "bank_accounts_(general)_031"},
    "lost_or_stolen_reported": {"question": "Does the customer say the card was lost or stolen?",
                                "answers": YES_NO, "source": "bank_accounts_(general)_031"},
} | {f"asked:{a}": {"question": f"Does the customer ask for this account action: {a.replace('_', ' ')}?",
                    "answers": YES_NO, "source": "Glint convention (REQUEST_ACTIONS)"} for a in REQUEST_ACTIONS}


def customer_turns(sim: dict) -> list[str]:
    """The customer's text turns of a saved simulation, in order (tool calls and empty turns left out)."""
    return [str(m.get("content")) for m in sim.get("messages") or [] if m.get("role") == "user" and m.get("content")]


def prompt(fact: str, turns: list[str]) -> str:
    spec = FACTS[fact]
    numbered = "\n".join(f"[{i}] {t}" for i, t in enumerate(turns, 1))
    return (f"Read the customer's turns of a bank support conversation.\n\n{numbered}\n\nQuestion: {spec['question']}\n"
            f"Answer with one JSON object: {{\"answer\": one of {list(spec['answers'])}, \"turn\": the number of the "
            "turn you rely on (0 if not_said), \"quote\": the exact words from that turn (empty if not_said)}.")


def check_answer(fact: str, reply: dict | None, turns: list[str]) -> dict:
    """The validated answer: {fact, answer, turn, quote, valid, why}. Invalid when the reply is not an object, the
    answer is outside the closed set, or the quote is not in the cited turn."""
    out = {"fact": fact, "answer": "invalid", "turn": 0, "quote": "", "valid": False, "why": ""}
    if reply is None:
        return out | {"why": "no JSON object in the reply"}
    answer, quote = str(reply.get("answer", "")), str(reply.get("quote") or "")
    try:
        turn = int(reply.get("turn") or 0)
    except (TypeError, ValueError):
        return out | {"why": "turn is not a number"}
    if answer not in FACTS[fact]["answers"]:
        return out | {"why": f"answer {answer!r} is outside the closed set"}
    if answer != "not_said":
        if not 1 <= turn <= len(turns):
            return out | {"why": f"turn {turn} does not exist"}
        if not quote or " ".join(quote.lower().split()) not in " ".join(turns[turn - 1].lower().split()):
            return out | {"why": f"the quote is not in turn {turn}"}
    return {"fact": fact, "answer": answer, "turn": turn, "quote": quote, "valid": True, "why": ""}


def extract(sim: dict, fact: str, model: ModelCall | None) -> dict:
    """One fact from one conversation (one model call)."""
    call = require(model, "T7 extraction")
    if fact not in FACTS:
        raise ValueError(f"unknown fact {fact!r}")
    turns = customer_turns(sim)
    return check_answer(fact, parse_object(call(prompt(fact, turns))), turns)


def request_record(sim: dict, model: ModelCall | None) -> list[dict]:
    """The extracted request record: one asked:<action> question per action; the actions answered yes, each with
    its cited turn and quote (the record_requests form, built without the agent)."""
    require(model, "T7 extraction")
    out = []
    for action in REQUEST_ACTIONS:
        a = extract(sim, f"asked:{action}", model)
        if a["valid"] and a["answer"] == "yes":
            out.append({"action": action, "target_id": "", "customer_words": a["quote"], "turn": a["turn"]})
    return out


def score(predictions: list[dict], labels: list[dict]) -> dict:
    """Accuracy against hand labels ({task_id, trial, fact, answer}), matched on (task_id, trial, fact): overall and
    per fact, with the invalid share and the confusion pairs (label, prediction)."""
    want = {(str(x["task_id"]), int(x.get("trial") or 0), x["fact"]): x["answer"] for x in labels}
    per: dict[str, Counter] = {}
    confusion: Counter = Counter()
    for p in predictions:
        k = (str(p["task_id"]), int(p.get("trial") or 0), p["fact"])
        if k not in want:
            continue
        c = per.setdefault(p["fact"], Counter())
        c["n"] += 1
        c["correct"] += p["answer"] == want[k]
        c["invalid"] += p["answer"] == "invalid"
        confusion[(want[k], p["answer"])] += 1
    n = sum(c["n"] for c in per.values())
    correct = sum(c["correct"] for c in per.values())
    return {"n": n, "accuracy": round(correct / n, 4) if n else None,
            "invalid": sum(c["invalid"] for c in per.values()),
            "per_fact": {f: {"n": c["n"], "accuracy": round(c["correct"] / c["n"], 4)} for f, c in sorted(per.items())},
            "confusion": {f"{a} -> {b}": v for (a, b), v in sorted(confusion.items())},
            "labels_unmatched": len(want.keys() - {(str(p["task_id"]), int(p.get("trial") or 0), p["fact"])
                                                   for p in predictions})}


def load_jsonl(path) -> list[dict]:
    from pathlib import Path
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
