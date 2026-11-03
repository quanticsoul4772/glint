"""Checks against the conversation text, over a saved simulation (appendix rows #36, #46, #53 and #59 of
research/fix-research-2026-11-02.md, "Field values"; the "Not before Nov 4" list's "checks against the conversation
text" and "customer facts extracted from the conversation"). The request record built by an extractor (#9) is
glint.gates.conversation's, on the rest-refusals-defaults branch.

The gates never see conversation text (`Enforcer.attach` wraps `env.get_response`, which carries tool calls only), so
these run offline over a run's saved messages: measurement, never a decision on a write. Deterministic checks are
keyword and string matches (Glint convention, no policy clause beyond the one each names). The extractor takes a
pluggable model call (`ModelCall`: prompt -> completion text) and is OFF by default: with no model it returns None,
and nothing here calls a paid service. Memo M12 applies: an extractor's accuracy is unmeasured until labeled.

- #36 `possession_conflicts`: a debit dispute's card_in_possession against what the customer said about the card
  (bank_accounts_(general)_031 argument 10 asks "Do you still have your physical debit card in your possession?").
- #46 `unsupplied_values`: write arguments (numbers, identifiers) that appear in no earlier customer turn or tool
  result (Latent Critic, reader 2 G).
- #53 `uncited_records`: record_requests / record_retention_answer / record_dispute_answers entries whose
  customer_words appear in no customer turn (turn-cited record validator, reader 2 P).
- #59 `dispute_without_filing`: the customer described a disputed charge and no dispute filing ran (reader 3 G1).
- #36 with a model: `extract_fact`, a single closed-question fact from the customer's turns (whether they still have
  the card), in place of the keyword match.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable

from glint.assist import as_flag
from glint.assist.banking import DISPUTE_ANSWERS_TOOL, REQUESTS_TOOL, RETENTION_TOOL
from glint.gates.summary import run_calls

ModelCall = Callable[[str], str]
DEBIT_DISPUTE, CREDIT_DISPUTE = "file_debit_card_transaction_dispute_6281", "file_credit_card_transaction_dispute_4829"
RECORD_TOOLS = (REQUESTS_TOOL, RETENTION_TOOL, DISPUTE_ANSWERS_TOOL)
# Glint convention: phrasings of a card no longer held, and of a card still held
LOST = re.compile(r"\b(lost|stolen|misplaced|can(?:no|')t find|cannot find|someone took|was taken|don'?t have (?:it|my "
                  r"card|the card)|no longer have)\b", re.IGNORECASE)
HELD = re.compile(r"\b(still have (?:it|my card|the card)|i have (?:it|my card|the card)|it'?s (?:right )?here|in my "
                  r"(?:wallet|possession|hand))\b", re.IGNORECASE)
NEGATED_LOST = re.compile(r"\b(not|never|haven'?t|didn'?t|wasn'?t)\s+(?:been\s+)?(lost|stolen)\b", re.IGNORECASE)
DISPUTE_TALK = re.compile(r"\b(dispute|didn'?t (?:make|authorize)|unauthori[sz]ed|charged twice|double[- ]charged|"
                          r"duplicate charge|fraud(?:ulent)?|never received|wrong amount|overcharged)\b", re.IGNORECASE)
_ID_LIKE = re.compile(r"\d")


def _turns(sim: dict, role: str) -> list[tuple[int, str]]:
    return [(i, str(m.get("content") or "")) for i, m in enumerate(sim.get("messages") or [])
            if m.get("role") == role and m.get("content")]


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w$.]+", " ", str(text).lower()).split())


def possession_said(text: str) -> bool | None:
    """True when the text says the customer still has the card, False when it says the card is lost or stolen, None
    when it says neither (or both)."""
    held = bool(HELD.search(text)) or bool(NEGATED_LOST.search(text))
    lost = bool(LOST.search(NEGATED_LOST.sub(" ", text)))
    return None if held == lost else held


def possession_conflicts(sim: dict) -> list[dict]:
    """#36: each debit dispute filing that ran whose card_in_possession differs from the customer's latest statement
    about the card before it."""
    said = [(i, v) for i, text in _turns(sim, "user") if (v := possession_said(text)) is not None]
    out = []
    for c in run_calls(sim):
        if c["tool"] != DEBIT_DISPUTE or c["error"] or not c["args"]:
            continue
        before = [v for i, v in said if i < c["index"]]
        if before and as_flag(c["args"].get("card_in_possession")) != before[-1]:
            out.append({"index": c["index"], "card_id": c["args"].get("card_id"),
                        "card_in_possession": c["args"].get("card_in_possession"), "customer_said_has_card": before[-1]})
    return out


def _variants(value) -> set[str]:
    if isinstance(value, bool):
        return set()
    if isinstance(value, (int, float)):
        x = float(value)
        return {f"{x:g}", f"{x:.2f}", f"{x:,.2f}", str(value)}
    return {str(value).lower()}


def unsupplied_values(sim: dict, tools: tuple[str, ...] | None = None) -> list[dict]:
    """#46: each scalar argument of a write that ran, a number or a value with a digit in it (ids, dates, amounts),
    that appears in no customer turn and no tool reply before the call. Enum words and booleans are not judged."""
    seen: list[tuple[int, str]] = _turns(sim, "user") + [(c["index"], c["content"]) for c in run_calls(sim)]
    out = []
    for c in run_calls(sim):
        if c["error"] or not c["args"] or c["tool"] in RECORD_TOOLS or (tools and c["tool"] not in tools):
            continue
        earlier = " ".join(text.lower() for i, text in seen if i < c["index"])
        for arg, value in c["args"].items():
            if isinstance(value, bool) or not (isinstance(value, (int, float)) or _ID_LIKE.search(str(value))):
                continue
            if not any(v in earlier for v in _variants(value)):
                out.append({"index": c["index"], "tool": c["tool"], "arg": arg, "value": value})
    return out


def _cited(words: str, customer: list[str]) -> bool:
    w = _norm(words)
    return bool(w) and any(w in _norm(t) for t in customer)


def _record_entries(call: dict) -> list[dict]:
    args = call["args"] or {}
    if call["tool"] == REQUESTS_TOOL:
        reqs = args.get("requests")
        if isinstance(reqs, str):
            try:
                reqs = json.loads(reqs)
            except json.JSONDecodeError:
                reqs = []
        return [r for r in reqs or [] if isinstance(r, dict)]
    return [args]


def uncited_records(sim: dict) -> list[dict]:
    """#53: each entry of a Glint record call whose customer_words is empty or is not in any customer turn before the
    call (case, spacing and punctuation ignored)."""
    users = _turns(sim, "user")
    out = []
    for c in run_calls(sim):
        if c["tool"] not in RECORD_TOOLS or c["args"] is None:
            continue
        customer = [t for i, t in users if i < c["index"]]
        for entry in _record_entries(c):
            words = str(entry.get("customer_words") or "")
            if not _cited(words, customer):
                out.append({"index": c["index"], "tool": c["tool"], "customer_words": words[:200],
                            "action": entry.get("action")})
    return out


def dispute_without_filing(sim: dict) -> dict | None:
    """#59: the customer's turns describe a disputed charge, and no dispute filing (debit or credit card) ran."""
    talk = [i for i, t in _turns(sim, "user") if DISPUTE_TALK.search(t)]
    filed = any(c["tool"] in (DEBIT_DISPUTE, CREDIT_DISPUTE) and not c["error"] and c["deny"] is None
                for c in run_calls(sim))
    return {"customer_turns": talk} if talk and not filed else None


FACT_PROMPT = ("Below are a bank customer's messages. Question: {question}\nAnswer with exactly one of: {choices}. "
               "Answer \"unknown\" if the messages do not say.\n\n{turns}")
POSSESSION_QUESTION = "Does the customer still have the physical debit card in their possession?"


def extract_fact(sim: dict, question: str, choices: tuple[str, ...], model: ModelCall | None = None) -> str | None:
    """A single closed-question fact from the customer's turns (single-fact extraction, T7); None when no model is
    given (off) or the answer is not one of the choices."""
    if model is None:
        return None
    turns = "\n".join(f"- {t}" for _, t in _turns(sim, "user"))
    answer = model(FACT_PROMPT.format(question=question, choices=", ".join(choices), turns=turns)).strip().strip(
        ".\"'").lower()
    return answer if answer in choices else None


def audit(sim: dict, model: ModelCall | None = None) -> dict:
    """Every check on one simulation; with a model, also #36's possession fact as the model extracts it."""
    out = {"task_id": sim.get("task_id"), "trial": sim.get("trial"),
           "possession_conflicts": possession_conflicts(sim), "unsupplied_values": unsupplied_values(sim),
           "uncited_records": uncited_records(sim), "dispute_without_filing": dispute_without_filing(sim)}
    if model is not None:
        out["possession_extracted"] = extract_fact(sim, POSSESSION_QUESTION, ("yes", "no", "unknown"), model)
    return out
