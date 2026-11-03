"""Checks that read the conversation text or ask a model: the pieces the write gates cannot run themselves.

The write gates see tool calls and the database only (research note fix-research-2026-11-02, fact T7). The research
appendix "Request gate, holds and defaults" failed several mechanisms on exactly that: they need the conversation, a
model, or a person. This module builds them as plain functions over a transcript, each with a pluggable model call,
so they can be tested with fakes and switched on later. glint.gates.stages wires them into the enforcer; every one is
off unless configured, and nothing here calls a paid service by itself.

- #9 request record built by an extractor (T7, D§8): `extract_requests`, single-fact and turn-cited.
- #8 address confirmed in the conversation (B1 §1): `address_confirmed`.
- #10 dialogue-grounded checklist verifier (PolicyGuard, R1 3b): `checklist_verify`, `CHECKLISTS`.
- #11 action trace against the stated goal (AlignmentCheck, R4 M2): `alignment_check`.
- #16 ask the customer instead of refusing (RTBAS, R2 B): `customer_confirmed`, `ask_question`.
- #26 cascade to a calibrated verifier, unsure cases asked (R4 K3, D§8): `Cascade`, `calibrate`, `binomial_upper`.

None of these was measured: offline they are unmeasurable without labels (research M12), so every result they give
is unvalidated until a labeled set exists.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from glint.assist.banking import REQUEST_ACTIONS

Turn = dict  # {"role": "user" | "assistant", "content": str}; "user" is the customer
ModelCall = Callable[[str], str]  # a prompt in, the model's reply text out; the caller picks the model and pays for it


def turns(messages: Iterable) -> list[Turn]:
    """The customer and agent turns with text, from τ²/LiteLLM message dicts or objects with role and content. Tool
    calls, tool results and system prompts are dropped: the checks read what was said."""
    out = []
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            out.append({"role": role, "content": content})
    return out


class TranscriptTap:
    """The conversation as the agent last saw it: wrap the agent's completion function, and each call keeps the
    message list it was sent (LiteLLM's `messages`, keyword or second positional argument). Calling the tap returns
    those messages as turns. `key` (default: one conversation) names the conversation a call belongs to, so
    conversations run concurrently each keep their own messages (tools_ab passes its run context)."""

    def __init__(self, key: Callable[[], object] | None = None):
        self._key = key or (lambda: None)
        self._messages: dict = {}

    def wrap(self, completion: Callable) -> Callable:
        def wrapped(*args, **kwargs):
            messages = kwargs.get("messages", args[1] if len(args) > 1 else None)
            if messages is not None:
                self._messages[self._key()] = list(messages)
            return completion(*args, **kwargs)
        return wrapped

    def __call__(self) -> list[Turn]:
        return turns(self._messages.get(self._key(), []))


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", str(text).lower()))


def said_by_customer(conversation: list[Turn], words: str) -> bool:
    """Whether `words` (case, spacing and punctuation aside) appear in one customer turn: the turn citation that keeps
    an extracted fact checkable (research #53's validator, in its deterministic form)."""
    wanted = _norm(words)
    return bool(wanted) and any(wanted in _norm(t["content"]) for t in conversation if t["role"] == "user")


def parse_json(reply: str):
    """The first JSON object or array in a model reply (a ```json fence allowed). Raises ValueError when there is
    none: a reply that cannot be read is never treated as an answer."""
    text = str(reply)
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError(f"no JSON in the model reply: {text[:120]!r}")
    try:
        value, _ = json.JSONDecoder().raw_decode(text[min(starts):])
    except json.JSONDecodeError as exc:
        raise ValueError(f"unreadable JSON in the model reply: {exc}") from exc
    return value


def _dialogue(conversation: list[Turn]) -> str:
    who = {"user": "customer", "assistant": "agent"}
    return "\n".join(f"[{k}] {who[t['role']]}: {t['content']}" for k, t in enumerate(conversation))


# ---- #9 request record built by an extractor -----------------------------------------------------------------------

EXTRACT_PROMPT = """You read a bank support conversation and list the account actions the customer asked for.
Allowed actions (action: the id it targets): {actions}.
Return only a JSON list of objects {{"action": ..., "target_id": ..., "customer_words": ...}}. customer_words must be
copied exactly from one customer turn. Leave target_id "" when the customer named no id. List nothing the agent
proposed and the customer did not ask for.

Conversation:
{dialogue}"""


def extract_requests(conversation: list[Turn], model: ModelCall) -> list[dict]:
    """The customer's requests as record_requests stores them, extracted by `model` (research #9: single-fact
    extraction; T7's premise held: the customer asked in all 10 test runs; accuracy untested). Only actions in
    REQUEST_ACTIONS whose customer_words appear in a customer turn are kept (turn-cited), each (action, target) once."""
    actions = ", ".join(f"{a}: {arg}" for a, (_, arg) in sorted(REQUEST_ACTIONS.items()))
    found = parse_json(model(EXTRACT_PROMPT.format(actions=actions, dialogue=_dialogue(conversation))))
    if not isinstance(found, list):
        raise TypeError(f"the extractor returned {type(found).__name__}, not a list")
    out, seen = [], set()
    for r in found:
        if not isinstance(r, dict):
            continue
        action, target, words = str(r.get("action", "")), str(r.get("target_id") or ""), str(r.get("customer_words",
                                                                                                    ""))
        if action in REQUEST_ACTIONS and said_by_customer(conversation, words) and (action, target) not in seen:
            seen.add((action, target))
            out.append({"action": action, "target_id": target, "customer_words": words})
    return out


# ---- #8 and #16: did the customer say yes ------------------------------------------------------------------------

AFFIRM = ("yes", "yeah", "yep", "yup", "correct", "that's right", "that is right", "sounds good", "please do",
          "go ahead", "that works", "sure", "confirm", "confirmed", "absolutely", "ok", "okay")
NEGATE = {"no", "not", "don't", "dont", "nope", "different", "another", "instead", "wrong", "change", "cancel",
          "wait", "never"}


def affirms(text: str) -> bool:
    """A customer turn that says yes and nothing that undoes it (a deterministic reading; Glint convention)."""
    words = _norm(text)
    padded = f" {words} "
    return any(f" {a} " in padded for a in AFFIRM) and not NEGATE & set(words.split())


def yes_no(reply: str) -> bool:
    """A model's yes/no answer. Raises ValueError on anything else."""
    first = (_norm(reply).split() or [""])[0]
    if first not in ("yes", "no"):
        raise ValueError(f"the model answered neither yes nor no: {str(reply)[:80]!r}")
    return first == "yes"


def _reply_after(conversation: list[Turn], asked: Callable[[str], bool]) -> str | None:
    """The customer's turn right after the last agent turn for which `asked` holds; None when there is none."""
    for k in range(len(conversation) - 1, -1, -1):
        t = conversation[k]
        if t["role"] == "assistant" and asked(t["content"]):
            after = [u for u in conversation[k + 1:] if u["role"] == "user"]
            return after[0]["content"] if after else None
    return None


CONFIRM_PROMPT = """The agent asked the customer: {question}
Agent's last turn: {agent}
Customer's reply: {reply}
Did the customer clearly agree? Answer yes or no."""


def address_confirmed(conversation: list[Turn], on_file: str, model: ModelCall | None = None) -> bool:
    """Research #8: the agent named the address on file (its street line) and the customer's next turn agreed.
    With `model`, the model judges the reply instead of the keyword reading."""
    street = _norm(str(on_file).split(",")[0])
    if not street:
        return False
    named = [t["content"] for t in conversation if t["role"] == "assistant" and street in _norm(t["content"])]
    reply = _reply_after(conversation, lambda text: street in _norm(text))
    if reply is None:
        return False
    if model is None:
        return affirms(reply)
    return yes_no(model(CONFIRM_PROMPT.format(question=f"ship to {on_file}?", agent=named[-1], reply=reply)))


def ask_question(tool: str, args: dict, reason: str) -> str:
    """The question the agent puts to the customer instead of a refusal (research #16)."""
    target = next((str(v) for k, v in sorted(args.items()) if k.endswith("_id") and k != "user_id" and v), "")
    return (f"Before I make this change ({tool}{' on ' + target if target else ''}): the bank's check found that "
            f"{reason}. Do you want me to go ahead?")


def customer_confirmed(conversation: list[Turn], question: str, model: ModelCall | None = None) -> bool:
    """Research #16: the customer's reply to the agent's last question agrees. The keyword reading by default, the
    model's yes/no with `model`. The gate cannot check that the agent asked this exact question, only that it asked
    one and was told yes; that the question matched stays a trust assumption (Glint convention)."""
    agent = [t["content"] for t in conversation if t["role"] == "assistant" and "?" in t["content"]]
    reply = _reply_after(conversation, lambda text: "?" in text)
    if reply is None:
        return False
    if model is None:
        return affirms(reply)
    return yes_no(model(CONFIRM_PROMPT.format(question=question, agent=agent[-1], reply=reply)))


# ---- #10 dialogue-grounded checklist verifier ----------------------------------------------------------------------

@dataclass(frozen=True)
class ChecklistItem:
    id: str
    question: str
    source: str


CHECKLISTS: dict[str, tuple[ChecklistItem, ...]] = {
    "close_credit_card_account_7834": (
        ChecklistItem("reason_asked", "Did the agent ask the customer why they want to close the account?",
                      "credit_card_account_logistics_003 step 3: \"Ask the customer why they want to close\""),
        ChecklistItem("offer_declined", "Did the customer decline the retention offer before the closure?",
                      "credit_card_account_logistics_003 steps 5-6"),
    ),
    "order_replacement_credit_card_7291": (
        ChecklistItem("address_confirmed", "Did the customer confirm the shipping address (primary or alternate)?",
                      "credit_card_replacements_001: \"Confirm the shipping address with the customer\""),
    ),
    "file_debit_card_transaction_dispute_6281": (
        ChecklistItem("possession_asked", "Did the agent ask whether the customer still has the physical card?",
                      "bank_accounts_(general)_031 argument 10"),
        ChecklistItem("pin_asked", "Did the agent ask whether the customer believes the PIN was compromised?",
                      "bank_accounts_(general)_031 argument 11"),
    ),
    "file_credit_card_transaction_dispute_4829": (
        ChecklistItem("merchant_asked", "Did the agent ask whether the customer tried to resolve it with the "
                      "merchant first?", "credit_cards_(general)_014 argument 9"),
    ),
}
CHECKLIST_PROMPT = """You check a bank agent's conversation against a policy checklist before a write runs.
Write: {tool} {args}
Checklist:
{items}
Answer only a JSON object mapping each id to "met", "unmet" or "unsure".

Conversation:
{dialogue}"""


def checklist_verify(conversation: list[Turn], tool: str, args: dict, model: ModelCall) -> dict:
    """Research #10 (PolicyGuard): each CHECKLISTS item for `tool` as met, unmet or unsure in the model's reading of
    the conversation. An item the reply leaves out, or answers with another word, is unsure."""
    items = CHECKLISTS.get(tool, ())
    if not items:
        return {"met": [], "unmet": [], "unsure": []}
    listed = "\n".join(f"- {i.id}: {i.question}" for i in items)
    reply = parse_json(model(CHECKLIST_PROMPT.format(tool=tool, args=json.dumps(args, sort_keys=True, default=str),
                                                     items=listed, dialogue=_dialogue(conversation))))
    if not isinstance(reply, dict):
        raise TypeError(f"the checklist verifier returned {type(reply).__name__}, not an object")
    out = {"met": [], "unmet": [], "unsure": []}
    for i in items:
        verdict = str(reply.get(i.id, "unsure")).lower()
        out[verdict if verdict in out else "unsure"].append(i.id)
    return out


# ---- #11 action trace against the stated goal ----------------------------------------------------------------------

ALIGNMENT_PROMPT = """The customer's turns state what they want. Below them are the account changes the agent made and
the one it proposes now. Is the proposed change in service of what the customer asked for?
Answer only a JSON object {{"aligned": true or false, "reason": "..."}}.

Customer turns:
{goal}

Changes made: {done}
Proposed: {tool} {args}"""


def alignment_check(conversation: list[Turn], done: list[tuple[str, dict]], tool: str, args: dict,
                    model: ModelCall) -> dict:
    """Research #11 (AlignmentCheck): whether the proposed write serves the customer's stated goal, judged by `model`
    from the customer turns and the write trace. Raises TypeError when `aligned` is not a JSON boolean."""
    goal = "\n".join(f"- {t['content']}" for t in conversation if t["role"] == "user")
    trace = "; ".join(f"{t} {json.dumps(a, sort_keys=True, default=str)}" for t, a in done) or "none"
    reply = parse_json(model(ALIGNMENT_PROMPT.format(goal=goal, done=trace, tool=tool,
                                                     args=json.dumps(args, sort_keys=True, default=str))))
    if not isinstance(reply, dict) or not isinstance(reply.get("aligned"), bool):
        raise TypeError(f"the alignment check gave no boolean 'aligned': {reply!r}"[:200])
    return {"aligned": reply["aligned"], "reason": str(reply.get("reason", ""))}


# ---- #26 cascade to a calibrated verifier --------------------------------------------------------------------------

def binomial_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """The one-sided Clopper-Pearson upper bound on a rate, from k events in n trials, at confidence 1 - alpha
    (research #24/N5's bound, here for a verifier's threshold)."""
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if n == 0 or k == n:
        return 1.0

    def cdf(p: float) -> float:
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))

    lo, hi = k / n, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) > alpha else (lo, mid)
    return hi


def calibrate(benign_scores: list[float], max_false_block: float, alpha: float = 0.05) -> float:
    """The highest deny threshold whose false-block rate on labeled benign writes (scores below it are blocked) has a
    Clopper-Pearson upper bound at most `max_false_block`. Raises ValueError when even blocking none of them cannot
    show that bound: too few labeled writes. The threshold holds only for the verifier and labels it was fitted on;
    any change voids it (research #25)."""
    scores = sorted(float(s) for s in benign_scores)
    n = len(scores)
    best = None
    for t in sorted(set(scores)):
        blocked = sum(1 for s in scores if s < t)
        if binomial_upper(blocked, n, alpha) <= max_false_block:
            best = t
    if best is None:
        raise ValueError(f"{n} labeled benign writes cannot bound the false-block rate at {max_false_block} "
                         f"(confidence {1 - alpha:g}); label more")
    return best


SCORE_PROMPT = """From the conversation, how likely is it (0 to 1) that this write follows the bank's policy and the
customer's request? Answer only a JSON object {{"p": number}}.
Write: {tool} {args}

Conversation:
{dialogue}"""


def model_score(model: ModelCall) -> Callable[[list[Turn], str, dict], float]:
    """A Cascade score function asking `model` for a probability."""
    def score(conversation: list[Turn], tool: str, args: dict) -> float:
        reply = parse_json(model(SCORE_PROMPT.format(tool=tool, args=json.dumps(args, sort_keys=True, default=str),
                                                     dialogue=_dialogue(conversation))))
        if not isinstance(reply, dict) or not isinstance(reply.get("p"), (int, float)):
            raise TypeError(f"the verifier gave no number 'p': {reply!r}"[:200])
        return float(reply["p"])
    return score


@dataclass(frozen=True)
class Cascade:
    """Research #26: the deterministic gates first, then a verifier score; at or above `allow_at` the write runs,
    below `deny_below` it is refused, in between the agent asks the customer (glint.gates.stages)."""
    score: Callable[[list[Turn], str, dict], float]
    deny_below: float
    allow_at: float

    def __post_init__(self):
        if not 0.0 <= self.deny_below <= self.allow_at <= 1.0:
            raise ValueError(f"need 0 <= deny_below <= allow_at <= 1, got {self.deny_below}, {self.allow_at}")

    def decide(self, conversation: list[Turn], tool: str, args: dict) -> tuple[str, float]:
        p = self.score(conversation, tool, args)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"the verifier score {p} is outside [0, 1]")
        return ("allow" if p >= self.allow_at else "deny" if p < self.deny_below else "ask"), p
