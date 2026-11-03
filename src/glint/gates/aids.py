"""Agent aids and two deny-only conventions on banking writes (fix research 2026-11-02, rows 14, N2 and N10).

- `credit_note` (policy, bank_accounts_(general)_017): the one-credit-per-checking-account fact, shown at the account
  reads (get_all_user_accounts_by_user_id_3847, get_bank_account_transactions_9173) with which of the customer's
  checking accounts were credited today. A function of the database, so the reply replays unchanged.
- `duplicate` (Glint convention, no clause): a write is refused when the last write that ran on the same target (the
  same `*_id` arguments) in this conversation is this exact call (same tool, identical arguments), on the enforcer's
  record of the writes that ran, in order (`toolkit.glint_ran`); freeze, unfreeze, freeze runs. Exempt: the initial
  transfer initial_transfer_to_human_agent_0218 and the credit limit increase submit, which τ²'s train gold repeats
  (032 the initial transfer; 051 the submit), and transfer_to_human_agents. Gold also repeats the user tool
  request_human_agent_transfer (034, 081), a user tool, which no agent gate sees.
- `provenance` (Glint convention): a write is refused when one of its identifier arguments (an argument named
  `*_id`) appeared in no earlier tool result of this conversation (`toolkit.glint_seen`). Only `*_id` arguments are
  identifiers here: values the customer says aloud, such as card last-4 digits, are not checked.

Both denies only refuse; neither lets a refused write run. Their records are rebuilt in τ²'s strict replay
(glint.gates.enforce): writes that ran replay, and the earlier results are taken from the replayed message history.
"""
from __future__ import annotations

import json

from glint.assist.banking import TODAY
from glint.gates import extend
from glint.gates.bankdb import as_date
from glint.gates.core import GateResult, allow, deny

CREDIT_TOOL = "apply_checking_account_credit_5829"
CREDIT_SOURCE = "bank_accounts_(general)_017"
CREDIT_TYPES = ("rebate_credit", "fee_refund")  # glint.gates.banking.CREDIT_TYPES: what the credit tool writes
ACCOUNTS_READ, TRANSACTIONS_READ = "get_all_user_accounts_by_user_id_3847", "get_bank_account_transactions_9173"


def credit_note(db, tool: str, args: dict) -> str | None:
    """The [policy] line for an account read: one credit per checking account per interaction (017), and, for each of
    the customer's checking accounts, whether it was credited today. None for other reads or a customer with no
    checking account."""
    if tool == ACCOUNTS_READ:
        user = str(args.get("user_id", ""))
    elif tool == TRANSACTIONS_READ:
        user = str((db.accounts.data.get(str(args.get("account_id", ""))) or {}).get("user_id", ""))
    else:
        return None
    checking = sorted(k for k, a in db.accounts.data.items()
                      if a.get("user_id") == user and str(a.get("class", "")).lower() == "checking")
    if not user or not checking:
        return None
    today = {t.get("account_id") for t in db.bank_account_transaction_history.data.values()
             if t.get("type") in CREDIT_TYPES and as_date(t.get("date")) == TODAY}
    credited = [a for a in checking if a in today]
    owed = [a for a in checking if a not in today]
    return (f"[policy] {CREDIT_SOURCE}: {CREDIT_TOOL} may be called ONCE per checking account per customer "
            "interaction, so each checking account that needs corrections gets its own single credit with the "
            "combined amount. Credited today: " + (", ".join(credited) or "none") + ". Not credited today: "
            + (", ".join(owed) or "none") + ".")


# ---- duplicate writes (N2) ----------------------------------------------------------------------------------------

# τ²'s train gold repeats initial_transfer_to_human_agent_0218 (task 032) and the submit (051); transfer_to_human_agents
# has been exempt since N2. The user tool request_human_agent_transfer (repeated in 034 and 081) is not an agent write.
DUPLICATE_EXEMPT = ("transfer_to_human_agents", "initial_transfer_to_human_agent_0218",
                    "submit_credit_limit_increase_request_7392")
DUPLICATE_SOURCE = "Glint convention, no clause: a write that already ran is not sent again"


def write_key(tool: str, args: dict | None) -> tuple[str, str]:
    """A write's identity for the duplicate check: the tool and its arguments, key order ignored."""
    return tool, json.dumps(args or {}, sort_keys=True, default=str)


def target_key(args: dict | None) -> str:
    """A write's target for the duplicate check: its identifier arguments (identifiers), key order ignored."""
    return json.dumps(identifiers(args or {}), sort_keys=True)


def duplicate(tool: str):
    """A gate for `tool`: refused when the last write that ran on the same target in this conversation is this exact
    call (tool and identical arguments). A different write on the target since (an unfreeze between two freezes)
    lets it run again."""
    def gate(toolkit, a: dict) -> GateResult:
        target = target_key(a)
        last = next((key for t, key in reversed(getattr(toolkit, "glint_ran", [])) if t == target), None)
        if last == write_key(tool, a):
            return deny("glint.duplicate_write", DUPLICATE_SOURCE, "this exact write, with the same arguments, is "
                        "the last write that ran on this record in this conversation", "Do not send it again. Tell "
                        "the customer it is already done; if they want a second, different change, send that "
                        "change's own arguments.", {"tool": tool}, consequence="Nothing changed: the earlier "
                        "identical write already took effect.",
                        facts=[f"{tool} with these arguments is the last write that ran on this record"])
        return allow("glint.duplicate_write", DUPLICATE_SOURCE)
    return gate


# ---- identifier provenance (N10) ----------------------------------------------------------------------------------

PROVENANCE_SOURCE = "Glint convention: every identifier a write names was shown by an earlier tool result"


def identifiers(args: dict) -> dict[str, str]:
    """The write's identifier arguments: every non-empty text argument whose name ends in `_id`."""
    return {k: v for k, v in sorted(args.items()) if k.endswith("_id") and isinstance(v, str) and v.strip()}


def provenance(tool: str):
    """A gate for `tool`: refused when one of its identifier arguments appeared in no earlier tool result."""
    def gate(toolkit, a: dict) -> GateResult:
        seen = getattr(toolkit, "glint_seen", None)
        if seen is None:  # no record kept (an environment the enforcer did not attach): nothing to judge on
            return allow("glint.identifier_provenance.unrecorded", PROVENANCE_SOURCE)
        text = "\n".join(seen)
        unseen = [f"{k} {v!r}" for k, v in identifiers(a).items() if v.strip() not in text]
        if unseen:
            return deny("glint.identifier_provenance", PROVENANCE_SOURCE, "an identifier in this write was not shown "
                        "by any earlier tool result in this conversation", "Look the record up first with the read "
                        "tool that lists it (the customer's accounts, cards or transactions), then send the write "
                        "with the identifier exactly as that result shows it.", {"tool": tool, "unseen": unseen},
                        consequence="Nothing changed. A write on an identifier no lookup returned may name a "
                        "record that does not exist or is not this customer's.",
                        facts=[f"not in any earlier tool result: {', '.join(unseen)}"])
        return allow("glint.identifier_provenance", PROVENANCE_SOURCE)
    return gate


def seen_before(history) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    """From a message history: each tool message id -> the earlier assistant tool results (not errors) before it, and
    all of them. The enforcer's live record keeps the same texts in the same order, so a replay judges each write on
    what the live check saw."""
    before, texts = {}, []
    for m in history or []:
        if getattr(m, "role", None) != "tool":
            continue
        before[str(m.id)] = tuple(texts)
        if counts_as_seen(m):
            texts.append(str(m.content))
    return before, tuple(texts)


def counts_as_seen(reply) -> bool:
    """A tool result counted as shown to the agent: a reply to the assistant that is not an error."""
    return (getattr(reply, "requestor", "assistant") == "assistant" and not reply.error
            and not str(reply.content).startswith("Error"))


# The records the two denies judge on, kept by enforcer hooks (glint.gates.extend).

@extend.hook("attach", order=30)
def _init_records(enforcer, env) -> None:
    env.tools.glint_ran = []  # (target, (tool, arguments)) of every write that ran, in order, for the duplicate deny
    env.tools.glint_seen = []  # every assistant tool result that is not an error, for identifier provenance (N10)
    env.glint_seen_before = {}  # in τ²'s replay: tool message id -> the results before it (seen_before)


@extend.hook("before_call", order=10)
def _replayed_seen(call: extend.Call) -> None:
    """In τ²'s replay, judge each call on the results the live check saw before it."""
    before = call.env.glint_seen_before
    if call.replay and str(call.message.id) in before:
        call.tools.glint_seen = list(before[str(call.message.id)])


@extend.hook("after_write", order=30)
def _record_ran(call: extend.Call, reply) -> None:
    if extend.ran(reply):
        call.tools.glint_ran.append((target_key(call.write[1]), write_key(*call.write)))


@extend.hook("reply", order=20)
def _record_seen(call: extend.Call, reply) -> None:
    if counts_as_seen(reply):
        call.tools.glint_seen.append(str(reply.content))
