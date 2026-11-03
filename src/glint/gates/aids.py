"""An agent aid and a deny-only convention on banking writes (fix research 2026-11-02, row 14 and N2).

- `credit_note` (policy, bank_accounts_(general)_017): the one-credit-per-checking-account fact, shown at the account
  reads (get_all_user_accounts_by_user_id_3847, get_bank_account_transactions_9173) with which of the customer's
  checking accounts were credited today. A function of the database, so the reply replays unchanged.
- `duplicate` (Glint convention, no clause): a write is refused when the same tool with identical arguments already
  ran in this conversation, on the enforcer's record of the writes that ran (`toolkit.glint_ran`). Transfers and the
  credit limit increase submit are exempt: τ²'s train gold repeats them (032, 034, 081 transfers; 051 submit).

The deny only refuses; it never lets a refused write run. Its record is rebuilt in τ²'s strict replay
(glint.gates.enforce): the writes that ran replay.
"""
from __future__ import annotations

import json

from glint.assist.banking import TODAY
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

DUPLICATE_EXEMPT = ("transfer_to_human_agents", "submit_credit_limit_increase_request_7392")
DUPLICATE_SOURCE = "Glint convention, no clause: a write that already ran is not sent again"


def write_key(tool: str, args: dict | None) -> tuple[str, str]:
    """A write's identity for the duplicate check: the tool and its arguments, key order ignored."""
    return tool, json.dumps(args or {}, sort_keys=True, default=str)


def duplicate(tool: str):
    """A gate for `tool`: refused when the same call (tool and identical arguments) already ran in this conversation."""
    def gate(toolkit, a: dict) -> GateResult:
        if write_key(tool, a) in getattr(toolkit, "glint_ran", set()):
            return deny("glint.duplicate_write", DUPLICATE_SOURCE, "this exact write, with the same arguments, "
                        "already ran in this conversation", "Do not send it again. Tell the customer it is already "
                        "done; if they want a second, different change, send that change's own arguments.",
                        {"tool": tool}, consequence="Nothing changed: the earlier identical write already took "
                        "effect.", facts=[f"{tool} with these arguments ran earlier in this conversation"])
        return allow("glint.duplicate_write", DUPLICATE_SOURCE)
    return gate

