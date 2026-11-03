"""The off switches of the fixes that live as module values, registered as enforcer options (glint.gates.extend).

The owner's decision of 2026-11-03: every fix built switched off (research fix-research-2026-11-02) is on by default,
each with an off switch, set through the registered options so every entry point gets the new defaults. Four of them
are module values read where the check runs; each option sets its module value when an Enforcer is built:

- `entry_wait_fraud` ("deny"; off: "flag"): glint.gates.debit.ENTRY_WAIT_FRAUD, the 48-hour entry-tier wait after a
  fraud_suspected closure (_029 vs _031).
- `credit_amount_mode` ("deny"; off: "flag" or "off"): glint.gates.fees.CREDIT_AMOUNT_MODE, the checking-credit
  amount rule (bank_accounts_(general)_017).
- `constraints` (True; off: False): glint.gates.constraints.ENABLED, the constrained-value gates from τ²'s tool
  docstrings. The gates are built before options are set up, so a change rebuilds them.
- `search_compact_after` (20; off: None): glint.assist.tau2_tools.SEARCH_COMPACT_AFTER, the knowledge-base searches
  before new searches answer with a diagnosis (appendix #72).

A module value holds for the whole process: the value of the last Enforcer built wins (Glint convention: one Enforcer
per run in tools_ab, the replays and each bank server).
"""
from __future__ import annotations

from glint.assist import tau2_tools
from glint.gates import constraints, debit, extend, fees
from glint.gates.core import assert_required
from glint.gates.rules import REQUIRED_GATES, build_registry

ENTRY_WAIT_MODES = ("flag", "deny")


def _setup_entry_wait(enforcer, mode: str) -> None:
    if mode not in ENTRY_WAIT_MODES:
        raise ValueError(f"unknown entry_wait_fraud {mode!r}; one of {ENTRY_WAIT_MODES}")
    enforcer.entry_wait_fraud = debit.ENTRY_WAIT_FRAUD = mode


def _setup_credit_amount(enforcer, mode: str) -> None:
    if mode not in fees.MODES:
        raise ValueError(f"unknown credit_amount_mode {mode!r}; one of {fees.MODES}")
    enforcer.credit_amount_mode = fees.CREDIT_AMOUNT_MODE = mode


def _setup_constraints(enforcer, on: bool) -> None:
    """Rebuilt when the gates were built under the other setting (request_gate is set up before, at order 50)."""
    enforcer.constraints = constraints.ENABLED = bool(on)
    built = any(g.name.startswith("constraints_") for g in enforcer.registry.gates())
    if enforcer.domain == "banking_knowledge" and built != constraints.ENABLED:
        enforcer.registry = build_registry(enforcer.domain, enforcer.request_gate)
        assert_required(enforcer.registry, REQUIRED_GATES[enforcer.domain], f"gates for {enforcer.domain}")


def _setup_search(enforcer, after: int | None) -> None:
    if after is not None and (isinstance(after, bool) or not isinstance(after, int) or after < 1):
        raise ValueError(f"search_compact_after {after!r}: a positive whole number, or None for off")
    enforcer.search_compact_after = tau2_tools.SEARCH_COMPACT_AFTER = after


extend.option("entry_wait_fraud", debit.ENTRY_WAIT_FRAUD, "the entry-tier wait after a fraud_suspected closure: "
              "\"deny\" (default) or \"flag\" (off)", order=60, owner=__name__, setup=_setup_entry_wait,
              manifest=lambda enforcer, mode: {"entry_wait_fraud": mode})
extend.option("credit_amount_mode", fees.CREDIT_AMOUNT_MODE, "the checking-credit amount rule: \"deny\" (default), "
              "\"flag\" or \"off\"", order=61, owner=__name__, setup=_setup_credit_amount,
              manifest=lambda enforcer, mode: {"credit_amount_mode": mode})
extend.option("constraints", constraints.ENABLED, "the constrained-value gates from τ²'s tool docstrings; on by "
              "default", order=62, owner=__name__, setup=_setup_constraints,
              manifest=lambda enforcer, on: {"constraints": bool(on)})
extend.option("search_compact_after", tau2_tools.SEARCH_COMPACT_AFTER, "knowledge-base searches before searches "
              "answer with a diagnosis; 20 by default, None is off", order=63, owner=__name__, setup=_setup_search,
              manifest=lambda enforcer, after: {"search_compact_after": after})
