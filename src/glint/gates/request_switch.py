"""The banking request gate switch (research row 6, fix-research-2026-11-02; PR #41), a Glint convention, registered as
an enforcer option and hook (glint.gates.extend).

`Enforcer(request_gate=False)`: no banking request gates (rules.build_registry(domain, False)), no request-record
reminder among the open items of the enforcer's replies (enforce._open_items(tools, request_gate)), and record_requests
taken off every attached toolkit (tau2_tools.drop_request_tool). On by default.
"""
from __future__ import annotations

from glint.assist.tau2_tools import drop_request_tool
from glint.gates import extend
from glint.gates.core import assert_required
from glint.gates.rules import REQUIRED_GATES, build_registry


def _setup(enforcer, on: bool) -> None:
    """The gates are built before options are set up; with the switch off they are rebuilt without the request gates,
    before the manifest is written."""
    enforcer.request_gate = on
    if not on:
        enforcer.registry = build_registry(enforcer.domain, False)
        assert_required(enforcer.registry, REQUIRED_GATES[enforcer.domain], f"gates for {enforcer.domain}")


extend.option("request_gate", True, "research row 6: False drops the banking request gates, the request-record "
              "reminder and record_requests; on by default", order=50, owner=__name__, setup=_setup,
              manifest=lambda enforcer, on: {"request_gate": on})


@extend.hook("attach", order=50)
def _drop_request_tool(enforcer, env) -> None:
    if enforcer.domain == "banking_knowledge" and not enforcer.request_gate:
        drop_request_tool(env)
