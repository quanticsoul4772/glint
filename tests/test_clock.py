"""The injectable clock (glint.clock, Enforcer(clock=...)): the default is τ²'s environment time, so recorded and
replayed decisions are unchanged; another clock changes every date-dependent decision (technical judge, round 2,
2.3: the gates ran only on τ²'s frozen date)."""
from datetime import date, datetime

import pytest
from tau2.data_model.message import ToolCall
from tau2.domains.banking_knowledge.tools import KnowledgeTools
from tau2.domains.banking_knowledge.utils import get_now
from test_gates_phase2 import _bank, _call, _denied, _ran

from glint import clock
from glint.assist import banking as assist
from glint.gates import Enforcer, EventLog, Gate, allow
from glint.gates import procedure as proc
from glint.gates.bankdb import days_since, within_year
from glint.gates.core import ACCOUNT_RECORDS, run_gate

LATER = clock.Clock(datetime(2026, 11, 3, 9, 0, 0), "EST")
USER = "a7c2f85d91"  # task_084's customer


def test_the_default_clock_is_tau2s_environment_time():
    assert clock.TAU2_BANKING.now == get_now()
    assert KnowledgeTools.get_current_time(None) == f"The current time is {clock.TAU2_BANKING.text}."
    assert (clock.today(), clock.now_text()) == (date(2025, 11, 14), "2025-11-14 03:40:00 EST")


def test_using_sets_the_clock_for_the_block_and_restores_it():
    with clock.using(LATER) as c:
        assert c is LATER and clock.current() is LATER and clock.now_text() == "2026-11-03 09:00:00 EST"
    assert clock.current() is clock.TAU2_BANKING
    with pytest.raises(TypeError, match="glint.clock.Clock"), clock.using("2026-11-03"):
        pass
    wall = clock.wall_clock("UTC")
    assert wall.zone == "UTC" and wall.now.tzinfo is None and wall.now.microsecond == 0


def test_date_rules_follow_the_clock_in_effect():
    assert (days_since("11/14/2025"), within_year("2025-01-02"), assist.waiver_expires()) == (0, True, "11/14/2026")
    with clock.using(LATER):
        assert (days_since("11/14/2025"), within_year("2025-01-02"), assist.waiver_expires()) == (
            354, False, "11/03/2027")


def test_a_gate_runs_under_the_callers_clock_in_the_pool_thread():
    gate = Gate("today", ("t",), "p", lambda db, a: allow("r", "s", {"today": clock.today()}), checks=ACCOUNT_RECORDS)
    assert run_gate(gate, None, {})[0].details["today"] == date(2025, 11, 14)
    with clock.using(LATER):
        assert run_gate(gate, None, {}, wait=True)[0].details["today"] == date(2026, 11, 3)


def test_the_enforcer_defaults_to_tau2s_clock_and_refuses_a_clock_of_another_type():
    enforcer = Enforcer("banking_knowledge", events := EventLog(":memory:"))
    assert enforcer.clock is clock.TAU2_BANKING
    assert "clock" not in events.events("gates.manifest")[-1]["payload"]  # recorded manifests are unchanged
    with pytest.raises(TypeError, match="glint.clock.Clock"):
        Enforcer("banking_knowledge", EventLog(":memory:"), clock="2026-11-03")


def _verification_call(env, text: str) -> ToolCall:
    user = env.tools.db.users.data[USER]
    return ToolCall(id="v", name="log_verification", requestor="assistant", arguments={
        "time_verified": text, **{k: user[k] for k in ("name", "user_id", "address", "email", "phone_number",
                                                       "date_of_birth")}})


def test_the_hooks_precheck_reads_the_enforcers_clock():
    """Enforcer.precheck (glint-bank's Muse Code PreToolUse hook) decides under the same clock as get_response."""
    env, _, _ = _bank("task_084", Enforcer("banking_knowledge", EventLog(":memory:"), clock=LATER))
    _call(env, "get_user_information_by_name", customer_name=env.tools.db.users.data[USER]["name"])
    refused = env.glint_enforcer.precheck(env, _verification_call(env, clock.TAU2_BANKING.text))
    assert _denied(refused, "is not the current time '2026-11-03 09:00:00 EST'")
    assert env.glint_enforcer.precheck(env, _verification_call(env, LATER.text)) is None
    assert clock.current() is clock.TAU2_BANKING


def _verify_at(env, text: str):
    user = env.tools.db.users.data[USER]
    _call(env, "get_user_information_by_name", customer_name=user["name"])
    return _call(env, "log_verification", time_verified=text,
                 **{k: user[k] for k in ("name", "user_id", "address", "email", "phone_number", "date_of_birth")})


def test_a_different_injected_date_changes_the_date_dependent_decisions():
    """Under the default clock τ²'s time verifies the customer. Under an injected clock the same call is refused (its
    time is not the current time), the injected time verifies, and the verified gate reads that day's record."""
    env, _, _ = _bank("task_084")
    assert _ran(_verify_at(env, clock.TAU2_BANKING.text))

    enforcer = Enforcer("banking_knowledge", events := EventLog(":memory:"), clock=LATER)
    assert events.events("gates.manifest")[-1]["payload"]["clock"] == LATER.text
    env, _, _ = _bank("task_084", enforcer)
    refused = _verify_at(env, clock.TAU2_BANKING.text)
    assert _denied(refused, "is not the current time '2026-11-03 09:00:00 EST'")
    assert clock.current() is clock.TAU2_BANKING  # the enforcer's clock does not leak past the call
    assert _ran(_verify_at(env, LATER.text))
    fired = [e["payload"]["decision"] for e in events.events("gate.fired")
             if e["payload"]["gate"] == "banking_log_verification_gate"]
    assert fired == ["deny", "allow"]
    assert not proc.verified(env.tools.db, {"user_id": USER}).allowed  # the default day has no record
    with clock.using(LATER):
        assert proc.verified(env.tools.db, {"user_id": USER}).allowed
    assert enforcer.check("change_user_email", {"user_id": USER, "new_email": "x@example.com"}, env.tools.db,
                          toolkit=env.tools).rule_id != "procedure.verified"
