"""The request gate switch (research row 6, fix-research-2026-11-02): `request_gate` False removes the banking request
gates, the request-record reminder and the record_requests tool; the default keeps all three."""
import anyio
import pytest
import tools_ab
from mcp.shared.exceptions import MCPError
from tau2.data_model.message import ToolCall
from test_gates_phase2 import _bank, _call, _hidden, _verify

from glint.assist.banking import REQUEST_TOOLS, REQUESTS_TOOL
from glint.assist.tau2_tools import drop_request_tool
from glint.bank import server as bank
from glint.gates import Enforcer, EventLog, rules
from glint.gates.replayplan import with_plans

REMINDER = "requests are not recorded yet"
CLOSE = {"card_id": "dbc_82b5d1a6f3", "reason": "fraud_suspected"}


def _request_gates(reg) -> list[str]:
    names = {f"banking_request_{tool}" for tool in REQUEST_TOOLS}
    return [g.name for g in reg.gates() if g.name in names]


def test_the_registry_leaves_out_only_the_request_gates_when_off():
    on, off = rules.build_registry("banking_knowledge"), rules.build_registry("banking_knowledge", request_gate=False)
    assert _request_gates(on) and not _request_gates(off)
    assert {g.name for g in on.gates()} - {g.name for g in off.gates()} == set(_request_gates(on))
    assert rules.REQUIRED_GATES["banking_knowledge"] <= off.covered()  # every write still has a gate
    assert {g.name for g in rules.build_registry("retail", request_gate=False).gates()} == \
        {g.name for g in rules.build_registry("retail").gates()}


def test_the_manifest_records_the_switch():
    for state in (True, False):
        events = EventLog(":memory:")
        assert Enforcer("banking_knowledge", events, request_gate=state).request_gate is state
        assert events.events("gates.manifest")[-1]["payload"]["request_gate"] is state


def test_gate_on_offers_the_tool_reminds_and_refuses_an_unrecorded_close():
    env, _, events = _bank("task_084")
    assert REQUESTS_TOOL in env.tools.get_tools()
    _verify(env, "a7c2f85d91")
    assert REMINDER in _call(env, "get_current_time").content
    reply = _hidden(env, "close_debit_card_4721", **CLOSE)
    assert reply.error and "record_requests" in reply.content
    assert any(e["payload"]["rule_id"].startswith("banking.request.") for e in events.events("gate.fired"))


def test_gate_off_drops_the_tool_the_reminder_and_the_refusal():
    events = EventLog(":memory:")
    env, _, _ = _bank("task_084", Enforcer("banking_knowledge", events, request_gate=False))
    assert REQUESTS_TOOL not in env.tools.get_tools() and not env.tools.has_tool(REQUESTS_TOOL)
    assert "plan_disputes" in env.tools.get_tools()  # the other Glint records stay
    _verify(env, "a7c2f85d91")
    assert REMINDER not in _call(env, "get_current_time").content
    assert _call(env, REQUESTS_TOOL, user_id="a7c2f85d91", requests=[]).error  # not a tool of this toolkit
    _hidden(env, "close_debit_card_4721", **CLOSE)
    fired = [e["payload"] for e in events.events("gate.fired")]
    request_gates = {f"banking_request_{tool}" for tool in REQUEST_TOOLS}
    assert fired and not any(p["gate"] in request_gates or p["rule_id"].startswith("banking.request") for p in fired)


def test_dropping_the_tool_is_idempotent():
    env, _, _ = _bank("task_084", Enforcer("banking_knowledge", EventLog(":memory:"), request_gate=False))
    cls = type(env.tools)
    assert type(drop_request_tool(env).tools) is cls


def test_with_plans_inserts_the_request_record_only_with_the_gate_on():
    call = ToolCall(id="gold-0", name="call_discoverable_agent_tool", requestor="assistant", arguments={
        "agent_tool_name": "close_debit_card_4721", "arguments": '{"card_id": "dbc_1", "reason": "lost", "user_id": "u1"}'})
    with_record = with_plans([call])
    assert [c.name for c in with_record] == [REQUESTS_TOOL, call.name]
    assert [c.name for c in with_plans([call], request_record=False)] == [call.name]


def test_tools_ab_passes_the_set_key(tmp_path):
    assert tools_ab._enforcer({}, "banking_knowledge", tmp_path / "on").request_gate is True
    assert tools_ab._enforcer({"request_gate": False}, "banking_knowledge", tmp_path / "off").request_gate is False


def test_the_bank_server_publishes_record_requests_only_with_the_gate_on(tmp_path):
    on = bank.BankServer(home=tmp_path / "on", sid="on")
    off = bank.BankServer(home=tmp_path / "off", sid="off", request_gate=False)
    assert REQUESTS_TOOL in {t.name for t in on.tool_list()}
    names = {t.name for t in off.tool_list()}
    assert REQUESTS_TOOL not in names and bank.PLAN_TOOL in names
    assert off.session().enforcer.request_gate is False
    with pytest.raises(MCPError):
        off.call(None, REQUESTS_TOOL, {"user_id": "u", "requests": []})
    decided = bank.BankServer(home=tmp_path / "dt", sid="dt", decision_tools=True, request_gate=False)
    assert REQUESTS_TOOL not in {t.name for t in decided.tool_list()}
    for s in (on, off, decided):
        s.session().events.close()


def test_the_cli_flag_turns_the_gate_off(monkeypatch):
    seen = {}

    class Fake:
        def __init__(self, decision_tools=False, request_gate=True):
            seen.update(decision_tools=decision_tools, request_gate=request_gate)

        async def run_stdio(self):
            return None

    monkeypatch.setattr(bank, "BankServer", Fake)
    assert bank.main(["--no-request-gate"]) == 0 and seen == {"decision_tools": False, "request_gate": False}
    assert bank.main([]) == 0 and seen["request_gate"] is True


def test_the_replays_run_with_the_gate_off():
    import bank_replay
    import gate_replay

    task = next(t for t in gate_replay.tasks_of("banking_knowledge") if t.id == "task_084")
    report = gate_replay.replay("banking_knowledge", [task], EventLog(":memory:"), request_gate=False)
    assert report["gold_writes"] and not report["false_refusals"]
    assert not any(r.startswith("banking.request") for r in report["rules"])
    g = anyio.run(bank_replay.gold, ["task_084"], None, False)
    assert g["gold_writes"] and not (g["false_refusals"] or g["hash_mismatches"] or g["result_mismatches"])
