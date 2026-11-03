"""glint.integrations.llamafirewall_scanner: the gate-decision mapping with fakes, the scanner factory with a fake
llamafirewall module (the real package is not installed in CI: it pulls in torch and pins another mcp), and one
real Enforcer on a τ² banking training task. A live LlamaFirewall run is not tested here (docs/llamafirewall.md)."""
import asyncio
import enum
import json
import sys
import types
from dataclasses import dataclass

import pytest
from tau2.data_model.message import ToolCall
from test_gates_phase2 import _bank, _call, _play  # the shared τ² banking setup

from glint.gates import Enforcer, EventLog, allow, deny
from glint.gates.rules import ADDRESS_ASK, EXPEDITE_ASK
from glint.integrations import llamafirewall_scanner as lfs


def _deny(rule_id: str):
    return deny(rule_id, "doc_x", "the check found a problem", "Do the next step.", consequence="Nothing ran.",
                facts=["fact one"])


class FakeEnforcer:
    """write_call treats every tool whose name starts with `write_` as a gated write; check returns the result set
    for that tool (allow by default) and records what it was asked."""
    wording = "plain"

    def __init__(self, results: dict | None = None):
        self.results, self.checked = results or {}, []

    def write_call(self, env, call: ToolCall):
        return (call.name, call.arguments) if call.name.startswith("write_") else None

    def check(self, tool, args, db, replay=False, toolkit=None):
        self.checked.append((tool, args, db, replay, toolkit))
        return self.results.get(tool, allow("rule_ok", "doc_ok"))


class FakeEnv:
    def __init__(self, enforcer=None):
        self.tools = types.SimpleNamespace(db="the-db")
        self.glint_enforcer = enforcer


def _openai(name: str, args, call_id: str = "c1") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def test_parse_reads_the_openai_and_the_flat_shape():
    call = lfs.parse_tool_call(_openai("write_a", {"x": 1}))
    assert (call.id, call.name, call.arguments, call.requestor) == ("c1", "write_a", {"x": 1}, "assistant")
    flat = lfs.parse_tool_call({"name": "read_b", "arguments": {"y": 2}}, index=3)
    assert (flat.id, flat.name, flat.arguments) == ("llamafirewall-3", "read_b", {"y": 2})
    assert lfs.parse_tool_call({"name": "read_c"}).arguments == {}


@pytest.mark.parametrize("args", ["not json", "[1, 2]", 7])
def test_parse_refuses_arguments_that_are_not_an_object(args):
    assert lfs.parse_tool_call({"name": "write_a", "function": {"name": "write_a", "arguments": args}}) is None


def test_refusal_blocks_with_the_agents_message():
    verdict = lfs.to_verdict(_deny("banking.some_rule"))
    assert verdict.decision == lfs.BLOCK and verdict.score == 1.0
    assert verdict.reason == _deny("banking.some_rule").message("plain")
    assert "What to do next: Do the next step." in verdict.reason
    assert lfs.to_verdict(_deny("r"), "terse").reason.startswith("Error: this write was blocked")
    assert "Fix: Do the next step." in lfs.to_verdict(_deny("r"), "terse").reason


@pytest.mark.parametrize("rule", [EXPEDITE_ASK, ADDRESS_ASK])
def test_an_ask_rule_is_a_hold_for_human_review(rule):
    verdict = lfs.to_verdict(_deny(rule))
    assert verdict.decision == lfs.HOLD and rule in verdict.reason


def test_allow_allows_with_the_rule():
    verdict = lfs.to_verdict(allow("rule_ok", "doc_ok"))
    assert verdict == lfs.Verdict(lfs.ALLOW, "allowed by Glint rule rule_ok (doc_ok)", 0.0)


def test_judge_call_checks_writes_on_the_live_database_and_passes_reads():
    enforcer, env = FakeEnforcer(), FakeEnv()
    assert lfs.judge_call(enforcer, env, ToolCall(id="1", name="read_a", arguments={})).decision == lfs.ALLOW
    assert enforcer.checked == []
    assert lfs.judge_call(enforcer, env, ToolCall(id="2", name="write_a", arguments={"k": 1})).decision == lfs.ALLOW
    assert enforcer.checked == [("write_a", {"k": 1}, "the-db", False, env.tools)]


def test_message_first_block_wins_then_hold_then_allow():
    enforcer = FakeEnforcer({"write_block": _deny("banking.r"), "write_hold": _deny(ADDRESS_ASK)})
    env = FakeEnv()
    calls = [_openai("write_ok", {}), _openai("write_hold", {}), _openai("write_block", {}), _openai("write_z", {})]
    assert lfs.judge_tool_calls(enforcer, env, calls).decision == lfs.BLOCK
    assert [c[0] for c in enforcer.checked] == ["write_ok", "write_hold", "write_block"]  # stops at the block
    assert lfs.judge_tool_calls(enforcer, env, calls[:2]).decision == lfs.HOLD
    assert lfs.judge_tool_calls(enforcer, env, calls[:1]) == lfs.Verdict(lfs.ALLOW, "no gated write was refused",
                                                                         0.0)
    assert lfs.judge_tool_calls(enforcer, env, None).decision == lfs.ALLOW


def test_unparsed_arguments_block_without_a_check():
    enforcer = FakeEnforcer()
    bad = {"id": "c", "function": {"name": "write_a", "arguments": "{oops"}}
    verdict = lfs.judge_tool_calls(enforcer, FakeEnv(), [_openai("read_a", {}), bad])
    assert verdict.decision == lfs.BLOCK and "tool call 1" in verdict.reason and enforcer.checked == []


# --- the factory, against a fake llamafirewall module with 1.0.3's names and shapes -------------------------------

class _Role(enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"


class _ScanDecision(enum.Enum):
    ALLOW = "allow"
    HUMAN_IN_THE_LOOP_REQUIRED = "human_in_the_loop_required"
    BLOCK = "block"


@dataclass
class _ScanResult:
    decision: _ScanDecision
    reason: str
    score: float


class _Scanner:
    def __init__(self, scanner_name: str, block_threshold: float = 1.0) -> None:
        self.name = scanner_name


@dataclass
class _Message:
    role: _Role
    content: str
    tool_calls: list | None = None


@pytest.fixture
def fake_llamafirewall(monkeypatch):
    registry = {}
    module = types.ModuleType("llamafirewall")
    module.Role, module.ScanDecision, module.ScanResult, module.Scanner = _Role, _ScanDecision, _ScanResult, _Scanner

    def register_llamafirewall_scanner(name):
        def decorator(cls):
            registry[name] = cls
            return cls
        return decorator

    module.register_llamafirewall_scanner = register_llamafirewall_scanner
    monkeypatch.setitem(sys.modules, "llamafirewall", module)
    return registry


def test_factory_registers_a_no_argument_scanner_that_maps_decisions(fake_llamafirewall):
    enforcer = FakeEnforcer({"write_block": _deny("banking.r"), "write_hold": _deny(EXPEDITE_ASK)})
    cls = lfs.make_scanner(enforcer, FakeEnv(enforcer), name="glint_test")
    assert fake_llamafirewall["glint_test"] is cls
    scanner = cls()  # LlamaFirewall's create_scanner calls scanner_class() with no arguments
    assert scanner.name == "glint_test"

    def scan(role, calls=None):
        return asyncio.run(scanner.scan(_Message(role, "text", calls), None))

    assert scan(_Role.ASSISTANT, [_openai("write_block", {})]).decision == _ScanDecision.BLOCK
    held = scan(_Role.ASSISTANT, [_openai("write_hold", {})])
    assert held.decision == _ScanDecision.HUMAN_IN_THE_LOOP_REQUIRED and held.score == 1.0
    assert scan(_Role.ASSISTANT, [_openai("write_ok", {})]).decision == _ScanDecision.ALLOW
    user = scan(_Role.USER, [_openai("write_block", {})])
    assert user.decision == _ScanDecision.ALLOW and "assistant tool calls only" in user.reason


def test_factory_refuses_an_environment_this_enforcer_does_not_gate(fake_llamafirewall):
    with pytest.raises(ValueError, match="enforcer.attach"):
        lfs.make_scanner(FakeEnforcer(), FakeEnv(enforcer=None))
    assert fake_llamafirewall == {}


# --- one real Enforcer on a banking training task -----------------------------------------------------------------

def test_real_enforcer_blocks_a_write_before_verification_and_changes_nothing():
    env, task, events = _bank("task_081")
    enforcer = env.glint_enforcer
    card = env.tools.db.debit_cards.data["dbc_tm92c4d7e8_blue"]
    status = card["status"]
    _call(env, "unlock_discoverable_agent_tool", agent_tool_name="freeze_debit_card_3892")
    freeze = {"id": "f1", "function": {"name": "call_discoverable_agent_tool", "arguments": json.dumps(
        {"agent_tool_name": "freeze_debit_card_3892", "arguments": json.dumps({"card_id": "dbc_tm92c4d7e8_blue"})})}}
    verdict = lfs.judge_tool_calls(enforcer, env, [freeze])
    assert verdict.decision == lfs.BLOCK and "has not been verified" in verdict.reason
    assert env.tools.db.debit_cards.data["dbc_tm92c4d7e8_blue"]["status"] == status  # nothing ran
    assert events.events("gate.fired")[-1]["payload"]["decision"] == "deny"
    read = {"name": "get_user_information_by_id", "arguments": {"user_id": "x"}}
    assert lfs.judge_tool_calls(enforcer, env, [read]).decision == lfs.ALLOW
    _play(env, task, "freeze_debit_card_3892")  # gold up to the freeze: the user is verified now
    assert lfs.judge_tool_calls(enforcer, env, [freeze]).decision == lfs.ALLOW
    assert env.tools.db.debit_cards.data["dbc_tm92c4d7e8_blue"]["status"] == status  # a scan never runs the write
    assert isinstance(enforcer, Enforcer) and isinstance(events, EventLog)
