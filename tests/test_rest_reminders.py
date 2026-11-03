"""Opt-in reminders and flags in the enforcer (fix-research appendix: #60 Declare reminder, #69 step-limit warning,
#68 transfer flagged when the tools cover the request) and the search diagnosis (#72). All default off."""
import json

import pytest
from loguru import logger
from tau2.data_model.message import ToolCall
from tau2.registry import registry
from test_gates_phase2 import _bank, _call, _ran, _verify

from glint import declare
from glint.assist import kb, tau2_tools
from glint.assist.tau2_tools import assist_environment
from glint.gates import Enforcer, EventLog
from glint.gates.enforce import (
    CLAUSES,
    STEP_WARN_NOTE,
    TOOLS_COVER,
    _observable,
    declare_reminder_clauses,
)

logger.remove()
USER = "u_nonexistent"


def _enforcer(**kw):
    events = EventLog(":memory:")
    return Enforcer("banking_knowledge", events, **kw), events


def _first_user(env):
    return next(iter(env.tools.db.users.data))


def test_reminders_are_off_by_default_and_unknown_ones_are_refused():
    e, events = _enforcer()
    assert e.declare_clauses == [] and e.step_warn is None
    assert events.events("gates.manifest")[-1]["payload"]["reminders"] == []
    with pytest.raises(ValueError, match="unknown reminders"):
        _enforcer(reminders=("nope",))


def test_the_declare_reminder_shows_pending_obligations_until_met():
    e, _ = _enforcer()
    e.declare_clauses = [declare.validate({"id": "T-1", "doc": "doc_t", "template": "response",
                                           "a": "log_verification", "b": "change_user_email",
                                           "correlate": [["user_id", "user_id"]]}),
                         declare.validate({"id": "T-2", "doc": "doc_t", "template": "response",
                                           "a": "log_verification", "b": "get_user_information_by_id"})]
    env, _, _ = _bank("task_087", e)
    user = _first_user(env)
    _verify(env, user)
    assert env.tools.glint_trace[-1]["tool"] == "log_verification"
    reply = _call(env, "get_user_information_by_id", user_id=user)
    # T-2's b is a plain read, which τ²'s evaluator does not replay: never shown
    assert "(T-1): after log_verification for " in reply.content and "T-2" not in reply.content
    _call(env, "change_user_email", user_id=user, new_email="x@example.com")
    assert "T-1" not in _call(env, "get_user_information_by_id", user_id=user).content


def test_the_step_warning_counts_replayed_calls_only():
    e, _ = _enforcer(step_warn=2)
    env, _, _ = _bank("task_087", e)
    user = _first_user(env)
    assert STEP_WARN_NOTE not in _call(env, "get_user_information_by_id", user_id=user).content  # a read: not counted
    assert env.tools.glint_replayed == 0
    _verify(env, user)
    assert STEP_WARN_NOTE not in _call(env, "get_user_information_by_id", user_id=user).content
    _call(env, "unlock_discoverable_agent_tool", agent_tool_name="freeze_debit_card_3892")
    assert env.tools.glint_replayed >= 2
    assert STEP_WARN_NOTE in _call(env, "get_user_information_by_id", user_id=user).content


def test_a_transfer_is_flagged_when_every_request_has_a_tool():
    e, events = _enforcer()
    env, _, _ = _bank("task_087", e)
    env.tools.glint_requests = [{"action": "close_debit_card", "target_id": "dbc_1"}]
    assert _ran(_call(env, "transfer_to_human_agents", summary="customer asked", reason="other"))
    flag = events.events("transfer.flagged")[-1]["payload"]
    assert flag["rule_id"] == TOOLS_COVER and flag["requests"] == [["close_debit_card", "dbc_1"]]
    env2, _, _ = _bank("task_087", e)
    env2.tools.glint_requests = [{"action": "something_without_a_tool", "target_id": ""}]
    n = len(events.events("transfer.flagged"))
    _call(env2, "transfer_to_human_agents", summary="customer asked", reason="other")
    env3, _, _ = _bank("task_087", e)  # no record: nothing to compare
    _call(env3, "transfer_to_human_agents", summary="customer asked", reason="other")
    assert len(events.events("transfer.flagged")) == n


def test_observable_activities_and_the_clause_file_loader(tmp_path):
    env, _, _ = _bank("task_087")
    assert _observable(env.tools, "freeze_debit_card_3892") and _observable(env.tools, "log_verification")
    assert not _observable(env.tools, "get_user_information_by_id")
    assert not _observable(env.tools, "no_such_tool")
    p = tmp_path / "rec.json"
    p.write_text(json.dumps({"agreed": [
        {"id": "a", "template": "response", "a": "x", "b": "y"},
        {"id": "b", "template": "precedence", "a": "x", "b": "y"},
        {"id": "c", "template": "existence", "a": "x", "condition_text": "when asked"}]}), encoding="utf-8")
    assert [c["id"] for c in declare_reminder_clauses(p)] == ["a"]
    if CLAUSES.exists():  # the shipped reconciled list loads and validates
        assert all(c["checkable"] for c in declare_reminder_clauses())
        assert _enforcer(reminders=("declare",))[0].declare_clauses == declare_reminder_clauses()


def test_the_trace_skips_errors_user_calls_and_unparsed_hidden_calls():
    env, _, _ = _bank("task_087")
    env.get_response(ToolCall(id="u", name="log_verification", arguments={}, requestor="user"))
    env.get_response(ToolCall(id="h", name="call_discoverable_agent_tool",
                              arguments={"agent_tool_name": "freeze_debit_card_3892", "arguments": "{bad"},
                              requestor="assistant"))
    assert env.tools.glint_trace == []


def _assisted():
    env = registry.get_env_constructor("banking_knowledge")(retrieval_variant="bm25_grep")
    return assist_environment(env, "banking_knowledge").tools


def test_search_diagnosis_is_off_by_default_and_replaces_results_after_the_budget(monkeypatch):
    tools = _assisted()
    assert "Document:" in tools.use_tool("KB_search", query="stolen card replacement")
    monkeypatch.setattr(tau2_tools, "SEARCH_COMPACT_AFTER", 1)
    second = tools.use_tool("grep", pattern="stolen|replacement")
    assert second.startswith("Search 2 in this conversation;") and "Document:" not in second
    assert "Diagnosis:" in second and "(ID: " in second
    assert tools.use_tool("grep", pattern="stolen|replacement").startswith("Same grep as your search 2")
    third = tools.use_tool("KB_search", query="dispute unauthorized fraudulent charge stolen card")
    assert "Already returned by an earlier search:" in third or "(ID: " in third


def test_diagnosis_text():
    class P:
        @staticmethod
        def get_document_title(d):
            return {"a": "Doc A"}.get(d)
    text = kb.diagnosis(P, [("a", 1.0), ("b", 0.5)], ["b"], 4)
    assert "Search 4 in this conversation; 1 distinct documents" in text
    assert "- Doc A (ID: a)" in text and "Already returned by an earlier search: b." in text
    assert "found no documents" in kb.diagnosis(P, [], [], 1)
