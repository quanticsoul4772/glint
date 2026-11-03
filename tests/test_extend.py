"""The enforcer's extension points (glint.gates.extend): registered options build Enforcer's keywords and the
gates.manifest event, and hooks run in (order, name) order at fixed points of the gated wrapper."""
import json

import pytest
from loguru import logger
from tau2.data_model.message import ToolCall, ToolMessage
from tau2.data_model.tasks import Task
from tau2.registry import registry

from glint import paths
from glint.gates import Enforcer, EventLog, aids, extend

logger.remove()
WRAPPER = "call_discoverable_agent_tool"
TASKS = {t["id"]: t for t in json.loads((paths.tau2_data() / "tau2" / "domains" / "banking_knowledge" /
                                         "tasks.json").read_text(encoding="utf-8"))}


def _manifest(events: EventLog) -> dict:
    return events.events("gates.manifest")[-1]["payload"]


def _reply(content: str = "ok", error: bool = False) -> ToolMessage:
    return ToolMessage(id="m", content=content, requestor="assistant", role="tool", error=error)


def test_the_existing_keywords_are_registered_options_with_their_defaults_and_one_line_docs():
    opts = {o.name: o for o in extend.options()}
    # every fix on by default (owner's decision, 2026-11-03); stages "auto" runs nothing with no model configured
    assert {"wording": "plain", "stages": "auto", "uncovered_reminder": True}.items() <= {
        n: o.default for n, o in opts.items()}.items()
    assert [o.name for o in extend.options()][:3] == ["wording", "stages", "uncovered_reminder"]
    assert all(o.doc and "\n" not in o.doc for o in opts.values())
    enforcer = Enforcer("retail", events := EventLog(":memory:"))
    assert (enforcer.wording, enforcer.stages, enforcer.uncovered_reminder) == ("plain", None, True)
    payload = _manifest(events)
    assert payload["wording"] == "plain" and "stages" not in payload and payload["uncovered_reminder"] is True
    off = Enforcer("retail", events := EventLog(":memory:"), stages=None, uncovered_reminder=False)
    assert (off.stages, off.uncovered_reminder) == (None, False) and "uncovered_reminder" not in _manifest(events)


def test_an_unknown_keyword_is_refused_and_a_bad_wording_still_raises():
    with pytest.raises(TypeError, match="unknown options"):
        Enforcer("retail", EventLog(":memory:"), no_such_option=True)
    with pytest.raises(ValueError, match="unknown deny wording"):
        Enforcer("retail", EventLog(":memory:"), wording="loud")


def test_a_registered_option_sets_the_attribute_and_its_manifest_keys(monkeypatch):
    monkeypatch.setattr(extend, "OPTIONS", dict(extend.OPTIONS))
    extend.option("demo_level", 1, "a test option", order=99, owner=__name__,
                  manifest=lambda enforcer, v: {"demo_level": v})
    seen = []
    extend.option("demo_setup", "x", "a test option with a setup", order=98, owner=__name__,
                  setup=lambda enforcer, v: seen.append((v, enforcer.demo_level if hasattr(enforcer, "demo_level")
                                                         else None)))
    enforcer = Enforcer("retail", events := EventLog(":memory:"), demo_level=3)
    assert enforcer.demo_level == 3 and _manifest(events)["demo_level"] == 3
    assert seen == [("x", None)]  # set up in order: 98 before 99
    assert not hasattr(enforcer, "demo_setup")  # a setup owns the attributes


def test_option_registration_refuses_a_second_owner_and_a_multiline_doc(monkeypatch):
    monkeypatch.setattr(extend, "OPTIONS", dict(extend.OPTIONS))
    with pytest.raises(ValueError, match="already registered"):
        extend.option("wording", "plain", "another wording", order=1, owner="elsewhere")
    with pytest.raises(ValueError, match="one-line doc"):
        extend.option("demo", 0, "two\nlines", order=1, owner=__name__)
    first = extend.OPTIONS["wording"]
    extend.option("wording", "plain", "reloaded", order=10, owner=first.owner)  # a reload replaces
    assert extend.OPTIONS["wording"].doc == "reloaded"


def test_hooks_run_by_order_then_name_whatever_the_registration_order(monkeypatch):
    monkeypatch.setattr(extend, "HOOKS", {p: {} for p in extend.PHASES})
    calls = []
    for name, order in (("c", 20), ("b", 10), ("a", 20)):
        extend.hook("before_call", order, name)(lambda call, n=name: calls.append(n))
    extend.run("before_call", None)
    assert calls == ["b", "a", "c"]
    with pytest.raises(ValueError, match="unknown hook phase"):
        extend.hook("no_phase", 1)


def test_a_reply_hook_replaces_the_reply_or_keeps_it_with_none(monkeypatch):
    monkeypatch.setattr(extend, "HOOKS", {p: {} for p in extend.PHASES})
    extend.hook("reply", 1, "keep")(lambda call, reply: None)
    extend.hook("reply", 2, "tag")(lambda call, reply: reply.model_copy(update={"content": reply.content + "+t"}))
    extend.hook("reply", 3, "tag2")(lambda call, reply: reply.model_copy(update={"content": reply.content + "+u"}))
    assert extend.run_reply("reply", None, _reply()).content == "ok+t+u"
    assert extend.ran(_reply()) and not extend.ran(_reply(error=True)) and not extend.ran(_reply("Error: x"))


def test_the_built_in_hooks_keep_the_order_the_wrapper_had():
    names = {p: [h.fn.__name__ for h in extend.hooks(p)] for p in extend.PHASES}
    assert names["after_write"] == ["_allow_note", "_increase_record", "_record_ran", "_uncovered_write",
                                    "_trace_write", "_flag_transfer"]
    assert names["reply"] == ["_with_open_items", "_record_seen"]
    assert names["after_read"] == ["_trace_read", "_after_read_records"]
    assert names["before_call"] == ["_count_replayed", "_replayed_seen"]
    assert names["before_write"] == ["_increase_snapshot"] and names["open_items"] == ["_extra_items"]
    # enforce, stages, aids, reminders, request_switch
    assert names["attach"] == ["_init_uncovered", "_init_records", "_init_records", "_init_records",
                               "_drop_request_tool"]
    assert [o.name for o in extend.options()][3:6] == ["reminders", "step_warn", "request_gate"]


def test_open_items_and_reply_hooks_reach_a_banking_reply_and_get_the_call(monkeypatch):
    monkeypatch.setattr(extend, "HOOKS", {p: dict(h) for p, h in extend.HOOKS.items()})
    got = []
    extend.hook("open_items", 20, "demo")(lambda call: ["[open] demo item"])
    extend.hook("after_read", 5, "demo")(lambda call, reply: got.append((call.message.id, call.write, call.replay)))
    extend.hook("reply", 30, "demo")(lambda call, reply: reply.model_copy(update={"content": reply.content + "|end"}))
    task = Task.model_validate(TASKS["task_072"])
    env = registry.get_env_constructor("banking_knowledge")(retrieval_variant="bm25_grep", task=task)
    Enforcer("banking_knowledge", EventLog(":memory:")).attach(env)
    init = task.initial_state
    env.set_state(init.initialization_data if init else None, init.initialization_actions if init else None, [],
                  strict=True)
    owner = next(iter(env.tools.db.accounts.data.values()))["user_id"]
    env.get_response(ToolCall(id="u1", name="unlock_discoverable_agent_tool", requestor="assistant",
                              arguments={"agent_tool_name": aids.ACCOUNTS_READ}))
    read = env.get_response(ToolCall(id="r1", name=WRAPPER, requestor="assistant", arguments={
        "agent_tool_name": aids.ACCOUNTS_READ, "arguments": json.dumps({"user_id": owner})}))
    assert read.content.endswith("[open] demo item|end")
    assert got[-1] == ("r1", None, False)
    assert env.tools.glint_seen[-1].endswith("[open] demo item")  # seen (order 20) ran before the order-30 hook
