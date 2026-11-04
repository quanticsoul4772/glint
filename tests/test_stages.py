"""The optional stages around the write gates (glint.gates.conversation, glint.gates.stages) and the uncovered-write
reminder (research fix-research-2026-11-02 appendix #8-#12, #15, #16, #26). Every model, approver and transcript here
is a fake: no paid service is called. Only TRAIN tasks are used."""
import json
from types import SimpleNamespace

import pytest
from tau2.data_model.message import ToolCall
from test_gates_phase2 import _bank, _call, _hidden, _play

from glint.gates import conversation as conv
from glint.gates.enforce import Enforcer
from glint.gates.eventlog import EventLog
from glint.gates.rules import ADDRESS_ASK, parse_hidden_arguments
from glint.gates.stages import REQUEST_RULES, ConversationAllowlist, StageRunner, Stages
from glint.integrations import llamafirewall_scanner as lfs

PAY = "pay_credit_card_from_checking_9182"


def _turns(*pairs):
    return [{"role": r, "content": c} for r, c in pairs]


def _model(reply):
    calls = []

    def model(prompt):
        calls.append(prompt)
        return reply(prompt) if callable(reply) else reply
    model.calls = calls
    return model


# ---- glint.gates.conversation ----------------------------------------------------------------------------------------

def test_turns_keep_customer_and_agent_text_only_and_the_tap_keeps_the_latest_messages():
    class Msg:
        role, content = "assistant", "hello"
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}, {"role": "tool", "content": "{}"},
            {"role": "assistant", "content": None}, {"role": "user", "content": "  "}, Msg()]
    assert conv.turns(msgs) == _turns(("user", "hi"), ("assistant", "hello"))
    tap = conv.TranscriptTap()
    assert tap() == []
    completion = tap.wrap(lambda *a, **k: "reply")
    assert completion("model", [{"role": "user", "content": "first"}]) == "reply"
    assert tap() == _turns(("user", "first"))
    completion(model="m", messages=[{"role": "user", "content": "second"}])
    completion(model="m")  # no messages: the last list stays
    assert tap() == _turns(("user", "second"))


def test_parse_json_reads_fenced_replies_and_refuses_unreadable_ones():
    assert conv.parse_json('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert conv.parse_json('sure: {"p": 0.5} done') == {"p": 0.5}
    with pytest.raises(ValueError, match="no JSON"):
        conv.parse_json("no idea")
    with pytest.raises(ValueError, match="unreadable"):
        conv.parse_json("{broken")


def test_the_extractor_keeps_only_known_actions_the_customer_said():
    talk = _turns(("user", "Please pay my credit card cc_1 from checking."), ("assistant", "Shall I also close it?"))
    model = _model(json.dumps([
        {"action": "pay_credit_card", "target_id": "cc_1", "customer_words": "pay my credit card"},
        {"action": "pay_credit_card", "target_id": "cc_1", "customer_words": "Pay my credit card"},  # duplicate
        {"action": "close_credit_card", "target_id": "cc_1", "customer_words": "also close it"},  # the agent said it
        {"action": "fly_to_mars", "target_id": "", "customer_words": "pay"}, "junk"]))
    assert conv.extract_requests(talk, model) == [
        {"action": "pay_credit_card", "target_id": "cc_1", "customer_words": "pay my credit card"}]
    assert "pay_credit_card: credit_card_account_id" in model.calls[0] and "[0] customer:" in model.calls[0]
    with pytest.raises(TypeError, match="not a list"):
        conv.extract_requests(talk, _model('{"action": "x"}'))
    assert not conv.said_by_customer(talk, "")


def test_affirms_and_yes_no():
    assert conv.affirms("Yes, that's right.") and conv.affirms("ok go ahead")
    assert not conv.affirms("No, ship it to my work address instead") and not conv.affirms("Hmm.")
    assert not conv.affirms("yes but wait")
    assert conv.yes_no("Yes.") and not conv.yes_no("no, not really")
    with pytest.raises(ValueError):
        conv.yes_no("maybe")


def test_address_confirmed_needs_the_agent_to_name_the_street_and_the_customer_to_agree():
    on_file = "12 Elm Street, Springfield, IL 62701"
    yes = _turns(("assistant", "Ship to 12 Elm Street, Springfield?"), ("user", "Yes, that's right."))
    assert conv.address_confirmed(yes, on_file)
    assert not conv.address_confirmed(yes[:1], on_file)  # no reply yet
    assert not conv.address_confirmed(_turns(("assistant", "Ship to 12 Elm Street?"), ("user", "No, work address")),
                                      on_file)
    assert not conv.address_confirmed(_turns(("assistant", "Where to?"), ("user", "Yes")), on_file)
    assert not conv.address_confirmed(yes, "")
    assert conv.address_confirmed(yes, on_file, _model("yes"))
    assert not conv.address_confirmed(yes, on_file, _model("no"))


def test_the_ask_question_names_the_target_and_confirmation_reads_the_reply_to_the_last_question():
    q = conv.ask_question(PAY, {"user_id": "u1", "credit_card_account_id": "cc_1", "amount": 5}, "no record")
    assert "on cc_1" in q and "no record" in q
    assert "on " not in conv.ask_question(PAY, {"amount": 5}, "x").split(":")[0]
    talk = _turns(("assistant", "Do you want me to go ahead?"), ("user", "Yes please do"))
    assert conv.customer_confirmed(talk, q)
    assert not conv.customer_confirmed(talk[:1], q)
    assert not conv.customer_confirmed(_turns(("assistant", "Done."), ("user", "yes")), q)
    model = _model("no")
    assert not conv.customer_confirmed(talk, q, model) and q in model.calls[0]


def test_the_checklist_verifier_reads_each_item_and_treats_missing_answers_as_unsure():
    tool = "file_debit_card_transaction_dispute_6281"
    v = conv.checklist_verify([], tool, {}, _model('{"possession_asked": "met", "pin_asked": "kinda"}'))
    assert v == {"met": ["possession_asked"], "unmet": [], "unsure": ["pin_asked"]}
    v = conv.checklist_verify([], tool, {}, _model('{"possession_asked": "UNMET"}'))
    assert v["unmet"] == ["possession_asked"] and v["unsure"] == ["pin_asked"]
    assert conv.checklist_verify([], "no_such_tool", {}, _model("{}")) == {"met": [], "unmet": [], "unsure": []}
    with pytest.raises(TypeError, match="not an object"):
        conv.checklist_verify([], tool, {}, _model("[1]"))
    assert all(i.source for items in conv.CHECKLISTS.values() for i in items)


def test_the_alignment_check_needs_a_boolean():
    talk = _turns(("user", "pay my card"), ("assistant", "sure"))
    model = _model('{"aligned": false, "reason": "nobody asked to close"}')
    v = conv.alignment_check(talk, [(PAY, {"amount": 5})], "close_credit_card_account_7834", {}, model)
    assert v == {"aligned": False, "reason": "nobody asked to close"}
    assert "- pay my card" in model.calls[0] and "sure" not in model.calls[0].split("Customer turns:")[1].split(
        "Changes")[0]
    with pytest.raises(TypeError, match="boolean"):
        conv.alignment_check(talk, [], PAY, {}, _model('{"aligned": "yes"}'))


def test_the_binomial_bound_and_the_calibrated_threshold():
    assert conv.binomial_upper(0, 0) == 1.0 and conv.binomial_upper(5, 5) == 1.0
    assert conv.binomial_upper(0, 59) == pytest.approx(1 - 0.05 ** (1 / 59), abs=1e-6)  # the rule of three
    assert conv.binomial_upper(1, 100) == pytest.approx(0.0466, abs=1e-3)
    with pytest.raises(ValueError):
        conv.binomial_upper(3, 2)
    scores = [0.1] + [0.9] * 199
    assert conv.calibrate(scores, 0.05) == 0.9  # blocking the one 0.1 score keeps the bound under 5%
    assert conv.calibrate(scores, 0.02) == 0.1  # it does not keep it under 2%: block nothing
    with pytest.raises(ValueError, match="label more"):
        conv.calibrate([0.5] * 10, 0.05)


def test_the_cascade_allows_asks_or_denies_by_score():
    c = conv.Cascade(lambda t, tool, a: a["p"], deny_below=0.2, allow_at=0.8)
    assert c.decide([], PAY, {"p": 0.9}) == ("allow", 0.9)
    assert c.decide([], PAY, {"p": 0.5})[0] == "ask" and c.decide([], PAY, {"p": 0.1})[0] == "deny"
    with pytest.raises(ValueError, match="outside"):
        c.decide([], PAY, {"p": 2})
    with pytest.raises(ValueError):
        conv.Cascade(c.score, 0.9, 0.1)
    score = conv.model_score(_model('{"p": 0.7}'))
    assert score([], PAY, {}) == 0.7
    with pytest.raises(TypeError, match="number"):
        conv.model_score(_model('{"p": "high"}'))([], PAY, {})


# ---- glint.gates.stages ----------------------------------------------------------------------------------------------

def test_stages_are_off_by_default_and_refuse_a_reading_stage_without_a_transcript():
    assert not Stages().enabled
    with pytest.raises(ValueError, match="transcript"):
        Stages(extractor=_model("[]"))
    with pytest.raises(ValueError, match="checklist_mode"):
        Stages(checklist_mode="maybe")
    with pytest.raises(ValueError, match="together"):
        Stages(approver=lambda *a: True)
    s = Stages(transcript=list, ask_rules=REQUEST_RULES)
    assert s.enabled and s.describe()["ask_rules"] == list(REQUEST_RULES)
    events = EventLog(":memory:")
    assert Enforcer("banking_knowledge", events, stages=Stages()).stages is None
    assert "stages" not in events.events("gates.manifest")[-1]["payload"]


def _toolkit(**extra):
    tk = SimpleNamespace(db=SimpleNamespace(users=SimpleNamespace(data={})), glint_requests=None,
                         glint_address_asked=set(), **extra)
    StageRunner.init(tk)
    return tk


def _runner(stages, context=None):
    events = EventLog(":memory:")
    return StageRunner(stages, events, context or (lambda: {"task_id": "t", "trial": 0})), events


def test_a_live_answer_is_kept_and_read_back_in_replay_without_calling_the_model():
    model = _model('{"possession_asked": "unmet", "pin_asked": "met"}')
    runner, events = _runner(Stages(transcript=list, checklist=model, checklist_mode="deny"))
    tool, args = "file_debit_card_transaction_dispute_6281", {"card_id": "dbc_1"}
    live = runner.review(_toolkit(), tool, args, replay=False)[0]
    assert live.rule_id == "stage.checklist" and "physical card" in live.fix
    replayed = runner.review(_toolkit(), tool, args, replay=True)[0]  # a fresh environment, as τ²'s evaluator builds
    assert replayed == live and len(model.calls) == 1
    assert runner.review(_toolkit(), tool, {"card_id": "other"}, replay=True) == (None, [])  # never asked live
    assert events.events("stage.flagged")[0]["payload"]["stage"] == "checklist"


def test_flag_mode_verifiers_log_and_attest_and_failures_follow_the_mode():
    tool = "file_debit_card_transaction_dispute_6281"
    runner, events = _runner(Stages(transcript=list, checklist=_model('{"possession_asked": "met"}')))
    assert runner.review(_toolkit(), tool, {}, False) == (None, ["conversation_flagged"])  # pin_asked unsure
    runner, _ = _runner(Stages(transcript=list, checklist=_model('{"possession_asked": "unmet", "pin_asked": "met"}')))
    assert runner.review(_toolkit(), tool, {}, False) == (None, ["conversation_flagged"])
    runner, _ = _runner(Stages(transcript=list, checklist=_model('{"possession_asked": "met", "pin_asked": "met"}')))
    assert runner.review(_toolkit(), tool, {}, False) == (None, [])
    runner, events = _runner(Stages(transcript=list, checklist=_model("garbage")))
    assert runner.review(_toolkit(), tool, {}, False) == (None, ["conversation_flagged"])
    assert "no JSON" in events.events("stage.error")[0]["payload"]["error"]
    runner, _ = _runner(Stages(transcript=list, checklist=_model("garbage"), checklist_mode="deny"))
    assert runner.review(_toolkit(), tool, {}, False)[0].rule_id == "stage.checklist.cannot_decide"


def test_the_alignment_stage_sees_the_writes_allowed_before():
    seen = []

    def model(prompt):
        seen.append(prompt)
        return '{"aligned": %s, "reason": "off goal"}' % ("false" if "close" in prompt.split("Proposed:")[1] else "true")
    runner, _ = _runner(Stages(transcript=lambda: _turns(("user", "pay my card")), alignment=model,
                               alignment_mode="deny"))
    tk = _toolkit()
    assert runner.review(tk, PAY, {"amount": 5}, False) == (None, [])
    refusal = runner.review(tk, "close_credit_card_account_7834", {}, False)[0]
    assert refusal.rule_id == "stage.alignment" and "off goal" in refusal.fix
    assert PAY in seen[1] and tk.glint_stage_trace == [(PAY, {"amount": 5})]
    runner, _ = _runner(Stages(transcript=list, alignment=_model('{"aligned": false}')))
    assert runner.review(_toolkit(), PAY, {}, False) == (None, ["conversation_flagged"])


def test_the_approver_and_the_per_conversation_allowlist():
    allow_list = ConversationAllowlist()
    ctx = {"task_id": "t", "trial": 0}
    allow_list.grant(ctx, PAY, "cc_1")
    allow_list.grant(ctx, "close_debit_card_4721")
    assert allow_list(PAY, {"credit_card_account_id": "cc_1"}, ctx)
    assert not allow_list(PAY, {"credit_card_account_id": "cc_2"}, ctx)
    assert allow_list("close_debit_card_4721", {"card_id": "any"}, ctx)
    assert not allow_list(PAY, {"credit_card_account_id": "cc_1"}, {"task_id": "t", "trial": 1})
    runner, _ = _runner(Stages(approver=allow_list, approval_tools=(PAY,)), context=lambda: dict(ctx))
    assert runner.review(_toolkit(), PAY, {"credit_card_account_id": "cc_1"}, False) == (None, ["approved"])
    refusal = runner.review(_toolkit(), PAY, {"credit_card_account_id": "cc_2"}, False)[0]
    assert refusal.rule_id == "stage.approval.refused"
    assert runner.review(_toolkit(), "other_tool", {}, False) == (None, [])
    runner, _ = _runner(Stages(approver=lambda *a: 1 / 0, approval_tools=(PAY,)))
    assert runner.review(_toolkit(), PAY, {}, False)[0].rule_id == "stage.approval.refused"  # an error is no approval


def test_the_cascade_stage_allows_denies_or_asks_once():
    talk = []
    runner, events = _runner(Stages(transcript=lambda: talk, cascade=conv.Cascade(lambda t, tool, a: a["p"], 0.2, 0.8)))
    assert runner.review(_toolkit(), PAY, {"p": 0.9}, False) == (None, [])
    assert runner.review(_toolkit(), PAY, {"p": 0.1}, False)[0].rule_id == "stage.cascade"
    tk = _toolkit()
    asked = runner.review(tk, PAY, {"p": 0.5}, False)[0]
    assert asked.rule_id == "stage.cascade.unsure.ask" and "Ask the customer" in asked.fix
    assert runner.review(tk, PAY, {"p": 0.5}, False)[0].rule_id == "stage.cascade.unsure.not_confirmed"
    talk += _turns(("assistant", "Shall I go ahead?"), ("user", "yes"))
    assert runner.review(tk, PAY, {"p": 0.5}, False) == (None, ["asked_confirmed"])
    runner, _ = _runner(Stages(transcript=list, cascade=conv.Cascade(lambda *a: 5, 0.2, 0.8)))
    assert runner.review(_toolkit(), PAY, {}, False)[0].rule_id == "stage.cascade.cannot_decide"
    assert [e["payload"]["decision"] for e in events.events("stage.cascade")] == ["allow", "deny", "ask", "ask", "ask"]


def test_prime_builds_the_record_and_marks_a_confirmed_address():
    users = {"u1": {"address": "12 Elm Street, Springfield"}}
    talk = _turns(("user", "I want to pay my credit card"), ("assistant", "Ship to 12 Elm Street?"), ("user", "yes"))
    extractor = _model('[{"action": "pay_credit_card", "target_id": "cc_1", "customer_words": "pay my credit card"}]')
    runner, events = _runner(Stages(transcript=lambda: talk, extractor=extractor, address_check=True))
    tk = _toolkit()
    tk.db.users.data = users
    runner.prime(tk, PAY, {"credit_card_account_id": "cc_1"}, False)
    assert tk.glint_requests[0]["action"] == "pay_credit_card"
    runner.prime(tk, PAY, {"credit_card_account_id": "cc_1"}, False)  # a record exists: no second extraction
    assert len(extractor.calls) == 1
    runner.prime(tk, "order_replacement_credit_card_7291", {"user_id": "u1", "credit_card_account_id": "cc_1"}, False)
    assert tk.glint_address_asked == {"cc_1"}
    assert {e["type"] for e in events.events()} >= {"requests.extracted", "address.confirmed_in_conversation"}
    runner, _ = _runner(Stages(transcript=list, extractor=_model("[]")))
    tk = _toolkit()
    runner.prime(tk, PAY, {}, False)
    assert tk.glint_requests is None  # nothing extracted: the gate decides as without the stage


# ---- through the gated environment (τ²'s train tasks) ----------------------------------------------------------------

def _gated(task_id, **kw):
    events = EventLog(":memory:")
    env, task, _ = _bank(task_id, Enforcer("banking_knowledge", events, **kw))
    return env, task, events


def test_an_ask_rule_turns_the_request_refusal_into_a_question_and_the_agreed_repeat_runs():
    talk = []
    env, task, events = _gated("task_051", stages=Stages(transcript=lambda: talk, ask_rules=REQUEST_RULES))
    args = _play(env, task, PAY)
    env.tools.glint_requests = None  # as if the agent never recorded the request
    first = _hidden(env, PAY, **args)
    assert first.error and "banking.request.no_record.ask" in first.content and "Ask the customer" in first.content
    talk += _turns(("assistant", "Do you want me to make this payment?"), ("user", "Yes, go ahead."))
    second = _hidden(env, PAY, **args)
    assert "was NOT executed" not in second.content
    assert "asked_confirmed" in events.events("write.attested")[-1]["payload"]["classes"]
    assert events.events("stage.ask_confirmed")


def _asked_twice():
    """task_051's payment with no request record, in an environment whose stages ask (#16) with a fake yes-model and
    review with a fake flag-mode alignment model; the customer already said yes to the agent's last question."""
    talk = _turns(("assistant", "Do you want me to make this payment?"), ("user", "Yes, go ahead."))
    confirm, align = _model("yes"), _model('{"aligned": true, "reason": ""}')
    env, task, events = _gated("task_051", stages=Stages(transcript=lambda: talk, ask_rules=REQUEST_RULES,
                                                         confirm_model=confirm, alignment=align))
    args = _play(env, task, PAY)
    env.tools.glint_requests = None
    _call(env, "unlock_discoverable_agent_tool", agent_tool_name=PAY)
    call = ToolCall(id="w", name="call_discoverable_agent_tool", requestor="assistant",
                    arguments={"agent_tool_name": PAY, "arguments": json.dumps(args)})
    return env, events, call, confirm, align


def _asked_key(call):
    """The call as #16 keeps it once asked (StageRunner._ask)."""
    args = parse_hidden_arguments(call.arguments["arguments"])
    return PAY, json.dumps(args, sort_keys=True, default=str), "banking.request.no_record"


def _stage_records(env):
    t = env.tools
    return (dict(t.glint_stage_seen), set(t.glint_stage_asked), list(t.glint_stage_trace), t.glint_requests,
            set(t.glint_address_asked))


def test_the_hooks_precheck_decides_as_the_logged_check_alone():
    """Code review 2026-11-03: precheck's unlogged check ran the stages too, so the ask marked the call asked and the
    logged check that followed confirmed it from the customer's earlier yes: the hook let through a write the server
    would have turned into a question. Now the dry check leaves the stage records as they were."""
    alone_env, alone_events, call, _, _ = _asked_twice()
    alone = alone_env.get_response(call)
    env, events, call, confirm, _ = _asked_twice()
    before = _stage_records(env)
    refusal = env.glint_enforcer.precheck(env, call)
    assert refusal is not None and refusal.content == alone.content
    assert "banking.request.no_record.ask" in refusal.content
    assert _stage_records(env) == _stage_records(alone_env) != before  # the logged check's records, once
    rules = [e["payload"]["rule_id"] for e in events.events("gate.fired")]
    assert rules == [e["payload"]["rule_id"] for e in alone_events.events("gate.fired")]
    # the same call again (the customer agreed) runs, with the confirm model asked once
    assert "was NOT executed" not in env.get_response(call).content and len(confirm.calls) == 1


def test_the_scanners_dry_check_leaves_no_stage_trace_and_asks_each_model_once():
    """The LlamaFirewall scanner's check (record=False) on an allowed write: the stage records are unchanged, no stage
    event is logged, and the write that then runs reads the scan's model answers instead of asking again; the
    write's records and events are those of the write run alone."""
    alone_env, alone_events, call, _, _ = _asked_twice()
    alone_env.tools.glint_stage_asked.add(_asked_key(call))
    alone_env.get_response(call)  # asked before: the customer's yes confirms it
    env, events, call, confirm, align = _asked_twice()
    env.tools.glint_stage_asked.add(_asked_key(call))
    before, kinds = _stage_records(env), [e["type"] for e in events.events()]
    asked = (len(confirm.calls), len(align.calls))  # the gold writes before it were reviewed too
    verdict = lfs.judge_call(env.glint_enforcer, env, call)
    assert verdict.decision == lfs.ALLOW, verdict.reason
    assert _stage_records(env) == before and [e["type"] for e in events.events()] == kinds
    assert "was NOT executed" not in env.get_response(call).content
    assert (len(confirm.calls) - asked[0], len(align.calls) - asked[1]) == (1, 1)
    assert _stage_records(env) == _stage_records(alone_env)
    stage_events = [e["type"] for e in events.events() if e["type"].startswith("stage.")]
    assert stage_events == [e["type"] for e in alone_events.events() if e["type"].startswith("stage.")]
    assert stage_events.count("stage.ask_confirmed") == 1


def test_a_dry_checks_model_error_is_logged_once_when_the_write_runs():
    runner, events = _runner(Stages(transcript=list, alignment=_model("not json"), alignment_mode="deny"))
    tk, tool, args = _toolkit(), "close_credit_card_1234", {"credit_card_account_id": "cc_1"}
    with runner.dry(tk):
        dry = runner.review(tk, tool, args, replay=False)[0]
    assert dry.rule_id == "stage.alignment.cannot_decide" and not events.events("stage.error")
    assert tk.glint_stage_seen == {} and runner.review(tk, tool, args, replay=False)[0] == dry
    assert len(events.events("stage.error")) == 1


def test_without_stages_the_same_write_is_refused_as_before():
    env, task, _ = _gated("task_051")
    args = _play(env, task, PAY)
    env.tools.glint_requests = None
    assert "banking.request.no_record" in _hidden(env, PAY, **args).content


def test_the_extractor_fills_the_record_in_the_gated_environment():
    talk = _turns(("user", "I'd like to pay my credit card balance from checking"))
    extractor = _model(lambda p: json.dumps([{"action": "pay_credit_card", "target_id": "",
                                              "customer_words": "pay my credit card balance"}]))
    env, task, events = _gated("task_051", stages=Stages(transcript=lambda: talk, extractor=extractor))
    args = _play(env, task, PAY)
    env.tools.glint_requests = None
    assert "was NOT executed" not in _hidden(env, PAY, **args).content
    assert events.events("requests.extracted")


def test_the_approver_refusal_is_logged_as_a_gate_decision():
    env, task, events = _gated("task_051", stages=Stages(approver=lambda *a: False, approval_tools=(PAY,)))
    args = _play(env, task, PAY)
    reply = _hidden(env, PAY, **args)
    assert reply.error and "stage.approval.refused" in reply.content
    assert events.events("gate.fired")[-1]["payload"]["gate"] == "stage_stage.approval.refused"


def test_a_confirmed_address_skips_the_address_hold():
    env, task, events = _gated("task_037", stages=Stages(transcript=lambda: talk, address_check=True))
    args = _play(env, task, "order_replacement_credit_card_7291")
    street = env.tools.db.users.data[args["user_id"]]["address"].split(",")[0]
    talk = _turns(("assistant", f"Shall I ship it to {street}?"), ("user", "Yes, that's right."))
    _hidden(env, "order_replacement_credit_card_7291", **args)
    rules = [e["payload"]["rule_id"] for e in events.events("gate.fired")
             if e["payload"]["tool"] == "order_replacement_credit_card_7291"]
    assert ADDRESS_ASK not in rules and events.events("address.confirmed_in_conversation")


def test_the_uncovered_reminder_lists_no_rule_writes_that_ran_and_is_off_by_default():
    for on in (True, False):
        env, task, events = _gated("task_051", uncovered_reminder=on)
        args = _play(env, task, PAY)
        reply = _hidden(env, PAY, **args)
        assert "was NOT executed" not in reply.content
        assert ("[review] pay_credit_card_from_checking_9182 ran on" in reply.content) is on
        assert ("uncovered_reminder" in events.events("gates.manifest")[-1]["payload"]) is on
