"""The request gate's allow-and-flag tier, the filled-in record call on its refusals, the widened action map
(glint.gates.banking.requested) and scripts/request_recount.py."""
import json
from types import SimpleNamespace

import request_recount
from test_gates_phase2 import _bank, _hidden, _play, _verify

from glint.gates.banking import UNRECORDED, _request_user, requested
from glint.gates.enforce import attestation_classes
from glint.gates.eventlog import EventLog


def _table(**rows):
    return SimpleNamespace(data=rows)


def _db(verified=()):
    return SimpleNamespace(
        debit_cards=_table(dbc_1={"account_id": "chk_1", "user_id": "u1"}, dbc_2={"account_id": "chk_2", "user_id": "u1"}),
        accounts=_table(chk_1={"user_id": "u1"}, chk_2={"user_id": "u1"}),
        credit_card_accounts=_table(cc_1={"user_id": "u1"}),
        verification_history=_table(**{f"v{k}": {"user_id": u, "time_verified": "2025-11-14 03:40:00 EST"}
                                       for k, u in enumerate(verified)}))


def _gate(tool, args, record, db=None):
    return requested(tool)(SimpleNamespace(db=db or _db(), glint_requests=record), args)


def _req(action, target=""):
    return {"action": action, "target_id": target}


def test_reversible_writes_without_a_covering_request_run_and_are_flagged():
    r = _gate("log_credit_card_closure_reason_4521", {"credit_card_account_id": "cc_1", "user_id": "u1"}, None)
    assert r.allowed and r.rule_id == UNRECORDED and r.details["would_deny"] == "banking.request.no_record"
    r = _gate("order_debit_card_5739", {"account_id": "chk_1"}, [_req("pay_credit_card", "cc_1")])
    assert r.allowed and r.rule_id == UNRECORDED and r.details["would_deny"] == "banking.request.not_requested"
    r = _gate("order_debit_card_5739", {"account_id": "chk_1"}, [_req("order_debit_card", "chk_2")])
    assert r.allowed and r.details["would_deny"] == "banking.request.other_target"


def test_a_close_request_covers_the_replacement_order_on_that_cards_account():
    assert _gate("order_debit_card_5739", {"account_id": "chk_1"}, [_req("close_debit_card", "dbc_1")]).rule_id == \
        "banking.request"
    assert _gate("order_debit_card_5739", {"account_id": "chk_1"}, [_req("close_debit_card")]).rule_id == \
        "banking.request"
    for target in ("dbc_2", "dbc_unknown"):
        assert _gate("order_debit_card_5739", {"account_id": "chk_1"},
                     [_req("close_debit_card", target)]).rule_id == UNRECORDED
    # the widening is one way: an order request does not cover a close
    assert not _gate("close_debit_card_4721", {"card_id": "dbc_1"}, [_req("order_debit_card", "chk_1")]).allowed


def test_closes_payments_and_filings_stay_refused_and_carry_the_record_call():
    r = _gate("pay_credit_card_from_checking_9182", {"credit_card_account_id": "cc_1", "user_id": "u9"}, None)
    assert r.rule_id == "banking.request.no_record"
    assert "record_requests(user_id='u9', requests=[{\"action\": \"pay_credit_card\", \"target_id\": \"cc_1\"" in r.fix
    r = _gate("close_debit_card_4721", {"card_id": "dbc_1"}, [_req("close_credit_card", "dbc_1")])
    assert r.rule_id == "banking.request.not_requested"
    assert "record_requests(user_id='u1', requests=[{\"action\": \"close_debit_card\", \"target_id\": \"dbc_1\"" in r.fix
    r = _gate("close_debit_card_4721", {"card_id": "dbc_1"}, [_req("close_debit_card", "dbc_2")])
    assert r.rule_id == "banking.request.other_target" and "\"target_id\": \"dbc_1\"" in r.fix
    r = _gate("file_debit_card_transaction_dispute_6281", {"transaction_id": "btxn_1", "user_id": "u1"}, None)
    assert not r.allowed and "dispute_debit_transaction" in r.fix and "<the customer's own words>" in r.fix


def test_the_record_call_names_the_customer_from_the_args_the_records_or_the_verification():
    assert _request_user(_db(), {"user_id": "u7"}) == "u7"
    assert _request_user(_db(), {"account_id": "chk_2"}) == "u1"
    assert _request_user(_db(verified=["u3"]), {"card_id": "dbc_x", "amount": 5}) == "u3"
    assert _request_user(_db(), {"card_id": "dbc_x"}) == "<user_id>"


def test_a_flagged_write_is_attested_as_request_unrecorded_not_customer_requested():
    gates = ["banking_request_order_debit_card_5739", "procedure_verified_gate"]
    assert attestation_classes(gates, gates[:1]) == ["uncovered", "request_unrecorded"]
    assert attestation_classes(gates + ["debit_order_gate"], gates[:1]) == ["rule_checked", "request_unrecorded"]
    assert attestation_classes(gates) == ["customer_requested"]


def test_a_closure_reason_without_a_record_runs_in_the_gated_environment_and_is_flagged():
    env, task, events = _bank("task_044")
    log = _play(env, task, "log_credit_card_closure_reason_4521")
    env.tools.glint_requests = None  # as if the agent never recorded the request
    reply = _hidden(env, "log_credit_card_closure_reason_4521", **log)
    assert "was NOT executed" not in reply.content
    fired = [e["payload"] for e in events.events("gate.fired") if e["payload"]["gate"].startswith("banking_request_")]
    assert fired[-1]["rule_id"] == UNRECORDED
    assert events.events("write.attested")[-1]["payload"]["classes"][-1] == "request_unrecorded"


def test_the_open_items_say_which_writes_need_the_record():
    env, _, _ = _bank("task_084")
    _verify(env, "a7c2f85d91")
    reply = env.get_response(__import__("tau2.data_model.message", fromlist=["ToolCall"]).ToolCall(
        id="t", name="get_current_time", arguments={}, requestor="assistant"))
    assert "closes, payments and filings are refused until recorded" in reply.content


# ---- scripts/request_recount.py ------------------------------------------------------------------------------------

def _log(path):
    log = EventLog(path)
    ctx = {"simulation_id": "s1", "task_id": "task_x", "trial": 0, "replay": False}

    def fired(tool, args, rule, decision="deny", **extra):
        log.emit("gate.fired", {"gate": f"banking_request_{tool}", "tool": tool, "args": args, "rule_id": rule,
                                "decision": decision, **ctx, **extra})
    fired("close_debit_card_4721", {"card_id": "dbc_1"}, "banking.request.no_record")
    fired("close_debit_card_4721", {"card_id": "dbc_1"}, "banking.request.no_record", replay=True)  # skipped
    log.emit("gate.fired", {"gate": "procedure_verified_gate", "tool": "x", "rule_id": "r", "decision": "deny", **ctx})
    log.emit("requests.recorded", {"requests": [["close_debit_card", "dbc_1"]], **ctx})
    log.emit("requests.recorded", {"requests": [["close_debit_card", "dbc_9"]], **ctx | {"replay": True}})
    fired("order_debit_card_5739", {"account_id": "chk_1"}, "banking.request.not_requested")
    fired("log_credit_card_closure_reason_4521", {"credit_card_account_id": "cc_1", "user_id": "u1"},
          "banking.request.not_requested")
    fired("close_debit_card_4721", {"card_id": "dbc_1"}, "banking.request", "allow")
    log.close()


def test_the_recount_reruns_each_live_request_decision_with_the_record_of_its_moment(tmp_path, capsys):
    path = tmp_path / "gates-banking_knowledge.sqlite"
    _log(path)
    decisions = request_recount.request_decisions(request_recount.read_log(path))
    assert [d["record"] is None for d in decisions] == [True, False, False, False]
    t = request_recount.tally(request_recount.recount(decisions, lambda task: _db()))
    assert t["old_denies"] == 3 and t["new_denies"] == 1
    assert t["new"] == {"banking.request": 2, "banking.request.no_record": 1, UNRECORDED: 1}
    assert t["changes"] == {
        "banking.request.not_requested -> banking.request (order_debit_card_5739)": 1,
        f"banking.request.not_requested -> {UNRECORDED} (log_credit_card_closure_reason_4521)": 1}
    out = tmp_path / "t.json"
    assert request_recount.main([str(path), "--json", str(out)], db_of=lambda task: _db()) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == t
    printed = capsys.readouterr().out
    assert "refusals recorded 3, under the current rules 1" in printed and "change    banking" in printed
    assert request_recount.main([]) == 2 and request_recount.main([str(path), "--out", "x"]) == 2


def test_the_recount_builds_the_tasks_initial_database():
    db = request_recount.task_db("task_084")
    assert db.debit_cards.data["dbc_82b5d1a6f3"]["account_id"]
    assert request_recount.task_db("task_084") is db
