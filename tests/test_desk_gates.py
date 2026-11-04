"""The Bank Desk's gate events: a policy-gate deny in a recorded run becomes a card the page shows in place of the
write, and export_scenarios pairs the same trial of two sets where alone failed and the gated run passed after one."""
import json

from loguru import logger

logger.remove()

import export_scenarios
import tools_ab

from glint.desk.events import parse_deny, timeline
from glint.gates.core import DENY_PREFIX

DENY = (f"{DENY_PREFIX} and was NOT executed. Nothing changed.\n"
        "Rule: banking.close_card.no_reason (credit_card_account_logistics_003 (closing a card))\n"
        "What the check found: no closure reason is on record for this account\n"
        "Facts it used:\n- account cc_1: no closure reason logged\n- a fact: with a colon in it\n"
        "Why it matters: The policy requires the reason before a card is closed.\n"
        "What to do next: Ask why the user is closing and log it.\n"
        'Verdict: {"ok": false}')


def test_the_desk_imports_without_tau2_and_its_deny_prefix_matches_the_gates():
    """Regression 2026-10-31: events imported glint.gates, which loads τ², so the desk could not start in its own
    image (Modal, the Pages build), which has no τ²."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    from glint.desk import events

    assert events.DENY_PREFIX == DENY_PREFIX
    src = Path(__file__).resolve().parents[1] / "src"
    code = ("import sys; sys.modules['tau2'] = None\n"  # any import of τ² now fails
            "import glint.desk.app; print('ok')")
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", code], capture_output=True, text=True, encoding="utf-8",
                       env=dict(os.environ, PYTHONPATH=str(src)), timeout=120, check=False)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-1500:]


def test_a_plain_deny_parses_into_its_card_fields():
    d = parse_deny(DENY)
    assert d["rule"] == "banking.close_card.no_reason"
    assert d["source"] == "credit_card_account_logistics_003 (closing a card)"  # nested parentheses kept whole
    assert d["found"] == "no closure reason is on record for this account"
    assert d["facts"] == ["account cc_1: no closure reason logged", "a fact: with a colon in it"]
    assert d["why"].startswith("The policy requires") and d["next"] == "Ask why the user is closing and log it."


def test_a_deny_in_another_layout_keeps_its_whole_text():
    terse = f"{DENY_PREFIX} (rule r, check c) and was NOT executed.\nReason: x\nFix: y"
    d = parse_deny(terse)
    assert d["rule"] == "" and "Reason: x" in d["found"] and "Fix: y" in d["found"]


def _call(cid, name, args):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": cid, "name": name, "arguments": args}]}


def test_a_denied_call_becomes_a_gate_event_in_place_of_its_action_or_lookup():
    messages = [
        {"role": "user", "content": "close my card"},
        _call("1", "call_discoverable_agent_tool",
              {"agent_tool_name": "close_credit_card_account_7834", "arguments": json.dumps({"id": "cc_1"})}),
        {"role": "tool", "id": "1", "content": DENY},
        _call("2", "log_verification", {"name": "x"}),  # not a write: a deny on it is still a gate event
        {"role": "tool", "id": "2", "content": DENY},
        _call("3", "get_user_information_by_id", {"user_id": "u"}),
        {"role": "tool", "id": "3", "content": "found"},
    ]
    kinds = [(e["kind"], e.get("tool")) for e in timeline(messages)]
    assert kinds == [("customer", None), ("gate", "close_credit_card_account_7834"), ("gate", "log_verification"),
                     ("lookup", "get_user_information_by_id")]
    gate = timeline(messages)[1]
    assert gate["args"] == {"id": "cc_1"} and gate["next"] == "Ask why the user is closing and log it."


def test_the_desk_write_list_is_tau2_bankings_write_tools():
    from tau2.domains.banking_knowledge.environment import get_environment

    from glint.desk.events import WRITES
    from glint.gates.enforce import write_tools

    assert WRITES == write_tools(get_environment(retrieval_variant="bm25_grep").tools) - {"example_agent_tool_0000"}


def _run(make_sim, trial, kind, deny=False):
    messages = [{"role": "user", "content": "close my card"}]
    if deny:
        messages += [_call("d", "close_credit_card_account_7834", {}), {"role": "tool", "id": "d", "content": DENY}]
    return make_sim("task_050", trial, kind, messages=messages)


def test_gated_pairs_use_one_trial_where_alone_failed_and_the_gated_run_passed_after_a_deny(
        run_set, make_sim, monkeypatch, tmp_path):
    # trial 0: gated passed but with no deny; trial 1: alone passed; trial 2 is the pair
    base = [_run(make_sim, 0, "fail"), _run(make_sim, 1, "pass"), _run(make_sim, 2, "fail")]
    tools = [_run(make_sim, 0, "pass"), _run(make_sim, 1, "pass", deny=True), _run(make_sim, 2, "pass", deny=True)]
    cfg = run_set({("base", "banking_knowledge"): base, ("tools", "banking_knowledge"): tools}, k=3, name="g_modal")
    monkeypatch.setattr(tools_ab, "for_backend", lambda c, modal: cfg)
    monkeypatch.setattr(export_scenarios, "DATA", cfg["data"].parent)
    monkeypatch.setattr(export_scenarios, "SOURCES", [])
    monkeypatch.setattr(export_scenarios, "GATED_SOURCES", [
        ("task_050-gates", "task_050", ("banking-other", "base"), ("banking-other", "tools"), False, True, "t", "n")])
    monkeypatch.setattr(export_scenarios, "OUT", tmp_path / "scenarios.json")
    export_scenarios.main()
    (s,) = json.loads((tmp_path / "scenarios.json").read_text(encoding="utf-8"))
    assert (s["id"], s["task_id"], s["variant"]) == ("task_050-gates", "task_050", "gates")
    assert s["alone"]["trial"] == s["tools"]["trial"] == 2 and s["tools"]["gate_blocks"] == 1
    assert s["set_result"] == {"set": "banking-other", "pairs": 3, "alone_passed": 1, "gated_passed": 3}


def test_a_gated_pair_reads_each_arm_from_its_own_set(run_set, make_sim, monkeypatch, tmp_path):
    """task_054's pair: alone from one set's base arm, gated from another set's tools arm, counts from each."""
    alone = run_set({("base", "banking_knowledge"): [_run(make_sim, 0, "fail"), _run(make_sim, 1, "fail")]}, k=2,
                    name="a_modal")
    gated = run_set({("tools", "banking_knowledge"): [_run(make_sim, 0, "pass"), _run(make_sim, 1, "pass", deny=True)]},
                    k=2, name="b_modal")
    by_set = {"banking-other": alone, "banking-failures": gated}
    monkeypatch.setattr(tools_ab, "for_backend", lambda c, modal: by_set[next(k for k, v in tools_ab.SETS.items()
                                                                              if v is c)])
    monkeypatch.setattr(export_scenarios, "DATA", alone["data"].parent)
    monkeypatch.setattr(export_scenarios, "SOURCES", [])
    monkeypatch.setattr(export_scenarios, "GATED_SOURCES", [
        ("task_050-gates", "task_050", ("banking-other", "base"), ("banking-failures", "tools"), True, False, "t", "n")])
    monkeypatch.setattr(export_scenarios, "OUT", tmp_path / "scenarios.json")
    export_scenarios.main()
    (s,) = json.loads((tmp_path / "scenarios.json").read_text(encoding="utf-8"))
    assert s["alone"]["trial"] == s["tools"]["trial"] == 1  # trial 0's gated pass had no deny
    assert s["recorded"] == {"base": {"passed": 0, "runs": 2}, "tools": {"passed": 2, "runs": 2}}
    assert s["source"].startswith("a_modal and b_modal") and "set_result" not in s


def test_a_gated_pair_counts_the_extra_gated_sets_it_lists(run_set, make_sim, monkeypatch, tmp_path):
    """task_089's pair: gated from v10, and its recorded gated runs also count the earlier gated set (RECORDED_ALSO)."""
    alone = run_set({("base", "banking_knowledge"): [_run(make_sim, 0, "fail"), _run(make_sim, 1, "fail")],
                     ("tools", "banking_knowledge"): [_run(make_sim, 0, "fail"), _run(make_sim, 1, "pass")]}, k=2,
                    name="a_modal")
    gated = run_set({("tools", "banking_knowledge"): [_run(make_sim, 0, "pass", deny=True), _run(make_sim, 1, "fail")]},
                    k=2, name="b_modal")
    by_set = {"banking-other": alone, "banking-failures": gated}
    monkeypatch.setattr(tools_ab, "for_backend", lambda c, modal: by_set[next(k for k, v in tools_ab.SETS.items()
                                                                              if v is c)])
    monkeypatch.setattr(export_scenarios, "DATA", alone["data"].parent)
    monkeypatch.setattr(export_scenarios, "SOURCES", [])
    monkeypatch.setattr(export_scenarios, "GATED_SOURCES", [
        ("task_050-gates", "task_050", ("banking-other", "base"), ("banking-failures", "tools"), False, True, "t", "n")])
    monkeypatch.setattr(export_scenarios, "RECORDED_ALSO", {"task_050-gates": [("banking-other", "tools")]})
    monkeypatch.setattr(export_scenarios, "OUT", tmp_path / "scenarios.json")
    export_scenarios.main()
    (s,) = json.loads((tmp_path / "scenarios.json").read_text(encoding="utf-8"))
    assert s["alone"]["trial"] == s["tools"]["trial"] == 0
    assert s["recorded"] == {"base": {"passed": 0, "runs": 2}, "tools": {"passed": 2, "runs": 4}}


def test_a_gated_source_without_such_a_trial_is_left_out_loudly(run_set, make_sim, monkeypatch, tmp_path, capsys):
    sims = [_run(make_sim, 0, "pass")]
    cfg = run_set({("base", "banking_knowledge"): sims, ("tools", "banking_knowledge"): sims}, k=1, name="g_modal")
    monkeypatch.setattr(tools_ab, "for_backend", lambda c, modal: cfg)
    monkeypatch.setattr(export_scenarios, "DATA", cfg["data"].parent)
    monkeypatch.setattr(export_scenarios, "SOURCES", [])
    monkeypatch.setattr(export_scenarios, "GATED_SOURCES", [
        ("task_050-gates", "task_050", ("banking-other", "base"), ("banking-other", "tools"), False, False, "t", "n")])
    monkeypatch.setattr(export_scenarios, "OUT", tmp_path / "scenarios.json")
    export_scenarios.main()
    assert "task_050-gates left out of the replay" in capsys.readouterr().out
