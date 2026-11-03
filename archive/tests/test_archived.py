"""The tests of the archived modules (archive/glint/conformal.py, archive/glint/verifiers.py), as they ran in
tests/ before the move of 2026-11-03. Kept for reference; not collected (pytest's testpaths is tests/). To run them,
put archive/ on PYTHONPATH and import the modules from there (`import conformal, verifiers`)."""
import math

import pytest
from test_field_values import CREDIT, _gold

import conformal
import verifiers


# ---- conformal ---------------------------------------------------------------------------------------------------

def test_conformal_thresholds_and_the_cascade():
    benign = [i / 10 for i in range(10)]  # 0.0 .. 0.9
    assert conformal.block_threshold(benign, 0.2) == 0.8  # ceil(11 * 0.8) = 9th smallest
    assert conformal.block_threshold(benign[:3], 0.1) == math.inf
    assert conformal.ask_threshold([0.5, 0.6, 0.7, 0.8], 0.4) == 0.6  # floor(5 * 0.4) = 2nd smallest
    assert conformal.ask_threshold([0.5], 0.1) == -math.inf
    for bad in (0, 1):
        with pytest.raises(ValueError):
            conformal.block_threshold(benign, bad)
        with pytest.raises(ValueError):
            conformal.ask_threshold(benign, bad)
    assert [conformal.cascade(s, 0.3, 0.8) for s in (0.1, 0.3, 0.8, 0.9)] == ["allow", "ask", "ask", "block"]
    with pytest.raises(ValueError):
        conformal.cascade(0.5, 0.9, 0.1)
    scores = [0.1, 0.4, 0.6, 0.9]
    lam = conformal.crc_threshold(lambda t: [1.0 if s > t else 0.0 for s in scores], [0.0, 0.5, 0.95], alpha=0.5)
    assert lam == 0.95  # (4 + 1) / 5 at 0.0 and (2 + 1) / 5 at 0.5 exceed 0.5; (0 + 1) / 5 at 0.95 does not
    assert conformal.crc_threshold(lambda t: [1.0] * 4, [0.0], alpha=0.1) is None
    with pytest.raises(ValueError):
        conformal.crc_threshold(lambda t: [2.0], [0.0], alpha=0.5)
    c = conformal.calibrate(benign, [0.85, 0.9, 0.95, 0.99], alpha=0.2, beta=0.4)
    assert c["block_at"] == 0.8 and c["ask_at"] == 0.8 and c["benign_blocked"] == 1 and c["offpolicy_allowed"] == 0


# ---- LLM stages (fakes only) --------------------------------------------------------------------------------------

def test_llm_stages_are_off_without_a_model():
    assert verifiers.provenance_guard([], 1.0, "schedule").status == "off"
    assert verifiers.draft_rules("text", "t") == []


def test_provenance_guard_with_a_fake_model():
    rows = [{"type": "atm_fee", "amount": -3.0}]
    assert verifiers.provenance_guard(rows, 3.0, "s", lambda p: '{"expected": 3.0, "why": "fee"}').status == "pass"
    assert verifiers.provenance_guard(rows, 3.0, "s", lambda p: '{"expected": 1.5}').status == "fail"
    assert verifiers.provenance_guard(rows, 3.0, "s", lambda p: "no idea").status == "unsure"
    assert verifiers.provenance_guard(rows, 3.0, "s", lambda p: '{"expected": "x"}').status == "unsure"
    assert verifiers.provenance_guard(rows, 3.0, "s", lambda p: '{"expected": 3.0,}').status == "unsure"


def test_drafted_rules_compile_and_are_accepted_only_without_gold_refusals():
    assert verifiers.compile_rule({"tool": "t", "arg": "a", "op": "bogus", "value": 1}) is None
    assert verifiers.compile_rule("not a dict") is None
    rule = verifiers.compile_rule({"tool": CREDIT, "arg": "amount", "op": "le", "value": 50})
    assert rule.allows(CREDIT, {"amount": 10}) and not rule.allows(CREDIT, {"amount": 60})
    assert rule.allows("other", {"amount": 60}) and rule.allows(CREDIT, {})
    assert not rule.allows(CREDIT, {"amount": "x"})
    gold = [(CREDIT, a) for _, _, a in _gold(CREDIT) if "amount" in a]
    probes = [(CREDIT, {"amount": 75.0})]
    verdict = verifiers.accept_drafted_rule(rule, gold, lambda t, a: True, probes)
    assert verdict["accepted"] and verdict["adds"] == probes
    strict = verifiers.compile_rule({"tool": CREDIT, "arg": "amount", "op": "lt", "value": 5})
    assert not verifiers.accept_drafted_rule(strict, gold, lambda t, a: True, probes)["accepted"]
    reply = 'Here: [{"tool": "' + CREDIT + '", "arg": "amount", "op": "gt", "value": 0, "source": "_017"}, {"op": 1}]'
    drafted = verifiers.draft_rules("text", CREDIT, lambda p: reply)
    assert len(drafted) == 1 and drafted[0].source == "_017"
    assert verifiers.draft_rules("text", CREDIT, lambda p: "[not json") == []
