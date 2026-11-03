"""Every fix on by default (owner's decision, 2026-11-03; back-on-track plan step 3), each with an off switch, through
the registered enforcer options (glint.gates.extend), and the entry points that pass them (tools_ab, the bank server).
No model, no network: a stage model here is a local fake."""
import sys
import types

import pytest
import tools_ab
from loguru import logger

from glint.assist import tau2_tools
from glint.bank import server as bank
from glint.gates import Enforcer, EventLog, build_registry, constraints, debit, extend, fees
from glint.gates import stages as stage_config
from glint.gates.conversation import TranscriptTap
from glint.gates.stages import AUTO, Stages, default_stages, resolve_model

logger.remove()


@pytest.fixture(autouse=True)
def _restore_module_values(monkeypatch):
    """An Enforcer sets the module values of its switches; each test gets them back as they were."""
    for module, name in ((debit, "ENTRY_WAIT_FRAUD"), (fees, "CREDIT_AMOUNT_MODE"), (constraints, "ENABLED"),
                         (tau2_tools, "SEARCH_COMPACT_AFTER"), (stage_config, "MODEL")):
        monkeypatch.setattr(module, name, getattr(module, name))


def _manifest(events: EventLog) -> dict:
    return events.events("gates.manifest")[-1]["payload"]


def _constraint_gates(enforcer: Enforcer) -> list[str]:
    return [g.name for g in enforcer.registry.gates() if g.name.startswith("constraints_")]


def test_every_switch_is_on_by_default():
    defaults = {o.name: o.default for o in extend.options()}
    assert {"entry_wait_fraud": "deny", "credit_amount_mode": "deny", "constraints": True,
            "uncovered_reminder": True, "reminders": ("declare",), "step_warn": 60, "search_compact_after": 20,
            "stages": AUTO}.items() <= defaults.items()
    assert (debit.ENTRY_WAIT_FRAUD, fees.CREDIT_AMOUNT_MODE, constraints.ENABLED, tau2_tools.SEARCH_COMPACT_AFTER) == (
        "deny", "deny", True, 20)
    e = Enforcer("banking_knowledge", events := EventLog(":memory:"))
    assert _constraint_gates(e) and e.stages is None  # no model configured: no stage runs
    assert {"entry_wait_fraud": "deny", "credit_amount_mode": "deny", "constraints": True,
            "search_compact_after": 20, "uncovered_reminder": True}.items() <= _manifest(events).items()


def test_the_off_switches_set_the_module_values_and_drop_the_constraint_gates():
    e = Enforcer("banking_knowledge", events := EventLog(":memory:"), entry_wait_fraud="flag",
                 credit_amount_mode="off", constraints=False, search_compact_after=None)
    assert (debit.ENTRY_WAIT_FRAUD, fees.CREDIT_AMOUNT_MODE, constraints.ENABLED, tau2_tools.SEARCH_COMPACT_AFTER) == (
        "flag", "off", False, None)
    assert _constraint_gates(e) == []
    assert {"entry_wait_fraud": "flag", "credit_amount_mode": "off", "constraints": False,
            "search_compact_after": None}.items() <= _manifest(events).items()
    # the next Enforcer with defaults turns them back on, the constraint gates included
    again = Enforcer("banking_knowledge", EventLog(":memory:"))
    assert _constraint_gates(again) and (debit.ENTRY_WAIT_FRAUD, constraints.ENABLED) == ("deny", True)
    # with the request gate off too, the rebuilt gates keep it off
    both = Enforcer("banking_knowledge", EventLog(":memory:"), request_gate=False, constraints=False)
    constraints.ENABLED = False  # restored by the fixture
    expected = [g.name for g in build_registry("banking_knowledge", False).gates()]
    assert [g.name for g in both.registry.gates()] == expected
    retail = Enforcer("retail", EventLog(":memory:"), constraints=False)
    assert _constraint_gates(retail) == []


@pytest.mark.parametrize("kw", [{"entry_wait_fraud": "off"}, {"credit_amount_mode": "strict"},
                                {"search_compact_after": 0}, {"search_compact_after": True},
                                {"search_compact_after": "20"}, {"stages": "on"}])
def test_bad_switch_values_are_refused(kw):
    with pytest.raises(ValueError):
        Enforcer("banking_knowledge", EventLog(":memory:"), **kw)


def _fake_model(prompt: str) -> str:
    return "{}"


def test_stages_run_each_configured_stage_and_nothing_without_a_model(monkeypatch):
    assert resolve_model() is None and resolve_model("") is None
    assert default_stages(None, list) == Stages() and default_stages(_fake_model, None) == Stages()
    full = default_stages(_fake_model, list)
    assert full.enabled and full.describe()["checklist"] == "flag" and full.describe()["alignment"] == "flag"
    assert full.describe()["cascade"] == list(stage_config.CASCADE_THRESHOLDS) and full.approver is None
    fake = types.ModuleType("glint_fake_stage_model")
    fake.call, fake.value = _fake_model, 3
    monkeypatch.setitem(sys.modules, "glint_fake_stage_model", fake)
    assert resolve_model("glint_fake_stage_model:call") is _fake_model
    with pytest.raises(ValueError, match="module:function"):
        resolve_model("glint_fake_stage_model")
    with pytest.raises(TypeError, match="not callable"):
        resolve_model("glint_fake_stage_model:value")
    # a model configured but no transcript (the enforcer's own default): still nothing runs
    monkeypatch.setattr(stage_config, "MODEL", "glint_fake_stage_model:call")
    assert Enforcer("banking_knowledge", EventLog(":memory:")).stages is None
    e = Enforcer("banking_knowledge", events := EventLog(":memory:"), stages=full)
    assert e.stages is not None and _manifest(events)["stages"]["checklist"] == "flag"


def test_the_tap_keeps_each_conversation_apart():
    current = ["a"]
    tap = TranscriptTap(key=lambda: current[0])
    completion = tap.wrap(lambda *a, **k: "reply")
    completion(model="m", messages=[{"role": "user", "content": "from a"}])
    current[0] = "b"
    assert tap() == []
    completion("m", [{"role": "user", "content": "from b"}])
    assert [t["content"] for t in tap()] == ["from b"]
    current[0] = "a"
    assert [t["content"] for t in tap()] == ["from a"]


def test_tools_ab_passes_the_defaults_and_each_off_key(tmp_path, monkeypatch):
    got = []
    monkeypatch.setattr(tools_ab, "Enforcer", lambda *a, **kw: got.append(kw) or kw)
    monkeypatch.setattr(tools_ab, "EventLog", lambda path: path)
    assert tools_ab._enforcer({"gates": False}, "banking_knowledge", tmp_path) is None
    tools_ab._enforcer({}, "banking_knowledge", tmp_path)
    assert {"reminders": ("declare",), "step_warn": 60, "uncovered_reminder": True, "stages": Stages(),
            "search_compact_after": 20}.items() <= got[-1].items()
    tools_ab._enforcer({"reminders": [], "step_warn": None, "uncovered_reminder": False, "stages": False,
                        "search_compact_after": None}, "banking_knowledge", tmp_path)
    assert {"reminders": (), "step_warn": None, "uncovered_reminder": False, "stages": None,
            "search_compact_after": None}.items() <= got[-1].items()
    fake = types.ModuleType("glint_fake_stage_model")
    fake.call = _fake_model
    monkeypatch.setitem(sys.modules, "glint_fake_stage_model", fake)
    tap = TranscriptTap()
    cfg = {"stage_model": "glint_fake_stage_model:call"}
    assert tools_ab._stage_model(cfg) is _fake_model
    assert tools_ab._stage_model(cfg | {"stages": False}) is None
    assert tools_ab._stage_model(cfg | {"gates": False}) is None
    tools_ab._enforcer(cfg, "banking_knowledge", tmp_path, tap)
    assert got[-1]["stages"].transcript is tap
    assert got[-1]["stages"].describe() == default_stages(_fake_model, tap).describe()


def test_bank_server_passes_the_defaults_and_its_off_flags(tmp_path, monkeypatch):
    monkeypatch.setenv("GLINT_BANK_HOME", str(tmp_path / "bank-home"))
    on = bank.BankServer(sid="on").session()
    assert on.enforcer.uncovered_reminder is True and on.enforcer.stages is None
    assert _manifest(on.events)["uncovered_reminder"] is True
    off = bank.BankServer(sid="off", uncovered_reminder=False, stages=False).session()
    assert off.enforcer.uncovered_reminder is False and "uncovered_reminder" not in _manifest(off.events)
    served = []

    async def run_stdio(self):
        served.append((self.uncovered_reminder, self.stages))

    monkeypatch.setattr(bank.BankServer, "run_stdio", run_stdio)
    assert bank.main([]) == 0 and bank.main(["--no-uncovered-reminder", "--no-stages"]) == 0
    assert served == [(True, True), (False, False)]
