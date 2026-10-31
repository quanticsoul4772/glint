"""Shared test setup.

Every test run gets its own empty workspace (GLINT_HOME), set before any glint module is imported: glint.paths reads
it at import, so the spend ledger, the desk's pair log and run data all land in a temporary folder, never in the real
workspace. (2026-10-27: the desk tests had written 12 live-run start rows into the real
desk_sessions.jsonl, which then tripped the daily pair limit.)

Fixtures build τ² results the way runs save them, so the analysis and export code is tested on known inputs.
"""
import atexit
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# The workspace lives in the repo (on D:, gitignored), not in the system temp folder on C:, and is deleted when the
# test session ends.
_ROOT = Path(__file__).resolve().parents[1]
(_ROOT / ".pytest_home").mkdir(exist_ok=True)
_HOME = Path(tempfile.mkdtemp(prefix="home-", dir=_ROOT / ".pytest_home"))
atexit.register(shutil.rmtree, _HOME, ignore_errors=True)
# the split files live in the repo (splits/) and keys are read only when a run starts, so the workspace starts empty
(_HOME / "research" / "5-measurements").mkdir(parents=True)
os.environ["GLINT_HOME"] = str(_HOME)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest


def sim(task_id: str, trial: int, kind: str = "pass", messages=None) -> dict:
    """One saved τ² simulation of the given outcome kind: pass, fail, context_overflow, glint_error, budget_stop,
    eval_error or infrastructure_error (see glint.runlog.outcome)."""
    s = {"task_id": task_id, "trial": trial, "termination_reason": "user_stop", "info": {},
         "reward_info": {"reward": 1.0 if kind == "pass" else 0.0}, "messages": messages or [], "duration": 1.0}
    errors = {"context_overflow": "litellm.ContextWindowExceededError: too long",
              "glint_error": "AtemParseError: host calls [] != generated", "budget_stop": "BudgetExceeded: cap",
              "infrastructure_error": "ConnectionError: reset"}
    if kind in errors:
        s |= {"termination_reason": "infrastructure_error", "info": {"error": errors[kind]}, "reward_info": None}
    elif kind == "eval_error":
        s["reward_info"] = None
    return s


@pytest.fixture
def make_sim():
    return sim


@pytest.fixture
def run_set(tmp_path):
    """A stand-in tools_ab set: `run_set({("base", "retail"): [sims], ...}, k=2, per_domain={"airline": 1})` writes
    the arm files and returns a cfg that tools_ab.analyze accepts."""
    def build(arms: dict, k: int, per_domain: dict | None = None, name: str = "t"):
        data = tmp_path / name
        data.mkdir(parents=True, exist_ok=True)
        for (arm, domain), sims in arms.items():
            (data / f"{arm}-{domain}.json").write_text(json.dumps({"simulations": sims}), encoding="utf-8")
        return {"data": data, "out": tmp_path / f"{name}-report.json", "trials": k, "save": name} | {
            f"{d}_trials": v for d, v in (per_domain or {}).items()}

    return build


@pytest.fixture(autouse=True)
def _restore_tau2_registry():
    """`install()` replaces τ²'s domain constructors process-wide; each test gets them back as they were, so one test's
    install cannot change another test's environments (2026-10-27: it made a later desk test assist twice)."""
    from tau2.registry import registry

    saved = dict(registry._domains)
    yield
    registry._domains.clear()
    registry._domains.update(saved)


@pytest.fixture(autouse=True)
def _paths_stay_in_the_test_home():
    """After every test glint.paths still points at the test workspace: a test that reloads it without GLINT_HOME
    and does not reload it back would send every later test to the real workspace."""
    yield
    paths = sys.modules.get("glint.paths")
    if paths is not None:
        assert paths.HOME == _HOME, f"glint.paths.HOME is {paths.HOME}, not the test workspace {_HOME}"
