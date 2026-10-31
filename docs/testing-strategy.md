# Glint test strategy

## Why the tests exist, and what matters most
Glint makes two kinds of claims a bug can break: what it **spends** (Modal GPU time, Meta and DeepInfra credit) and what it **measured** (the numbers in the README and on the Results tab). The tests protect those first. On 2026-10-27 a refactor moved the tool-call count inside the wrong block of `tools_ab.analyze`, and only a new pipeline test caught it.

## Coverage goal
1. **Spend and measurement code: 90% coverage of lines and branches** (`branch = true`: every if/else arm counts).
   - Spend: `meter`, `glimmer_serve`, `desk/live`.
   - Measurement: `runlog`, `calllog`, `tools_ab`, `export_results`, `export_scenarios`, `gate_labels`, `wrong_writes`.
2. **Every other module: at least 70%.**
3. **Total: at least 85%** across the library and the measurement pipeline. Two kinds of code are left out:
   - the finished one-off pilot and check scripts;
   - code that runs only inside Modal (`glimmer_modal`, `desk/modal_app`).

   The `omit` list in `pyproject.toml` names them.
4. **No regressions.** A module below its goal is held at a floor (its measured number, rounded down) in `scripts/coverage_gate.py`. Floors only go up; a module's entry is deleted when it reaches its goal.
5. **Enforced in CI on every PR** (`.github/workflows/tests.yml`). The PR fails when the total drops below 85% or any module drops below its floor, and the message names the module and its number.

## What a test may and may not touch
- **Never the real workspace.** `tests/conftest.py` gives each run its own `GLINT_HOME` under `.pytest_home/` (on D:, gitignored) before any glint module is imported. That workspace is deleted at exit. pytest's `tmp_path` folders go to `.pytest_tmp/`.

  Rule: no test writes to the real workspace (`GLINT_HOME`). The desk tests once wrote 12 live-run rows into the real `desk_sessions.jsonl`.
- **Never a paid or remote service.**
  - No Modal, GPU, DeepInfra, Meta API or Baseten calls.
  - The modal CLI is a stand-in for `subprocess.run`; HTTP goes through `httpx.MockTransport`; τ²'s batch is a stand-in `_run`; LiteLLM loggers are called directly.
  - No keys: scripts read them only when a run starts (`m2_pilot.require_keys()`), and the test workspace has no `.env`.
- **No dependence on the clock or ordering.** Background-thread code (watchers, meter loops) is tested by calling its step function (`tick()`, `poll()`) directly, or with short intervals and a deadline.
- **Fixed inputs live in `tests/data/`.** The task split partitions (`splits/`) and the recorded wrong-write runs (`data-public/`) are in the repo itself, since the scripts read them too.

## The levels
1. **Unit.** Pure functions with known answers: outcome classification, price math, verdict rules, parsing (`atem`), timeline building (`desk/events`).
2. **Component.** One module with its edges replaced:
   - the Modal session with a fake CLI;
   - the meter with a broken ledger;
   - the desk with fake panes;
   - the runner's `generate` with a fake τ² batch.

   These carry the fail-fast checks. Each failure path asserts that it fails loudly: the exception, the message, and the state left behind (for example, no arm file after a cap stop).
3. **Pipeline.** The measurement path from saved τ² runs to the published numbers: `tools_ab.analyze` → `export_results` / `export_scenarios`, and `gate_labels`.
   - Inputs come from the `make_sim` and `run_set` fixtures, one simulation per outcome kind.
   - Every counting rule in the project rules has a test: report by task; context overflows are failures; infrastructure, evaluation and cap stops are excluded and counted; Glint errors count against the tools; trials beyond k are left out and noted.
4. **Tool correctness.** The decision tools are checked against τ²'s gold actions on recorded tasks (`test_assist`, `test_decisions`). A tool fix comes with the task that exposed it. Tools are frozen after 2026-10-27; a held-out defect is reported in the notes, not fixed in place.
5. **Live checks are not tests.** Deploys and runs are verified by the `deploy-desk` and `run-ab` skills (the five URL checks; the approval block and monitor). They are never part of `pytest`.

## Rules for every change
- **Every bug fix ships with a regression test.** The test reproduces the failure, and a comment names the date or incident.
- **Every new fail-fast path gets a test** that triggers it and checks the message.
- **Tests read like the code they test.** Use the shared fixtures in `conftest.py`, not new one-off scaffolding; extend the fixtures when a new shape is needed.
- **When a change lowers a module's coverage,** add tests in the same PR. Never lower a floor.

## Running it
```bash
uv sync --extra dev --python 3.12                                     # the worktree's own .venv
.venv/Scripts/python.exe -X utf8 -m pytest -q --cov --cov-report=term --cov-report=json:coverage.json
.venv/Scripts/python.exe scripts/coverage_gate.py
.venv/Scripts/ruff.exe check src tests scripts --line-length 120
```
`.python-version` pins Python 3.12: τ² imports `audioop`, which 3.13 removed.

## Out of scope
- τ²-bench itself (pinned at v1.0.1).
- The one-off pilot scripts (finished; they will not run again).
- Code that runs only in Modal containers. The serving wrapper around it (`glimmer_serve`) is in scope and tested with a fake CLI.
