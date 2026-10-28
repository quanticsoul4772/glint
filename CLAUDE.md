# Glint: project rules

Glint: MCP decision tools that take over analytical decisions for Meta's open-weight Muse Glimmer 30B on τ²-bench (Meta Global AI Developer Hackathon, Indie track; due 2026-11-04 9 PM ET). Research notes live outside this repo in `<workspace>\research`; run data in `<workspace>\measure`.

## Money (hard rules)
- **Only free compute.** Glimmer runs on Modal's free credit (`glint.glimmer_serve`). Never start RunPod or Kaggle GPU time. DeepInfra and Meta credit are nearly spent: ask before using them.
- **Every run has caps** (`glint.meter`), and Modal sessions go through `glimmer_serve.session()` (mid-session cap, retried stop, spend written every 60 s).
- **Ask before every run** with: the set, runs, k, arms, caps, expected $ and wall time.

## Experiments
- **Both arms of a comparison run on the same backend** (never mix Modal and DeepInfra runs).
- **Report by task, not by run.** Runs of one task are not independent. Keep tuned and untuned tasks apart, and never pool them into one p-value.
- **Held-out means untouched.** Once a task's failures drive a tool change, it is design data. τ²'s test split is reserved for the final check.
- **Recompute every number in a note from the data on disk** before writing it. Never read cost from a running total; `tools_ab` prints each arm's own spend.
- **A failed run's cause is in `info.error`.** A context overflow is a model failure, not an infrastructure error.

## Code and git
- **One branch per change, from `origin/main`.** Push with `scripts/push_pr.sh`: it refuses a branch whose PR is merged or closed (two commits were lost that way).
- **Before a PR:** `.venv/Scripts/python.exe -X utf8 -m pytest -q --cov --cov-report=json:coverage.json`, then `.venv/Scripts/python.exe scripts/coverage_gate.py`, then `.venv/Scripts/ruff.exe check src tests scripts --line-length 120`. CI runs the same three steps on every PR.
- **Coverage goal** (`docs/testing-strategy.md`): spend and measurement code 90%, every other module 70%, total 85%; never lower a floor. Every bug fix ships with a regression test. Tests never touch the real workspace or a paid service.
- **No secrets in the repo.** Keys live in `<workspace>\.env`; never print them.
- **Never edit the folder a run loads.** While a run is running or queued (a chained run starts each later program fresh, loading the folder's code at that moment), keep `<workspace>\glint` on `main` and develop in a worktree: `git worktree add ../glint-dev -b BRANCH origin/main`, with tests run as `PYTHONPATH=<worktree>/src .venv/Scripts/python.exe -m pytest` so they test the worktree's code.
- **Long jobs:** start them only with `scripts/launch.sh LOG -- command ...`, which detaches just the job and checks it is alive. Run checks (git pull, greps) as separate commands first; never chain them into a background `&` list, which backgrounds the checks too and loses their output. Keep the machine awake (the driver runs locally).

## Writing
- American spelling. No filler.
- State what was measured and how sure it is. Correct a wrong number in place, and say it was corrected.
