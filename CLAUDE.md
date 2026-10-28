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
- **Before a PR:** `.venv/Scripts/python.exe -X utf8 -m pytest -q` and `.venv/Scripts/ruff.exe check src tests scripts --line-length 120`.
- **No secrets in the repo.** Keys live in `<workspace>\.env`; never print them.
- **Long jobs:** run them detached with a log under `measure/data/`, watch the log, and keep the machine awake (the driver runs locally).

## Writing
- American spelling. No filler.
- State what was measured and how sure it is. Correct a wrong number in place, and say it was corrected.
