# Glint: project rules

Glint: deterministic policy gates that check every write of Meta's open-weight Muse Glimmer 30B against a bank's written policy, as code, before it runs, on τ²-bench, with a hash-chained gate log (Meta Global AI Developer Hackathon, Indie track). Working files (run data, ledger, results, keys) live outside the repo, under `GLINT_HOME`.

## Experiments
- **Every paid run has caps** (`glint.meter`); Modal sessions go through `glimmer_serve.session()` (mid-session cap, retried stop, spend written every 60 s).
- **Both arms of a comparison run on the same backend** (never mix Modal and DeepInfra runs).
- **Report by task, not by run.** Runs of one task are not independent. Keep tuned and untuned tasks apart, and never pool them into one p-value.
- **Held-out means untouched.** Once a task's failures drive a tool change, it is design data. τ²'s test split is reserved for the final check.
- **Recompute every number in a note from the data on disk** before writing it. Never read cost from a running total; `tools_ab` prints each arm's own spend.
- **A failed run's cause is in `info.error`.** A context overflow is a model failure, not an infrastructure error.

## Code and git
- **One branch per change, from `origin/main`.** Push with `scripts/push_pr.sh`: it refuses a branch whose PR is merged or closed (two commits were lost that way).
- **Before a PR:** `python -X utf8 -m pytest -q --cov --cov-report=json:coverage.json`, then `python scripts/coverage_gate.py`, then `ruff check src tests scripts --line-length 120`. CI runs the same three steps on every PR.
- **Coverage goal** (`docs/testing-strategy.md`): spend and measurement code 90%, every other module 70%, total 85%; never lower a floor. Every bug fix ships with a regression test. Tests never touch the real workspace or a paid service.
- **No secrets in the repo.** Keys stay in the workspace's `.env`; never print them.
- **Never edit the checkout a run loads.** A chained run starts each later program fresh and loads the folder's code at that moment; develop in a git worktree instead.
- **Long jobs:** start them only with `scripts/launch.sh LOG -- command ...`, which detaches just the job and checks it is alive. Run checks (git pull, greps) as separate commands first; never chain them into a background `&` list, which backgrounds the checks too and loses their output.

## Writing
- American spelling. No filler. No first person in docs ("what was measured", not "what we measured").
- State what was measured and how sure it is. Correct a wrong number in place, and say it was corrected.
