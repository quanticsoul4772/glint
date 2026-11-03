# Scripts

Run them from the repo root with `uv run python scripts/NAME.py` (each script's docstring has its usage). Scripts that call a model or a GPU have spend caps (`glint.meter`); none of them run in CI.

## Current: gates, comparisons and results
| Script | What it does |
|---|---|
| `tools_ab.py` | Glimmer with vs without Glint on the same tasks and seeds; the tools arm enforces the write gates unless the set says otherwise. Checkpointed; `--analyze` reports by task |
| `status.py` | Where every comparison set stands: progress per arm, outcomes, error causes, spend, ETA |
| `gate_replay.py` | False-refusal replay: τ²'s gold actions for every training task through the gates |
| `gate_catch.py` | Replays recorded failed runs through the gates: would the first wrong write have been stopped? |
| `gate_summary.py` | The per-arm gate summary line for a set's data folder |
| `bank_replay.py` | glint-bank replays through the MCP tool-call path (gold actions and recorded wrong writes) |
| `plan_sim.py` | Would dispute planning have set gold's provisional-credit flags in recorded runs? |
| `wrong_writes.py` | Writes per arm that Glint's checks judge wrong, in every with/without comparison |
| `export_results.py` | Regenerates the Bank Desk results tab (`src/glint/desk/results.json`) |
| `export_scenarios.py` | Regenerates the Bank Desk replay pairs (`src/glint/desk/scenarios.json`) |
| `lf_scan.py` | Glint's gates as a LlamaFirewall 1.0.3 custom scanner on task_054: a $4,000 increase is `BLOCK`, $2,500 is `ALLOW` (the 2026-11-03 run in the demo video). Needs `llamafirewall==1.0.3` in a separate venv; not run in CI |
| `muse_check.py` | Did every expected MCP server start in a Muse Code run? |
| `serve_smoke.py` | One tool-call request to Glimmer served on Modal, after the endpoint is ready |
| `partition.py` | The one-time split of τ²'s train tasks (pilot / fit / reserve); the files it wrote are fixed |

## Development
| Script | What it does |
|---|---|
| `coverage_gate.py` | Per-module coverage floors, run after `pytest --cov` (also in CI) |
| `push_pr.sh` | Pushes the current branch only if its PR is still open |
| `pr_edit.sh` | Edits an open PR's title or body (REST); refuses a merged or closed PR |
| `wt.sh` | Runs pytest or a script against this checkout's `src` with the shared venv, from any worktree |
| `wait_run.sh` | Waits for a launch.sh job's log to show the job ended, then prints the ending lines (run in the background) |
| `launch.sh` | Starts one long job detached and checks it is alive |

## Measurement history (Oct 26–27)
These produced the measurements that led to the current design: the first plan routed hard decisions by reading Glimmer's activations, then the decision tools, then the gates. They are kept so every earlier number can be reproduced; they are not part of the current pipeline.

| Script | What it did |
|---|---|
| `label_shipped.py` | M0: Glint's step labeler on τ²'s shipped retail trajectories |
| `usersim_check.py` | Does Spark Open on Baseten work as τ²'s user simulator and judge? |
| `m1_prefix_check.py` | M1: token-prefix and tokenization checks |
| `m2_pilot.py` | M2: the retail pilot (its model settings and key check are shared by the later runners) |

The M0 dry run, the other M1 checks (`m1_deepinfra_checks.py`, `m1_c2_rerun.py`) and the M2 judge pass (`m2_judge.py`) are in `archive/scripts/` (`archive/README.md`).
| `planb_k16.py` | Plan B: all 74 retail train tasks at k=16 |
| `headroom_pilot.py`, `banking_glimmer.py`, `cost_check.py` | Does Glimmer fail often enough, and Muse Wonder do better, for routing to matter (banking and retail)? |
| `twin_pilot.py`, `unedited_rerun.py` | How often Glimmer contradicts a preference the user stated |
| `build_jobs.py`, `gate_prompts.py`, `gate_labels.py` | Inputs for the activation-probe gate: per-step prompts and labels for every recorded write |
| `mock_meta_server.py` | A logging mock of the Meta Model API, to learn the protocol Muse Code speaks |
