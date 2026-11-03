# Archive

Code that nothing in `src/` or `scripts/` imports, moved out of the package and the scripts on 2026-11-03 (judge panel 2026-11-03, technical report: dead code inside the coverage base). It is kept so earlier notes and numbers can be traced; it is not installed, tested, linted or measured for coverage.

| File | What it was | Why it is here |
|---|---|---|
| `glint/conformal.py` | Conformal thresholds for a calibrated verifier cascade (appendix #26) | No verifier was ever wired in; imported only by its own test |
| `glint/verifiers.py` | Field-value LLM stages with a pluggable model call, off by default (appendix #44, #45) | Never wired into the gates; imported only by its own tests |
| `scripts/dryrun_m0.py` | M0: a dry run of τ²-bench v1.0.1 with hosted Muse Spark on 2 retail tasks | Finished one-off measurement (Oct 26); imported by nothing |
| `scripts/m1_deepinfra_checks.py`, `scripts/m1_c2_rerun.py` | M1: Glimmer on DeepInfra; tool-call parsing checks | Finished one-off measurements; imported by nothing |
| `scripts/m2_judge.py` | M2: an offline Spark judge pass over the retail pilot's tool calls | Finished one-off measurement; imported by nothing |
| `tests/test_archived.py` | The tests of `conformal.py` and `verifiers.py`, as they ran in `tests/` | Moved with the modules |

`scripts/m1_prefix_check.py` and `scripts/m2_pilot.py` stay in `scripts/`: `build_jobs.py`, `gate_prompts.py`, `tools_ab.py` and other scripts import them.
