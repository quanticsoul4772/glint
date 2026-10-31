"""One-task cost check (spend-guard rule): Glimmer on one τ-Knowledge banking task, k=1, no Wonder. Contributor
Spark user simulator and NL judge. Glimmer output capped at 8,192 tokens (τ²'s default 65,536 halves the
prompt room); retrieval at the default top_k=10. Caps: Glimmer $1, Meta $0.50, both through one meter logger (PR 8).
Reports Glimmer and Meta dollars, tokens and calls for the run, and the reward.
Usage: python -X utf8 cost_check.py [TASK_ID]
"""
import json
import sys
import time

import litellm
import tau2.evaluator.evaluator_nl_assertions as nl
from headroom_pilot import DOMAINS, GLIMMER_MODELS, META_MODELS, select
from m2_pilot import GLIMMER, META, SPARK, glimmer_args, require_keys
from tau2.data_model.simulation import TextRunConfig
from tau2.data_model.tasks import Task
from tau2.runner.batch import run_tasks
from tau2.utils import llm_utils

from glint import paths
from glint.atem import reparsing
from glint.meter import Caps, Meter, guarded, litellm_callback
from glint.ratelimit import RateLimiter

require_keys()
task_id = sys.argv[1] if len(sys.argv) > 1 else select()["banking_knowledge"][0]
stamp = time.strftime("%Y%m%d-%H%M%S")  # a fresh phase per run, so earlier runs are not summed in
glim = Meter(Caps(meta_usd=1.0, gpu_usd=0.0), phase=f"costcheck-glimmer-{task_id}-{stamp}")
meta = Meter(Caps(meta_usd=0.5, gpu_usd=0.0), phase=f"costcheck-meta-{task_id}-{stamp}")
litellm.callbacks = [litellm_callback((glim, GLIMMER_MODELS), (meta, META_MODELS))]
base, glimmer = llm_utils.completion, reparsing(llm_utils.completion)


def route(*args, **kwargs):
    return (glimmer if "Muse-Glimmer" in kwargs.get("model", "") else base)(*args, **kwargs)


llm_utils.completion = guarded(guarded(route, glim, only=GLIMMER_MODELS), meta, RateLimiter(per_minute=120),
                               only=META_MODELS)
nl.DEFAULT_LLM_NL_ASSERTIONS, nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = SPARK, {**META, "temperature": 0.0}

tasks = {str(t["id"]): t for t in json.loads((DOMAINS / "banking_knowledge" / "tasks.json").read_text(encoding="utf-8"))}
config = TextRunConfig(domain="banking_knowledge", agent="llm_agent", llm_agent=GLIMMER,
                       llm_args_agent={**glimmer_args("medium"), "max_tokens": 8192}, user="user_simulator", llm_user=SPARK,
                       llm_args_user=META, num_trials=1, max_concurrency=1, seed=3100, retrieval_config="bm25_grep",
                       save_to=f"glint-costcheck-{task_id}", auto_review=False, max_retries=0)
results = run_tasks(config, [Task.model_validate(tasks[task_id])])
sim = results.simulations[0]
paths.MEASURE.joinpath(f"costcheck-{task_id}-{stamp}.json").write_text(
    results.model_dump_json(), encoding="utf-8")
rows = [json.loads(line) for line in glim.ledger.read_text(encoding="utf-8").splitlines()]
mine = [r for r in rows if r.get("phase") in (glim.phase, meta.phase)]
report = {
    "task": task_id, "termination": str(sim.termination_reason),
    "reward": sim.reward_info.reward if sim.reward_info else None,
    "glimmer_usd": round(glim.meta_usd, 4), "meta_usd": round(meta.meta_usd, 4),
    "glimmer_calls": sum(r["phase"] == glim.phase for r in mine),
    "glimmer_prompt_tokens": sum(r["prompt"] for r in mine if r["phase"] == glim.phase),
    "glimmer_completion_tokens": sum(r["completion"] for r in mine if r["phase"] == glim.phase),
    "peak_prompt_tokens": max((r["prompt"] for r in mine if r["phase"] == glim.phase), default=0),
    "max_output_tokens": 8192,
}
out = paths.RESULTS / f"costcheck-banking-{task_id}-{stamp}.json"
out.write_text(json.dumps(report, indent=1), encoding="utf-8")
print(json.dumps(report, indent=1))
