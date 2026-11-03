"""M0 step 3: dry-run τ²-bench v1.0.1 on Windows with hosted Muse Spark contributor as agent, user simulator
and NL-assertion judge, on 2 retail pilot tasks. Purpose: validate the harness and that glint.meter matches
hand-computed cost from the raw usage fields. Meta API cap for this run: $1.
"""
import json
import sys

import litellm
import tau2.evaluator.evaluator_nl_assertions as nl
from dotenv import dotenv_values
from tau2.data_model.simulation import TextRunConfig
from tau2.runner.batch import run_domain
from tau2.utils import llm_utils

import splits
from glint import paths
from glint.meter import Caps, Meter, guarded, litellm_callback, meta_cost
from glint.ratelimit import RateLimiter

MODEL = "openai/muse-spark-1.3-contributor"
KEY = dotenv_values(str(paths.ENV_FILE))["MODEL_API_KEY"]
LLM_ARGS = {"api_base": "https://api.meta.ai/v1", "api_key": KEY, "temperature": 1.0}
PARTITION = json.loads(splits.PARTITION.read_text())
TASKS = PARTITION["retail"]["pilot"][:2]

meter = Meter(Caps(meta_usd=1.0, gpu_usd=0.0), phase="m0-dryrun")
litellm.callbacks = [litellm_callback(meter)]
llm_utils.completion = guarded(llm_utils.completion, meter, RateLimiter(per_minute=120))
# τ² hardcodes the NL-assertion judge to gpt-4.1; R4 puts it on Spark contributor.
nl.DEFAULT_LLM_NL_ASSERTIONS = MODEL
nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {**LLM_ARGS, "temperature": 0.0}

config = TextRunConfig(
    domain="retail", task_split_name="train", task_ids=TASKS, agent="llm_agent",
    llm_agent=MODEL, llm_args_agent=LLM_ARGS, user="user_simulator", llm_user=MODEL, llm_args_user=LLM_ARGS,
    num_trials=1, max_concurrency=1, save_to="glint-m0-dryrun", auto_review=False,
)
results = run_domain(config)

# Hand math: recompute cost from the ledger's raw token fields, independent of the meter's running total.
ledger = [json.loads(line) for line in meter.ledger.read_text(encoding="utf-8").splitlines()]
ledger = [r for r in ledger if r.get("phase") == "m0-dryrun" and r["kind"] == "meta"]
hand = sum(meta_cost(r["model"], r["prompt"], r["cached"], r["completion"]) for r in ledger)
summary = {
    "tasks": TASKS,
    # paired by task_id: τ² does not return simulations in task order
    "reward_by_task": {str(s.task_id): (s.reward_info.reward if s.reward_info else None) for s in results.simulations},
    "termination_by_task": {str(s.task_id): str(s.termination_reason) for s in results.simulations},
    "calls_metered": len(ledger),
    "meter_total_usd": round(meter.meta_usd, 6),
    "hand_math_usd": round(hand, 6),
    "agent_cost_reported_by_tau2": [s.agent_cost for s in results.simulations],
    "user_cost_reported_by_tau2": [s.user_cost for s in results.simulations],
}
out = paths.RESULTS / "M0-dryrun.json"
out.write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
sys.exit(0 if abs(meter.meta_usd - hand) < 1e-9 else 1)
