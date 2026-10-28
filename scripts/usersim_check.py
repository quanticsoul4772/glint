"""Pre-M2 check: does Muse Spark Open on Baseten work as τ² user simulator and NL-assertion judge?
Same 2 retail pilot tasks as the M0 dry run (agent stays Spark contributor), k=2, so it can be compared with
M0-dryrun.json on reward, termination, turn counts and cost. Spend cap for this run: $2 (Meta + Baseten).
"""
import json
from collections import defaultdict

import litellm
import tau2.evaluator.evaluator_nl_assertions as nl
from dotenv import dotenv_values
from tau2.data_model.simulation import TextRunConfig
from tau2.runner.batch import run_domain
from tau2.utils import llm_utils

from glint import paths
from glint.meter import Caps, Meter, guarded, litellm_callback
from glint.ratelimit import RateLimiter

ENV = dotenv_values(str(paths.ENV_FILE))
AGENT = "openai/muse-spark-1.3-contributor"
AGENT_ARGS = {"api_base": "https://api.meta.ai/v1", "api_key": ENV["MODEL_API_KEY"], "temperature": 1.0}
USER = "openai/meta-models/muse-spark-open-1026"
USER_ARGS = {"api_base": "https://inference.baseten.co/v1", "api_key": ENV["BASETEN_API_KEY"], "temperature": 1.0}
PARTITION = json.loads((paths.RESULTS / "split-partition.json").read_text())
TASKS = PARTITION["retail"]["pilot"][:2]

meter = Meter(Caps(meta_usd=2.0, gpu_usd=0.0), phase="pre-m2-usersim")
litellm.callbacks = [litellm_callback(meter)]
llm_utils.completion = guarded(llm_utils.completion, meter, RateLimiter(per_minute=120))
nl.DEFAULT_LLM_NL_ASSERTIONS = USER
nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {**USER_ARGS, "temperature": 0.0}

config = TextRunConfig(
    domain="retail", task_split_name="train", task_ids=TASKS, agent="llm_agent",
    llm_agent=AGENT, llm_args_agent=AGENT_ARGS, user="user_simulator", llm_user=USER, llm_args_user=USER_ARGS,
    num_trials=2, max_concurrency=2, save_to="glint-pre-m2-usersim", auto_review=False,
)
results = run_domain(config)

ledger = [json.loads(line) for line in meter.ledger.read_text(encoding="utf-8").splitlines()]
ledger = [r for r in ledger if r.get("phase") == "pre-m2-usersim" and r["kind"] == "meta"]
by_model = defaultdict(lambda: {"calls": 0, "prompt": 0, "cached": 0, "completion": 0, "reasoning": 0, "usd": 0.0})
for r in ledger:
    m = by_model[r["model"].split("/")[-1]]
    m["calls"] += 1
    for k in ("prompt", "cached", "completion", "reasoning", "usd"):
        m[k] += r[k]


def turns(sim, role):
    return sum(1 for msg in sim.messages if msg.role == role)


summary = {
    "tasks": TASKS,
    "sims": [{"task": str(s.task_id), "trial": s.trial, "reward": s.reward_info.reward if s.reward_info else None,
              "termination": str(s.termination_reason), "user_turns": turns(s, "user"),
              "agent_turns": turns(s, "assistant")} for s in results.simulations],
    "by_model": {k: {**v, "usd": round(v["usd"], 6)} for k, v in by_model.items()},
    "total_usd": round(meter.meta_usd, 6),
}
out = paths.RESULTS / "pre-M2-usersim-spark-open.json"
out.write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
