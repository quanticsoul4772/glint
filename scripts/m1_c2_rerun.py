"""M1 C2 rerun with Glint's ATEM parsing (glint.atem): the same 3 retail pilot tasks, Glimmer on DeepInfra as agent,
contributor Spark as user simulator. Compare with M1-deepinfra-checks.json, where DeepInfra's parser turned every zip
into a number and all three tasks scored 0. Spend cap $0.50.
"""
import json
from pathlib import Path

import litellm
from dotenv import dotenv_values
from tau2.data_model.simulation import TextRunConfig
from tau2.runner.batch import run_domain
from tau2.utils import llm_utils

from glint.atem import reparsing
from glint.meter import Caps, Meter, guarded, litellm_callback
from glint.ratelimit import RateLimiter

ENV = dotenv_values(r"<workspace>\.env")
GLIMMER = "openai/meta-models/Muse-Glimmer-30B"
GLIMMER_ARGS = {"api_base": "https://api.deepinfra.com/v1/openai", "api_key": ENV["DEEPINFRA_API_KEY"],
                "temperature": 1.0, "top_p": 0.95, "logprobs": True, "top_logprobs": 5,
                "extra_body": {"top_k": 64, "chat_template_kwargs": {"reasoning_strength": "medium"}}}
USER = "openai/muse-spark-1.3-contributor"
USER_ARGS = {"api_base": "https://api.meta.ai/v1", "api_key": ENV["MODEL_API_KEY"], "temperature": 1.0}
PARTITION = json.loads(Path(r"<workspace>\research\5-measurements\split-partition.json").read_text())
TASKS = PARTITION["retail"]["pilot"][:3]

meter = Meter(Caps(meta_usd=0.5, gpu_usd=0.0), phase="m1-c2-rerun")
litellm.callbacks = [litellm_callback(meter)]
llm_utils.completion = guarded(reparsing(llm_utils.completion), meter, RateLimiter(per_minute=120))

config = TextRunConfig(
    domain="retail", task_split_name="train", task_ids=TASKS, agent="llm_agent",
    llm_agent=GLIMMER, llm_args_agent=GLIMMER_ARGS, user="user_simulator", llm_user=USER, llm_args_user=USER_ARGS,
    num_trials=1, max_concurrency=3, save_to="glint-m1-c2-rerun", auto_review=False,
)
results = run_domain(config)
Path(r"<workspace>\measure\data\m1-c2-rerun-simulations.json").write_text(results.model_dump_json(), encoding="utf-8")
calls = [tc for s in results.simulations for m in s.messages if m.role == "assistant" for tc in (m.tool_calls or [])]
summary = {
    "tasks": TASKS,
    "reward_by_task": {str(s.task_id): s.reward_info.reward if s.reward_info else None for s in results.simulations},
    "termination_by_task": {str(s.task_id): str(s.termination_reason) for s in results.simulations},
    "tool_calls": len(calls),
    "lookup_results": [m.content[:60] for s in results.simulations for m in s.messages
                       if m.role == "tool" and "user" in (m.content or "").lower()][:6],
    "spend_usd": round(meter.meta_usd, 6),
}
Path(r"<workspace>\research\5-measurements\M1-c2-rerun.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
