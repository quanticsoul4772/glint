"""M1 checks on Glimmer hosted by DeepInfra (measurement-plan-v2 C2, C3, C7). Spend cap $1.

C2: τ² retail pilot tasks with Glimmer as the agent (user simulator: contributor Spark); every agent turn must
    parse (tool_calls or plain text, no leaked call markup). Full simulations are saved for C5.
C3: reasoning_strength low vs high through chat_template_kwargs changes reasoning length.
C7: 16 concurrent requests; count 429s and other errors.
"""
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import litellm
from dotenv import dotenv_values
from tau2.data_model.simulation import TextRunConfig
from tau2.runner.batch import run_domain
from tau2.utils import llm_utils

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
OUT = Path(r"<workspace>\research\5-measurements\M1-deepinfra-checks.json")
SIMS = Path(r"<workspace>\measure\data\m1-c2-simulations.json")

meter = Meter(Caps(meta_usd=1.0, gpu_usd=0.0), phase="m1-deepinfra")
litellm.callbacks = [litellm_callback(meter)]
llm_utils.completion = guarded(llm_utils.completion, meter, RateLimiter(per_minute=120))
summary = {}

# C2
config = TextRunConfig(
    domain="retail", task_split_name="train", task_ids=TASKS, agent="llm_agent",
    llm_agent=GLIMMER, llm_args_agent=GLIMMER_ARGS, user="user_simulator", llm_user=USER, llm_args_user=USER_ARGS,
    num_trials=1, max_concurrency=3, save_to="glint-m1-c2", auto_review=False,
)
results = run_domain(config)
SIMS.write_text(results.model_dump_json(), encoding="utf-8")
turns, failures = [], []
for sim in results.simulations:
    for i, msg in enumerate(sim.messages):
        if msg.role != "assistant":
            continue
        choice = (msg.raw_data or {}).get("choices", [{}])[0]
        text = msg.content or ""
        leaked = any(t in text for t in ("<|", "function_calls", "</", "invoke"))
        turns.append({"task": str(sim.task_id), "idx": i, "tool_call": bool(msg.tool_calls),
                      "finish": choice.get("finish_reason"), "leaked_markup": leaked,
                      "reasoning_chars": len(choice.get("message", {}).get("reasoning_content") or ""),
                      "logprob_tokens": len((choice.get("logprobs") or {}).get("content") or [])})
        if leaked or (not msg.tool_calls and not text.strip()):
            failures.append(turns[-1])
summary["C2"] = {
    "tasks": TASKS,
    "reward_by_task": {str(s.task_id): s.reward_info.reward if s.reward_info else None for s in results.simulations},
    "termination_by_task": {str(s.task_id): str(s.termination_reason) for s in results.simulations},
    "agent_turns": len(turns), "tool_call_turns": sum(t["tool_call"] for t in turns),
    "parse_failures": failures, "finish_length": sum(t["finish"] == "length" for t in turns),
    "turns_with_reasoning": sum(t["reasoning_chars"] > 0 for t in turns),
    "turns_with_logprobs": sum(t["logprob_tokens"] > 0 for t in turns),
}
print(json.dumps(summary["C2"], indent=1))

# C3 and C7 call the completion function directly, through the same guard.
call = llm_utils.completion
PROBE = [{"role": "system", "content": "You are a retail support agent. Follow policy. Use tools."},
         {"role": "user", "content": "I want to return two items from order #W2378156 and exchange a third for a "
                                     "different size. My zip is 75243. Can you do it all today?"}]


def reasoning_tokens(strength):
    args = {**GLIMMER_ARGS, "extra_body": {"top_k": 64, "chat_template_kwargs": {"reasoning_strength": strength}}}
    r = call(model=GLIMMER, messages=PROBE, max_tokens=8000, **args)
    return r.usage.completion_tokens


summary["C3"] = {s: [reasoning_tokens(s) for _ in range(3)] for s in ("low", "high")}
summary["C3"]["median_ratio_high_over_low"] = round(
    statistics.median(summary["C3"]["high"]) / max(statistics.median(summary["C3"]["low"]), 1), 2)
print(json.dumps(summary["C3"]))


def one(_):
    try:
        call(model=GLIMMER, messages=[{"role": "user", "content": "Reply with one word: ok"}], max_tokens=300,
             **GLIMMER_ARGS)
        return "ok"
    except Exception as e:  # recorded, not hidden: the point of C7 is to count failures by type
        return type(e).__name__ + ": " + str(e)[:120]


direct = guarded(litellm.completion, meter)  # C7 bypasses the 120 RPM limiter to test the provider's own limits
call = direct
with ThreadPoolExecutor(16) as pool:
    outcomes = list(pool.map(one, range(16)))
summary["C7"] = {"requests": 16, "ok": outcomes.count("ok"), "errors": [o for o in outcomes if o != "ok"]}
print(json.dumps(summary["C7"]))

summary["spend_usd"] = round(meter.meta_usd, 6)
OUT.write_text(json.dumps(summary, indent=1), encoding="utf-8")
print("spend", summary["spend_usd"])
