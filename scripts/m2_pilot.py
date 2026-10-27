"""M2 pilot, generation and analysis (measurement-plan-v2 §M2; v1 §M2). Hosted APIs only; capture is a separate
Modal session. Runs (user simulator and NL judge: contributor Spark):
  retail-medium   10 pilot retail tasks x k=2, Glimmer (DeepInfra) medium
  airline-medium  5 pilot airline tasks x k=2, Glimmer medium
  retail-low/high first 3 pilot retail tasks x k=1, Glimmer low / high
  retail-wonder   10 pilot retail tasks x k=1, always-Wonder (Meta API)
Spend cap $12 across Meta and DeepInfra. Usage: python m2_pilot.py [run ...] | python m2_pilot.py --analyze
"""
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import litellm
import tau2.evaluator.evaluator_nl_assertions as nl
from dotenv import dotenv_values
from tau2.data_model.simulation import TextRunConfig
from tau2.runner.batch import run_domain
from tau2.utils import llm_utils

from glint.atem import generated_text, reparsing
from glint.labeler import Labeler
from glint.labels import FAB, VIOL, is_resid_positive, passes_r0
from glint.meter import Caps, Meter, guarded, litellm_callback
from glint.ratelimit import RateLimiter

ENV = dotenv_values(r"<workspace>\.env")
DATA = Path(r"<workspace>\measure\data\m2")
OUT = Path(r"<workspace>\research\5-measurements\M2-pilot.json")
PARTITION = json.loads(Path(r"<workspace>\research\5-measurements\split-partition.json").read_text())
RETAIL, AIRLINE = PARTITION["retail"]["pilot"], PARTITION["airline"]["pilot"]

GLIMMER = "openai/meta-models/Muse-Glimmer-30B"
META = {"api_base": "https://api.meta.ai/v1", "api_key": ENV["MODEL_API_KEY"], "temperature": 1.0}
SPARK = "openai/muse-spark-1.3-contributor"
WONDER = "openai/muse-wonder-1.0"


def glimmer_args(strength):
    return {"api_base": "https://api.deepinfra.com/v1/openai", "api_key": ENV["DEEPINFRA_API_KEY"],
            "temperature": 1.0, "top_p": 0.95, "logprobs": True, "top_logprobs": 5,
            "extra_body": {"top_k": 64, "chat_template_kwargs": {"reasoning_strength": strength}}}


RUNS = {
    "retail-medium": ("retail", RETAIL, 2, GLIMMER, glimmer_args("medium")),
    "airline-medium": ("airline", AIRLINE, 2, GLIMMER, glimmer_args("medium")),
    "retail-low": ("retail", RETAIL[:3], 1, GLIMMER, glimmer_args("low")),
    "retail-high": ("retail", RETAIL[:3], 1, GLIMMER, glimmer_args("high")),
    "retail-wonder": ("retail", RETAIL, 1, WONDER, META),
}


def install(meter):
    litellm.callbacks = [litellm_callback(meter)]
    base, glimmer = llm_utils.completion, reparsing(llm_utils.completion)

    def completion(*args, **kwargs):
        return (glimmer if "Muse-Glimmer" in kwargs.get("model", "") else base)(*args, **kwargs)

    llm_utils.completion = guarded(completion, meter, RateLimiter(per_minute=120))
    nl.DEFAULT_LLM_NL_ASSERTIONS = SPARK
    nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {**META, "temperature": 0.0}


def generate(names):
    meter = Meter(Caps(meta_usd=12.0, gpu_usd=0.0), phase="m2-pilot")
    install(meter)
    DATA.mkdir(parents=True, exist_ok=True)
    for name in names:
        domain, tasks, k, agent, agent_args = RUNS[name]
        t0 = time.time()
        config = TextRunConfig(
            domain=domain, task_split_name="train", task_ids=tasks, agent="llm_agent", llm_agent=agent,
            llm_args_agent=agent_args, user="user_simulator", llm_user=SPARK, llm_args_user=META,
            num_trials=k, max_concurrency=10, save_to=f"glint-m2-{name}", auto_review=False,
        )
        results = run_domain(config)
        (DATA / f"{name}.json").write_text(results.model_dump_json(), encoding="utf-8")
        print(f"{name}: {len(results.simulations)} sims in {time.time() - t0:.0f}s; spend so far ${meter.meta_usd:.3f}",
              flush=True)


def analyze_run(name):
    data = json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))
    domain, _, _, agent, _ = RUNS[name]
    tasks = {str(t["id"]): t for t in data["tasks"]}
    sims = data["simulations"]
    out = {"sims": len(sims)}
    rewards = [(s.get("reward_info") or {}).get("reward", 0.0) for s in sims]
    out["pass_rate"] = round(sum(r == 1.0 for r in rewards) / len(sims), 3)
    out["pass_by_task"] = {t: [r for s, r in zip(sims, rewards) if str(s["task_id"]) == t] for t in tasks}
    agent_msgs = [m for s in sims for m in s["messages"] if m["role"] == "assistant" and m.get("raw_data")]
    usage = [m["raw_data"].get("usage") or {} for m in agent_msgs]
    per_task_out = [sum((m["raw_data"].get("usage") or {}).get("completion_tokens", 0)
                        for m in s["messages"] if m["role"] == "assistant" and m.get("raw_data")) for s in sims]
    out["agent_calls"] = len(agent_msgs)
    out["agent_output_tokens_per_task_median"] = statistics.median(per_task_out)
    out["agent_prompt_tokens_total"] = sum(u.get("prompt_tokens", 0) for u in usage)
    out["finish_length"] = sum(m["raw_data"]["choices"][0].get("finish_reason") == "length" for m in agent_msgs)
    out["user_turns_per_agent_turn"] = round(
        sum(m["role"] == "user" for s in sims for m in s["messages"]) / max(len(agent_msgs), 1), 3)
    out["minutes_per_task_median"] = round(statistics.median(s.get("duration", 0) for s in sims) / 60, 2)
    if "Muse-Glimmer" in agent:
        kinds = Counter()
        for m in agent_msgs:
            reasoning = generated_text(m["raw_data"]["choices"][0]).startswith(" to=self")
            kinds[("tool_call" if m.get("tool_calls") else "reply", reasoning)] += 1
        calls = kinds[("tool_call", True)] + kinds[("tool_call", False)]
        steps = sum(kinds.values())
        out["reasoning_rate_steps"] = round((kinds[("tool_call", True)] + kinds[("reply", True)]) / steps, 3)
        out["reasoning_rate_tool_calls"] = round(kinds[("tool_call", True)] / max(calls, 1), 3)
    if domain == "retail":
        counts, steps = Counter(), 0
        fab = viol = resid = viol_after_r0 = 0
        for s in sims:
            for r in Labeler(tasks[str(s["task_id"])]).label(s["messages"], trial=s.get("trial", 0)):
                steps += 1
                counts.update(r.labels)
                fab += bool(r.labels & FAB)
                viol += bool(r.labels & VIOL)
                # a violation the pre-call floor cannot catch (R0 holds every VIOL rule except post-call R14)
                viol_after_r0 += bool(r.labels & VIOL) and passes_r0(r.labels)
                resid += is_resid_positive(r.labels)
        out["labeled_steps"] = steps
        out["FAB_steps"], out["VIOL_steps"], out["RESID_positive"] = fab, viol, resid
        out["VIOL_after_R0"] = viol_after_r0
        out["per_label"] = {k.value: v for k, v in counts.most_common()}
    return out


def analyze():
    report = {name: analyze_run(name) for name in RUNS if (DATA / f"{name}.json").exists()}
    ledger = [json.loads(line) for line in Path(r"<workspace>\measure\meter\ledger.jsonl").read_text().splitlines()]
    m2 = [r for r in ledger if r.get("phase") == "m2-pilot"]
    report["spend_usd_by_model"] = {}
    for r in m2:
        key = r["model"].split("/")[-1]
        report["spend_usd_by_model"][key] = round(report["spend_usd_by_model"].get(key, 0) + r["usd"], 4)
    per_min = Counter(int(r["ts"] // 60) for r in m2 if "contributor" in r["model"])
    report["peak_contributor_rpm"] = max(per_min.values()) if per_min else 0
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    if sys.argv[1:] == ["--analyze"]:
        analyze()
    else:
        generate(sys.argv[1:] or list(RUNS))
        analyze()
