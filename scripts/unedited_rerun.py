"""Are retail 20 and 93 real Glimmer errors? Their failures came from twin copies with an appended preference
sentence; this reruns the ORIGINAL tasks: 2 tasks x k=4, Glimmer (DeepInfra, medium, glint.atem), no decision tools,
contributor Spark user simulator. Caps Glimmer $0.50, Meta $0.20, one meter logger, fresh phases.
Usage: python unedited_rerun.py
"""
import json
import time

import litellm
import tau2.evaluator.evaluator_nl_assertions as nl
from m2_pilot import GLIMMER, META, SPARK, glimmer_args, require_keys
from tau2.data_model.simulation import TextRunConfig
from tau2.runner.batch import run_domain
from tau2.utils import llm_utils

from glint import paths
from glint.atem import reparsing
from glint.meter import Caps, Meter, guarded, litellm_callback
from glint.ratelimit import RateLimiter

TASKS = ["20", "93"]
require_keys()
stamp = time.strftime("%Y%m%d-%H%M%S")
glim = Meter(Caps(meta_usd=0.5, gpu_usd=0.0), phase=f"unedited-glimmer-{stamp}")
meta = Meter(Caps(meta_usd=0.2, gpu_usd=0.0), phase=f"unedited-meta-{stamp}")
litellm.callbacks = [litellm_callback((glim, ("Muse-Glimmer",)), (meta, ("muse-spark", "muse-wonder")))]
base, glimmer = llm_utils.completion, reparsing(llm_utils.completion)


def route(*args, **kwargs):
    return (glimmer if "Muse-Glimmer" in kwargs.get("model", "") else base)(*args, **kwargs)


llm_utils.completion = guarded(guarded(route, glim, only=("Muse-Glimmer",)), meta, RateLimiter(per_minute=120),
                               only=("muse-spark", "muse-wonder"))
nl.DEFAULT_LLM_NL_ASSERTIONS, nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = SPARK, {**META, "temperature": 0.0}
config = TextRunConfig(domain="retail", task_split_name="train", task_ids=TASKS, agent="llm_agent", llm_agent=GLIMMER,
                       llm_args_agent=glimmer_args("medium"), user="user_simulator", llm_user=SPARK,
                       llm_args_user=META, num_trials=4, max_concurrency=4, seed=4000,
                       save_to=f"glint-unedited-{stamp}", auto_review=False, max_retries=0)
results = run_domain(config)
out = paths.MEASURE / f"unedited-20-93-{stamp}.json"
out.write_text(results.model_dump_json(), encoding="utf-8")
report = {"tasks": TASKS, "glimmer_usd": round(glim.meta_usd, 4), "meta_usd": round(meta.meta_usd, 4), "by_task": {}}
for s in results.simulations:
    writes = [(tc.name, tc.arguments) for m in s.messages if m.role == "assistant" for tc in (m.tool_calls or [])
              if tc.name.startswith(("exchange", "return", "cancel", "modify"))]
    report["by_task"].setdefault(str(s.task_id), []).append(
        {"trial": s.trial, "reward": s.reward_info.reward if s.reward_info else None,
         "termination": str(s.termination_reason), "writes": writes})
(paths.RESULTS / "unedited-20-93.json").write_text(json.dumps(report, indent=1))
print(json.dumps({k: v for k, v in report.items() if k != "by_task"}),
      {t: [x["reward"] for x in v] for t, v in report["by_task"].items()})
