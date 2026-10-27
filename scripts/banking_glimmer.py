"""Glimmer's half of the banking headroom pilot: the same 20 banking tasks, retrieval (bm25_grep, top_k 10) and seed
as Wonder's run, Glimmer k=2 with an 8,192-token output cap (the default 65,536 overflowed the 131K context).
Concurrency 4 to limit overshoot past a cap; no retries; caps Glimmer $3, Meta $1, one meter logger, fresh phases.
Compares per task with Wonder's saved results. Usage: python -X utf8 banking_glimmer.py [--analyze]
"""
import json
import statistics
import sys
import time

import litellm
import tau2.evaluator.evaluator_nl_assertions as nl
from headroom_pilot import DATA, DOMAINS, GLIMMER_MODELS, META_MODELS, RESULTS, select
from m2_pilot import GLIMMER, META, SPARK, glimmer_args
from tau2.data_model.simulation import TextRunConfig
from tau2.data_model.tasks import Task
from tau2.runner.batch import run_tasks
from tau2.utils import llm_utils

from glint.atem import reparsing
from glint.meter import Caps, Meter, guarded, litellm_callback
from glint.ratelimit import RateLimiter

OUT = DATA / "banking_knowledge-glimmer-8k.json"


def generate():
    stamp = time.strftime("%Y%m%d-%H%M%S")
    glim = Meter(Caps(meta_usd=3.0, gpu_usd=0.0), phase=f"banking-glimmer-{stamp}")
    meta = Meter(Caps(meta_usd=1.0, gpu_usd=0.0), phase=f"banking-meta-{stamp}")
    litellm.callbacks = [litellm_callback((glim, GLIMMER_MODELS), (meta, META_MODELS))]
    base, glimmer = llm_utils.completion, reparsing(llm_utils.completion)

    def route(*args, **kwargs):
        return (glimmer if "Muse-Glimmer" in kwargs.get("model", "") else base)(*args, **kwargs)

    llm_utils.completion = guarded(guarded(route, glim, only=GLIMMER_MODELS), meta, RateLimiter(per_minute=120),
                                   only=META_MODELS)
    nl.DEFAULT_LLM_NL_ASSERTIONS, nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = SPARK, {**META, "temperature": 0.0}
    all_tasks = {str(t["id"]): t for t in
                 json.loads((DOMAINS / "banking_knowledge" / "tasks.json").read_text(encoding="utf-8"))}
    tasks = [Task.model_validate(all_tasks[t]) for t in select()["banking_knowledge"]]
    config = TextRunConfig(domain="banking_knowledge", agent="llm_agent", llm_agent=GLIMMER,
                           llm_args_agent={**glimmer_args("medium"), "max_tokens": 8192}, user="user_simulator",
                           llm_user=SPARK, llm_args_user=META, num_trials=2, max_concurrency=4, seed=3000,
                           retrieval_config="bm25_grep", save_to=f"glint-banking-glimmer-8k-{stamp}",
                           auto_review=False, max_retries=0)
    results = run_tasks(config, tasks)
    OUT.write_text(results.model_dump_json(), encoding="utf-8")
    print(f"{len(results.simulations)} sims; Glimmer ${glim.meta_usd:.3f}, Meta ${meta.meta_usd:.3f}", flush=True)


def analyze():
    g = json.loads(OUT.read_text(encoding="utf-8"))["simulations"]
    w = json.loads((DATA / "banking_knowledge-wonder.json").read_text(encoding="utf-8"))["simulations"]
    rate = lambda sims: round(statistics.mean((s.get("reward_info") or {}).get("reward") == 1.0 for s in sims), 3)
    by = lambda sims: {t: [(s.get("reward_info") or {}).get("reward") == 1.0 for s in sims if str(s["task_id"]) == t]
                       for t in sorted({str(s["task_id"]) for s in sims})}
    gt, wt = by(g), by(w)
    report = {
        "glimmer_sims": len(g), "glimmer_pass": rate(g), "wonder_pass": rate(w),
        "glimmer_termination": dict(__import__("collections").Counter(str(s.get("termination_reason")) for s in g)),
        "wonder_passes_glimmer_fails_some": sorted(t for t in gt if all(wt.get(t) or [False]) and not all(gt[t])),
        "wonder_passes_glimmer_fails_all": sorted(t for t in gt if all(wt.get(t) or [False]) and not any(gt[t])),
        "glimmer_passes_wonder_fails": sorted(t for t in gt if any(gt[t]) and not any(wt.get(t) or [True])),
        "glimmer_mixed_tasks": sorted(t for t in gt if any(gt[t]) and not all(gt[t])),
        "by_task": {t: {"glimmer": gt[t], "wonder": wt.get(t)} for t in gt},
    }
    (RESULTS / "M2-banking-headroom.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "by_task"}, indent=1))


if __name__ == "__main__":
    if sys.argv[1:] != ["--analyze"]:
        generate()
    analyze()
