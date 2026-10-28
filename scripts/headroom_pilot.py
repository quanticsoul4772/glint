"""Headroom pilot: does Glimmer fail often enough, and Wonder do better, for task-level routing to matter?
Domains: τ-Knowledge banking (local bm25_grep retrieval, no embedding API) and τ² airline. Per domain 20 tasks,
Glimmer (DeepInfra, medium, glint.atem parsing) k=2 and Wonder (Meta API) k=1; contributor Spark user simulator and
NL judge. Banking has no published split, so a seeded partition holds out 20 test tasks that are never run here.
Caps: Glimmer $5, Meta credit $20 (separate pools). Usage: python headroom_pilot.py [--analyze]
"""
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict

import litellm
import tau2.evaluator.evaluator_nl_assertions as nl
from m2_pilot import GLIMMER, META, SPARK, WONDER, glimmer_args
from tau2.data_model.simulation import TextRunConfig
from tau2.data_model.tasks import Task
from tau2.runner.batch import run_tasks
from tau2.utils import llm_utils

from glint import paths
from glint.atem import reparsing
from glint.meter import Caps, Meter, guarded, litellm_callback
from glint.ratelimit import RateLimiter

DOMAINS = paths.tau2_data() / "tau2" / "domains"
DATA = paths.MEASURE / "headroom"
RESULTS = paths.RESULTS
SEED = "glint-headroom-2026-10-26"
META_MODELS, GLIMMER_MODELS = ("muse-spark", "muse-wonder"), ("Muse-Glimmer",)


def _order(ids, tag):
    return sorted(ids, key=lambda t: hashlib.sha256(f"{SEED}:{tag}:{t}".encode()).hexdigest())


def select():
    banking_raw = (DOMAINS / "banking_knowledge" / "tasks.json").read_bytes()
    banking_ids = [str(t["id"]) for t in json.loads(banking_raw)]
    order = _order(banking_ids, "banking")
    partition = {"seed": SEED, "tasks_sha256": hashlib.sha256(banking_raw).hexdigest(),
                 "test_untouched": order[:20], "pilot": order[20:40], "rest": order[40:]}
    out = RESULTS / "split-partition-banking.json"
    if out.exists():
        assert json.loads(out.read_text()) == partition, "banking partition changed"
    else:
        out.write_text(json.dumps(partition, indent=1))
    airline_train = json.loads((DOMAINS / "airline" / "split_tasks.json").read_text())["train"]
    return {"banking_knowledge": partition["pilot"], "airline": _order(airline_train, "airline")[:20]}


RUNS = [("banking_knowledge", "glimmer", 2), ("banking_knowledge", "wonder", 1),
        ("airline", "glimmer", 2), ("airline", "wonder", 1)]


def generate():
    glim = Meter(Caps(meta_usd=5.0, gpu_usd=0.0), phase="headroom-glimmer")
    meta = Meter(Caps(meta_usd=20.0, gpu_usd=0.0), phase="headroom-meta")
    # one logger for both pools: LiteLLM keeps a single instance per logger class
    litellm.callbacks = [litellm_callback((glim, GLIMMER_MODELS), (meta, META_MODELS))]
    base, glimmer = llm_utils.completion, reparsing(llm_utils.completion)

    def route(*args, **kwargs):
        return (glimmer if "Muse-Glimmer" in kwargs.get("model", "") else base)(*args, **kwargs)

    llm_utils.completion = guarded(guarded(route, glim, only=GLIMMER_MODELS), meta, RateLimiter(per_minute=120),
                                   only=META_MODELS)
    nl.DEFAULT_LLM_NL_ASSERTIONS, nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = SPARK, {**META, "temperature": 0.0}
    chosen = select()
    DATA.mkdir(parents=True, exist_ok=True)
    for domain, agent, k in RUNS:
        out = DATA / f"{domain}-{agent}.json"
        if out.exists():
            continue
        all_tasks = {str(t["id"]): t for t in json.loads((DOMAINS / domain / "tasks.json").read_text(encoding="utf-8"))}
        tasks = [Task.model_validate(all_tasks[t]) for t in chosen[domain]]
        config = TextRunConfig(
            domain=domain, agent="llm_agent", llm_agent=GLIMMER if agent == "glimmer" else WONDER,
            llm_args_agent=glimmer_args("medium") if agent == "glimmer" else META, user="user_simulator",
            llm_user=SPARK, llm_args_user=META, num_trials=k, max_concurrency=20, seed=3000,
            retrieval_config="bm25_grep" if domain == "banking_knowledge" else None,
            save_to=f"glint-headroom-{domain}-{agent}", auto_review=False)
        results = run_tasks(config, tasks)
        out.write_text(results.model_dump_json(), encoding="utf-8")
        print(f"{domain} {agent}: {len(results.simulations)} sims; Glimmer ${glim.meta_usd:.2f}, Meta ${meta.meta_usd:.2f}",
              flush=True)


def analyze():
    report = {}
    for domain in ("banking_knowledge", "airline"):
        per = {}
        for agent in ("glimmer", "wonder"):
            path = DATA / f"{domain}-{agent}.json"
            if not path.exists():
                continue
            sims = json.loads(path.read_text(encoding="utf-8"))["simulations"]
            by_task, failed_checks, term = defaultdict(list), Counter(), Counter()
            for s in sims:
                ri = s.get("reward_info") or {}
                r = ri.get("reward", 0.0)
                by_task[str(s["task_id"])].append(r)
                term[str(s.get("termination_reason"))] += 1
                if r != 1.0:
                    if not (ri.get("db_check") or {}).get("db_match", True):
                        failed_checks["db"] += 1
                    if any(not c.get("met") for c in (ri.get("communicate_checks") or [])):
                        failed_checks["communicate"] += 1
                    if any(not a.get("met") for a in (ri.get("nl_assertions") or [])):
                        failed_checks["nl_assertion"] += 1
            per[agent] = {"sims": len(sims), "pass_rate": round(statistics.mean(r == 1.0 for v in by_task.values()
                                                                               for r in v), 3),
                          "by_task": {t: v for t, v in by_task.items()}, "failed_checks": dict(failed_checks),
                          "termination": dict(term)}
        if "glimmer" in per and "wonder" in per:
            g, w = per["glimmer"]["by_task"], per["wonder"]["by_task"]
            headroom = [t for t in g if statistics.mean(x == 1.0 for x in g[t]) < 1 and all(x == 1.0 for x in w.get(t, []))
                        and w.get(t)]
            both_fail = [t for t in g if not any(x == 1.0 for x in g[t]) and not any(x == 1.0 for x in w.get(t, [1.0]))]
            per["tasks_glimmer_fails_some_wonder_passes"] = headroom
            per["tasks_both_fail"] = both_fail
        report[domain] = per
    ledger = [json.loads(line) for line in paths.LEDGER.read_text().splitlines()]
    spend = Counter()
    for r in ledger:
        if r.get("phase", "").startswith("headroom"):
            spend[r["model"].split("/")[-1]] += r["usd"]
    report["spend_usd"] = {k: round(v, 3) for k, v in spend.items()}
    (RESULTS / "M2-headroom-pilot.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({d: {a: {k: v for k, v in x.items() if k != "by_task"} if isinstance(x, dict) else x
                          for a, x in p.items()} if isinstance(p, dict) else p for d, p in report.items()}, indent=1))


if __name__ == "__main__":
    if sys.argv[1:] != ["--analyze"]:
        generate()
    analyze()
