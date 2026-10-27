"""Glimmer with vs without the decision tools. Same tasks and seeds in both arms: arm "base" runs first with τ²'s own
tools; arm "tools" then installs glint.assist (check_order_action, find_user_items, find_variants,
check_reservation_action, get_upcoming_trips_cost). Glimmer on DeepInfra (medium, glint.atem); contributor Spark user
simulator and NL judge. One meter logger.

Sets:
- `design` (default): the tasks the tools were built from (research/3-project/tool-offload-research.md): retail 20,
  93, 82 and airline 7; k=8; caps Glimmer $2, Meta $0.50.
- `heldout`: 10 retail + 10 airline train tasks drawn at random (seed 2026) after excluding every task used to design
  the tools or classified in the failure inventory; test-40 is never used; k=4; caps Glimmer $3.50, Meta $0.50
  (size chosen with parallax `decide`).

Usage: python tools_ab.py [design|heldout] [--analyze]
"""
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import litellm
import tau2.evaluator.evaluator_nl_assertions as nl
from m2_pilot import GLIMMER, META, SPARK, glimmer_args
from tau2.data_model.simulation import TextRunConfig
from tau2.registry import registry
from tau2.runner.batch import run_domain
from tau2.utils import llm_utils

from glint.assist.tau2_tools import install
from glint.atem import reparsing
from glint.meter import Caps, Meter, guarded, litellm_callback
from glint.ratelimit import RateLimiter

TOOL_NAMES = {"check_order_action", "find_user_items", "find_variants", "check_reservation_action",
              "get_upcoming_trips_cost"}
# Tasks the tools were designed from, or that are in the failure inventory
DESIGN_TASKS = {"retail": {"20", "93", "82", "52", "98", "103", "104"}, "airline": {"7"}}
MEASURE = Path(r"<workspace>\measure\data")
RESULTS = Path(r"<workspace>\research\5-measurements")


def heldout_tasks(n: int = 10, seed: int = 2026) -> dict[str, list[str]]:
    rng = random.Random(seed)
    out = {}
    for domain in ("retail", "airline"):
        ids = [t.id for t in registry.get_tasks_loader(domain)("train") if t.id not in DESIGN_TASKS[domain]]
        out[domain] = sorted(rng.sample(ids, n), key=int)
    return out


SETS = {
    "design": {"tasks": lambda: {"retail": ["20", "93", "82"], "airline": ["7"]}, "trials": 8, "glimmer_usd": 2.0,
               "meta_usd": 0.5, "data": MEASURE / "tools_ab", "out": RESULTS / "tools-ab.json", "save": "tools-ab"},
    "heldout": {"tasks": heldout_tasks, "trials": 4, "glimmer_usd": 3.5, "meta_usd": 0.5,
                "data": MEASURE / "tools_ab_heldout", "out": RESULTS / "tools-ab-heldout.json",
                "save": "tools-ab-heldout"},
}


def generate(cfg: dict) -> None:
    # One stamp per comparison: a resumed run keeps the same meter phases (so the caps count earlier spend) and the
    # same τ² save_to (so finished simulations are resumed, not paid for again)
    data = cfg["data"]
    data.mkdir(parents=True, exist_ok=True)
    stamp_file = data / "stamp.txt"
    if not stamp_file.exists():
        stamp_file.write_text(time.strftime("%Y%m%d-%H%M%S"), encoding="utf-8")
    stamp = stamp_file.read_text(encoding="utf-8").strip()
    tasks = cfg["tasks"]()
    (data / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
    glim = Meter(Caps(meta_usd=cfg["glimmer_usd"], gpu_usd=0.0), phase=f"{cfg['save']}-glimmer-{stamp}")
    meta = Meter(Caps(meta_usd=cfg["meta_usd"], gpu_usd=0.0), phase=f"{cfg['save']}-meta-{stamp}")
    litellm.callbacks = [litellm_callback((glim, ("Muse-Glimmer",)), (meta, ("muse-spark", "muse-wonder")))]
    base, glimmer = llm_utils.completion, reparsing(llm_utils.completion)

    def route(*args, **kwargs):
        return (glimmer if "Muse-Glimmer" in kwargs.get("model", "") else base)(*args, **kwargs)

    llm_utils.completion = guarded(guarded(route, glim, only=("Muse-Glimmer",)), meta, RateLimiter(per_minute=120),
                                   only=("muse-spark", "muse-wonder"))
    nl.DEFAULT_LLM_NL_ASSERTIONS, nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = SPARK, {**META, "temperature": 0.0}
    for arm in ("base", "tools"):
        for domain, ids in tasks.items():
            out = data / f"{arm}-{domain}.json"
            if out.exists():
                continue
            if arm == "tools":
                install(domain)
            config = TextRunConfig(domain=domain, task_split_name="train", task_ids=ids, agent="llm_agent",
                                   llm_agent=GLIMMER, llm_args_agent=glimmer_args("medium"), user="user_simulator",
                                   llm_user=SPARK, llm_args_user=META, num_trials=cfg["trials"], max_concurrency=4,
                                   seed=5000, save_to=f"glint-{cfg['save']}-{arm}-{domain}-{stamp}",
                                   auto_review=False, max_retries=0, auto_resume=True)
            results = run_domain(config)
            out.write_text(results.model_dump_json(), encoding="utf-8")
            print(f"{arm} {domain}: {len(results.simulations)} sims; Glimmer ${glim.meta_usd:.3f}, "
                  f"Meta ${meta.meta_usd:.3f}", flush=True)


def analyze(cfg: dict) -> None:
    report = {}
    for arm in ("base", "tools"):
        for domain in ("retail", "airline"):
            path = cfg["data"] / f"{arm}-{domain}.json"
            if not path.exists():
                continue
            for s in json.loads(path.read_text(encoding="utf-8"))["simulations"]:
                key = f"{domain}-{s['task_id']}"
                row = report.setdefault(key, {}).setdefault(arm, {"passes": 0, "runs": 0, "tool_calls": Counter(),
                                                                   "infra_errors": 0})
                if str(s.get("termination_reason")) == "infrastructure_error":
                    row["infra_errors"] += 1
                    continue
                row["runs"] += 1
                row["passes"] += (s.get("reward_info") or {}).get("reward") == 1.0
                for m in s["messages"]:
                    for tc in (m.get("tool_calls") or []) if m["role"] == "assistant" else []:
                        if tc["name"] in TOOL_NAMES:
                            row["tool_calls"][tc["name"]] += 1
    for arms in report.values():
        for row in arms.values():
            row["tool_calls"] = dict(row["tool_calls"])
            row["pass_rate"] = round(row["passes"] / row["runs"], 3) if row["runs"] else None
    cfg["out"].write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    args = sys.argv[1:]
    cfg = SETS[next((a for a in args if a in SETS), "design")]
    if "--analyze" not in args:
        generate(cfg)
    analyze(cfg)
