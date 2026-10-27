"""Glimmer with vs without the decision tools. Same tasks and seeds in both arms: arm "base" runs first with τ²'s own
tools; arm "tools" then installs glint.assist (retail: check_order_action, find_user_items, find_variants; airline:
check_reservation_action, get_upcoming_trips_cost; banking: check_credit_limit_increase, check_card_dispute).
Glimmer at medium strength through glint.atem; contributor Spark user simulator and NL judge. One meter logger.

Backends:
- DeepInfra (default): hosted Glimmer, paid per token.
- `--modal`: Glimmer served by glint.glimmer_serve on Modal's free credit. The whole run is one `session()`
  (deploy, wait for /health with the log watch, stop and meter the GPU time at the end), and both arms run there.
  Runs from the two backends are never mixed in one comparison (data under `<set>_modal`).

Sets:
- `design`: the tasks the tools were built from (research/3-project/tool-offload-research.md): retail 20, 93, 82 and
  airline 7; k=8; caps Glimmer $2, Meta $0.50.
- `heldout`: 10 retail + 10 airline train tasks drawn at random (seed 2026) after excluding every task used to design
  the tools or classified in the failure inventory; test-40 is never used; caps Glimmer $3.50, Meta $0.50 (size
  chosen with parallax `decide`). Retail k=4; airline k=2, because about $2 was left on DeepInfra: trials 0-1 of the
  base arm were already done, and τ² derives trial seeds as a prefix-stable sequence, so k=2 reuses them unchanged.
- `banking`: the banking tasks whose gold has a credit card dispute or a credit limit increase, outside the untouched
  test split (`5-measurements/split-partition-banking.json`); 038 and 054 are the recorded failures. These are also
  the tasks whose gold calls the banking tools are checked against (tests/test_assist.py), so they are design tasks,
  not held-out. k=4, retrieval bm25_grep, output capped at 8,192 tokens; caps Meta $1, GPU $6.

Runs go domain by domain (base then tools), so a domain that finishes has both arms.

- `banking-failures`: 038 and 054 only, run first; shares `banking`'s stamp and checkpoint.

Usage: python tools_ab.py [design|heldout|banking|banking-failures] [--modal] [--analyze]
"""
import json
import os
import random
import sys
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import litellm
import tau2.evaluator.evaluator_nl_assertions as nl
from m2_pilot import GLIMMER, META, SPARK, glimmer_args
from tau2.data_model.simulation import TextRunConfig
from tau2.data_model.tasks import Task
from tau2.registry import registry
from tau2.runner.batch import run_domain, run_tasks
from tau2.utils import llm_utils

from glint import glimmer_serve
from glint.assist.tau2_tools import install
from glint.atem import reparsing
from glint.meter import Caps, Meter, guarded, litellm_callback
from glint.ratelimit import RateLimiter

TOOL_NAMES = {"check_order_action", "find_user_items", "find_variants", "check_reservation_action",
              "get_upcoming_trips_cost", "check_credit_limit_increase", "check_card_dispute"}
# Tasks the tools were designed from, or that are in the failure inventory
DESIGN_TASKS = {"retail": {"20", "93", "82", "52", "98", "103", "104"}, "airline": {"7"}}
MEASURE = Path(r"<workspace>\measure\data")
RESULTS = Path(r"<workspace>\research\5-measurements")
BANKING_TASKS = Path(r"<workspace>\vendor\tau2-bench\data\tau2\domains\banking_knowledge\tasks.json")
BANKING_SPLIT = RESULTS / "split-partition-banking.json"
BANKING_WRITES = ("file_credit_card_transaction_dispute_4829", "submit_credit_limit_increase_request_7392")
MODAL_CLI = str(Path(__file__).resolve().parents[1] / ".venv" / "Scripts" / "modal.exe")


def heldout_tasks(n: int = 10, seed: int = 2026) -> dict[str, list[str]]:
    rng = random.Random(seed)
    out = {}
    for domain in ("retail", "airline"):
        ids = [t.id for t in registry.get_tasks_loader(domain)("train") if t.id not in DESIGN_TASKS[domain]]
        out[domain] = sorted(rng.sample(ids, n), key=int)
    return out


def banking_tasks() -> dict[str, list[str]]:
    untouched = set(json.loads(BANKING_SPLIT.read_text(encoding="utf-8"))["test_untouched"])
    ids = [t["id"] for t in json.loads(BANKING_TASKS.read_text(encoding="utf-8"))
           if t["id"] not in untouched and any(a["name"] == "call_discoverable_agent_tool"
                                               and a["arguments"].get("agent_tool_name") in BANKING_WRITES
                                               for a in t["evaluation_criteria"].get("actions") or [])]
    return {"banking_knowledge": sorted(ids)}


SETS = {
    "design": {"tasks": lambda: {"retail": ["20", "93", "82"], "airline": ["7"]}, "trials": 8, "glimmer_usd": 2.0,
               "meta_usd": 0.5, "gpu_usd": 4.0, "data": MEASURE / "tools_ab", "out": RESULTS / "tools-ab.json",
               "save": "tools-ab"},
    "heldout": {"tasks": heldout_tasks, "trials": 4, "airline_trials": 2, "glimmer_usd": 3.5, "meta_usd": 0.5,
                "gpu_usd": 6.0, "data": MEASURE / "tools_ab_heldout", "out": RESULTS / "tools-ab-heldout.json",
                "save": "tools-ab-heldout"},
    "banking": {"tasks": banking_tasks, "trials": 4, "glimmer_usd": 3.0, "meta_usd": 1.0, "gpu_usd": 6.0,
                "data": MEASURE / "tools_ab_banking", "out": RESULTS / "tools-ab-banking.json",
                "save": "tools-ab-banking"},
    # The two recorded failures first. Same stamp and τ² save_to as `banking`, so the full set later resumes these
    # simulations instead of running them again
    "banking-failures": {"tasks": lambda: {"banking_knowledge": ["task_038", "task_054"]}, "trials": 4,
                         "glimmer_usd": 3.0, "meta_usd": 1.0, "gpu_usd": 6.0,
                         "data": MEASURE / "tools_ab_banking_failures", "stamp_dir": MEASURE / "tools_ab_banking",
                         "out": RESULTS / "tools-ab-banking-failures.json", "save": "tools-ab-banking"},
}


def for_backend(cfg: dict, modal: bool) -> dict:
    if not modal:
        return cfg
    stamp_dir = cfg.get("stamp_dir", cfg["data"])
    return cfg | {"modal": True, "data": cfg["data"].with_name(cfg["data"].name + "_modal"),
                  "stamp_dir": stamp_dir.with_name(stamp_dir.name + "_modal"),
                  "out": cfg["out"].with_name(cfg["out"].stem + "-modal.json"), "save": cfg["save"] + "-modal"}


def _trials(cfg: dict, domain: str) -> int:
    return cfg.get(f"{domain}_trials", cfg["trials"])


def _agent(cfg: dict, domain: str, endpoint: str | None) -> tuple[str, dict]:
    if endpoint:
        args = glimmer_serve.litellm_args(endpoint, os.environ["MODAL_PROXY_KEY"], os.environ["MODAL_PROXY_SECRET"])
        model = f"openai/{glimmer_serve.MODEL_NAME}"
    else:
        model, args = GLIMMER, glimmer_args("medium")
    if domain == "banking_knowledge":
        args = {**args, "max_tokens": 8192}  # without the cap the default 65,536 overflowed the context
    return model, args


def _run(cfg: dict, domain: str, ids: list[str], arm: str, stamp: str, endpoint: str | None):
    model, args = _agent(cfg, domain, endpoint)
    common = {"domain": domain, "agent": "llm_agent", "llm_agent": model, "llm_args_agent": args,
              "user": "user_simulator", "llm_user": SPARK, "llm_args_user": META, "num_trials": _trials(cfg, domain),
              "max_concurrency": 8 if endpoint else 4, "seed": 5000,
              "save_to": f"glint-{cfg['save']}-{arm}-{domain}-{stamp}", "auto_review": False, "max_retries": 0,
              "auto_resume": True}
    if domain == "banking_knowledge":
        by_id = {t["id"]: t for t in json.loads(BANKING_TASKS.read_text(encoding="utf-8"))}
        config = TextRunConfig(**common, retrieval_config="bm25_grep")
        return run_tasks(config, [Task.model_validate(by_id[i]) for i in ids])
    return run_domain(TextRunConfig(**common, task_split_name="train", task_ids=ids))


def generate(cfg: dict) -> None:
    # One stamp per comparison: a resumed run keeps the same meter phases (so the caps count earlier spend) and the
    # same τ² save_to (so finished simulations are resumed, not paid for again)
    data = cfg["data"]
    data.mkdir(parents=True, exist_ok=True)
    stamp_dir = cfg.get("stamp_dir", data)
    stamp_dir.mkdir(parents=True, exist_ok=True)
    stamp_file = stamp_dir / "stamp.txt"
    if not stamp_file.exists():
        stamp_file.write_text(time.strftime("%Y%m%d-%H%M%S"), encoding="utf-8")
    stamp = stamp_file.read_text(encoding="utf-8").strip()
    tasks = cfg["tasks"]()
    (data / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
    pending = [(d, ids, arm) for d, ids in tasks.items() for arm in ("base", "tools")
               if not (data / f"{arm}-{d}.json").exists()]
    if not pending:
        return
    glim = Meter(Caps(meta_usd=cfg["glimmer_usd"], gpu_usd=0.0), phase=f"{cfg['save']}-glimmer-{stamp}")
    meta = Meter(Caps(meta_usd=cfg["meta_usd"], gpu_usd=0.0), phase=f"{cfg['save']}-meta-{stamp}")
    gpu = Meter(Caps(meta_usd=0.0, gpu_usd=cfg["gpu_usd"]), phase=f"{cfg['save']}-gpu-{stamp}")
    litellm.callbacks = [litellm_callback((glim, ("Muse-Glimmer",)), (meta, ("muse-spark", "muse-wonder")))]
    base, glimmer = llm_utils.completion, reparsing(llm_utils.completion)

    def route(*args, **kwargs):
        return (glimmer if "Muse-Glimmer" in kwargs.get("model", "") else base)(*args, **kwargs)

    llm_utils.completion = guarded(guarded(route, glim, only=("Muse-Glimmer",)), meta, RateLimiter(per_minute=120),
                                   only=("muse-spark", "muse-wonder"))
    nl.DEFAULT_LLM_NL_ASSERTIONS, nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = SPARK, {**META, "temperature": 0.0}
    with ExitStack() as stack:
        endpoint = None
        if cfg.get("modal"):
            env = os.environ | {"MODAL_CONFIG_PATH": r"<workspace>\.modal.toml", "MODAL_PROFILE": "quanticsoul4772",
                                "PYTHONIOENCODING": "utf-8"}
            endpoint = glimmer_serve.deploy(MODAL_CLI, env)
            stack.enter_context(glimmer_serve.session(gpu, MODAL_CLI, env, tag=cfg["save"]))
            log = stack.enter_context(glimmer_serve.LogWatch(MODAL_CLI, env))
            glimmer_serve.wait_ready(endpoint, os.environ["MODAL_PROXY_KEY"], os.environ["MODAL_PROXY_SECRET"],
                                     failed=log.failure, log_lines=lambda: len(log.lines))
        for domain, ids, arm in pending:
            if arm == "tools":
                install(domain)
            results = _run(cfg, domain, ids, arm, stamp, endpoint)
            (data / f"{arm}-{domain}.json").write_text(results.model_dump_json(), encoding="utf-8")
            print(f"{arm} {domain}: {len(results.simulations)} sims; Glimmer ${glim.meta_usd:.3f}, "
                  f"Meta ${meta.meta_usd:.3f}", flush=True)


def analyze(cfg: dict) -> None:
    report = {}
    # Domains come from the result files themselves (the design run predates tasks.json)
    domains = sorted({p.stem.split("-", 1)[1] for p in cfg["data"].glob("*-*.json") if p.stem != "tasks"})
    if not domains:
        raise FileNotFoundError(f"no results in {cfg['data']}")
    for arm in ("base", "tools"):
        for domain in domains:
            path = cfg["data"] / f"{arm}-{domain}.json"
            if not path.exists():
                continue
            for s in json.loads(path.read_text(encoding="utf-8"))["simulations"]:
                if s["trial"] >= _trials(cfg, domain):  # a resumed checkpoint can hold trials beyond this k
                    continue
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
    cfg = for_backend(SETS[next((a for a in args if a in SETS), "design")], "--modal" in args)
    if "--analyze" not in args:
        generate(cfg)
    analyze(cfg)
