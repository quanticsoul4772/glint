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
- `heldout2`: a fresh draw of 10 retail + 10 airline train tasks (seed 2027) excluding the design tasks, the whole
  failure inventory (twin-B 21, 44, 78 included) and every task of the first held-out draw; k=4.
- `banking-other`: the other 10 banking tasks of `banking` (not tuned on), k=2.
- `gate-banking`: activation-gate training data, no-tools arm only: the 12 banking tasks x 25 trials, concurrency 12.

Optional set key `connections`: the size of LiteLLM's shared HTTP connection pool. When present, `generate()`
replaces τ²'s shared clients (sync and async; 10 connections) with ones of that size before any call, still following
Modal's redirects and carrying the logging hooks (`glimmer_serve.use_connections`). τ² runs `concurrency` simulations,
plus the user simulator's calls, on that one pool, so a smaller pool queues requests silently (a POOL WAIT line in the
log reports it). Sets without the key keep τ²'s client unchanged, so their runs stay comparable; no existing set has
it.

Usage: python tools_ab.py [design|heldout|heldout2|banking|banking-failures|banking-other|gate-banking]
       [--modal] [--analyze] [--allow-mixed-code] [--rerun-infrastructure-errors]

`--rerun-infrastructure-errors` (needs `--allow-mixed-code` when the code changed): the runs that ended in an
infrastructure error are taken out of τ²'s checkpoint, the arm file and events.jsonl, and kept in
`released-<arm>-<domain>-<time>.json` with the commit that releases them; the resume then reruns exactly those
(task, trial) pairs with their own seeds.
"""
import json
import os
import random
import sys
import threading
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

from glint import glimmer_serve, paths
from glint.assist.tau2_tools import install
from glint.atem import reparsing
from glint.calllog import recording
from glint.meter import BudgetExceeded, Caps, Meter, guarded, litellm_callback, register_litellm_prices
from glint.ratelimit import RateLimiter
from glint.runlog import RunEvents, outcome

TOOL_NAMES = {"check_order_action", "find_user_items", "find_variants", "check_reservation_action",
              "get_upcoming_trips_cost", "check_credit_limit_increase", "check_card_dispute"}
# Tasks the tools were designed from, or that are in the failure inventory
DESIGN_TASKS = {"retail": {"20", "93", "82", "52", "98", "103", "104"}, "airline": {"7"}}
MEASURE = paths.MEASURE
RESULTS = paths.RESULTS
BANKING_TASKS = paths.tau2_data() / "tau2" / "domains" / "banking_knowledge" / "tasks.json"
BANKING_SPLIT = RESULTS / "split-partition-banking.json"
BANKING_WRITES = ("file_credit_card_transaction_dispute_4829", "submit_credit_limit_increase_request_7392")
MODAL_CLI = str(Path(__file__).resolve().parents[1] / ".venv" / "Scripts" / "modal.exe")


def heldout_tasks(n: int = 10, seed: int = 2026, exclude: dict[str, set[str]] | None = None) -> dict[str, list[str]]:
    """The first held-out draw used DESIGN_TASKS only, which misses the twin-B tasks of the failure inventory (21, 44,
    78); task 21 was drawn. Kept as it was so the recorded draw reproduces."""
    rng = random.Random(seed)
    out = {}
    for domain in ("retail", "airline"):
        skip = DESIGN_TASKS[domain] | (exclude or {}).get(domain, set())
        ids = [t.id for t in registry.get_tasks_loader(domain)("train") if t.id not in skip]
        out[domain] = sorted(rng.sample(ids, n), key=int)
    return out


# Tasks the tools or their fixes have been tuned on, or that are in the failure inventory: the design tasks, the twin-B
# tasks, and the first held-out draw (its failures produced the airline-hint and return/exchange fixes)
INVENTORY = {"retail": {"21", "44", "78"}, "airline": set()}


def heldout2_tasks(n: int = 10, seed: int = 2027) -> dict[str, list[str]]:
    first = heldout_tasks()
    return heldout_tasks(n, seed, {d: INVENTORY[d] | set(first[d]) for d in ("retail", "airline")})


def banking_test_tasks() -> dict[str, list[str]]:
    """τ²'s untouched banking test split (all 20 tasks): the final check, never used to change the tools."""
    return {"banking_knowledge": sorted(json.loads(BANKING_SPLIT.read_text(encoding="utf-8"))["test_untouched"])}


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
    "heldout2": {"tasks": heldout2_tasks, "trials": 4, "concurrency": 12, "glimmer_usd": 3.5, "meta_usd": 1.0,
                 "gpu_usd": 8.0,
                 "data": MEASURE / "tools_ab_heldout2", "out": RESULTS / "tools-ab-heldout2.json",
                 "save": "tools-ab-heldout2"},
    # fresh folders (2026-10-28): the earlier banking stamp folder holds runs from before the checkpoint fix
    "banking": {"tasks": banking_tasks, "trials": 4, "concurrency": 12, "glimmer_usd": 3.0, "meta_usd": 1.5,
                "gpu_usd": 12.0, "data": MEASURE / "tools_ab_banking12", "out": RESULTS / "tools-ab-banking12.json",
                "save": "tools-ab-banking12"},
    # The final check on τ²'s untouched banking test split; the tools are frozen
    "banking-test": {"tasks": banking_test_tasks, "trials": 4, "concurrency": 12, "glimmer_usd": 3.0, "meta_usd": 2.0,
                     "gpu_usd": 20.0, "data": MEASURE / "tools_ab_banking_test",
                     "out": RESULTS / "tools-ab-banking-test.json", "save": "tools-ab-banking-test"},
    # The 10 banking tasks not tuned on (the banking tools were changed after reading 038 and 054 runs only); k=2 to
    # keep Modal credit for activation capture
    "banking-other": {"tasks": lambda: {"banking_knowledge": [t for t in banking_tasks()["banking_knowledge"]
                                                              if t not in ("task_038", "task_054")]},
                      "trials": 2, "glimmer_usd": 3.0, "meta_usd": 1.0, "gpu_usd": 5.0,
                      "data": MEASURE / "tools_ab_banking_other", "out": RESULTS / "tools-ab-banking-other.json",
                      "save": "tools-ab-banking-other"},
    # Activation-gate data (research/3-project/activation-gate-design.md): no-tools runs only, labelled afterwards by
    # glint.decisions; 12 tasks x 25 trials
    "gate-banking": {"tasks": banking_tasks, "trials": 25, "arms": ("base",), "concurrency": 12, "glimmer_usd": 3.0,
                     "meta_usd": 3.0, "gpu_usd": 30.0, "data": MEASURE / "gate_banking",
                     "out": RESULTS / "gate-banking-runs.json", "save": "gate-banking"},
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


def _save_dir(cfg: dict, arm: str, domain: str, stamp: str) -> Path:
    """τ²'s checkpoint folder for one arm, the same path run_domain derives from save_to."""
    from tau2.utils.utils import DATA_DIR

    return DATA_DIR / "simulations" / f"glint-{cfg['save']}-{arm}-{domain}-{stamp}"


def _run(cfg: dict, domain: str, ids: list[str], arm: str, stamp: str, endpoint: str | None):
    model, args = _agent(cfg, domain, endpoint)
    common = {"domain": domain, "agent": "llm_agent", "llm_agent": model, "llm_args_agent": args,
              "user": "user_simulator", "llm_user": SPARK, "llm_args_user": META, "num_trials": _trials(cfg, domain),
              "max_concurrency": cfg.get("concurrency", 8) if endpoint else 4, "seed": 5000,
              "save_to": f"glint-{cfg['save']}-{arm}-{domain}-{stamp}", "auto_review": False, "max_retries": 0,
              "auto_resume": True,
              # the run log gets only τ²'s warnings and errors; each simulation's full debug log goes to its own
              # artifacts/task_<id>/sim_<id>/task.log in the checkpoint folder (95% of the run log was τ² debug)
              "log_level": "WARNING", "verbose_logs": True}
    if domain == "banking_knowledge":
        by_id = {t["id"]: t for t in json.loads(BANKING_TASKS.read_text(encoding="utf-8"))}
        config = TextRunConfig(**common, retrieval_config="bm25_grep")
        # run_tasks persists nothing without save_path (its docstring), so banking runs had no checkpoint: a crash
        # lost every finished run and resume could not work. Pass the paths run_domain would derive.
        save_dir = _save_dir(cfg, arm, domain, stamp)
        return run_tasks(config, [Task.model_validate(by_id[i]) for i in ids], save_path=save_dir / "results.json",
                         save_dir=save_dir)
    return run_domain(TextRunConfig(**common, task_split_name="train", task_ids=ids))


def _commit() -> str:
    import subprocess

    # check=True and a non-empty commit: an empty commit would switch the mixed-code guard off for good
    repo = str(Path(__file__).resolve().parents[1])
    out = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                         check=True).stdout.strip()
    if not out:
        raise SystemExit(f"git rev-parse gave no commit for {repo}; the run cannot record which code it measures")
    dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain", "src"], capture_output=True, text=True,
                           check=True).stdout.strip()
    return out + ("-dirty" if dirty else "")


KEPT_ON_RESUME = ("context_overflow", "glint_error")  # failed runs: results, never rerun


def keep_failed_runs_on_resume() -> None:
    """τ²'s resume (tau2.runner.checkpoint.try_resume) reruns every simulation whose termination reason is
    infrastructure_error, and rewrites the checkpoint without them. Glint splits that reason (glint.runlog.outcome):
    a context overflow or a Glint error is a failed run, a result, and must not be rerun until it passes. This wraps
    the resume τ²'s batch runner calls so those runs stay as recorded, on disk and in the runs it skips; only
    Glint's infrastructure_error and budget_stop runs (not results) are rerun."""
    from tau2.data_model.simulation import Results
    from tau2.runner import batch

    if getattr(batch.try_resume, "glint_keeps_failures", False):
        return
    resume = batch.try_resume

    def wrapped(save_path, simulation_results, tasks, num_trials, auto_resume=False, results_format="json"):
        save_path = Path(save_path)
        before = Results.load(save_path).simulations if save_path.exists() else []
        if save_path.exists() and Results._detect_format(save_path) != "json":
            raise RuntimeError(f"{save_path}: τ²'s dir checkpoint format; the failed-run keeper handles json only")
        keep = [s for s in before if outcome(s.model_dump(mode="json")) in KEPT_ON_RESUME]
        results, done, tasks = resume(save_path, simulation_results, tasks, num_trials, auto_resume, results_format)
        present = {s.id for s in results.simulations}
        back = [s for s in keep if s.id not in present]
        if back:
            results.simulations.extend(back)
            done |= {(s.trial, s.task_id, s.seed) for s in back}
            save_path.write_text(results.model_dump_json(indent=2), encoding="utf-8")
            print(f"KEPT {len(back)} failed runs (context overflow or Glint error) as recorded; τ²'s resume would "
                  f"have rerun them: {sorted((s.task_id, s.trial) for s in back)}", flush=True)
        return results, done, tasks

    wrapped.glint_keeps_failures = True
    batch.try_resume = wrapped


SAVE_TRIES, SAVE_WAIT_S = 6, 0.5


def retry_locked_checkpoint_saves() -> None:
    """τ² saves its checkpoint by renaming a temp file over results.json, once. On Windows the rename fails with
    "[WinError 5] Access is denied" while another process has results.json open (the event watcher reads it every
    30 s; antivirus and indexers also open files), and τ² then drops the finished run as an infrastructure error
    (2026-10-28, banking-test task_022 trial 3). This wraps the save τ²'s batch runner uses: a refused save is
    retried a few times, each refusal printed; if the lock outlasts the retries the error is raised as before."""
    from tau2.runner import batch

    if getattr(batch.create_checkpoint_fns, "glint_retries_saves", False):
        return
    create = batch.create_checkpoint_fns

    def retrying(fn, what):
        def call(*args, **kwargs):
            for attempt in range(1, SAVE_TRIES + 1):
                try:
                    return fn(*args, **kwargs)
                except PermissionError as e:
                    print(f"CHECKPOINT LOCKED: τ²'s {what} was refused (attempt {attempt}/{SAVE_TRIES}): {e}",
                          flush=True)
                    if attempt == SAVE_TRIES:
                        raise
                    time.sleep(SAVE_WAIT_S)
        return call

    def wrapped(save_path, lock):
        save_fn, replace_fn = create(save_path, lock)
        return retrying(save_fn, "checkpoint save"), retrying(replace_fn, "checkpoint replace")

    wrapped.glint_retries_saves = True
    batch.create_checkpoint_fns = wrapped


def release_infrastructure_errors(cfg: dict, stamp: str, commit: str) -> int:
    """Take the infrastructure-error runs out of each arm's checkpoint so a resume reruns them (see the docstring
    above). Nothing is deleted: the released runs are written to a file first. Returns how many were released."""
    data, released = cfg["data"], 0
    tasks = json.loads((data / "tasks.json").read_text(encoding="utf-8")) if (data / "tasks.json").exists() else {}
    for domain in tasks:
        for arm in cfg.get("arms", ("base", "tools")):
            checkpoint = _save_dir(cfg, arm, domain, stamp) / "results.json"
            if not checkpoint.exists():
                continue
            results = json.loads(checkpoint.read_text(encoding="utf-8"))
            drop = [s for s in results["simulations"] if outcome(s) == "infrastructure_error"]
            if not drop:
                continue
            keys = {(s["task_id"], s["trial"]) for s in drop}
            side = data / f"released-{arm}-{domain}-{time.strftime('%Y%m%d-%H%M%S')}.json"
            side.write_text(json.dumps({"released_on_commit": commit, "simulations": drop}), encoding="utf-8")
            results["simulations"] = [s for s in results["simulations"] if (s["task_id"], s["trial"]) not in keys]
            checkpoint.write_text(json.dumps(results), encoding="utf-8")
            (data / f"{arm}-{domain}.json").unlink(missing_ok=True)  # the arm runs again; τ² skips the kept runs
            events = data / "events.jsonl"
            if events.exists():
                rows = [json.loads(x) for x in events.read_text(encoding="utf-8").splitlines()]
                events.write_text("".join(json.dumps(r) + "\n" for r in rows if not (
                    r["label"] == f"{arm} {domain}" and r["outcome"] == "infrastructure_error"
                    and (r["task_id"], r["trial"]) in keys)), encoding="utf-8")
            with (cfg.get("stamp_dir", data) / "reruns.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.time(), "commit": commit, "arm": arm, "domain": domain,
                                    "runs": sorted(keys)}) + "\n")
            print(f"RELEASED {len(drop)} infrastructure-error runs of {arm} {domain} for a rerun on {commit}: "
                  f"{sorted(keys)}; kept in {side.name}", flush=True)
            released += len(drop)
    return released


def generate(cfg: dict) -> None:
    # One stamp per comparison: a resumed run keeps the same meter phases (so the caps count earlier spend) and the
    # same τ² save_to (so finished simulations are resumed, not paid for again)
    data = cfg["data"]
    data.mkdir(parents=True, exist_ok=True)
    stamp_dir = cfg.get("stamp_dir", data)
    stamp_dir.mkdir(parents=True, exist_ok=True)
    stamp_file = stamp_dir / "stamp.txt"
    commit_file = stamp_dir / "commit.txt"
    commit = _commit()
    if not stamp_file.exists():
        stamp_file.write_text(time.strftime("%Y%m%d-%H%M%S"), encoding="utf-8")
        commit_file.write_text(commit, encoding="utf-8")
    stamp = stamp_file.read_text(encoding="utf-8").strip()
    # A resume on different code would merge runs of two code versions into one comparison (the first held-out
    # arms span four commits)
    started = commit_file.read_text(encoding="utf-8").strip() if commit_file.exists() else None
    if started == "" and "--allow-mixed-code" not in sys.argv:
        raise SystemExit(f"{commit_file} is empty: which code this comparison started on is unknown. Write the "
                         "starting commit into it, or pass --allow-mixed-code to resume knowingly")
    if started and started != commit and "--allow-mixed-code" not in sys.argv:
        raise SystemExit(f"this comparison started on commit {started}, the code is now {commit}; resuming would mix "
                         "versions. Start a fresh stamp, or pass --allow-mixed-code.")
    keep_failed_runs_on_resume()
    retry_locked_checkpoint_saves()
    if "--rerun-infrastructure-errors" in sys.argv and not release_infrastructure_errors(cfg, stamp, commit):
        raise SystemExit("--rerun-infrastructure-errors: no infrastructure-error runs to rerun")
    tasks = cfg["tasks"]()
    (data / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
    pending = [(d, ids, arm) for d, ids in tasks.items() for arm in cfg.get("arms", ("base", "tools"))
               if not (data / f"{arm}-{d}.json").exists()]
    if not pending:
        return
    if cfg.get("connections"):  # before any call: the pool size every call of the run shares (see the docstring)
        glimmer_serve.use_connections(cfg["connections"])
    glim = Meter(Caps(meta_usd=cfg["glimmer_usd"], gpu_usd=0.0), phase=f"{cfg['save']}-glimmer-{stamp}")
    meta = Meter(Caps(meta_usd=cfg["meta_usd"], gpu_usd=0.0), phase=f"{cfg['save']}-meta-{stamp}")
    gpu = Meter(Caps(meta_usd=0.0, gpu_usd=cfg["gpu_usd"]), phase=f"{cfg['save']}-gpu-{stamp}")
    register_litellm_prices()
    litellm.callbacks = [litellm_callback((glim, ("Muse-Glimmer",)), (meta, ("muse-spark", "muse-wonder")))]
    base, glimmer = llm_utils.completion, reparsing(llm_utils.completion)

    halt = threading.Event()  # set by glimmer_serve.session() when the GPU cap trips or its watcher fails

    def route(*args, **kwargs):
        if "Muse-Glimmer" not in kwargs.get("model", ""):
            return base(*args, **kwargs)
        if halt.is_set():
            raise BudgetExceeded("the Modal session was halted (GPU cap reached or the GPU watcher failed)")
        return glimmer(*args, **kwargs)

    # every Glimmer call's exact prompt and completion token ids (τ²'s saved runs keep neither), for activation capture
    recorded = recording(route, data / "calls.jsonl", require_ids=bool(cfg.get("modal")))
    llm_utils.completion = guarded(guarded(recorded, glim, only=("Muse-Glimmer",)), meta, RateLimiter(per_minute=120),
                                   only=("muse-spark", "muse-wonder"))
    nl.DEFAULT_LLM_NL_ASSERTIONS, nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = SPARK, {**META, "temperature": 0.0}
    with ExitStack() as stack:
        endpoint = None
        if cfg.get("modal"):
            env = os.environ | {"MODAL_CONFIG_PATH": str(paths.MODAL_CONFIG), "MODAL_PROFILE": "quanticsoul4772",
                                "PYTHONIOENCODING": "utf-8"}
            endpoint = glimmer_serve.deploy(MODAL_CLI, env)
            halt = stack.enter_context(glimmer_serve.session(gpu, MODAL_CLI, env, tag=cfg["save"]))
            log = stack.enter_context(glimmer_serve.LogWatch(MODAL_CLI, env))
            glimmer_serve.wait_ready(endpoint, os.environ["MODAL_PROXY_KEY"], os.environ["MODAL_PROXY_SECRET"],
                                     failed=log.failure, log_lines=lambda: len(log.lines))
        for domain, ids, arm in pending:
            if arm == "tools":
                install(domain)
            glim_before, meta_before = glim.meta_usd, meta.meta_usd
            events = RunEvents(_save_dir(cfg, arm, domain, stamp) / "results.json", data / "events.jsonl",
                               label=f"{arm} {domain}")
            with events:
                results = _run(cfg, domain, ids, arm, stamp, endpoint)
            if halt.is_set():  # the drained runs are not results: no arm file, so a resume reruns this arm
                raise BudgetExceeded(f"{arm} {domain}: the Modal session was halted mid-arm; the arm file is not "
                                     "written and a resumed run starts this arm again")
            kinds = Counter(outcome(s.model_dump(mode="json")) for s in results.simulations)
            if kinds["budget_stop"]:  # cap-stopped runs are not results: no arm file, so a resume reruns this arm
                raise BudgetExceeded(f"{arm} {domain}: {kinds['budget_stop']} runs were stopped by a spend cap; the "
                                     "arm file is not written and a resumed run starts this arm again")
            expected = len(ids) * _trials(cfg, domain)
            if len(results.simulations) < expected:
                print(f"WARNING: {arm} {domain}: {len(results.simulations)} simulations, expected {expected} "
                      f"({len(ids)} tasks x k={_trials(cfg, domain)}); the missing runs are not in this arm",
                      flush=True)
            (data / f"{arm}-{domain}.json").write_text(results.model_dump_json(), encoding="utf-8")
            if kinds["glint_error"]:
                print(f"WARNING: {arm} {domain}: {kinds['glint_error']} of {len(results.simulations)} runs failed in "
                      "Glint's own code (atem or call recording); they count as failed runs. Causes are in "
                      "info.error and events.jsonl", flush=True)
            if kinds["eval_error"]:
                print(f"WARNING: {arm} {domain}: {kinds['eval_error']} of {len(results.simulations)} runs have no "
                      "reward (τ²'s evaluation failed); they are excluded from the analysis", flush=True)
            if kinds["infrastructure_error"]:
                print(f"WARNING: {arm} {domain}: {kinds['infrastructure_error']} of {len(results.simulations)} runs "
                      "ended in an infrastructure error; they are excluded from the analysis and are not retried",
                      flush=True)
            if kinds["context_overflow"]:
                print(f"WARNING: {arm} {domain}: {kinds['context_overflow']} of {len(results.simulations)} runs "
                      "overflowed the context; they count as failed runs", flush=True)
            print(f"{arm} {domain}: {len(results.simulations)} sims; this arm Glimmer "
                  f"${glim.meta_usd - glim_before:.3f}, Meta ${meta.meta_usd - meta_before:.3f}; phase totals Glimmer "
                  f"${glim.meta_usd:.3f}, "
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
                print(f"WARNING: no {arm} arm for {domain} ({path.name} missing): its tasks have one arm only and "
                      "are left out of the comparison", flush=True)
                continue
            beyond_k = 0
            for s in json.loads(path.read_text(encoding="utf-8"))["simulations"]:
                if s["trial"] >= _trials(cfg, domain):  # a resumed checkpoint can hold trials beyond this k
                    beyond_k += 1
                    continue
                key = f"{domain}-{s['task_id']}"
                row = report.setdefault(key, {}).setdefault(arm, {"passes": 0, "runs": 0, "tool_calls": Counter(),
                                                                   "infra_errors": 0, "context_overflows": 0,
                                                                   "glint_errors": 0, "eval_errors": 0,
                                                                   "budget_stops": 0})
                kind = outcome(s)
                if kind in ("infrastructure_error", "eval_error", "budget_stop"):  # not results; counted below
                    row[{"infrastructure_error": "infra_errors", "eval_error": "eval_errors",
                         "budget_stop": "budget_stops"}[kind]] += 1
                    continue
                # a context overflow is the model's own conversation outgrowing the context, and a Glint error is
                # Glint's own failure: both are failed runs (τ² files them as infrastructure errors; the cause is in
                # info.error)
                row["context_overflows"] += kind == "context_overflow"
                row["glint_errors"] += kind == "glint_error"
                row["runs"] += 1
                row["passes"] += kind == "pass"
                for m in s["messages"]:
                    for tc in (m.get("tool_calls") or []) if m["role"] == "assistant" else []:
                        if tc["name"] in TOOL_NAMES:
                            row["tool_calls"][tc["name"]] += 1
            if beyond_k:
                print(f"NOTE: {arm} {domain}: {beyond_k} runs beyond k={_trials(cfg, domain)} left out (a resumed "
                      "checkpoint)", flush=True)
    for arms in report.values():
        for row in arms.values():
            row["tool_calls"] = dict(row["tool_calls"])
            row["pass_rate"] = round(row["passes"] / row["runs"], 3) if row["runs"] else None
    cfg["out"].write_text(json.dumps(report, indent=1), encoding="utf-8")
    # a tools arm that never called a Glint tool most likely ran without them (an install that did not take)
    for domain in domains:
        rows = [arms["tools"] for key, arms in report.items() if key.startswith(f"{domain}-") and "tools" in arms]
        if rows and not any(row["tool_calls"] for row in rows):
            print(f"WARNING: the tools arm for {domain} made 0 Glint tool calls in {sum(r['runs'] for r in rows)} "
                  "runs: check that the tools were installed", flush=True)
    print(json.dumps(report, indent=1))
    # Loud summary: excluded runs (infrastructure, evaluation, cap) can leave arms with different run counts, a task
    # can have one arm only, and Glint errors count against the tools
    for key, arms in sorted(report.items()):
        if set(arms) != {"base", "tools"}:
            print(f"WARNING: {key}: only the {', '.join(arms)} arm has runs; the task is not compared", flush=True)
        runs = {arm: row["runs"] for arm, row in arms.items()}
        excluded = {arm: {k: row[k] for k in ("infra_errors", "eval_errors", "budget_stops") if row[k]}
                    for arm, row in arms.items()}
        excluded = {arm: e for arm, e in excluded.items() if e}
        glint = {arm: row["glint_errors"] for arm, row in arms.items() if row["glint_errors"]}
        if excluded or glint or len(set(runs.values())) > 1:
            print(f"WARNING: {key}: runs per arm {runs}, excluded {excluded or 'none'}, Glint errors (failed runs) "
                  f"{glint or 'none'}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    cfg = for_backend(SETS[next((a for a in args if a in SETS), "design")], "--modal" in args)
    if "--analyze" not in args:
        generate(cfg)
    analyze(cfg)
