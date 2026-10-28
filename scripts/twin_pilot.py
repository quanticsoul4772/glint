"""User-intent twin yield pilot: how often does Glimmer contradict a preference the user stated verbatim?
15 fit/pilot retail tasks with a choice point (up to 5 payment, the rest variant), twins A and B, k=2, two
conditions: plain (the user states the preference when it comes up) and early (the user states it in the first
message, many turns before the write). Glimmer on DeepInfra at medium with glint.atem parsing; contributor Spark
user simulator. Spend cap $3. Usage: python twin_pilot.py [--analyze]
"""
import hashlib
import json
import sys
from collections import Counter

from m2_pilot import GLIMMER, META, SPARK, glimmer_args, install
from tau2.data_model.simulation import TextRunConfig
from tau2.data_model.tasks import Task
from tau2.runner.batch import run_tasks

from glint import paths
from glint.meter import Caps, Meter
from glint.twins import (
    ChoicePoint,
    at_choice_point,
    contradicts,
    find_choice_points,
    make_twins,
    stated_phrases,
)

RETAIL = paths.tau2_data() / "tau2" / "domains" / "retail"
DATA = paths.MEASURE / "twins"
OUT = paths.RESULTS / "M2-twin-pilot.json"
EARLY = " State this preference in your very first message, before anything else."


def build():
    db = json.loads((RETAIL / "db.json").read_text(encoding="utf-8"))
    tasks = {str(t["id"]): t for t in json.loads((RETAIL / "tasks.json").read_text(encoding="utf-8"))}
    part = json.loads((paths.RESULTS / "split-partition.json").read_text())["retail"]
    order = sorted(part["pilot"] + part["fit"], key=lambda t: hashlib.sha256(f"twin-pilot:{t}".encode()).hexdigest())
    chosen, n_pay = [], 0
    for tid in order:
        points = find_choice_points(tasks[tid], db)
        pay = [p for p in points if p.kind == "payment"]
        var = [p for p in points if p.kind == "variant"]
        if pay and n_pay < 5:
            chosen.append((tid, pay[0]))
            n_pay += 1
        elif var and len(chosen) - n_pay < 10:
            chosen.append((tid, var[0]))
        if len(chosen) == 15:
            break
    twins = {"plain": [], "early": []}
    for tid, point in chosen:
        for twin in make_twins(tasks[tid], point, db):
            twins["plain"].append(twin)
            early = json.loads(json.dumps(twin))
            early["id"] += "-early"
            early["user_scenario"]["instructions"]["reason_for_call"] += EARLY
            twins["early"].append(early)
    return twins


def generate():
    meter = Meter(Caps(meta_usd=3.0, gpu_usd=0.0), phase="twin-pilot")
    install(meter)
    DATA.mkdir(parents=True, exist_ok=True)
    twins = build()
    (DATA / "twins.json").write_text(json.dumps(twins, indent=1), encoding="utf-8")
    for cond, items in twins.items():
        out = DATA / f"{cond}.json"
        if out.exists():
            continue
        tasks = [Task.model_validate({k: v for k, v in t.items() if k != "glint_twin"}) for t in items]
        config = TextRunConfig(domain="retail", agent="llm_agent", llm_agent=GLIMMER,
                               llm_args_agent=glimmer_args("medium"), user="user_simulator", llm_user=SPARK,
                               llm_args_user=META, num_trials=2, max_concurrency=25, seed=2000,
                               save_to=f"glint-twin-pilot-{cond}", auto_review=False)
        results = run_tasks(config, tasks)
        out.write_text(results.model_dump_json(), encoding="utf-8")
        print(f"{cond}: {len(results.simulations)} sims; spend so far ${meter.meta_usd:.2f}", flush=True)


def analyze():
    db = json.loads((RETAIL / "db.json").read_text(encoding="utf-8"))
    twins = json.loads((DATA / "twins.json").read_text(encoding="utf-8"))
    report = {}
    for cond, items in twins.items():
        for t in items:
            # re-derive (option name, value) pairs: twins.json may predate the whole-word check
            g = t["glint_twin"]
            point = ChoicePoint(g["source_task"], 0, g["kind"], g["order_id"], g["old_item"], g["stated_value"], ())
            g["stated_phrases"] = [list(p) for p in stated_phrases(point, g["stated_value"], db)]
        meta = {t["id"]: t for t in items}
        sims = json.loads((DATA / f"{cond}.json").read_text(encoding="utf-8"))["simulations"]
        by_twin = {"A": Counter(), "B": Counter()}
        per_task = Counter()
        for s in sims:
            twin = meta[str(s["task_id"])]
            c = by_twin[twin["id"].removesuffix("-early")[-1]]
            c["episodes"] += 1
            # an episode's label is its first write at the choice point once the preference was stated;
            # later writes there (retries after a tool error, corrections) are not counted again
            first = None
            for i, m in enumerate(s["messages"]):
                if m["role"] != "assistant" or first is not None:
                    continue
                for tc in m.get("tool_calls") or []:
                    if not at_choice_point(twin, tc["name"], tc["arguments"]):
                        continue
                    v = contradicts(twin, tc["name"], tc["arguments"], s["messages"][:i])
                    if v is None:
                        c["writes_before_preference_stated"] += 1
                    else:
                        first = v
                        break
            if first is None:
                c["episodes_without_a_labeled_write"] += 1
            else:
                c["labeled_episodes"] += 1
                c["contradicting_episodes"] += first
                if first:
                    per_task[twin["glint_twin"]["source_task"]] += 1
        report[cond] = {"sims": len(sims), "twin_A": dict(by_twin["A"]), "twin_B": dict(by_twin["B"]),
                        "tasks_with_a_contradiction": len(per_task), "by_task": dict(per_task)}
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    if sys.argv[1:] != ["--analyze"]:
        generate()
    analyze()
