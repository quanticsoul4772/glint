"""Export Bank Desk replay scenarios from recorded with/without banking runs (research/3-project/demo-spec.md).

For each task, a recorded no-tools run that failed is paired with a with-tools run of the same task that passed
(both arms on Modal, same tools version). Each run becomes a timeline of events: customer and agent messages, lookups,
actions (hidden bank tools, unwrapped), Glint verdicts (the decision-tool calls and their results), and the outcome.
The no-tools run's actions are labelled with glint.decisions, so the page can show which ones a Glint tool would
have overruled and why. Writes src/glint/desk/scenarios.json (τ² banking data, MIT licensed).

Usage: python export_scenarios.py
"""
import json
from pathlib import Path

import tools_ab  # its analysis gives the counts the Results tab shows
from tau2.data_model.tasks import Task
from tau2.domains.banking_knowledge.environment import get_environment

from glint import paths
from glint.decisions import label_run
from glint.desk.events import customer_summary, outcome, timeline
from glint.runlog import NOT_RESULTS
from glint.runlog import outcome as run_outcome

DATA = paths.MEASURE
TASKS = paths.tau2_data() / "tau2" / "domains" / "banking_knowledge" / "tasks.json"
OUT = Path(__file__).resolve().parents[1] / "src" / "glint" / "desk" / "scenarios.json"
# (task, folder, tuned) for tasks whose base arm failed and tools arm passed at least once; tuned = the tools or the
# gates were refined after reading the task's runs (054 and 038 for the tools; 052 for the gates since 2026-10-30,
# by_task.TUNED)
SOURCES = [("task_054", "banking-failures", True), ("task_038", "banking-failures", True),
           ("task_050", "banking-other", False), ("task_052", "banking-other", True)]
# Gated pairs (decision 2026-10-31): Glimmer alone beside Glimmer with Glint's tools and every write gated, the same
# trial (seed) in both arms, where alone failed and the gated run passed after at least one policy-gate deny. The arms
# come from different sets: (scenario id, task, (alone set, arm), (gated set, arm), tuned, picked, title, note).
# banking-plan-gated's own tools arm (an earlier version of the gates) passed task_054 4/4 on the same seeds with no
# deny, so a pair shows the gate stopping a wrong write and the agent following its fix; it is not evidence that the
# gate caused the pass (claims review 2026-10-31).
# The untuned unseen pair is listed first, so the desk opens on it (judge panel 2026-11-03: a tuned default reads as
# cherry-picked); the tuned pair stays available.
GATED_SOURCES = [
    ("task_045-unseen", "task_045", ("banking-unseen-lean-gated", "base"), ("banking-unseen-lean-gated", "tools"),
     False, True, "Card closure on unseen task 045 (gated)",
     ("Both arms from banking-unseen-gated (2026-10-30 to 31), same seed, a task whose runs no gate was built from. "
      "Across the set, alone's misses include 30 context overflows, and the gated arm hit τ²'s step limit in 41 of "
      "its 108 runs (research note of 2026-10-31).")),
    ("task_054-gates", "task_054", ("banking-plan-gated", "base"), ("banking-plan-v9-gated", "tools"), True, False,
     "Fraud dispute, replacement card and a credit limit increase (gated)",
     ("Alone: the ungated arm of banking-plan-gated (2026-10-29). Gated: v9 (2026-10-30), same seed. An earlier gated "
      "version (banking-plan-gated's tools arm) passed this task 4/4 on the same seeds without blocking anything.")),
]
TITLES = {"task_054": "Fraud dispute, replacement card and a credit limit increase",
          "task_038": "Three disputed charges on a card that is being reissued",
          "task_050": "Credit limit increase request",
          "task_052": "Credit limit increase with a recent prior increase"}


def _arm(set_name: str, arm: str, cache: dict) -> tuple[list[dict], dict, str]:
    """One arm of one set: its result runs (trials below k, no run that is not a result), its analysis counts by
    task, and its folder name."""
    if set_name not in cache:
        cfg = tools_ab.for_backend(tools_ab.SETS[set_name], True)
        tools_ab.analyze(cfg)
        cache[set_name] = (cfg, json.loads(cfg["out"].read_text(encoding="utf-8")))
    cfg, report = cache[set_name]
    k = tools_ab._trials(cfg, "banking_knowledge")
    sims = [s for s in json.loads((DATA / cfg["data"].name / f"{arm}-banking_knowledge.json").read_text(
        encoding="utf-8"))["simulations"] if s["trial"] < k and run_outcome(s) not in NOT_RESULTS]
    return sims, report, cfg["data"].name


def _denies(sim: dict) -> int:
    return sum(e["kind"] == "gate" for e in timeline(sim["messages"]))


def gated_scenarios(tasks: dict, cache: dict) -> list[dict]:
    """GATED_SOURCES as scenarios. Each pair is one trial (seed) where alone failed and the gated run passed after a
    gate deny; `set_result` is the whole set's matched result (every trial both arms finished), shown beside a
    picked example."""
    out = []
    for sid, task_id, (a_set, a_arm), (g_set, g_arm), tuned, picked, title, note in GATED_SOURCES:
        a_sims, a_report, a_folder = _arm(a_set, a_arm, cache)
        g_sims, g_report, g_folder = _arm(g_set, g_arm, cache)
        alone_by = {s["trial"]: s for s in a_sims if s["task_id"] == task_id and s["messages"]}
        gated_by = {s["trial"]: s for s in g_sims if s["task_id"] == task_id and s["messages"]}
        trial = next((t for t in sorted(alone_by) if t in gated_by and not outcome(alone_by[t])["passed"]
                      and outcome(gated_by[t])["passed"] and _denies(gated_by[t])), None)
        if trial is None:
            print(f"WARNING: {sid} left out of the replay: no trial where alone failed and the gated run passed after "
                  f"a deny", flush=True)
            continue
        alone, gated = alone_by[trial], gated_by[trial]
        task = Task.model_validate(tasks[task_id])
        decisions = label_run(alone, task, get_environment(retrieval_variant="bm25_grep"))
        overrules = {d.message_index: {"tool": d.tool, "reasons": d.reasons}
                     for d in decisions if d.label == "overruled" and d.executed}
        a_counts = a_report[f"banking_knowledge-{task_id}"][a_arm]
        g_counts = g_report[f"banking_knowledge-{task_id}"][g_arm]
        scenario = {
            "id": sid, "task_id": task_id, "title": title, "tuned": tuned, "variant": "gates", "picked": picked,
            "customer": customer_summary(str(task.user_scenario.instructions)),
            "source": f"{a_folder if a_folder == g_folder else f'{a_folder} and {g_folder}'} (Glimmer 30B bf16 on "
                      f"Modal). {note}",
            "recorded": {"base": {"passed": a_counts["passes"], "runs": a_counts["runs"]},
                         "tools": {"passed": g_counts["passes"], "runs": g_counts["runs"]}},
            "alone": {"trial": trial, "events": timeline(alone["messages"], overrules), "outcome": outcome(alone),
                      "overruled_writes": len(overrules)},
            "tools": {"trial": trial, "events": timeline(gated["messages"]), "outcome": outcome(gated),
                      "gate_blocks": _denies(gated)},
        }
        if picked:
            alone_all = {(s["task_id"], s["trial"]): s for s in a_sims}
            gated_all = {(s["task_id"], s["trial"]): s for s in g_sims}
            matched = [(alone_all[key], gated_all[key]) for key in alone_all if key in gated_all]
            scenario["set_result"] = {"set": g_set, "pairs": len(matched),
                                      "alone_passed": sum(outcome(a)["passed"] for a, _ in matched),
                                      "gated_passed": sum(outcome(g)["passed"] for _, g in matched)}
        out.append(scenario)
        print(f"{sid}: trial {trial}, {len(overrules)} overruled writes alone, {scenario['tools']['gate_blocks']} "
              f"gate blocks")
    return out


def main():
    tasks = {t["id"]: t for t in json.loads(TASKS.read_text(encoding="utf-8"))}
    reports = {}
    scenarios = gated_scenarios(tasks, {})  # listed first: the gated pairs are what Glint ships now
    for task_id, set_name, tuned in SOURCES:
        cfg = tools_ab.for_backend(tools_ab.SETS[set_name], True)
        if set_name not in reports:
            tools_ab.analyze(cfg)
            reports[set_name] = json.loads(cfg["out"].read_text(encoding="utf-8"))
        counts = reports[set_name][f"banking_knowledge-{task_id}"]
        folder, k = cfg["data"].name, tools_ab._trials(cfg, "banking_knowledge")
        task = Task.model_validate(tasks[task_id])
        # the same runs the analysis counts: trials below k, and no run that is not a result
        arms = {arm: [s for s in json.loads((DATA / folder / f"{arm}-banking_knowledge.json")
                                            .read_text(encoding="utf-8"))["simulations"]
                      if s["task_id"] == task_id and s["trial"] < k and run_outcome(s) not in NOT_RESULTS]
                for arm in ("base", "tools")}
        alone = next((s for s in arms["base"] if not outcome(s)["passed"] and s["messages"]), None)
        helped = next((s for s in arms["tools"] if outcome(s)["passed"] and s["messages"]), None)
        if alone is None or helped is None:
            print(f"WARNING: {task_id} left out of the replay: no failed-alone / passed-with-tools pair in "
                  f"{folder}", flush=True)
            continue
        decisions = label_run(alone, task, get_environment(retrieval_variant="bm25_grep"))
        overrules = {d.message_index: {"tool": d.tool, "reasons": d.reasons}
                     for d in decisions if d.label == "overruled" and d.executed}
        scenarios.append({
            "id": task_id, "task_id": task_id, "title": TITLES.get(task_id, task_id), "tuned": tuned, "variant": "tools",
            "customer": customer_summary(str(task.user_scenario.instructions)),
            "source": f"{folder} (Glimmer 30B bf16 on Modal; recorded 2026-10-27)",
            # the pair is chosen (a failed run alone, a passed run with tools); these are all the recorded runs,
            # taken from the same analysis as the Results tab
            "recorded": {arm: {"passed": counts[arm]["passes"], "runs": counts[arm]["runs"]} for arm in arms},
            "alone": {"trial": alone["trial"], "events": timeline(alone["messages"], overrules), "outcome": outcome(alone),
                      "overruled_writes": len(overrules)},
            "tools": {"trial": helped["trial"], "events": timeline(helped["messages"]), "outcome": outcome(helped)},
        })
        print(f"{task_id}: alone trial {alone['trial']} ({len(overrules)} overruled writes), "
              f"tools trial {helped['trial']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(scenarios, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(scenarios)} scenarios to {OUT}")


if __name__ == "__main__":
    main()
