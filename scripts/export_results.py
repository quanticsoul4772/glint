"""Export the measured with/without results for the Bank Desk results tab (src/glint/desk/results.json).

Re-runs tools_ab's analysis for each comparison set (so context overflows count as failed runs), then writes, per
set: its role (held-out, untuned, tuned, design), backend, k, and per task the pass counts of both arms and whether
the task improved, tied or got worse; plus the set's totals by task. Numbers come only from the run data on disk.

Usage: python export_results.py
"""
import json
from pathlib import Path

from loguru import logger

logger.remove()  # τ² logs its registry at DEBUG on import

import by_task
import tools_ab  # after the log setup above
from glint.runlog import NOT_RESULTS

OUT = Path(__file__).resolve().parents[1] / "src" / "glint" / "desk" / "results.json"
TUNED_BANKING = {"task_038", "task_054"}  # the tools were refined after reading these tasks' runs
# the gates, the dispute plan and the open items were shaped by these tasks' runs or gold: 037-041, 054, 081, and
# since 2026-10-30 031, 051 and 052
TUNED_GATES = set(by_task.TUNED)
# a tools-only set -> the set whose base arm it is compared with (same tasks, trials and seeds; checked in main)
BASE_FROM = {"banking-plan-v9-gated": "banking-plan-gated", "banking-unseen-v10-gated": "banking-unseen-gated",
             "banking-test-v2-gated": "banking-test-gated"}
GATED_NOTE = ("Glint's decision tools with every write gated (a denied write does not run), on the code at PR 147, "
              "against the ungated arm without tools of banking-plan-gated (same 12 tasks, seeds and backend). Eight "
              "earlier reported gated versions are in docs/results.md; v9 is the latest, and every version is "
              "compared with one baseline draw. Later changes (PR 148, PR 149: refusal wording and which requests count "
              "as a credit limit increase) are not in this run. k=4. Glimmer on Modal.")
# The banking test split is design data: some debit rules came from its failures on 2026-10-28, and the rules of
# PRs 29-44 were designed from its 2026-11-02 gated run. Every set on it is labeled tuned or design, never held-out.
TEST_SPLIT_DESIGN = ("The rules of PRs 29-44 were designed from this split's 2026-11-02 gated run, so the split is "
                     "design data and no held-out banking test result is possible from here on.")
SETS = [  # (set, Modal backend, title, role, note[, only these tasks]): tuned and untuned tasks are never pooled
    ("banking-test-v2-gated", True,
     "Banking test split, gated arm after the rules designed from it (2026-11-03; tuned, in-sample)", "tuned",
     ("τ²'s 20 banking test tasks, k=2, the gated tools arm only, on the code with every fix on (main at 7bf75cb, PRs "
      "29-45). Its alone column is the 2026-11-02 alone arm of the run below: a cross-day comparison, and the alone "
      "arm was not rerun. Tuned and in-sample: " + TEST_SPLIT_DESIGN + " Passes 11/40 vs alone's 7/40, by task 3 "
      "better, 0 worse (sign test p = 0.25, two-sided); may be chance. Against τ²'s offline re-grade of the alone "
      "arm (10/40), 3 better, 1 worse (p = 0.63). Not re-graded. Step-limit endings 6 (the 2026-11-02 gated arm: 14). "
      "The gate refused 40 writes; 25 refused a write τ² expects, and 4 of those 25 were false against gold "
      "(request, verification once x2, closure reason). Glimmer on Modal.")),
    ("banking-test-gated", True, "Banking test split, every write gated (2026-11-02; design data for PRs 29-44)",
     "tuned",
     ("τ²'s 20 banking test tasks, k=2, Glimmer alone vs Glimmer with the gates and decision tools, on the "
      "code after PRs 20-25 (the request record, the retention answer, the savings interest rules). Some debit "
      "rules were written from this split's failures on 2026-10-28, and every conversation of this run was then "
      "read: " + TEST_SPLIT_DESIGN + " Passes 7/40 vs 7/40 (by task 2 better, 2 worse). "
      "Writes that differ from τ²'s expected ones: 34 alone vs 25 gated, extra writes 13 vs 4 (by task 5 fewer, 2 "
      "more; may be chance); wrong values 21 vs 21. 14 gated runs hit τ²'s step limit (10 knowledge-base "
      "search loops, 4 with request-record refusals); alone had 7 context overflows. Glimmer on Modal.")),
    ("banking-unseen-v10-gated", True, "Banking, 59 unseen tasks, gates plus the search fix (tuned on their failures)",
     "tuned",
     ("The same 59 tasks and seeds, k=2, the gated tools arm only, on the code after PR 158-174: the v10 fixes and the "
      "search fix of PR 171, which was designed after reading these tasks' step-limit failures, so the set is tuned "
      "on them. Its alone column is banking-unseen-gated's Glimmer-alone arm. No gain: on the old gated arm's 108 matched "
      "runs, 14 passed with the fix vs 19 before, and the step limit was hit in 34 vs 41; what keeps these runs "
      "searching is not established. Glimmer on Modal.")),
    ("banking-unseen-lean-gated", True, "Banking, 59 unseen tasks, write gates enforced", "untuned",
     ("The 59 banking training tasks outside the 12 the gates were built on, whose correct answers make a gated write; "
      "no gate, plan or tool change came from their runs. k=2, Glimmer on Modal, gated tools arm on the code at "
      "PR 151 (before the search fix of PR 171). No gain: 5 tasks better, 8 worse (sign test p = 0.58, two-sided); "
      "task_064 is no data because every run in both arms overflowed. Alone's misses include 30 context overflows; "
      "the gated arm hit τ²'s step limit in 41 of 108 runs (cause likely, not tested). The gate refused 21 writes: "
      "0 false (6 refused a write τ² expects until a required step was done); the other 15 refused writes τ² does "
      "not expect and are not judged.")),
    ("banking-plan-v9-gated", True, "Banking, untuned tasks, write gates enforced (v9)", "untuned",
     ("Tasks 050 and 053, which no gate, plan or open-item change came from. 053's failures are likely the simulated "
      "customer's answers (not checked); two tasks cannot reach p < 0.05 by sign test. " + GATED_NOTE),
     lambda task: task not in TUNED_GATES),
    ("banking-plan-v9-gated", True, "Banking, tuned tasks, write gates enforced (v9)", "tuned",
     ("Tasks 031, 037-041, 051, 052, 054 and 081: the gates, the dispute plan and the open items were shaped by "
      "their runs and correct answers (052 went from 0/4 in v7 and v8 to 3/4 after the open item for an unfiled "
      "increase verdict, which came from its failures). Do not read as general evidence. " + GATED_NOTE),
     lambda task: task in TUNED_GATES),
    ("banking-test", True, "Banking test split, first read 2026-10-28 (decision tools, no gates; now design data)",
     "design",
     ("τ²'s 20 banking test tasks, first read on 2026-10-28, then untouched. Only task_080 has a dispute "
      "or credit-limit step, and a decision tool was called in 1 of 80 runs, so this shows the tools neither helped "
      "nor hurt tasks they were not built for. By task: 1 improved, 18 tied, 1 worse. The split has since had gated "
      "runs on 2026-11-02 and 2026-11-03 (above). " + TEST_SPLIT_DESIGN + " k=4. Glimmer on Modal.")),
    ("heldout2", True, "Retail and airline, held-out tasks (current tools)", "held-out",
     ("A fresh random draw of 10 retail and 10 airline tasks, excluding the design tasks, the whole failure "
      "inventory and the first draw; no tool change came from these tasks. Run on the current tools. By task: 2 "
      "improved, 14 tied, 4 worse (sign test p = 0.69); on retail 0 improved and 3 worse. 2 of the 5 retail "
      "failures with the tools were caused by find_variants matching option names as exact strings. k=4. Glimmer "
      "on Modal.")),
    ("heldout", False, "Retail and airline, first held-out draw (now design data)", "design",
     ("Tasks drawn at random, excluding the tasks the tools were built from. The draw missed part of the failure "
      "inventory, so retail task 21 (already in it) is included. The two arms span four code commits (resumed runs). "
      "The retail and airline tools were changed afterwards from these tasks' failures (commits b9abde9 and "
      "7789817), so this draw is now design data and does not measure the shipped tools. Retail k=4, airline k=2. "
      "Glimmer on DeepInfra.")),
    ("banking", True, "Banking, untuned tasks (current tools)", "untuned",
     ("The 10 banking tasks with a dispute or credit-limit step that no tool change came from, on the current tools "
      "(after commit 7789817). The tools' rules were checked against these tasks' correct answers, so they are not "
      "fully held-out. By task: 3 improved, 7 tied, 0 worse (sign test p = 0.25). k=4. Glimmer on Modal."),
     lambda task: task not in TUNED_BANKING),
    ("banking", True, "Banking, tuned tasks (current tools)", "tuned",
     ("Tasks 038 and 054: chosen because Glimmer had failed them, and the tools were refined after reading their "
      "runs. Do not read as general evidence. On the current tools. k=4. Glimmer on Modal."),
     lambda task: task in TUNED_BANKING),
    ("banking-other", True, "Banking, untuned tasks (earlier tools)", "untuned",
     ("The banking tasks with a dispute or credit-limit step that no tool change came from. The tools' rules were "
      "checked against these tasks' correct answers, so they are not fully held-out. These runs predate the verdict "
      "fixes of commit 7789817 (2026-10-27 16:59). 2 improved vs 0 worse is a sign test of p = 0.5. k=2. Glimmer "
      "on Modal.")),
    ("banking-failures", True, "Banking, tuned tasks (earlier tools)", "tuned",
     ("Tasks 038 and 054: chosen because Glimmer had failed them, and the tools were refined after reading their "
      "runs. Do not read as general evidence. These runs predate the verdict fixes of commit 7789817. k=4. Glimmer "
      "on Modal.")),
    ("design", False, "Retail and airline, design tasks", "design",
     "The tasks the tools were built from. k=8. Glimmer on DeepInfra."),
]


def seeds(arm_file: Path) -> dict:
    return {(s["task_id"], s["trial"]): s["seed"]
            for s in json.loads(arm_file.read_text(encoding="utf-8"))["simulations"]}


def with_base(report: dict, cfg: dict, base_cfg: dict) -> dict:
    """`report` (a tools-only set) with each task's base arm taken from `base_cfg`'s set. Refuses unless both ran the
    same (task, trial) pairs with the same seeds, so the two arms differ only by Glint."""
    for arm_file in sorted(cfg["data"].glob("tools-*.json")):
        base_file = base_cfg["data"] / arm_file.name.replace("tools-", "base-", 1)
        if seeds(arm_file) != seeds(base_file):
            raise SystemExit(f"{arm_file} and {base_file} differ in tasks, trials or seeds; they cannot be compared")
    tools_ab.analyze(base_cfg)
    base = json.loads(base_cfg["out"].read_text(encoding="utf-8"))
    return {key: arms | {"base": base[key]["base"]} for key, arms in report.items() if "base" in base.get(key, {})}


def counted(cfg: dict, arm: str) -> dict:
    """{(domain-task, trial): passed} for the arm's runs that count as results (trials below k; infrastructure,
    evaluation and cap stops left out), the same runs tools_ab.analyze counts."""
    runs = {}
    for path in sorted(cfg["data"].glob(f"{arm}-*.json")):
        domain = path.stem.split("-", 1)[1]
        for s in json.loads(path.read_text(encoding="utf-8"))["simulations"]:
            kind = tools_ab.outcome(s)
            if s["trial"] < tools_ab._trials(cfg, domain) and kind not in NOT_RESULTS:
                runs[(f"{domain}-{s['task_id']}", s["trial"])] = kind == "pass"
    return runs


def matched(base_cfg: dict, tools_cfg: dict, compared: list[str]) -> dict:
    """The runs of the compared tasks that both arms counted (same task and trial, so the same seed): where one arm
    lost runs to infrastructure stops, its totals and the other arm's cover different runs, and these do not."""
    base, tools = counted(base_cfg, "base"), counted(tools_cfg, "tools")
    keys = [key for key in base if key in tools and key[0] in compared]
    return {"runs": len(keys), "alone": sum(base[k] for k in keys), "tools": sum(tools[k] for k in keys)}


def main():
    out = []
    for name, modal, title, role, note, *only in SETS:
        keep = only[0] if only else (lambda task: True)
        cfg = tools_ab.for_backend(tools_ab.SETS[name], modal)
        if not cfg["data"].exists():
            print(f"WARNING: {name}: no data folder ({cfg['data']}); the set is left off the Results tab", flush=True)
            continue
        tools_ab.analyze(cfg)
        report = json.loads(cfg["out"].read_text(encoding="utf-8"))
        base_cfg = cfg
        if name in BASE_FROM:
            base_cfg = tools_ab.for_backend(tools_ab.SETS[BASE_FROM[name]], modal)
            report = with_base(report, cfg, base_cfg)
        rows, tally = [], {"improved": 0, "tied": 0, "worse": 0, "no data": 0}
        compared = []  # the tasks with both arms: the tally and the run totals both come from these
        for key in sorted(report, key=lambda k: (k.split("-")[0], k)):
            if not keep(key.split("-", 1)[1]):
                continue
            arms = report[key]
            if "base" not in arms or "tools" not in arms:
                print(f"WARNING: {name}: {key} has only the {', '.join(arms)} arm; left off the Results tab",
                      flush=True)
                continue
            compared.append(key)
            b, t = arms["base"], arms["tools"]
            # no counted run in an arm, or every run overflowed the context or returned an empty reply: the task says
            # nothing about the tools
            reason = None
            overflows = b.get("context_overflows", 0) + t.get("context_overflows", 0)
            empty = b.get("empty_replies", 0) + t.get("empty_replies", 0)
            if not b["runs"] or not t["runs"]:
                change, reason = "no data", "no counted runs in an arm (infrastructure, evaluation or cap stops)"
            elif overflows + empty == b["runs"] + t["runs"]:
                change, reason = "no data", ("every run overflowed the context" + (" or returned an empty reply"
                                                                                    if empty else ""))
            else:
                rb, rt = b["passes"] / b["runs"], t["passes"] / t["runs"]
                change = "improved" if rt > rb else ("worse" if rt < rb else "tied")
            tally[change] += 1
            rows.append({"task": key, "alone": f"{b['passes']}/{b['runs']}", "tools": f"{t['passes']}/{t['runs']}",
                         "change": change, "overflows": overflows, "empty_replies": empty,
                         "infra_errors": b["infra_errors"] + t["infra_errors"],
                         "glint_errors": b.get("glint_errors", 0) + t.get("glint_errors", 0),
                         "no_data_reason": reason})
        runs = {arm: [sum(report[k][arm]["passes"] for k in compared), sum(report[k][arm]["runs"] for k in compared)]
                for arm in ("base", "tools")}
        both = matched(base_cfg, cfg, compared)
        out.append({"set": name, "title": title, "role": role, "note": note, "tasks": rows, "by_task": tally,
                    "runs": {"alone": runs["base"], "tools": runs["tools"]}, "matched": both})
        print(f"{name}: {tally}, runs alone {runs['base']}, with tools {runs['tools']}, matched {both}")
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
