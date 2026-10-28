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

import tools_ab  # after the log setup above

OUT = Path(__file__).resolve().parents[1] / "src" / "glint" / "desk" / "results.json"
SETS = [  # (set, Modal backend, title, role, note)
    ("heldout2", True, "Retail and airline, held-out tasks (current tools)", "held-out",
     ("A fresh random draw of 10 retail and 10 airline tasks, excluding the design tasks, the whole failure "
      "inventory and the first draw; no tool change came from these tasks. Run on the current tools. By task: 2 "
      "improved, 14 tied, 4 worse (sign test p = 0.69); on retail 0 improved and 3 worse. 2 of the 5 retail "
      "failures with the tools were caused by find_variants matching option names as exact strings. k=4. Glimmer "
      "on Modal.")),
    ("heldout", False, "Retail and airline, first held-out draw (now design data)", "design",
     ("Tasks drawn at random, excluding the tasks the tools were built from. The draw missed part of the failure "
      "inventory, so retail task 21 (already in it) is included. The two arms span four code commits (resumed runs). "
      "The retail and airline tools were changed afterwards from these tasks' failures (commits aa1dadc and "
      "ce7b242), so this draw is now design data and does not measure the shipped tools. Retail k=4, airline k=2. "
      "Glimmer on DeepInfra.")),
    ("banking-other", True, "Banking, untuned tasks", "untuned",
     ("The banking tasks with a dispute or credit-limit step that no tool change came from. The tools' rules were "
      "checked against these tasks' correct answers, so they are not fully held-out. These runs predate the verdict "
      "fixes of commit ce7b242 (2026-10-27 16:59). 2 improved vs 0 worse is a sign test of p = 0.5. k=2. Glimmer "
      "on Modal.")),
    ("banking-failures", True, "Banking, tuned tasks", "tuned",
     ("Tasks 038 and 054: chosen because Glimmer had failed them, and the tools were refined after reading their "
      "runs. Do not read as general evidence. These runs predate the verdict fixes of commit ce7b242. k=4. Glimmer "
      "on Modal.")),
    ("design", False, "Retail and airline, design tasks", "design",
     "The tasks the tools were built from. k=8. Glimmer on DeepInfra."),
]


def main():
    out = []
    for name, modal, title, role, note in SETS:
        cfg = tools_ab.for_backend(tools_ab.SETS[name], modal)
        if not cfg["data"].exists():
            print(f"WARNING: {name}: no data folder ({cfg['data']}); the set is left off the Results tab", flush=True)
            continue
        tools_ab.analyze(cfg)
        report = json.loads(cfg["out"].read_text(encoding="utf-8"))
        rows, tally = [], {"improved": 0, "tied": 0, "worse": 0, "no data": 0}
        compared = []  # the tasks with both arms: the tally and the run totals both come from these
        for key in sorted(report, key=lambda k: (k.split("-")[0], k)):
            arms = report[key]
            if "base" not in arms or "tools" not in arms:
                print(f"WARNING: {name}: {key} has only the {', '.join(arms)} arm; left off the Results tab",
                      flush=True)
                continue
            compared.append(key)
            b, t = arms["base"], arms["tools"]
            # no counted run in an arm, or every run overflowed the context: the task says nothing about the tools
            reason = None
            if not b["runs"] or not t["runs"]:
                change, reason = "no data", "no counted runs in an arm (infrastructure, evaluation or cap stops)"
            elif b.get("context_overflows", 0) + t.get("context_overflows", 0) == b["runs"] + t["runs"]:
                change, reason = "no data", "every run overflowed the context"
            else:
                rb, rt = b["passes"] / b["runs"], t["passes"] / t["runs"]
                change = "improved" if rt > rb else ("worse" if rt < rb else "tied")
            tally[change] += 1
            rows.append({"task": key, "alone": f"{b['passes']}/{b['runs']}", "tools": f"{t['passes']}/{t['runs']}",
                         "change": change, "overflows": b.get("context_overflows", 0) + t.get("context_overflows", 0),
                         "infra_errors": b["infra_errors"] + t["infra_errors"],
                         "glint_errors": b.get("glint_errors", 0) + t.get("glint_errors", 0),
                         "no_data_reason": reason})
        runs = {arm: [sum(report[k][arm]["passes"] for k in compared), sum(report[k][arm]["runs"] for k in compared)]
                for arm in ("base", "tools")}
        out.append({"set": name, "title": title, "role": role, "note": note, "tasks": rows, "by_task": tally,
                    "runs": {"alone": runs["base"], "tools": runs["tools"]}})
        print(f"{name}: {tally}, runs alone {runs['base']}, with tools {runs['tools']}")
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
