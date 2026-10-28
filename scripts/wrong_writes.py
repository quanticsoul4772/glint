"""Wrong writes per arm: in every with/without comparison, how many of Glimmer's writes Glint's checks judge wrong.

Each write in each recorded run (both arms) is labelled by glint.decisions: "overruled" means the decision tool's
verdict contradicts the write. Counted per set, arm and task, and split into writes the environment executed and
writes it rejected itself. Runs that are not results (glint.runlog.NOT_RESULTS) are left out and counted. Writes
`research/5-measurements/wrong-writes.json` and prints one line per set and arm. $0: no model is called.

Usage: python wrong_writes.py [SET ...]        (no SET: every set with both arm files)
"""
import json
import sys
from collections import Counter, defaultdict

from loguru import logger

from glint import paths
from glint.decisions import label_run
from glint.runlog import NOT_RESULTS, outcome

# τ² logs its whole registry at DEBUG when imported; this command prints only its own report
logger.remove()
logger.add(sys.stderr, level="WARNING")

import gate_labels  # after the log setup above
import tools_ab

OUT = paths.RESULTS / "wrong-writes.json"
ARMS = ("base", "tools")


def count_set(data, domains) -> dict:
    """{domain: {arm: {"runs", "writes", "overruled", "overruled_executed", "by_task", "dropped", "unjudged"}}}
    for the arm files in `data`; an arm whose file is missing is left out."""
    out = {}
    for domain in domains:
        tasks = gate_labels.tasks(domain)
        for arm in ARMS:
            f = data / f"{arm}-{domain}.json"
            if not f.exists():
                continue
            c, by_task, dropped, unjudged = Counter(), defaultdict(Counter), Counter(), {}
            for s in json.loads(f.read_text(encoding="utf-8"))["simulations"]:
                kind = outcome(s)
                if kind in NOT_RESULTS:
                    dropped[kind] += 1
                    continue
                if str(s["task_id"]) not in tasks:
                    dropped["unknown task id"] += 1
                    continue
                c["runs"] += 1
                for d in label_run(s, tasks[str(s["task_id"])], gate_labels.env(domain), unjudged):
                    c["writes"] += 1
                    by_task[str(s["task_id"])]["writes"] += 1
                    if d.label == "overruled":
                        c["overruled"] += 1
                        c["overruled_executed"] += d.executed
                        by_task[str(s["task_id"])]["overruled"] += 1
                        by_task[str(s["task_id"])]["overruled_executed"] += d.executed
                    elif d.label == "uncertain":
                        c["uncertain"] += 1
            out.setdefault(domain, {})[arm] = dict(c) | {"by_task": {t: dict(v) for t, v in sorted(by_task.items())},
                                                         "dropped": dict(dropped), "unjudged": unjudged}
    return out


def main():
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(tools_ab.SETS)
    report = {}
    for name in names:
        for modal in (False, True):
            cfg = tools_ab.for_backend(tools_ab.SETS[name], modal)
            if not (cfg["data"] / "tasks.json").exists():
                continue
            domains = json.loads((cfg["data"] / "tasks.json").read_text(encoding="utf-8"))
            key = f"{name}{' --modal' if modal else ''}"
            counts = count_set(cfg["data"], domains)
            if not any(len(arms) == len(ARMS) for arms in counts.values()):
                print(f"NOTE: {key}: no domain has both arm files; not counted", flush=True)
                continue
            report[key] = counts
            for domain, arms in counts.items():
                for arm, c in arms.items():
                    line = (f"{key:28} {domain:17} {arm:5} {c.get('runs', 0):3} runs, {c.get('writes', 0):4} writes, "
                            f"{c.get('overruled', 0):3} overruled ({c.get('overruled_executed', 0)} executed)")
                    if c.get("uncertain"):
                        line += f", {c['uncertain']} uncertain"
                    if c["dropped"]:
                        line += f"; left out {c['dropped']}"
                    if c["unjudged"]:
                        line += f"; not judged {c['unjudged']}"
                    print(line, flush=True)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
