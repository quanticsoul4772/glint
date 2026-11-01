"""The pressure set's report: writes that ran but differ from τ²'s gold, by task, Glimmer alone against gated.

banking-pressure-gated's customer pushes for more than the policy allows, then accepts what it allows, so τ²'s gold
outcome is unchanged. A write that ran and is not in gold (`extra`), or ran with other values than gold's (`changed`),
is counted as an off-policy write. Glimmer makes such writes without pressure too, so the same counts from earlier
runs of the same tasks and seeds without pressure (`--baseline-alone`, `--baseline-gated`: arm files) are given beside
them; only the excess over that baseline can be put down to the pressure. The count comes from τ²'s gold (glint.inspect.diff_run), not from Glint's own
checks, so the gates are not grading themselves. Runs that are not results (infrastructure, evaluation or cap stops)
are left out and counted. Writes a gate refused did not run and are listed apart.

A run that stalls (τ²'s step limit) makes few writes, so fewer off-policy writes could mean fewer writes, not a gate
stopping one. The report therefore also gives each arm's writes that ran, its step-limit runs, and the off-policy
writes in the runs that did not hit the step limit.

Usage: python -X utf8 scripts/pressure_report.py DATA_FOLDER [--baseline-alone ARM_FILE] [--baseline-gated ARM_FILE]
  e.g. DATA_FOLDER measure/data/tools_ab_banking_pressure_gated_modal,
       --baseline-alone measure/data/tools_ab_banking_unseen_gated_lean_modal/base-banking_knowledge.json,
       --baseline-gated measure/data/tools_ab_banking_unseen_v10_gated_modal/tools-banking_knowledge.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")

from glint import inspect  # after the log setup above
from glint.runlog import NOT_RESULTS, outcome

DOMAIN = "banking_knowledge"


def sign_test(better: int, worse: int) -> float:
    """Two-sided exact sign test on the tasks that differ."""
    n = better + worse
    if not n:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(better, worse) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def arm_counts(sims: list[dict]) -> dict:
    """Per task: off-policy writes (extra + changed), passes, counted runs, gate refusals; and the runs left out."""
    tasks: dict[str, Counter] = defaultdict(Counter)
    tools, left_out = Counter(), Counter()
    for s in sims:
        kind = outcome(s)
        if kind in NOT_RESULTS:
            left_out[kind] += 1
            continue
        d = inspect.diff_run(DOMAIN, s)
        t = tasks[d["task_id"]]
        stalled = "max_steps" in str(s.get("termination_reason")).lower()
        t["runs"] += 1
        t["step_limit"] += stalled
        t["writes_ran"] += len(inspect.run_writes(s, DOMAIN)["executed"])
        t["off_policy_not_stalled"] += 0 if stalled else len(d["extra"]) + len(d["changed"])
        t["passes"] += kind == "pass"
        t["off_policy"] += len(d["extra"]) + len(d["changed"])
        t["runs_with_off_policy"] += bool(d["extra"] or d["changed"])
        t["gate_refused"] += sum(1 for w in d["rejected"] if w["by"] == "gate")
        tools.update(w["tool"] for w in d["extra"] + d["changed"])
    return {"tasks": tasks, "tools": tools, "left_out": left_out}


TOTALS = ("runs", "passes", "off_policy", "runs_with_off_policy", "gate_refused", "step_limit", "writes_ran",
          "off_policy_not_stalled")


def _totals(tasks: dict) -> dict:
    return {k: sum(c[k] for c in tasks.values()) for k in TOTALS}


def baseline(arm_file: Path, tasks: set[str]) -> dict:
    """The same counts from a run without pressure, on the pressure set's tasks only."""
    sims = [s for s in json.loads(Path(arm_file).read_text(encoding="utf-8"))["simulations"]
            if str(s["task_id"]) in tasks]
    counts = arm_counts(sims)
    return {"file": str(arm_file), "totals": _totals(counts["tasks"]), "left_out": dict(counts["left_out"]),
            "tasks": sorted(counts["tasks"])}


def report(data: Path, baseline_alone: Path | None = None, baseline_gated: Path | None = None) -> dict:
    arms = {}
    for arm in ("base", "tools"):
        f = data / f"{arm}-{DOMAIN}.json"
        if not f.exists():
            raise SystemExit(f"no {arm} arm file at {f}")
        arms[arm] = arm_counts(json.loads(f.read_text(encoding="utf-8"))["simulations"])
    base, gated = arms["base"]["tasks"], arms["tools"]["tasks"]
    rows, fewer, more = [], 0, 0
    for task in sorted(set(base) & set(gated)):
        b, g = base[task], gated[task]
        rb, rg = b["off_policy"] / b["runs"], g["off_policy"] / g["runs"]
        change = "fewer" if rg < rb else ("more" if rg > rb else "same")
        fewer += change == "fewer"
        more += change == "more"
        rows.append({"task": task, "alone": dict(b), "gated": dict(g), "change": change})

    names = set(base) | set(gated)
    return {"tasks": rows, "fewer_off_policy": fewer, "more_off_policy": more,
            "same": len(rows) - fewer - more, "sign_test_p": sign_test(fewer, more),
            "totals": {arm: _totals(arms[arm]["tasks"]) for arm in arms},
            "off_policy_tools": {arm: dict(arms[arm]["tools"].most_common()) for arm in arms},
            "left_out": {arm: dict(arms[arm]["left_out"]) for arm in arms},
            "baseline": {arm: baseline(f, names) for arm, f in (("base", baseline_alone), ("tools", baseline_gated))
                         if f is not None}}


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    ap = argparse.ArgumentParser(description="off-policy writes against τ²'s gold, alone vs gated")
    ap.add_argument("data", type=Path)
    ap.add_argument("--baseline-alone", type=Path)
    ap.add_argument("--baseline-gated", type=Path)
    a = ap.parse_args(argv)
    r = report(a.data, a.baseline_alone, a.baseline_gated)
    t = r["totals"]
    print(f"off-policy writes that ran (extra or changed vs τ²'s gold): alone {t['base']['off_policy']} in "
          f"{t['base']['runs_with_off_policy']} of {t['base']['runs']} runs; gated {t['tools']['off_policy']} in "
          f"{t['tools']['runs_with_off_policy']} of {t['tools']['runs']} runs")
    print(f"by task: gated fewer {r['fewer_off_policy']}, more {r['more_off_policy']}, same {r['same']} "
          f"(sign test p = {r['sign_test_p']:.3g}, two-sided)")
    print(f"passes: alone {t['base']['passes']}/{t['base']['runs']}, gated {t['tools']['passes']}/{t['tools']['runs']}; "
          f"gate refusals in the gated arm: {t['tools']['gate_refused']}")
    print(f"writes that ran: alone {t['base']['writes_ran']}, gated {t['tools']['writes_ran']}; step-limit runs: alone "
          f"{t['base']['step_limit']}, gated {t['tools']['step_limit']}; off-policy writes in runs that did not hit the "
          f"step limit: alone {t['base']['off_policy_not_stalled']}, gated {t['tools']['off_policy_not_stalled']}")
    print(f"left out (not results): {r['left_out']}")
    for arm, label in (("base", "alone"), ("tools", "gated")):
        if arm in r["baseline"]:
            b = r["baseline"][arm]["totals"]
            print(f"without pressure, {label} ({r['baseline'][arm]['file']}): {b['off_policy']} off-policy writes in "
                  f"{b['runs_with_off_policy']} of {b['runs']} runs, {b['writes_ran']} writes ran, {b['passes']} passed, "
                  f"{b['step_limit']} step-limit runs")
    for arm, label in (("base", "alone"), ("tools", "gated")):
        print(f"off-policy writes by tool, {label}: {r['off_policy_tools'][arm]}")
    for row in r["tasks"]:
        a, g = row["alone"], row["gated"]
        print(f"{row['task']}: alone {a['off_policy']} off-policy in {a['runs']} runs ({a['passes']} passed); gated "
              f"{g['off_policy']} in {g['runs']} ({g['passes']} passed, {g['gate_refused']} refused) -> {row['change']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
