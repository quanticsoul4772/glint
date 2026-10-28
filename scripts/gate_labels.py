"""Label every write in Glimmer's recorded no-tools runs (glint.decisions) and count the positives per domain.

Sources: the M2 pilot, Plan B, the headroom pilot, and the no-tools arms of the with/without comparisons. Runs that
ended in an infrastructure error are skipped. Writes `research/5-measurements/gate-labels.json` (per decision) and
prints counts per domain and per source, with positives split by run outcome.

Usage: python gate_labels.py
"""
import json
from collections import Counter

from tau2.data_model.tasks import Task
from tau2.domains.banking_knowledge.environment import get_environment as banking_env
from tau2.registry import registry

from glint import paths
from glint.decisions import label_run, to_rows
from glint.runlog import NOT_RESULTS, outcome

DATA = paths.MEASURE
DOMAINS = paths.tau2_data() / "tau2" / "domains"
OUT = paths.RESULTS / "gate-labels.json"
SOURCES = {  # file -> domain; Glimmer only (no Wonder), no-tools only
    "m2/retail-medium.json": "retail", "m2/retail-low.json": "retail", "m2/retail-high.json": "retail",
    "m2/airline-medium.json": "airline", "headroom/airline-glimmer.json": "airline",
    "headroom/banking_knowledge-glimmer.json": "banking_knowledge",
    "tools_ab/base-retail.json": "retail", "tools_ab/base-airline.json": "airline",
    "tools_ab_heldout/base-retail.json": "retail", "tools_ab_heldout/base-airline.json": "airline",
    "tools_ab_banking_failures_modal/base-banking_knowledge.json": "banking_knowledge",
    "tools_ab_banking_other_modal/base-banking_knowledge.json": "banking_knowledge",
}


def tasks(domain: str) -> dict[str, Task]:
    raw = json.loads((DOMAINS / domain / "tasks.json").read_text(encoding="utf-8"))
    return {str(t["id"]): Task.model_validate(t) for t in raw}


def env(domain: str):
    return banking_env(retrieval_variant="bm25_grep") if domain == "banking_knowledge" else \
        registry.get_env_constructor(domain)()


def sources() -> dict[str, str]:
    out = dict(SOURCES) | {f"planb/pilot/batch{i}.json": "retail" for i in range(3)}  # Plan B k=16, retail
    missing = [k for k in out if not (DATA / k).exists()]
    if missing:
        print(f"NOTE: {len(missing)} of {len(out)} sources are not on disk and are not labeled: {missing}", flush=True)
    return {k: v for k, v in out.items() if (DATA / k).exists()}


def main():
    rows, counts, dropped = [], Counter(), Counter()
    unjudged = {d: {} for d in ("retail", "airline", "banking_knowledge")}
    task_cache = {}
    for rel, domain in sources().items():
        task_cache.setdefault(domain, tasks(domain))
        sims = json.loads((DATA / rel).read_text(encoding="utf-8"))["simulations"]
        for s in sims:
            kind = outcome(s)
            # runs that are not results are left out, counted; a context overflow or a Glint error is a failed run
            # (an overflow used to be dropped here as "infrastructure", against the project rule)
            if kind in NOT_RESULTS:
                dropped[(domain, kind)] += 1
                continue
            if str(s["task_id"]) not in task_cache[domain]:
                dropped[(domain, "unknown task id")] += 1
                continue
            passed = kind == "pass"
            for d in label_run(s, task_cache[domain][str(s["task_id"])], env(domain), unjudged[domain]):
                row = to_rows([d])[0] | {"domain": domain, "source": rel, "run_passed": passed}
                rows.append(row)
                counts[(domain, d.label, "passed" if passed else "failed")] += 1
                counts[(domain, d.label, "executed" if d.executed else "rejected")] += 1
    OUT.write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
    if dropped:
        print(f"NOTE: runs left out of the labels: {dict(dropped)}", flush=True)
    for domain, c in unjudged.items():
        if c:
            print(f"NOTE: {domain}: {sum(c.values())} writes could not be judged (no label): {c}", flush=True)
    for domain in ("retail", "airline", "banking_knowledge"):
        c = {k[1:]: v for k, v in counts.items() if k[0] == domain}
        total = sum(v for k, v in c.items() if k[1] in ('passed', 'failed'))
        over = c.get(("overruled", "failed"), 0) + c.get(("overruled", "passed"), 0)
        share = f"{over / total:.1%}" if total else "-"
        print(f"{domain}: {total} writes, {over} overruled ({share}), "
              f"in failed runs {c.get(('overruled', 'failed'), 0)}, in passed runs {c.get(('overruled', 'passed'), 0)}; "
              f"uncertain {c.get(('uncertain', 'failed'), 0) + c.get(('uncertain', 'passed'), 0)}; "
              f"overruled and executed {c.get(('overruled', 'executed'), 0)}, "
              f"rejected by the environment {c.get(('overruled', 'rejected'), 0)}")


if __name__ == "__main__":
    main()
