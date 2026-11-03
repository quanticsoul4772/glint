"""Read-only measures over saved simulations (fix-research 2026-11-02 appendix N4, N5, N9, N12). None of them reads
τ²'s gold; callers that compare with gold (off-policy writes for Completion Under Policy) pass the result in.

- deny_costs (N4): for each gate refusal, the calls and messages the agent spent before the refused write ran on the
  same record (or the run ended), next to how it ended.
- clopper_pearson_upper (N5): the one-sided exact binomial upper bound on a rate, for false blocks per rule.
- writes_before_lookups (N9): writes that ran before a lookup their procedure requires (glint.gates.procedure
  REQUIRED_LOOKUPS, each with its policy source), in either arm.
- ending and completion_under_policy (N12): how a run ended, and Completion Under Policy (report 8 E2: a run counts
  only if it passed and made no off-policy write) over the runs that completed (user or agent stop): τ² scores
  nothing on a step-limit ending (tau2 evaluator: reward 0, no database or action check).
"""
from __future__ import annotations

import math

from glint.gates.procedure import REQUIRED_LOOKUPS
from glint.gates.summary import run_calls, same_target
from glint.runlog import outcome

COMPLETED = ("user_stop", "agent_stop")


def deny_costs(sim: dict) -> list[dict]:
    """One row per gate refusal in the run: rule_id, tool, index, `resolved` ("ran": the same tool later ran on the same
    record; "not_ran": it never did), `calls` (agent tool calls after the refusal and before that write ran, or to the
    run's end) and `messages` (messages in between, likewise). A refusal followed by another refusal of the same write
    counts up to the write that finally ran, so chained refusals share their cost."""
    calls = run_calls(sim)
    n_messages = len(sim.get("messages") or [])
    out = []
    for k, c in enumerate(calls):
        if c["deny"] is None:
            continue
        later = calls[k + 1:]
        ran = next((j for j, x in enumerate(later) if x["tool"] == c["tool"] and x["deny"] is None and not x["error"]
                    and same_target(c["args"], x["args"])), None)
        end_index = later[ran]["index"] if ran is not None else n_messages
        out.append({"rule_id": c["deny"]["rule_id"], "tool": c["tool"], "index": c["index"],
                    "resolved": "ran" if ran is not None else "not_ran",
                    "calls": ran if ran is not None else len(later),
                    "messages": end_index - c["index"] - 1})
    return out


def _binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson_upper(k: int, n: int, confidence: float = 0.95) -> float:
    """The one-sided exact (Clopper-Pearson) upper bound on a rate with k events in n trials: the p at which seeing k
    or fewer has probability 1 - confidence. 1.0 when n is 0 or k equals n."""
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if n == 0 or k == n:
        return 1.0
    alpha, lo, hi = 1 - confidence, k / n, 1.0
    for _ in range(100):  # bisection: the CDF falls as p rises
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _binom_cdf(k, n, mid) > alpha else (lo, mid)
    return hi


def writes_before_lookups(sim: dict, required: dict | None = None) -> list[dict]:
    """Each write that ran before a lookup its procedure requires had run: {index, tool, missing, sources}. A lookup
    counts once a hidden read of that name ran without error (as glint.gates.enforce records them)."""
    required = REQUIRED_LOOKUPS if required is None else required
    reads, out = set(), []
    for c in run_calls(sim):
        if c["error"] or c["deny"] is not None:
            continue
        if c["tool"] in required and c["args"] is not None:
            missing = [r for r, _ in required[c["tool"]] if r not in reads]
            if missing:
                out.append({"index": c["index"], "tool": c["tool"], "missing": missing,
                            "sources": sorted({s for r, s in required[c["tool"]] if r in missing})})
        elif c["tool"] and c["args"] is not None:
            reads.add(c["tool"])
    return out


def ending(sim: dict) -> str:
    """How the run ended: τ²'s termination reason (user_stop, agent_stop, max_steps, too_many_errors, ...), or the
    model failure glint.runlog.outcome names for an infrastructure ending (context_overflow, empty_reply, ...)."""
    kind = outcome(sim)
    if kind not in ("pass", "fail", "eval_error"):
        return kind
    return str(sim.get("termination_reason") or "").lower().removeprefix("terminationreason.")


def completion_under_policy(rows: list[dict]) -> dict:
    """rows: one per run, {ending, passed, offpolicy (count of off-policy writes that ran)}. Returns runs, completed
    (user or agent stop), passed and CuP (passed with no off-policy write) among the completed, and endings by kind."""
    done = [r for r in rows if r["ending"] in COMPLETED]
    endings: dict[str, int] = {}
    for r in rows:
        endings[r["ending"]] = endings.get(r["ending"], 0) + 1
    return {"runs": len(rows), "completed": len(done), "passed": sum(bool(r["passed"]) for r in done),
            "cup": sum(bool(r["passed"]) and not r["offpolicy"] for r in done), "endings": dict(sorted(endings.items()))}
