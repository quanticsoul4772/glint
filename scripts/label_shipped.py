"""M0: run glint's labeler on τ²'s shipped retail trajectories (4 trials × 114 tasks per model).

Reports per-label step counts, steps per episode, and agreement between the labeler's TOOL_ERROR and
the recorded tool errors (a replay-fidelity check). Not probe data: these are not Glimmer's activations.
"""
import json
import sys
from collections import Counter
from pathlib import Path

from glint import paths
from glint.labeler import Labeler
from glint.labels import FAB, VIOL, Label, is_resid_positive

RESULTS = paths.tau2_data() / "tau2" / "results" / "final"


def run(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = {str(t["id"]): t for t in data["tasks"]}
    counts, n_steps, n_eps, mismatch, write_steps = Counter(), 0, 0, 0, 0
    resid_pos, viol_pos, fab_pos, rewards = 0, 0, 0, []
    for sim in data["simulations"]:
        task = tasks[str(sim["task_id"])]
        rows = Labeler(task).label(sim["messages"], trial=sim.get("trial", 0))
        n_eps += 1
        rewards.append((sim.get("reward_info") or {}).get("reward", 0.0))
        recorded_err = {m["id"]: bool(m.get("error")) for m in sim["messages"] if m.get("role") == "tool"}
        for r in rows:
            n_steps += 1
            counts.update(r.labels)
            fab_pos += bool(r.labels & FAB)
            viol_pos += bool(r.labels & VIOL)
            resid_pos += is_resid_positive(r.labels)
            write_steps += r.tool in {"cancel_pending_order", "exchange_delivered_order_items",
                                      "modify_pending_order_items", "modify_pending_order_payment",
                                      "modify_pending_order_address", "return_delivered_order_items",
                                      "modify_user_address"}
        # replay fidelity: our executed error flags vs recorded ones, by call order
        calls = [tc for m in sim["messages"] if m.get("role") == "assistant" for tc in (m.get("tool_calls") or [])]
        ours = [Label.TOOL_ERROR in r.labels for r in rows]
        theirs = [recorded_err.get(tc.get("id"), False) for tc in calls]
        mismatch += sum(a != b for a, b in zip(ours, theirs))
    return {
        "file": path.name, "episodes": n_eps, "steps": n_steps, "write_steps": write_steps,
        "mean_reward": round(sum(rewards) / max(n_eps, 1), 3),
        "steps_with_FAB": fab_pos, "steps_with_VIOL": viol_pos, "RESID_positive_after_R0": resid_pos,
        "replay_error_mismatches": mismatch,
        "per_label": {k.value: v for k, v in sorted(counts.items(), key=lambda kv: -kv[1])},
    }


if __name__ == "__main__":
    files = sorted(RESULTS.glob("*_retail_*4trials.json"))
    out = [run(p) for p in files]
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    text = json.dumps(out, indent=1)
    if dest:
        dest.write_text(text)
    print(text)
