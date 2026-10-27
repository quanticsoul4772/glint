"""Build glint_capture job lines from recorded τ² simulations, with each step's prompt exactly as τ² sent it
(earlier reasoning removed; token counts verified against DeepInfra in M1-prefix-check.json).

Per agent step k: append the rest of prompt_k (read at its last token: turn_start), append Glimmer's generated
tokens (read at the <|eom|> closing the reasoning channel: P2, and at the last token: P4), then truncate back to
the longest prefix shared with prompt_{k+1}. Generated tokens are re-tokenized from the logprob stream text and
checked against the stream's token count.
Usage: python build_jobs.py SIMS.json OUT.jsonl MAX_STEPS [TASK_ID]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m1_prefix_check import generated_text, ids, render, to_template_messages
from tau2.data_model.simulation import Results

sims_path, out_path, max_steps = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
only_task = sys.argv[4] if len(sys.argv) > 4 else None
EOM = ids("<|eom|>")
assert len(EOM) == 1, EOM


def p2_index(gen_text, gen_ids):
    """Index of the <|eom|> that closes the reasoning channel, or None when the turn has no reasoning."""
    # generation starts right after '<|start|>assistant', so a reasoning turn begins with ' to=self'
    if not gen_text.startswith(" to=self"):
        return None
    return gen_ids.index(EOM[0])


results = Results.model_validate_json(sims_path.read_text(encoding="utf-8"))

report, n_lines = [], 0
with out_path.open("w", encoding="utf-8") as out:
    for sim in results.simulations:
        if only_task and str(sim.task_id) != only_task:
            continue
        steps = [i for i, m in enumerate(sim.messages) if m.role == "assistant" and m.raw_data][:max_steps]
        ops, cur, prev_target = [], [], []
        for n, i in enumerate(steps):
            msg = sim.messages[i]
            target = ids(render(to_template_messages(sim.messages[:i], keep_reasoning=False)))
            if target[:len(prev_target)] != prev_target:
                raise ValueError(f"task {sim.task_id} step {n}: prompt is not an extension of the previous prompt")
            common = next((p for p, (a, b) in enumerate(zip(cur, target)) if a != b), min(len(cur), len(target)))
            if common < len(cur):
                ops.append({"truncate": common})
            rest = target[common:]
            ops.append({"append": rest, "read": [[len(rest) - 1, f"{n}:turn_start"]]})
            gen_text = generated_text(msg)
            gen = ids(gen_text)
            n_stream = len(((msg.raw_data["choices"][0].get("logprobs") or {}).get("content")) or [])
            reads = [[len(gen) - 1, f"{n}:P4"]]
            p2 = p2_index(gen_text, gen)
            if p2 is not None:
                reads.insert(0, [p2, f"{n}:P2"])
            ops.append({"append": gen, "read": reads})
            cur, prev_target = target + gen, target
            report.append({"task": str(sim.task_id), "step": n, "prompt_tokens": len(target), "gen_tokens": len(gen),
                           "stream_tokens": n_stream, "retokenized_equal": len(gen) == n_stream, "has_P2": p2 is not None})
        out.write(json.dumps({"traj": f"{sim.task_id}:{sim.trial}", "ops": ops}) + "\n")
        n_lines += 1

print(json.dumps(report, indent=1))
print("trajectories", n_lines, "retokenized_equal", sum(r["retokenized_equal"] for r in report), "/", len(report))
