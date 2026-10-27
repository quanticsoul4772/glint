"""Build capture job lines (glint.glimmer_modal) from recorded τ² simulations, with each step's prompt exactly as τ²
sent it (earlier reasoning removed).

Per agent step k: append the rest of prompt_k (read at its last token: turn_start), append Glimmer's generated tokens
(reads: P2 = the <|eom|> closing the reasoning channel, when there is reasoning; P3a = the <|message|> opening the
first tool-call message, when there is a call; P4 = the last token), then truncate back to the longest prefix shared
with prompt_{k+1}. Fails loudly unless every step's prompt length equals the host's usage.prompt_tokens and the
re-tokenized generated text equals the streamed token count.
Usage: python build_jobs.py SIMS.json OUT.jsonl MAX_STEPS DOMAIN STRENGTH PREFIX [TASK_ID]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m1_prefix_check import generated_text, ids, render, to_template_messages
from tau2.data_model.simulation import Results

EOM = ids("<|eom|>")
assert len(EOM) == 1, EOM


def p2_index(gen_text, gen_ids):
    """Index of the <|eom|> that closes the reasoning channel, or None when the turn has no reasoning."""
    # generation starts right after '<|start|>assistant', so a reasoning turn begins with ' to=self'
    if not gen_text.startswith(" to=self"):
        return None
    return gen_ids.index(EOM[0])


def p3a_index(stream_tokens):
    """Index of the <|message|> token right before the first <atem:function_calls>, or None without a call."""
    text, starts = "", []
    for t in stream_tokens:
        starts.append(len(text))
        text += t
    at = text.find("<|message|><atem:function_calls>")
    if at < 0:
        return None
    return starts.index(at)


def main(argv):
    sims_path, out_path, max_steps = Path(argv[1]), Path(argv[2]), int(argv[3])
    domain, strength, prefix = argv[4], argv[5], argv[6]
    only_task = argv[7] if len(argv) > 7 else None
    results = Results.model_validate_json(sims_path.read_text(encoding="utf-8"))
    report, n_traj = [], 0
    with out_path.open("w", encoding="utf-8") as out:
        for sim in results.simulations:
            if only_task and str(sim.task_id) != only_task:
                continue
            n_traj += 1
            steps = [i for i, m in enumerate(sim.messages) if m.role == "assistant" and m.raw_data][:max_steps]
            ops, cur, prev_target = [], [], []
            for n, i in enumerate(steps):
                msg = sim.messages[i]
                choice = msg.raw_data["choices"][0]
                target = ids(render(to_template_messages(sim.messages[:i], keep_reasoning=False), domain, strength))
                server = (msg.raw_data.get("usage") or {}).get("prompt_tokens")
                if len(target) != server:
                    raise ValueError(f"{sim.task_id}:{sim.trial} step {n}: rendered {len(target)} != host {server}")
                if target[:len(prev_target)] != prev_target:
                    raise ValueError(f"{sim.task_id}:{sim.trial} step {n}: prompt does not extend the previous one")
                common = next((p for p, (a, b) in enumerate(zip(cur, target)) if a != b), min(len(cur), len(target)))
                if common < len(cur):
                    ops.append({"truncate": common})
                rest = target[common:]
                ops.append({"append": rest, "read": [[len(rest) - 1, f"{n}:turn_start"]]})
                stream = [t["token"] for t in (choice.get("logprobs") or {}).get("content") or []]
                gen_text = generated_text(msg)
                gen = ids(gen_text)
                if len(gen) != len(stream):
                    raise ValueError(f"{sim.task_id}:{sim.trial} step {n}: re-tokenized {len(gen)} != stream {len(stream)}")
                reads = []
                p2, p3a = p2_index(gen_text, gen), p3a_index(stream)
                if p2 is not None:
                    reads.append([p2, f"{n}:P2"])
                if p3a is not None:
                    reads.append([p3a, f"{n}:P3a"])
                reads.append([len(gen) - 1, f"{n}:P4"])
                ops.append({"append": gen, "read": reads})
                cur, prev_target = target + gen, target
                report.append({"has_P2": p2 is not None, "has_P3a": p3a is not None, "prompt": len(target),
                               "gen": len(gen)})
            out.write(json.dumps({"traj": f"{prefix}:{sim.task_id}:{sim.trial}", "ops": ops}) + "\n")
    summary = {"trajectories": n_traj, "steps": len(report), "with_P2": sum(r["has_P2"] for r in report),
               "with_P3a": sum(r["has_P3a"] for r in report),
               "replay_tokens": sum(len(op["append"]) for line in out_path.read_text().splitlines()
                                    for op in json.loads(line)["ops"] if "append" in op)}
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    main(sys.argv)
