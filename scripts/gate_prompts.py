"""Rebuild the prompt Glimmer saw before each recorded banking write, and check it against the meter ledger.

research/3-project/activation-gate-design.md, step 2 (offline probe). The banking runs saved no token ids, logprobs
or per-message usage, so the prompt is re-rendered from the saved messages with Glimmer's own chat template and
tokenizer (m1_prefix_check), using the banking environment the runs used (retrieval bm25_grep). The check: the
rebuilt prompt's token count must equal the prompt_tokens of a Glimmer call in the meter ledger, made within that
run's time window. Writes research/5-measurements/gate-prompts.json (per decision: token count, verified) and prints
the match rate; only verified prompts go to capture.

Usage: python gate_prompts.py
"""
import json
from datetime import datetime
from pathlib import Path

from m1_prefix_check import AGENT_INSTRUCTION, SYSTEM_PROMPT, ids, template, bos, to_template_messages
from tau2.data_model.simulation import Results
from tau2.domains.banking_knowledge.environment import get_environment

DATA = Path(r"<workspace>\measure\data")
LABELS = Path(r"<workspace>\research\5-measurements\gate-labels.json")
LEDGER = Path(r"<workspace>\measure\meter\ledger.jsonl")
OUT = Path(r"<workspace>\research\5-measurements\gate-prompts.json")


def banking_context():
    env = get_environment(retrieval_variant="bm25_grep")
    tools = [t.openai_schema for t in env.get_tools()]
    system = {"role": "system", "content": SYSTEM_PROMPT.format(domain_policy=env.get_policy(),
                                                               agent_instruction=AGENT_INSTRUCTION)}
    return tools, system


def _ts(value) -> float:
    return datetime.fromisoformat(str(value)).timestamp()


def main():
    tools, system = banking_context()
    rows = [r for r in json.loads(LABELS.read_text(encoding="utf-8")) if r["domain"] == "banking_knowledge"]
    glimmer = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines()]
    glimmer = [g for g in glimmer if g.get("kind") == "meta" and "Muse-Glimmer" in g.get("model", "")]
    sims = {}
    out = []
    for r in rows:
        if r["source"] not in sims:
            res = Results.model_validate_json((DATA / r["source"]).read_text(encoding="utf-8"))
            sims[r["source"]] = {(str(s.task_id), s.trial): s for s in res.simulations}
        sim = sims[r["source"]][(str(r["task_id"]), r["trial"])]
        text = template.render(messages=[system, *to_template_messages(sim.messages[:r["message_index"]], False)],
                               tools=tools, add_generation_prompt=True, bos_token=bos, reasoning_strength="medium")
        n = len(ids(text))
        lo, hi = _ts(sim.start_time) - 5, _ts(sim.end_time) + 5
        window = [g["prompt"] for g in glimmer if lo <= g["ts"] <= hi]
        out.append({k: r[k] for k in ("source", "task_id", "trial", "message_index", "tool", "label", "executed")}
                   | {"prompt_tokens": n, "verified": n in window,
                      "nearest_ledger": min(window, key=lambda p: abs(p - n)) if window else None})
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    ok = sum(o["verified"] for o in out)
    pos = [o for o in out if o["label"] == "overruled" and o["executed"]]
    print(f"{ok}/{len(out)} rebuilt prompts match a ledger call; positives verified "
          f"{sum(o['verified'] for o in pos)}/{len(pos)}")
    misses = [o["prompt_tokens"] - o["nearest_ledger"] for o in out if not o["verified"] and o["nearest_ledger"]]
    if misses:
        print(f"unverified: offset from the nearest ledger count, first 15: {misses[:15]}")


if __name__ == "__main__":
    main()
