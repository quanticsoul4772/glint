"""M1 checks C5 (token-prefix property) and C6 (tokenization fidelity) on the C2 Glimmer trajectories.

Renders each agent step's prompt with Glimmer's own chat template (pinned revision) and tokenizer, no weights.
C6: the locally counted prompt tokens vs DeepInfra's usage.prompt_tokens, per step.
C5a (history as τ² sends it, reasoning dropped): is step k's prompt an exact token prefix of step k+1's?
C5b (history with each turn's reasoning kept): is prompt_k + Glimmer's generated text for step k an exact prefix
     of step k+1's prompt? If yes, one prefill of the final transcript covers every step's read positions.
"""
import importlib
import json

from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
from tau2.data_model.simulation import Results
from tau2.utils.llm_utils import to_litellm_messages
from tokenizers import Tokenizer

from glint import paths

SNAP = paths.HF_CACHE / "hub" / "models--meta-models--Muse-Glimmer-30B" / "snapshots" / \
    "cec94fcfd6512f2cfe708138c32533b346ea4f69"
SIMS = paths.MEASURE / "m1-c2-simulations.json"
OUT = paths.RESULTS / "M1-prefix-check.json"


def raise_exception(msg):
    raise ValueError(msg)


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    # Same as Transformers' chat-template tojson (jinja2's builtin HTML-escapes)
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


jenv = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
jenv.filters["tojson"] = tojson
jenv.globals["raise_exception"] = raise_exception
template = jenv.from_string((SNAP / "chat_template.jinja").read_text(encoding="utf-8"))
bos = json.loads((SNAP / "tokenizer_config.json").read_text(encoding="utf-8")).get("bos_token") or ""
bos = bos["content"] if isinstance(bos, dict) else bos
tok = Tokenizer.from_file(str(SNAP / "tokenizer.json"))

_contexts = {}


def domain_context(domain):
    """(tools, system message) exactly as τ²'s LLMAgent sends them for `domain`."""
    if domain not in _contexts:
        env = importlib.import_module(f"tau2.domains.{domain}.environment").get_environment()
        _contexts[domain] = (
            [t.openai_schema for t in env.get_tools()],
            {"role": "system", "content": SYSTEM_PROMPT.format(domain_policy=env.get_policy(),
                                                               agent_instruction=AGENT_INSTRUCTION)},
        )
    return _contexts[domain]


def to_template_messages(messages, keep_reasoning):
    out = []
    for msg, lm in zip(messages, to_litellm_messages(messages)):
        if lm.get("tool_calls"):
            for tc in lm["tool_calls"]:
                tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
        if keep_reasoning and msg.role == "assistant":
            rc = ((msg.raw_data or {}).get("choices", [{}])[0].get("message") or {}).get("reasoning_content")
            if rc:
                lm["reasoning_content"] = rc
        out.append(lm)
    return out


def render(messages, domain="retail", strength="medium"):
    tools, system = domain_context(domain)
    return template.render(messages=[system, *messages], tools=tools, add_generation_prompt=True,
                           bos_token=bos, reasoning_strength=strength)


def ids(text):
    return tok.encode(text, add_special_tokens=False).ids


def generated_text(msg):
    lp = ((msg.raw_data or {}).get("choices", [{}])[0].get("logprobs") or {}).get("content") or []
    return "".join(t["token"] for t in lp)



def main():
    results = Results.model_validate_json(SIMS.read_text(encoding="utf-8"))
    rows = []
    for sim in results.simulations:
        steps = [i for i, m in enumerate(sim.messages) if m.role == "assistant"]
        for n, i in enumerate(steps):
            msg = sim.messages[i]
            stripped = ids(render(to_template_messages(sim.messages[:i], keep_reasoning=False)))
            row = {"task": str(sim.task_id), "step": n, "ours": len(stripped),
                   "server": ((msg.raw_data or {}).get("usage") or {}).get("prompt_tokens")}
            if n + 1 < len(steps):
                j = steps[n + 1]
                nxt = ids(render(to_template_messages(sim.messages[:j], keep_reasoning=False)))
                row["C5a_prefix"] = nxt[:len(stripped)] == stripped
                kept_k = render(to_template_messages(sim.messages[:i], keep_reasoning=True))
                kept_next = render(to_template_messages(sim.messages[:j], keep_reasoning=True))
                full = kept_k + generated_text(msg)
                row["C5b_text_prefix"] = kept_next.startswith(full)
                full_ids, next_ids = ids(full), ids(kept_next)
                row["C5b_token_prefix"] = next_ids[:len(full_ids)] == full_ids
                if not row["C5b_text_prefix"]:
                    k = next((p for p, (a, b) in enumerate(zip(full, kept_next)) if a != b), min(len(full), len(kept_next)))
                    row["C5b_diverge"] = {"ours": full[max(0, k - 60):k + 60], "next": kept_next[max(0, k - 60):k + 60]}
            rows.append(row)

    paired = [r for r in rows if "C5a_prefix" in r]
    summary = {
        "template_revision": SNAP.name,
        "steps": len(rows),
        "C6_count_match": sum(r["ours"] == r["server"] for r in rows),
        "C6_diffs": [r["ours"] - r["server"] for r in rows if r["server"] is not None and r["ours"] != r["server"]],
        "C5a_prefix": f'{sum(r["C5a_prefix"] for r in paired)}/{len(paired)}',
        "C5b_text_prefix": f'{sum(r["C5b_text_prefix"] for r in paired)}/{len(paired)}',
        "C5b_token_prefix": f'{sum(r["C5b_token_prefix"] for r in paired)}/{len(paired)}',
        "C5b_first_divergences": [r["C5b_diverge"] for r in paired if "C5b_diverge" in r][:3],
    }
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
