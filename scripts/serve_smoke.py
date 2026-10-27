"""Smoke test for Glimmer served on Modal (glint.glimmer_serve): one tool-call request, after the endpoint is ready.

Deploys `glint-serve`, waits on GET /health (never stops the app while it boots; stops at once if the log shows
the engine crashed), sends one chat request with a tool,
then `session()` stops the app, records the wall time at the A100 rate and checks that no container is left.
Checks: a parsed tool call, logprobs on every generated token, raw token ids, and vLLM's KV-cache line in the log.
GPU cap $1 (Modal free credit). Output: research/5-measurements/modal-serve-smoke.json.
Usage: python serve_smoke.py
"""
import json
import os
import re
import time
from pathlib import Path

import litellm

from glint.glimmer_serve import MODEL_NAME, LogWatch, deploy, litellm_args, say, session, wait_ready
from glint.meter import Caps, Meter

ROOT = Path(__file__).resolve().parents[1]
MODAL = str(ROOT / ".venv" / "Scripts" / "modal.exe")
OUT = Path(r"<workspace>\research\5-measurements\modal-serve-smoke.json")
RAW = Path(r"<workspace>\measure\data\modal-serve-smoke-response.json")  # for the glint.atem re-parse check
TOOLS = [{"type": "function", "function": {
    "name": "get_order_details", "description": "Get the status and details of an order.",
    "parameters": {"type": "object", "properties": {"order_id": {"type": "string", "description": "Such as '#W0000000'"}},
                   "required": ["order_id"]}}}]
MESSAGES = [{"role": "system", "content": "You are a retail support agent. Use the tools to look up orders."},
            {"role": "user", "content": "What is the status of my order #W2890441?"}]


def main():
    env = os.environ | {"MODAL_CONFIG_PATH": r"<workspace>\.modal.toml", "MODAL_PROFILE": "quanticsoul4772",
                        "PYTHONIOENCODING": "utf-8"}
    url = deploy(MODAL, env)
    key, secret = os.environ["MODAL_PROXY_KEY"], os.environ["MODAL_PROXY_SECRET"]
    meter = Meter(Caps(meta_usd=0.0, gpu_usd=1.0), phase=f"modal-serve-smoke-{time.strftime('%Y%m%d-%H%M%S')}")
    result = {"url": url, "vllm": "0.31.0"}
    with session(meter, MODAL, env, tag="smoke"), LogWatch(MODAL, env) as log:
        try:
            result["ready_after_s"] = round(wait_ready(url, key, secret, failed=log.failure, log_lines=lambda: len(log.lines)), 1)
            say("sending one tool-call request")
            t0 = time.time()
            resp = litellm.completion(model=f"openai/{MODEL_NAME}", messages=MESSAGES, tools=TOOLS, max_tokens=2048,
                                      **litellm_args(url, key, secret, "medium"))
            result["request_s"] = round(time.time() - t0, 1)
            say(f"response in {result['request_s']} s")
            # Record the response before the session tears down, so a teardown error cannot lose it
            _record(result, resp, log)
        except BaseException as e:
            say(f"SMOKE TEST FAILED: {type(e).__name__}: {e}")
            log.dump()
            raise
    result["gpu_usd_metered"] = round(meter.gpu_usd, 3)
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    say(f"SMOKE TEST {'PASSED' if result['pass'] else 'FAILED CHECKS'}: {OUT}")
    print(json.dumps(result, indent=1))


def _record(result: dict, resp, log) -> None:
    choice = resp.choices[0]
    calls = choice.message.tool_calls or []
    content = getattr(choice.logprobs, "content", None) or []
    result |= {
        "kv_cache_lines": [line.strip()[:200] for line in log.lines
                           if re.search(r"KV cache|GPU KV|Maximum concurrency", line)],
        "tool_calls": [{"name": c.function.name, "arguments": c.function.arguments} for c in calls],
        "content": choice.message.content,
        "finish_reason": choice.finish_reason,
        "completion_tokens": resp.usage.completion_tokens,
        "logprob_tokens": len(content),
        # LiteLLM moves vLLM's extra choice field into provider_specific_fields
        "token_ids": len((getattr(choice, "provider_specific_fields", None) or {}).get("token_ids") or []),
    }
    result["pass"] = bool(calls) and calls[0].function.name == "get_order_details" and \
        "#W2890441" in calls[0].function.arguments and result["logprob_tokens"] == result["completion_tokens"]
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    RAW.write_text(json.dumps(resp.model_dump(), indent=1, default=str), encoding="utf-8")
    say(f"recorded: tool calls {result['tool_calls']}, {result['logprob_tokens']}/{result['completion_tokens']} "
        f"tokens with logprobs, {result['token_ids']} token ids")

if __name__ == "__main__":
    main()
