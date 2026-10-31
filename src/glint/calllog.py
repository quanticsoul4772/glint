"""Record every Glimmer call's size and timing, tied to the run it belongs to.

`recording(fn, path, context=...)` wraps the completion function and appends one line per Glimmer call: the time, the
model, `request_key` of the messages sent, the number of messages, the prompt and completion token counts, and the
task and trial of the simulation that made it (`context`, the run context the gate log uses). A recorded call is
matched to its turn in a saved run by recomputing `request_key` from the run's messages with τ²'s own
`to_litellm_messages`. The peak prompt size per run shows how close a run came to Glimmer's context limit.

Until 2026-10-30 each line also held the prompt and completion token ids and the logprob tokens, for activation
capture (about 61 MB per 48-run set); that idea was cut on Oct 27 and nothing read them, so they are no longer kept
(decision 2026-10-30). Older calls.jsonl files still carry them.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path


class CallLogError(RuntimeError):
    # the class name goes first (τ² keeps only str(exception)); runlog.outcome counts it as Glint's failure
    def __str__(self) -> str:
        return f"CallLogError: {super().__str__()}"


def request_key(litellm_messages: list[dict]) -> str:
    """Stable hash of the conversation of one request. A leading system message is skipped: τ²'s agent prepends its
    system prompt to every request, and a saved run does not store it."""
    msgs = list(litellm_messages)
    if msgs and msgs[0].get("role") == "system":
        msgs = msgs[1:]
    return hashlib.sha256(json.dumps(msgs, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def key_for_turn(messages) -> str:
    """request_key of the request that produced the assistant turn after `messages` (τ² Message objects from a
    saved run), computed with τ²'s own conversion."""
    from tau2.utils.llm_utils import to_litellm_messages

    return request_key(to_litellm_messages(messages))


def _get(obj, *path):
    for key in path:
        if obj is None:
            return None
        obj = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
    return obj


def call_record(response, messages: list[dict], model: str, context: Callable[[], tuple] | None = None) -> dict:
    """One recorded call. Token counts come from the response's usage, else from the token ids a vLLM endpoint
    returns; the task and trial from `context()` ("unknown" without one)."""
    dump = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    choice = (dump.get("choices") or [{}])[0]
    usage = dump.get("usage") or {}
    prompt = usage.get("prompt_tokens") or len(dump.get("prompt_token_ids") or [])
    completion = usage.get("completion_tokens") or len(
        _get(choice, "provider_specific_fields", "token_ids") or choice.get("token_ids") or [])
    # (task, trial) or (task, trial, simulation_id): tools_ab's run_context gives the attempt's id too
    task, trial, *rest = context() if context is not None else ("unknown", "unknown")
    return {"ts": time.time(), "model": model, "task_id": task, "trial": trial,
            **({"simulation_id": rest[0]} if rest else {}), "request_key": request_key(messages),
            "n_messages": len(messages), "prompt_tokens": int(prompt), "completion_tokens": int(completion)}


# What a Glimmer response carries for atem and the call log but τ² need not keep: τ² saves every assistant message's
# raw response (`raw_data=response.to_dict()`) in its checkpoint and rewrites the whole file after each run. On
# 2026-10-31 the unseen tools arm's checkpoint reached 5.4 GB for 100 runs (about 54 MB a run; in six v9 runs these
# fields were 24 of 25 MB): each save took minutes, finished runs queued behind it for over an hour, the GPU idled and
# scaled down, and the driver held 28 GB of memory.
BULKY_TOP = ("prompt_token_ids",)
BULKY_CHOICE = ("logprobs",)
BULKY_CHOICE_FIELDS = ("token_ids",)  # inside a choice's provider_specific_fields


def lean_saved(completion_fn, only: str = "Muse-Glimmer"):
    """Wrap a completion function so the saved copy of each response (its `to_dict()`, what τ² stores as raw_data)
    leaves out the prompt's token ids, the per-token logprobs and the generated token ids. The response object keeps
    them, so anything reading it live (atem's re-parse, the call log) is unchanged. Wrap outside both."""
    def _call(*args, **kwargs):
        response = completion_fn(*args, **kwargs)
        if only in kwargs.get("model", "") and callable(getattr(response, "to_dict", None)):
            full = response.to_dict

            def lean(*a, **k):
                d = full(*a, **k)
                for key in BULKY_TOP:
                    d.pop(key, None)
                for choice in d.get("choices") or []:
                    for key in BULKY_CHOICE:
                        choice.pop(key, None)
                    fields = choice.get("provider_specific_fields")
                    if isinstance(fields, dict):
                        for key in BULKY_CHOICE_FIELDS:
                            fields.pop(key, None)
                return d

            object.__setattr__(response, "to_dict", lean)  # the instance's own to_dict; the class is untouched
        return response

    return _call


def recording(completion_fn, path: Path, only: str = "Muse-Glimmer", context: Callable[[], tuple] | None = None):
    """Wrap a completion function so every call to a model whose name contains `only` is appended to `path`, with the
    task and trial `context()` names (tools_ab's current_run)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def _call(*args, **kwargs):
        response = completion_fn(*args, **kwargs)
        model = kwargs.get("model", "")
        if only in model:
            try:
                row = call_record(response, kwargs.get("messages") or [], model, context)
            except Exception as e:  # any recording bug after a paid call is Glint's failure, named as such
                raise CallLogError(f"recording a {model} call failed: {type(e).__name__}: {e}") from e
            with lock, path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        return response

    return _call


def load(path: Path) -> dict[str, dict]:
    """Recorded calls by request_key (the last call wins if a request was retried; the number replaced is printed)."""
    import sys

    out, replaced = {}, 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        replaced += row["request_key"] in out
        out[row["request_key"]] = row
    if replaced:
        print(f"calllog: {replaced} of {len(out) + replaced} recorded calls repeat an earlier request (retries); "
              "the last of each is kept", file=sys.stderr, flush=True)
    return out
