"""Record every Glimmer call exactly as the server saw and produced it, for activation capture.

τ²'s saved runs keep neither token ids nor logprobs nor usage for the agent's messages (2026-10-27: `raw_data` empty
in every recorded run), so a write turn's prompt could only be re-rendered, and only 34 of 119 banking prompts
re-rendered to the server's token count. `recording(fn, path)` wraps the completion function and appends one line per
Glimmer call: the prompt token ids and completion token ids vLLM returned (`return_token_ids`), the logprob tokens,
and `request_key` of the messages sent. A recorded call is matched to its turn in a saved run by recomputing
`request_key` from the run's messages with τ²'s own `to_litellm_messages`. Token ids are zlib-compressed int32.
"""
from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
import zlib
from array import array
from pathlib import Path


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


def pack(ids: list[int]) -> str:
    return base64.b64encode(zlib.compress(array("i", ids).tobytes(), 6)).decode("ascii")


def unpack(blob: str) -> list[int]:
    out = array("i")
    out.frombytes(zlib.decompress(base64.b64decode(blob)))
    return out.tolist()


def _get(obj, *path):
    for key in path:
        if obj is None:
            return None
        obj = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
    return obj


class CallLogError(RuntimeError):
    # the class name goes first (τ² keeps only str(exception)); runlog.outcome counts it as Glint's failure
    def __str__(self) -> str:
        return f"CallLogError: {super().__str__()}"


def call_record(response, messages: list[dict], model: str, require_ids: bool = False) -> dict:
    """One recorded call. With `require_ids` (a vLLM endpoint asked for `return_token_ids`), a response without
    token ids raises: activation capture would otherwise get silently empty records."""
    dump = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    choice = (dump.get("choices") or [{}])[0]
    prompt_ids = dump.get("prompt_token_ids") or []
    completion_ids = _get(choice, "provider_specific_fields", "token_ids") or choice.get("token_ids") or []
    if require_ids and not (prompt_ids and completion_ids):
        raise CallLogError(f"{model}: the response has no token ids (prompt {len(prompt_ids)}, completion "
                           f"{len(completion_ids)}); the endpoint must be asked for return_token_ids")
    logprobs = [t.get("token") for t in (_get(choice, "logprobs", "content") or [])]
    return {"ts": time.time(), "model": model, "request_key": request_key(messages), "n_messages": len(messages),
            "prompt_tokens": len(prompt_ids), "completion_tokens": len(completion_ids),
            "prompt_ids": pack(prompt_ids), "completion_ids": pack(completion_ids), "logprob_tokens": logprobs}


def recording(completion_fn, path: Path, only: str = "Muse-Glimmer", require_ids: bool = False):
    """Wrap a completion function so every call to a model whose name contains `only` is appended to `path`.
    `require_ids`: see `call_record` (set it for the vLLM endpoint, which returns token ids; hosted APIs do not)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def _call(*args, **kwargs):
        response = completion_fn(*args, **kwargs)
        model = kwargs.get("model", "")
        if only in model:
            try:
                row = call_record(response, kwargs.get("messages") or [], model, require_ids)
            except CallLogError:
                raise
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
