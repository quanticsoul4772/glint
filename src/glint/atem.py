"""Parse a Glimmer turn (ATEM format) from its raw generated text, keeping schema-string arguments as strings.

Hosted Glimmer is parsed server-side by vLLM's `muse_glimmer` tool parser, which sets
`_preserve_string_parameters = False`: every argument goes through `json.loads`, so a zip code "80273" arrives as
the number 80273 (and "02134" would lose its leading zero). vLLM's newer `muse` parser (#62207, a port of Meta's
HF `response_schema` contract) keeps parameters whose schema type is "string" verbatim. This module ports that
non-streaming logic (vllm/tool_parsers/muse_tool_parser.py, Apache-2.0) so Glint parses the exact text Glimmer
generated, taken from the logprob token stream.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

# "<|start|>assistant to=tool<|message|>", " to=self<|message|>" (first message of a turn) or a bare "<|message|>"
_MSG_HEADER_RE = re.compile(
    r"(?:<\|start\|>\s*assistant)?[^\S\n]*(?:to=(?P<rcpt>[A-Za-z0-9_.\-]+))?<\|message\|>"
)
_MSG_END_RE = re.compile(r"<\|eom\|>|<\|eot\|>")
_INVOKE_RE = re.compile(r"(<atem:invoke\b.*?</atem:invoke>)", re.DOTALL)
_NAME_RE = re.compile(r'<atem:invoke\b[^>]*?\bname="([^"]+)"')
_PARAM_RE = re.compile(
    r'<atem:parameter\b[^>]*?\bname="(?P<key>[^"]+)"[^>]*?>(?P<value>.*?)</atem:parameter>', re.DOTALL
)
_REASONING = (None, "self")
_CONTENT = ("user", "commentary")
_NON_TOOL = ("self", "user", "commentary", "developer")


class AtemParseError(ValueError):
    pass


@dataclass
class ParsedCall:
    name: str
    arguments: dict


@dataclass
class ParsedTurn:
    reasoning: str = ""
    content: str = ""
    tool_calls: list[ParsedCall] = field(default_factory=list)


def _iter_messages(text: str) -> Iterator[tuple[str | None, str]]:
    """Yield (recipient, body) per message. A message also ends where the next header starts, so a reasoning
    block missing its <|eom|> cannot swallow the tool call after it."""
    pos = 0
    while pos < len(text):
        header = _MSG_HEADER_RE.search(text, pos)
        if header is None:
            return
        start = header.end()
        end = _MSG_END_RE.search(text, start)
        nxt = _MSG_HEADER_RE.search(text, start)
        body_end = end.start() if end is not None else len(text)
        if nxt is not None and nxt.start() < body_end:
            body_end, pos = nxt.start(), nxt.start()
        else:
            pos = end.end() if end is not None else len(text)
        body = text[start:body_end]
        cut = body.find("<|start|>")
        yield header.group("rcpt"), body if cut == -1 else body[:cut]


def _decode(raw: str):
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def _schemas(tools: list[dict]) -> dict[str, dict]:
    out = {}
    for tool in tools:
        fn = tool.get("function", tool)
        out[fn["name"]] = fn.get("parameters") or {}
    return out


def _normalize_name(emitted: str, registered: set[str]) -> str:
    """Collapse the doubled form `get_x.get_x` the template's `"get_x.*"` recipient invites; nothing else.

    A name that is not registered passes through as Glimmer wrote it: the environment answers it as an unknown tool
    and Glimmer can recover. Raising here turned such a call into an infrastructure error that ended the whole run
    (2026-10-27: Glimmer called the hidden banking tool get_user_dispute_history_7291 by name, in 2 of 8 runs)."""
    if emitted in registered:
        return emitted
    head, sep, tail = emitted.partition(".")
    if sep and head == tail:
        return head
    return emitted


def parse_turn(text: str, tools: list[dict]) -> ParsedTurn:
    """Parse one generated assistant turn. `tools` are the OpenAI-format tool definitions sent with the request."""
    schemas = _schemas(tools)
    turn = ParsedTurn()
    tool_bodies = []
    for rcpt, body in _iter_messages(text):
        if rcpt in _REASONING:
            turn.reasoning += body
        elif rcpt in _CONTENT:
            turn.content += body
        elif rcpt not in _NON_TOOL:
            tool_bodies.append(body)
    for invoke in _INVOKE_RE.findall("\n".join(tool_bodies)):
        name_m = _NAME_RE.search(invoke)
        if not name_m:
            raise AtemParseError(f"invoke without a name: {invoke[:120]!r}")
        name = _normalize_name(name_m.group(1), set(schemas))
        props = schemas.get(name, {}).get("properties", {})  # unregistered: no schema, every value JSON-decoded
        strings = {k for k, s in props.items() if isinstance(s, dict) and s.get("type") == "string"}
        args = {}
        for pm in _PARAM_RE.finditer(invoke):
            key, raw = pm.group("key"), pm.group("value")
            args[key] = raw if key in strings else _decode(raw)
        turn.tool_calls.append(ParsedCall(name, args))
    return turn


def generated_text(choice: dict) -> str:
    """The exact generated text of a chat-completion choice, rebuilt from its logprob token stream."""
    stream = (choice.get("logprobs") or {}).get("content")
    if not stream:
        raise AtemParseError("response has no logprob token stream; request logprobs=True")
    return "".join(t["token"] for t in stream)


def reparse_response(response, tools: list[dict]) -> None:
    """Replace a LiteLLM ModelResponse's tool-call arguments, content and reasoning with `parse_turn` of the
    generated text. The host's call ids are kept; the number and names of calls must agree with the host's."""
    choice = response.choices[0]
    turn = parse_turn(generated_text(choice.model_dump() if hasattr(choice, "model_dump") else choice), tools)
    host_calls = choice.message.tool_calls or []
    if [c.function.name for c in host_calls] != [c.name for c in turn.tool_calls]:
        raise AtemParseError(f"host calls {[c.function.name for c in host_calls]} != "
                             f"generated calls {[c.name for c in turn.tool_calls]}")
    for host, ours in zip(host_calls, turn.tool_calls):
        host.function.arguments = json.dumps(ours.arguments, ensure_ascii=False)
    choice.message.content = turn.content or None
    choice.message.reasoning_content = turn.reasoning or None


def reparsing(completion_fn):
    """Wrap a completion function so every response with tools is re-parsed from Glimmer's generated text."""
    def _call(*args, **kwargs):
        response = completion_fn(*args, **kwargs)
        if kwargs.get("tools"):
            reparse_response(response, kwargs["tools"])
        return response

    return _call
