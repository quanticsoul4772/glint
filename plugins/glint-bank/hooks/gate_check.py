# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""glint-bank's Muse Code PreToolUse hook: `uv run --script hooks/gate_check.py` (the plugin's hook command).

Muse runs it before every tool call, with the call as JSON on stdin (hook_event_name, tool_name, tool_input,
tool_use_id, session_id, ...). Plugin hooks have no matcher, so it sees every tool, Muse's own included:
- a tool that is not one of glint-bank's write tools (WRITE_TOOLS, under the plugin's `mcp__plugin_glint_bank_b__`
  prefix) is let through: the hook prints nothing and exits 0 (Muse accepts a bare allow only with updatedInput, so
  "no decision" is how a hook lets a call through);
- a glint-bank write is sent to the bank server's gates (glint.bank.hook_endpoint: POST /decide on the port and with
  the token `<GLINT_BANK_HOME>/sessions/<MUSE_SESSION_ID>/hook.json` names). On a deny the hook prints
  {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
  "permissionDecisionReason": <the gate's refusal>}}, which blocks the call; the agent sees the refusal.
- Fail closed: when the decision cannot be had (no hook.json, the server unreachable or slow, a non-200 answer, an
  unreadable answer, unreadable hook input) the call is denied with the reason.

The server checks every call it receives as well; the hook is a second enforcement point, in Muse's permission path.
Standard library only, so uv starts it without installing anything.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

PREFIX = "mcp__plugin_glint_bank_b__"  # mcp__plugin_<plugin id, - as _>_<server id>__ (live probe, 2026-10-28)
# the published tools glint.gates.enforce gates: τ²'s WRITE tools (call_discoverable_agent_tool's hidden tool is
# judged by the server, which lets a hidden read through) and the gated GENERIC tools (rules.GATED_GENERIC);
# tests/test_bank_hook.py checks this set against the server's tool list
WRITE_TOOLS = frozenset({"call_discoverable_agent_tool", "change_user_email", "log_verification",
                         "transfer_to_human_agents", "give_discoverable_user_tool"})
TIMEOUT_S = 8.0  # below the hook's timeoutMs in plugin.json (10000)
HOOK_FILE = "hook.json"


def bank_home() -> Path:
    """GLINT_BANK_HOME, else ~/.glint-bank, as glint.bank.server.bank_home finds it."""
    if os.environ.get("GLINT_BANK_HOME"):
        return Path(os.environ["GLINT_BANK_HOME"])
    return (Path(os.environ["HOME"]) if os.environ.get("HOME") else Path.home()) / ".glint-bank"


def session_id(payload: dict) -> str:
    """The session key the bank server uses: MUSE_SESSION_ID (Muse gives stdio servers and hooks the top-level chat's
    id), else the payload's session_id."""
    return os.environ.get("MUSE_SESSION_ID") or str(payload.get("session_id") or "")


def _safe(sid: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", sid) or "_"  # as glint.bank.server._safe


def deny(reason: str) -> dict:
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                   "permissionDecisionReason": reason}}


def blocked(why: str) -> dict:
    """The fail-closed refusal: the write was not sent, and why."""
    return deny(f"Error: this write was blocked and was NOT executed. Nothing changed. Glint's PreToolUse hook could "
                f"not get the policy gates' decision ({why}), and no glint-bank write runs without one. Check that the "
                f"glint-bank server is running in this session.")


def ask_gates(tool: str, arguments: dict, call_id: str | None, sid: str) -> dict | None:
    """The hook output for one glint-bank write: None to let it through, else a deny."""
    path = bank_home() / "sessions" / _safe(sid) / HOOK_FILE
    try:
        where = json.loads(path.read_text(encoding="utf-8"))
        port, token = int(where["port"]), str(where["token"])
    except FileNotFoundError:
        return blocked(f"no gate endpoint for session {sid!r}: {path} does not exist")
    except (OSError, ValueError, KeyError, TypeError) as e:
        return blocked(f"{path} is unreadable: {type(e).__name__}: {e}")
    body = json.dumps({"token": token, "tool": tool, "arguments": arguments, "call_id": call_id}).encode("utf-8")
    request = urllib.request.Request(f"http://127.0.0.1:{port}/decide", data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            answer = json.loads(response.read())
    except urllib.error.HTTPError as e:
        return blocked(f"the gate endpoint answered HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace')}")
    except (OSError, ValueError) as e:  # refused, timed out, reset; or not JSON
        return blocked(f"the gate endpoint on 127.0.0.1:{port} could not be reached or read: {type(e).__name__}: {e}")
    if not isinstance(answer, dict):
        return blocked(f"the gate endpoint's answer is not an object: {answer!r:.200}")
    if answer.get("decision") == "allow":
        return None
    if answer.get("decision") == "deny" and isinstance(answer.get("reason"), str) and answer["reason"]:
        return deny(answer["reason"])
    return blocked(f"the gate endpoint's answer has no decision: {answer!r:.200}")


def decide(payload: object) -> dict | None:
    """The hook output for one PreToolUse payload: None to let the call through, else a deny."""
    if not isinstance(payload, dict):
        return blocked("the hook input is not a JSON object")
    name = payload.get("tool_name")
    if not isinstance(name, str) or not name.startswith(PREFIX) or name[len(PREFIX):] not in WRITE_TOOLS:
        return None
    arguments = payload.get("tool_input")
    if not isinstance(arguments, dict):
        return blocked(f"tool_input of {name} is not an object")
    call_id = payload.get("tool_use_id") if isinstance(payload.get("tool_use_id"), str) else None
    return ask_gates(name[len(PREFIX):], arguments, call_id, session_id(payload))


def main(stdin=None, stdout=None) -> int:
    """Read the payload as UTF-8 bytes (Windows' console encoding would garble non-ASCII arguments) and write the
    decision as ASCII JSON; exit 0 either way (a deny is a decision, not a hook failure)."""
    stdin, stdout = stdin or sys.stdin.buffer, stdout or sys.stdout
    try:
        payload = json.loads(stdin.read().decode("utf-8-sig"))
    except ValueError:  # not UTF-8, or not JSON
        payload = None
    out = decide(payload)
    if out is not None:
        stdout.write(json.dumps(out))
        stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
