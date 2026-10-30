"""glint-bank: one live τ² banking_knowledge environment per Muse session, served over MCP (stdio), every assistant
write gated by glint.gates.

Run: python -m glint.bank.server [--decision-tools]    (stdio MCP server)
     python -m glint.bank.server --selftest           (warm up, then a stdio tools/list round trip; exit 0 when ok)
     python -m glint.bank.server brief TASK_ID        (the customer card for a human playing the customer)
     python -m glint.bank.server log SESSION_ID       (a session's gate log, and its hash-chain check)

- **Sessions.** The session key is MUSE_SESSION_ID (Muse passes it to stdio servers), else `local-<pid>`. Each session
  holds an environment built with `get_environment(retrieval_variant="bm25_grep")`, set to a task's initial state, and
  passed through `assist_environment` with the session's `Enforcer`, so every assistant write through `get_response`
  is checked first (`glint.gates.enforce`). The gate log is `$GLINT_BANK_HOME/sessions/<sid>/gates.sqlite`
  (hash-chained; GLINT_BANK_HOME defaults to ~/.glint-bank).
- **Tools.** Generated from the toolkit: every visible τ² agent tool with its name, docstring and JSON schema, a
  read-only annotation on each tool that does not mutate state, and a destructive one on each WRITE tool. Plus the
  server's `load_scenario(task_id)` (fresh environment at the task's initial state) and `get_bank_policy`. Glint's
  banking decision tools are left out unless `--decision-tools` is given: with them, the agent may check first and
  never make the wrong write the demo shows being denied. `plan_disputes` is always published: the plan gate denies
  every credit card dispute filed without its plan, so a client without the tool could file none. It is annotated
  read-only (it never changes the bank) although τ² types it as mutating so its evaluator replays it.
- **Results.** A gate deny and a τ² tool error both return `is_error=true` with the message text (a deny carries the
  rule, what the check found, the facts it used, why it matters, the next step and the verdict). An unknown tool is
  a protocol error.
- **stdout.** τ² prints to stdout (a task file that fails to load is reported with print()), which would corrupt the
  stdio JSON-RPC stream: every τ² load and call runs with stdout redirected to stderr.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import anyio
import mcp_types as types
from mcp.server import Server
from mcp.shared.exceptions import MCPError


@contextlib.contextmanager
def quiet_stdout():
    """Send anything printed to stdout to stderr instead, for the duration."""
    with contextlib.redirect_stdout(sys.stderr):
        yield


with quiet_stdout():
    from tau2.data_model.message import ToolCall
    from tau2.domains.banking_knowledge.environment import get_environment, get_tasks
    from tau2.environment.toolkit import ToolType
    from tau2.runner.build import _derive_read_log_allowlist

    from glint.assist.banking import PLAN_TOOL
    from glint.assist.tau2_tools import assist_environment
    from glint.gates import Enforcer, EventLog

DOMAIN = "banking_knowledge"
RETRIEVAL = "bm25_grep"
LOAD, POLICY = "load_scenario", "get_bank_policy"
SERVER_NAME = "glint-bank"
INSTRUCTIONS = ("A live bank (τ²-bench banking_knowledge). You are the bank's agent; the user is the customer. Call "
                "get_bank_policy first and follow it. Every write is checked by a policy gate before it runs: a "
                "blocked write returns an error naming the rule, the reason and the fix, and was NOT executed. Follow the Fix.")


def bank_home() -> Path:
    """GLINT_BANK_HOME, else ~/.glint-bank. Muse starts stdio servers in a cleared environment that keeps HOME but not
    USERPROFILE, and on Windows Path.home() reads only USERPROFILE, so HOME is read first."""
    if os.environ.get("GLINT_BANK_HOME"):
        return Path(os.environ["GLINT_BANK_HOME"])
    return (Path(os.environ["HOME"]) if os.environ.get("HOME") else Path.home()) / ".glint-bank"


def default_session_id() -> str:
    return os.environ.get("MUSE_SESSION_ID") or f"local-{os.getpid()}"


def _safe(sid: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", sid) or "_"


def session_dir(home: Path, sid: str) -> Path:
    return home / "sessions" / _safe(sid)


@cache
def tasks() -> dict:
    with quiet_stdout():
        return {t.id: t for t in get_tasks()}


def build_environment(task=None):
    """An ungated τ² banking environment at `task`'s initial state (the base database with no task), with the read log
    τ²'s runner gives it: the hidden reads `task`'s gold makes are recorded in the scored database."""
    with quiet_stdout():
        allowlist = {"read_log_allowlist": _derive_read_log_allowlist(task)} if task is not None else {}
        env = get_environment(retrieval_variant=RETRIEVAL, task=task, **allowlist)
        init = task.initial_state if task is not None else None
        if init is not None:
            env.set_state(init.initialization_data, init.initialization_actions, init.message_history or [])
    return env


@dataclass
class BankSession:
    sid: str
    events: EventLog
    enforcer: Enforcer | None = None
    env: object = None
    task_id: str = ""
    call_id: str = ""
    published: set = field(default_factory=set)  # the τ² tool names this session's tool list carries
    lock: threading.Lock = field(default_factory=threading.Lock)

    def context(self) -> dict:
        return {"session_id": self.sid, "task_id": self.task_id, "call_id": self.call_id}


class BankServer:
    """The sessions and the MCP handlers. Every τ² call runs under its session's lock (τ²'s environment is not
    thread-safe, and Muse may issue calls in parallel)."""

    def __init__(self, home: Path | None = None, decision_tools: bool = False, sid: str | None = None):
        self.home = Path(home) if home is not None else bank_home()
        self.decision_tools = decision_tools
        self.sid = sid or default_session_id()
        self.sessions: dict[str, BankSession] = {}
        self._lock = threading.Lock()

    # ---- sessions -------------------------------------------------------------------------------------------------

    def session(self, sid: str | None = None) -> BankSession:
        """The session, created on first use, with a base environment so tools work before any scenario."""
        sid = sid or self.sid
        with self._lock:
            s = self.sessions.get(sid)
            if s is None:
                s = BankSession(sid, EventLog(session_dir(self.home, sid) / "gates.sqlite"))
                s.enforcer = Enforcer(DOMAIN, s.events, context=s.context)
                self.sessions[sid] = s
        with s.lock:
            if s.env is None:
                self._install(s, build_environment(None))
        return s

    def _install(self, s: BankSession, env) -> None:
        base = set(env.tools.get_tools())
        if self.decision_tools:  # the tools arm's toolkit, its excerpt KB_search and grep and KB_read included
            assist_environment(env, DOMAIN, s.enforcer)
        else:  # τ²'s own tools, gated; attach() adds plan_disputes, which the plan gate needs
            s.enforcer.attach(env)
        s.env = env
        s.published = set(env.tools.get_tools()) if self.decision_tools else base | {PLAN_TOOL}

    def load_scenario(self, sid: str | None, task_id: str) -> tuple[str, bool]:
        """A fresh environment at the task's initial state (the base database for an empty task_id). Returns only the
        task id and an opening line: never the customer's instructions or the evaluation criteria."""
        s = self.session(sid)
        task_id = (task_id or "").strip()
        if task_id and task_id not in tasks():
            return (f"Error: unknown task_id {task_id!r}; the 97 banking task ids run from task_001 to task_102, with "
                    "gaps."), True
        task = tasks()[task_id] if task_id else None
        env = build_environment(task)
        with s.lock:
            s.task_id = task_id
            self._install(s, env)
            s.events.emit("scenario.loaded", {"session_id": s.sid, "task_id": task_id}, actor="glint.bank")
        where = f"scenario {task_id}" if task_id else "the base bank database"
        return json.dumps({"task_id": task_id, "opening": (
            f"The bank is set to {where}. A customer is on the line and will describe what they need. Verify their "
            "identity as the policy requires before acting on any account.")}), False

    # ---- tools ----------------------------------------------------------------------------------------------------

    def tool_list(self, sid: str | None = None) -> list[types.Tool]:
        s = self.session(sid)
        with s.lock:
            toolkit = s.env.tools
            out = []
            for name, tool in toolkit.get_tools().items():
                if name not in s.published:
                    continue
                schema = tool.openai_schema["function"]
                out.append(types.Tool(name=name, description=schema.get("description") or "",
                                      input_schema=schema["parameters"], annotations=types.ToolAnnotations(
                                          read_only_hint=name == PLAN_TOOL or not toolkit.tool_mutates_state(name),
                                          destructive_hint=toolkit.tool_type(name) == ToolType.WRITE)))
        return out + [
            types.Tool(name=LOAD, description=(
                "Load a banking scenario: a fresh bank set to the task's starting state (an empty task_id loads the "
                "base database). Calling it again resets everything."), input_schema={
                "type": "object", "properties": {"task_id": {"type": "string", "description": "e.g. task_054"}}},
                annotations=types.ToolAnnotations(read_only_hint=False, destructive_hint=True)),
            types.Tool(name=POLICY, description="The bank's policy for its agent: read it before acting.",
                       input_schema={"type": "object", "properties": {}},
                       annotations=types.ToolAnnotations(read_only_hint=True, destructive_hint=False))]

    def call(self, sid: str | None, name: str, arguments: dict | None, call_id: str | None = None
             ) -> types.CallToolResult:
        """One tool call. τ² tools go through the session environment's gated `get_response` as an assistant call."""
        arguments = arguments or {}
        if name == LOAD:
            text, error = self.load_scenario(sid, str(arguments.get("task_id", "")))
            return _result(text, error)
        s = self.session(sid)
        if name == POLICY:
            with s.lock:
                return _result(s.env.get_policy(), False)
        if name not in s.published:
            raise MCPError(types.INVALID_PARAMS, f"unknown tool {name!r}")
        call = ToolCall(id=call_id or uuid.uuid4().hex, name=name, arguments=arguments, requestor="assistant")
        with s.lock, quiet_stdout():
            s.call_id = call.id
            try:
                reply = s.env.get_response(call)
            finally:
                s.call_id = ""
        return _result(str(reply.content), bool(reply.error))

    # ---- MCP ------------------------------------------------------------------------------------------------------

    def mcp_server(self) -> Server:
        async def on_list_tools(ctx, params) -> types.ListToolsResult:
            return types.ListToolsResult(tools=await anyio.to_thread.run_sync(self.tool_list, None))

        async def on_call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
            meta = params.meta or {}
            call_id = meta.get("callId") if isinstance(meta, dict) else None
            return await anyio.to_thread.run_sync(self.call, None, params.name, params.arguments, call_id)

        return Server(SERVER_NAME, instructions=INSTRUCTIONS, on_list_tools=on_list_tools, on_call_tool=on_call_tool)

    async def run_stdio(self) -> None:
        from mcp.server.stdio import stdio_server

        server = self.mcp_server()
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())


def _result(text: str, error: bool) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(text=text)], is_error=error)


# ---- CLIs ---------------------------------------------------------------------------------------------------------

def brief(task_id: str) -> str:
    """The customer card: what the human playing the customer knows and wants (never an agent tool)."""
    task = tasks().get(task_id)
    if task is None:
        raise SystemExit(f"unknown task_id {task_id!r}")
    return f"Customer card for {task_id}\n\n{task.user_scenario.instructions}"


def show_log(sid: str, home: Path | None = None) -> str:
    path = session_dir(Path(home) if home is not None else bank_home(), sid) / "gates.sqlite"
    if not path.exists():
        raise SystemExit(f"no gate log for session {sid!r} at {path}")
    events = EventLog(path)
    try:
        lines = []
        for e in events.events():
            p = e["payload"]
            if e["type"] == "gate.fired":
                lines.append(f"gate.fired  {p['decision']:5} {p['rule_id']:40} {p['tool']}")
            elif e["type"] == "scenario.loaded":
                lines.append(f"scenario.loaded  {p['task_id'] or '(base database)'}")
            else:
                lines.append(e["type"])
        lines.append(f"verify(): OK, {events.verify()} events, hash chain intact")
        return "\n".join(lines)
    finally:
        events.close()


async def _selftest_round_trip(env: dict) -> dict:
    from mcp.client.client import Client
    from mcp.client.stdio import StdioServerParameters

    params = StdioServerParameters(command=sys.executable, args=["-X", "utf8", "-m", "glint.bank.server"], env=env)
    async with Client(params, read_timeout_seconds=120) as client:
        listed = await client.list_tools()
        loaded = await client.call_tool(LOAD, {"task_id": "task_054"})
        policy = await client.call_tool(POLICY, {})
        found = await client.call_tool("get_user_information_by_name", {"customer_name": "Sofia Papadopoulos"})
    return {"tools": [t.name for t in listed.tools], "load_error": loaded.is_error, "policy_error": policy.is_error,
            "policy_chars": len(policy.content[0].text), "lookup_error": found.is_error}


def selftest() -> int:
    """Warm up (imports, the tasks, an environment), then start this server as a stdio subprocess and list its
    tools, load task_054, read the policy and run one read. Its gate log goes to a temporary folder."""
    n_tasks = len(tasks())
    build_environment(tasks()["task_054"])
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        env = dict(os.environ) | {"GLINT_BANK_HOME": tmp, "MUSE_SESSION_ID": "selftest"}
        report = anyio.run(_selftest_round_trip, env)
    ok = (n_tasks == 97 and len(report["tools"]) >= 18 and {LOAD, POLICY} <= set(report["tools"])
          and not (report["load_error"] or report["policy_error"] or report["lookup_error"]))
    print(json.dumps({"ok": ok, "tasks": n_tasks} | report), file=sys.stderr)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="glint-bank", description=__doc__.split("\n\n")[0])
    parser.add_argument("command", nargs="?", choices=("serve", "brief", "log"), default="serve")
    parser.add_argument("target", nargs="?", help="brief: a task id; log: a session id")
    parser.add_argument("--decision-tools", action="store_true", help="also publish Glint's banking decision tools")
    parser.add_argument("--selftest", action="store_true", help="warm up and run a stdio round trip, then exit")
    args = parser.parse_args(argv)
    from loguru import logger  # τ² logs its whole registry at DEBUG; the server keeps warnings only, on stderr

    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    if args.selftest:
        return selftest()
    if args.command in ("brief", "log"):
        if not args.target:
            parser.error(f"{args.command} needs a {'task id' if args.command == 'brief' else 'session id'}")
        print(brief(args.target) if args.command == "brief" else show_log(args.target))
        return 0
    anyio.run(BankServer(decision_tools=args.decision_tools).run_stdio)
    return 0


if __name__ == "__main__":
    sys.exit(main())
