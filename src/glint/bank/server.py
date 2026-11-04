"""glint-bank: one live τ² banking_knowledge environment per Muse session, served over MCP (stdio), every assistant
write gated by glint.gates.

Run: python -m glint.bank.server [--decision-tools] [--no-request-gate] [--no-uncovered-reminder] [--no-stages]
                                                     (stdio MCP server)
     python -m glint.bank.server --selftest           (warm up, then a stdio round trip: the tool list, and on
                                                     task_054 a $4,000 increase denied and $2,500 allowed; exit 0
                                                     when ok)
     python -m glint.bank.server brief TASK_ID        (the customer card for a human playing the customer)
     python -m glint.bank.server log SESSION_ID       (a session's gate log, and its hash-chain check)

- **Sessions.** The session key is MUSE_SESSION_ID (Muse passes it to stdio servers), else `local-<pid>`. Each session
  holds an environment built with `get_environment(retrieval_variant="bm25_grep")`, set to a task's initial state, and
  passed through `assist_environment` with the session's `Enforcer`, so every assistant write through `get_response`
  is checked first (`glint.gates.enforce`). The gate log is `$GLINT_BANK_HOME/sessions/<sid>/gates.sqlite`
  (GLINT_BANK_HOME defaults to ~/.glint-bank), HMAC-SHA256 hash-chained under GLINT_LOG_KEY when set, else under a
  per-install key generated on first use in GLINT_LOG_KEY_FILE (default `$GLINT_BANK_HOME/log.key`, outside every
  session folder). The key protects the chain only while the agent cannot read that file: put it where the agent's
  file tools do not reach. `log SESSION_ID` verifies keyed and older unkeyed logs alike and says which it is.
- **Tools.** Generated from the toolkit: every visible τ² agent tool with its name, docstring and JSON schema, a
  read-only annotation on each tool that does not mutate state, and a destructive one on each WRITE tool. Plus the
  server's `load_scenario(task_id)` (fresh environment at the task's initial state; refused to the agent once a gate
  has decided a write in the session, so a reload cannot clear the enforcer's records) and `get_bank_policy`. Glint's
  banking decision tools (the search fix, KB_read and the check tools) are published with `--decision-tools`, which
  the Muse plugin's server command passes; without the flag the server has the gates only. `plan_disputes` is always published: the plan gate denies
  every credit card dispute filed without its plan, so a client without the tool could file none. It is annotated
  read-only (it never changes the bank) although τ² types it as mutating so its evaluator replays it.
  `--no-request-gate` turns the request gate off (research row 6): no request gates, no request-record reminder, and
  record_requests is not published.
- **Every fix on by default** (owner's decision, 2026-11-03): the session's Enforcer takes every registered option's
  default (glint.gates.extend). `--no-uncovered-reminder` turns the [review] open item after a no_rule_yet write off;
  `--no-stages` turns the stages off (glint.gates.stages; the server sees no conversation, so even with a model
  configured no conversation stage has a transcript and none runs).
- **The PreToolUse hook.** While the stdio server runs it serves its session's gate decisions on 127.0.0.1
  (glint.bank.hook_endpoint, named by `<session dir>/hook.json`), which the Muse plugin's PreToolUse hook
  (plugins/glint-bank/hooks/gate_check.py) asks before each write reaches the server: a second enforcement point, in
  Muse's own permission path. `decide` runs the same gates on the same records and runs nothing; the server still
  checks every call it receives.
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


def warnings_only() -> None:
    """Keep loguru's warnings only, on stderr: τ² logs its whole registry at DEBUG when it is imported."""
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")


if __name__ == "__main__":  # run as a program (python -m glint.bank.server): quiet τ²'s import-time logs as well
    warnings_only()

with quiet_stdout():
    from tau2.data_model.message import ToolCall
    from tau2.domains.banking_knowledge.environment import get_environment, get_tasks
    from tau2.environment.toolkit import ToolType
    from tau2.runner.build import _derive_read_log_allowlist

    from glint.assist.banking import ADDRESS_TOOL, DISPUTE_ANSWERS_TOOL, PLAN_TOOL, REQUESTS_TOOL, RETENTION_TOOL
    from glint.assist.tau2_tools import assist_environment
    from glint.bank.hook_endpoint import FILE as HOOK_FILE
    from glint.bank.hook_endpoint import HookEndpoint
    from glint.gates import ChainBroken, Enforcer, EventLog
    from glint.gates.eventlog import KEY_ENV, install_key, log_key, recorded_scheme
    from glint.gates.stages import AUTO

DOMAIN = "banking_knowledge"
RETRIEVAL = "bm25_grep"
READ_ONLY_GENERIC = (PLAN_TOOL, RETENTION_TOOL, REQUESTS_TOOL, DISPUTE_ANSWERS_TOOL, ADDRESS_TOOL,
                     "check_credit_limit_increase")
LOAD, POLICY = "load_scenario", "get_bank_policy"
# published only with --decision-tools (the search fix replaces KB_search's body; it adds no name)
DECISION_TOOLS = frozenset({"check_credit_limit_increase", "check_card_dispute", "check_verification", "KB_read"})
SERVER_NAME = "glint-bank"
LOAD_LOCKED = ("Error: load_scenario is refused once a write has been made in this session: a reload would clear the "
               "gate's records of this conversation. Nothing changed. Carry on with the current scenario; to start "
               "another one, start a new session.")
INSTRUCTIONS = ("A live bank (τ²-bench banking_knowledge). You are the bank's agent; the user is the customer. Call "
                "get_bank_policy first and follow it. Every write is checked by a policy gate before it runs: a "
                "blocked write returns an error naming the rule, the reason and the fix, and was NOT executed. Follow the Fix.")


def bank_home() -> Path:
    """GLINT_BANK_HOME, else ~/.glint-bank. Muse starts stdio servers in a cleared environment that keeps HOME but not
    USERPROFILE, and on Windows Path.home() reads only USERPROFILE, so HOME is read first."""
    if os.environ.get("GLINT_BANK_HOME"):
        return Path(os.environ["GLINT_BANK_HOME"])
    return (Path(os.environ["HOME"]) if os.environ.get("HOME") else Path.home()) / ".glint-bank"


KEY_FILE_ENV = "GLINT_LOG_KEY_FILE"


def key_file(home: Path) -> Path:
    """GLINT_LOG_KEY_FILE, else `<home>/log.key`: outside every session folder, so no session's files include it."""
    return Path(os.environ[KEY_FILE_ENV]) if os.environ.get(KEY_FILE_ENV) else Path(home) / "log.key"


def session_key(home: Path) -> bytes:
    """The key every session log is chained with: GLINT_LOG_KEY when set, else the per-install key file (created on
    first use). Session logs are always keyed."""
    return log_key() or install_key(key_file(home))


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
    loaded_at: int = 0  # the gate log's event count when this environment was installed
    lock: threading.Lock = field(default_factory=threading.Lock)

    def context(self) -> dict:
        return {"session_id": self.sid, "task_id": self.task_id, "call_id": self.call_id}


class BankServer:
    """The sessions and the MCP handlers. Every τ² call runs under its session's lock (τ²'s environment is not
    thread-safe, and Muse may issue calls in parallel)."""

    def __init__(self, home: Path | None = None, decision_tools: bool = False, sid: str | None = None,
                 request_gate: bool = True, uncovered_reminder: bool = True, stages: bool = True):
        self.home = Path(home) if home is not None else bank_home()
        self.decision_tools, self.request_gate = decision_tools, request_gate
        self.uncovered_reminder, self.stages = uncovered_reminder, stages
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
                try:
                    events = EventLog(session_dir(self.home, sid) / "gates.sqlite", key=session_key(self.home))
                except ChainBroken as e:  # a resumed session whose log has another key, or none (before keying)
                    raise MCPError(types.INTERNAL_ERROR, f"{e}. Restart the server with {KEY_ENV} (or "
                                   f"{KEY_FILE_ENV}) as it was when this session's log was created, or start a new "
                                   "session.") from e
                s = BankSession(sid, events)
                s.enforcer = Enforcer(DOMAIN, s.events, context=s.context, request_gate=self.request_gate,
                                      uncovered_reminder=self.uncovered_reminder,
                                      stages=AUTO if self.stages else None)
                self.sessions[sid] = s
        with s.lock:
            if s.env is None:
                self._install(s, build_environment(None))
        return s

    def _install(self, s: BankSession, env) -> None:
        base = set(env.tools.get_tools())
        # the tools arm's toolkit: its budgeted KB_search and grep (best document whole, the rest as excerpts), KB_read
        if self.decision_tools:
            assist_environment(env, DOMAIN, s.enforcer)
        else:  # τ²'s own tools, gated; attach() adds plan_disputes and record_retention_answer, which gates need
            s.enforcer.attach(env)
        s.env = env
        s.loaded_at = s.events.count()
        s.published = set(env.tools.get_tools()) if self.decision_tools else base | {
            PLAN_TOOL, RETENTION_TOOL, DISPUTE_ANSWERS_TOOL, ADDRESS_TOOL} | ({REQUESTS_TOOL} if self.request_gate else set())

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

    def gated_since_load(self, sid: str | None = None) -> bool:
        """Whether a gate has decided a write in this session's current environment. From then on the agent's
        load_scenario is refused: a reload rebuilds the environment and with it the enforcer's records (the refusals
        the repeat escalation reads, the requests, the decided increases), so an agent could otherwise clear them
        between attempts (judge panel 2026-11-01 and 2026-11-03, technical 2.6). The operator's
        `BankServer.load_scenario` (the replays, scripts) is not limited."""
        s = self.session(sid)
        return any(not e["payload"].get("replay") for e in s.events.events("gate.fired", after=s.loaded_at))

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
                                          # both are GENERIC with mutates_state=True only so τ²'s replay rebuilds
                                          # the enforcer's records; neither writes the database
                                          read_only_hint=name in READ_ONLY_GENERIC
                                          or not toolkit.tool_mutates_state(name),
                                          destructive_hint=toolkit.tool_type(name) == ToolType.WRITE)))
        return out + [
            types.Tool(name=LOAD, description=(
                "Load a banking scenario: a fresh bank set to the task's starting state (an empty task_id loads the "
                "base database). Call it once, before any write: after the first write it is refused."), input_schema={
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
            if self.gated_since_load(sid):
                return _result(LOAD_LOCKED, True)
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

    def decide(self, sid: str | None, name: str, arguments: dict | None, call_id: str | None = None) -> str | None:
        """The gates' decision on a tool call before it runs, for the PreToolUse hook (glint.bank.hook_endpoint): the
        refusal text `call` would return, or None when it would not be refused by a gate (a read, an allowed write,
        the server's own tools, an unknown name: `call` answers those). Runs nothing; a refusal leaves the same log
        and records as the server's own (Enforcer.precheck) and adds a hook.denied event."""
        s = self.session(sid)
        if name not in s.published:
            return None
        call = ToolCall(id=call_id or uuid.uuid4().hex, name=name, arguments=arguments or {}, requestor="assistant")
        with s.lock, quiet_stdout():
            s.call_id = call.id
            try:
                refusal = s.enforcer.precheck(s.env, call)
            finally:
                s.call_id = ""
            if refusal is None:
                return None
            s.events.emit("hook.denied", {"tool": name, "call_id": call.id, "session_id": s.sid,
                                          "task_id": s.task_id}, actor="glint.bank.hook")
        return str(refusal.content)

    # ---- MCP ------------------------------------------------------------------------------------------------------

    def mcp_server(self) -> Server:
        async def on_list_tools(ctx, params) -> types.ListToolsResult:
            return types.ListToolsResult(tools=await anyio.to_thread.run_sync(self.tool_list, None))

        async def on_call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
            meta = params.meta or {}
            call_id = meta.get("callId") if isinstance(meta, dict) else None
            return await anyio.to_thread.run_sync(self.call, None, params.name, params.arguments, call_id)

        return Server(SERVER_NAME, instructions=INSTRUCTIONS, on_list_tools=on_list_tools, on_call_tool=on_call_tool)

    def hook_endpoint(self) -> HookEndpoint:
        """The decision endpoint for this server's session (unstarted): `<session dir>/hook.json` names it."""
        return HookEndpoint(lambda name, args, call_id: self.decide(None, name, args, call_id),
                            session_dir(self.home, self.sid) / HOOK_FILE)

    async def run_stdio(self) -> None:
        from mcp.server.stdio import stdio_server

        server = self.mcp_server()
        with self.hook_endpoint():
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
    """A session's events and its chain check. A keyed log is verified with the session key; an unkeyed one (written
    before session logs were keyed) is verified unkeyed and said to be unkeyed."""
    home = Path(home) if home is not None else bank_home()
    path = session_dir(home, sid) / "gates.sqlite"
    if not path.exists():
        raise SystemExit(f"no gate log for session {sid!r} at {path}")
    unkeyed = recorded_scheme(path) == "sha256"
    events = EventLog(path, key=None if unkeyed else session_key(home))
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
        n, head = events.head()
        lines.append(f"verify(): OK, {events.verify()} events, hash chain intact ({events.scheme})")
        if unkeyed:
            lines.append("unkeyed: anyone who can write this file can recompute its chain. Session logs are keyed "
                         "since 2026-11-03, so an unkeyed log is either older or was replaced.")
        lines.append(f"head: event {n}, {head}  (keep this line off this machine; verify against it to detect truncation)")
        return "\n".join(lines)
    finally:
        events.close()


async def _selftest_round_trip(env: dict, decision_tools: bool = False) -> dict:
    from mcp.client.client import Client
    from mcp.client.stdio import StdioServerParameters

    flags = ["--decision-tools"] if decision_tools else []
    params = StdioServerParameters(command=sys.executable, args=["-X", "utf8", "-m", "glint.bank.server", *flags],
                                   env=env)
    verification = next(a.arguments for a in tasks()[SELFTEST_TASK].evaluation_criteria.actions
                        if a.name == "log_verification")
    async with Client(params, read_timeout_seconds=120) as client:
        listed = await client.list_tools()
        loaded = await client.call_tool(LOAD, {"task_id": SELFTEST_TASK})
        policy = await client.call_tool(POLICY, {})
        found = await client.call_tool("get_user_information_by_name", {"customer_name": verification["name"]})
        # the demo's gate check: the customer verified, then a $4,000 increase on her $5,000 Gold card (above the
        # policy's maximum) and a $2,500 one, through the server's own gated tools
        await client.call_tool("get_credit_card_accounts_by_user", {"user_id": verification["user_id"]})
        verified = await client.call_tool("log_verification", verification)
        await client.call_tool("unlock_discoverable_agent_tool", {"agent_tool_name": SELFTEST_SUBMIT})
        submits = {}
        for amount in (4000, 2500):
            submits[amount] = await client.call_tool("call_discoverable_agent_tool", {
                "agent_tool_name": SELFTEST_SUBMIT,
                "arguments": json.dumps(SELFTEST_ACCOUNT | {"requested_increase_amount": amount})})
    rule = re.search(r"^Rule: (\S+) \(", submits[4000].content[0].text, re.MULTILINE)
    return {"tools": [t.name for t in listed.tools], "load_error": loaded.is_error, "policy_error": policy.is_error,
            "policy_chars": len(policy.content[0].text), "lookup_error": found.is_error,
            "verification_error": verified.is_error, "deny_4000": submits[4000].is_error,
            "deny_4000_rule": rule.group(1) if rule else None, "allow_2500": not submits[2500].is_error}


SELFTEST_TASK, SELFTEST_SUBMIT = "task_054", "submit_credit_limit_increase_request_7392"
SELFTEST_ACCOUNT = {"user_id": "584f9c5d00", "credit_card_account_id": "cc_584f9c5d00_gold"}
SELFTEST_RULE = "banking.cli.submit.reduce_amount"  # the rule that denies an increase above the policy's maximum


def selftest(decision_tools: bool = False) -> int:
    """Warm up (imports, the tasks, an environment), then start this server as a stdio subprocess (with
    --decision-tools when given) and list its tools, load task_054, read the policy and run one read. Then the gate
    check the demo shows: the customer verified, a $4,000 credit-limit increase must be denied by SELFTEST_RULE and a
    $2,500 one allowed. With the flag the listing must hold the decision tools. Its gate log goes to a temporary
    folder."""
    n_tasks = len(tasks())
    build_environment(tasks()["task_054"])
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        env = dict(os.environ) | {"GLINT_BANK_HOME": tmp, "MUSE_SESSION_ID": "selftest"}
        report = anyio.run(_selftest_round_trip, env, decision_tools)
    expected = {LOAD, POLICY} | (DECISION_TOOLS if decision_tools else set())
    ok = (n_tasks == 97 and len(report["tools"]) >= 18 and expected <= set(report["tools"])
          and not (report["load_error"] or report["policy_error"] or report["lookup_error"]
                   or report["verification_error"])
          and report["deny_4000"] and report["deny_4000_rule"] == SELFTEST_RULE and report["allow_2500"])
    print(json.dumps({"ok": ok, "tasks": n_tasks} | report), file=sys.stderr)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="glint-bank", description=__doc__.split("\n\n")[0])
    parser.add_argument("command", nargs="?", choices=("serve", "brief", "log"), default="serve")
    parser.add_argument("target", nargs="?", help="brief: a task id; log: a session id")
    parser.add_argument("--decision-tools", action="store_true", help="also publish Glint's banking decision tools")
    parser.add_argument("--no-request-gate", action="store_true",
                        help="turn the request gate off: no request gates, record_requests not published")
    parser.add_argument("--no-uncovered-reminder", action="store_true",
                        help="turn the [review] open item after a no_rule_yet write off")
    parser.add_argument("--no-stages", action="store_true", help="turn the stages around the gates off")
    parser.add_argument("--selftest", action="store_true", help="warm up and run a stdio round trip, then exit")
    args = parser.parse_args(argv)
    warnings_only()
    if args.selftest:
        return selftest(args.decision_tools)
    if args.command in ("brief", "log"):
        if not args.target:
            parser.error(f"{args.command} needs a {'task id' if args.command == 'brief' else 'session id'}")
        print(brief(args.target) if args.command == "brief" else show_log(args.target))
        return 0
    anyio.run(BankServer(decision_tools=args.decision_tools, request_gate=not args.no_request_gate,
                         uncovered_reminder=not args.no_uncovered_reminder, stages=not args.no_stages).run_stdio)
    return 0


if __name__ == "__main__":
    sys.exit(main())
