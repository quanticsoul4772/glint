"""glint-bank's Muse Code PreToolUse hook (plugins/glint-bank/hooks/gate_check.py) and the gate decision endpoint it
asks (glint.bank.hook_endpoint, BankServer.decide, Enforcer.precheck). The hook inputs are the payload shapes Muse
Code 1.4.14 sent a PreToolUse hook in the live probes of 2026-10-28 (research: muse-code-plugin-and-hackathon-
findings-2026-10-28, `muse-check/hooklog/pretooluse.jsonl`). No Muse and no model: the hook runs in-process, and once
as a subprocess, against a real bank server's endpoint or none."""
import importlib.util
import io
import json
import os
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from loguru import logger
from tau2.environment.toolkit import ToolType

from glint.bank import hook_endpoint
from glint.bank import server as bank
from glint.gates import rules
from glint.gates.enforce import REPEAT_NOTE

logger.remove()
ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "glint-bank"
HOOK = PLUGIN / "hooks" / "gate_check.py"
SID = "01a19364-01c9-75a1-968f-751e805a0b1d"
PREFIX = "mcp__plugin_glint_bank_b__"
SOFIA = {"user_id": "584f9c5d00", "credit_card_account_id": "cc_584f9c5d00_gold"}


@pytest.fixture
def hook(monkeypatch, tmp_path):
    monkeypatch.setenv("GLINT_BANK_HOME", str(tmp_path / "bank-home"))
    monkeypatch.setenv("MUSE_SESSION_ID", SID)
    spec = importlib.util.spec_from_file_location("glint_bank_gate_check", HOOK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def served(hook, tmp_path):
    """A bank server for session SID with its hook endpoint running, as `run_stdio` runs it."""
    srv = bank.BankServer(home=tmp_path / "bank-home", decision_tools=True, sid=SID)
    with srv.hook_endpoint() as endpoint:
        yield srv, endpoint


def payload(tool: str, tool_input: dict, n: int = 1, session_id: str = SID) -> dict:
    """A PreToolUse hook input as Muse Code sends it (every key the 2026-10-28 probe recorded)."""
    return {"hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": tool_input,
            "tool_use_id": f"call_01a193642e3d7234af2f825eddcaf3{n:02d}", "session_id": session_id,
            "turn_id": "01a19364-06e5-7092-8efe-3ac67d36a12b", "cwd": "D:\\Projects\\meta\\muse-check",
            "transcript_path": None, "model": "muse-spark-1.3-contributor", "permission_mode": "default",
            "model_provider": "meta"}


def run_hook(hook, data) -> str:
    """The hook's stdout for one input (bytes, or an object sent as UTF-8 JSON)."""
    raw = data if isinstance(data, bytes) else json.dumps(data).encode("utf-8")
    out = io.StringIO()
    assert hook.main(io.BytesIO(raw), out) == 0
    return out.getvalue()


def denial(hook, data) -> str:
    out = json.loads(run_hook(hook, data))
    assert set(out) == {"hookSpecificOutput"}
    spec = out["hookSpecificOutput"]
    assert spec["hookEventName"] == "PreToolUse" and spec["permissionDecision"] == "deny"
    assert set(spec) == {"hookEventName", "permissionDecision", "permissionDecisionReason"}
    assert spec["permissionDecisionReason"]  # Muse rejects a deny without a reason
    return spec["permissionDecisionReason"]


def _verification(task_id: str = "task_054") -> dict:
    return next(a.arguments for a in bank.tasks()[task_id].evaluation_criteria.actions if a.name == "log_verification")


def _lookups() -> list[tuple[str, dict]]:
    v = _verification()
    return [("get_user_information_by_name", {"customer_name": v["name"]}),
            ("get_credit_card_accounts_by_user", {"user_id": v["user_id"]})]


def _hidden(tool: str, **args) -> dict:
    return {"agent_tool_name": tool, "arguments": json.dumps(args)}


# ---- tools the hook lets through without asking ---------------------------------------------------------------------

@pytest.mark.parametrize("tool, tool_input", [
    ("submit_reminder_decision", {"decision": "none", "reason": "no reminder condition met", "skill_id": "x"}),
    ("shell", {"command": "ls"}),
    ("mcp__glint__log_verification", {"user_id": "584f9c5d00"}),  # another server's tool of the same name
    (PREFIX + "load_scenario", {"task_id": "task_054"}),
    (PREFIX + "get_bank_policy", {}),
    (PREFIX + "KB_search", {"query": "credit limit increase"}),
    (PREFIX + "get_user_information_by_name", {"customer_name": "Sofia Papadopoulos"}),
    (PREFIX + "plan_disputes", {"user_id": "584f9c5d00"}),
])
def test_tools_that_are_not_glint_bank_writes_pass_without_asking_the_gates(hook, monkeypatch, tool, tool_input):
    monkeypatch.setattr(hook, "ask_gates", lambda *a: pytest.fail("the gates were asked"))
    assert run_hook(hook, payload(tool, tool_input)) == ""  # no output: no decision, the call goes ahead


def test_the_hooks_write_tools_are_the_servers_gated_tools(hook, tmp_path):
    """WRITE_TOOLS is every published tool the enforcer gates (τ²'s WRITE tools and rules.GATED_GENERIC), with and
    without the decision tools; every other published tool is let through."""
    for decision_tools in (False, True):
        s = bank.BankServer(home=tmp_path / f"h{decision_tools}", decision_tools=decision_tools, sid="w").session()
        tk = s.env.tools
        gated = {n for n in s.published
                 if tk.tool_type(n) == ToolType.WRITE or n in rules.GATED_GENERIC["banking_knowledge"]}
        assert hook.WRITE_TOOLS == gated
        s.events.close()


def test_the_manifest_declares_the_hook_and_its_file_exists(hook):
    manifest = json.loads((PLUGIN / ".muse-plugin" / "plugin.json").read_text(encoding="utf-8"))
    (entry,) = manifest["capabilities"]["hooks"]
    assert entry["event"] == "PreToolUse" and entry["command"] == ["uv", "run", "--script", "hooks/gate_check.py"]
    assert (PLUGIN / entry["command"][-1]).is_file()
    assert entry["timeoutMs"] > hook.TIMEOUT_S * 1000  # the hook answers (fail closed) before Muse gives up on it
    # the reminders name the same write tools, under the same prefix
    assert {t.removeprefix(hook.PREFIX) for t in manifest["capabilities"]["reminders"][0]["tools"]} <= hook.WRITE_TOOLS
    assert '# dependencies = []\n' in HOOK.read_text(encoding="utf-8")  # uv starts it without installing anything


# ---- fail closed ----------------------------------------------------------------------------------------------------

def test_a_write_with_no_gate_endpoint_is_blocked(hook, tmp_path):
    reason = denial(hook, payload(PREFIX + "log_verification", _verification()))
    assert reason.startswith("Error: this write was blocked and was NOT executed.")
    assert "hook.json does not exist" in reason and SID in reason


def test_a_write_is_blocked_when_the_endpoint_is_unreachable(hook, tmp_path):
    with socket.socket() as s:  # a port nothing listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    where = tmp_path / "bank-home" / "sessions" / SID / "hook.json"
    where.parent.mkdir(parents=True)
    where.write_text(json.dumps({"port": port, "token": "t", "pid": 1}), encoding="utf-8")
    assert "could not be reached or read" in denial(hook, payload(PREFIX + "change_user_email", {"user_id": "u"}))
    where.write_text("{not json", encoding="utf-8")
    assert "hook.json is unreadable: JSONDecodeError" in denial(hook, payload(PREFIX + "change_user_email", {}))
    where.write_text(json.dumps({"token": "t"}), encoding="utf-8")
    assert "unreadable: KeyError" in denial(hook, payload(PREFIX + "change_user_email", {}))


def test_a_write_is_blocked_on_a_wrong_token_or_a_failed_decision(hook, served, tmp_path):
    _, endpoint = served
    where = endpoint.path
    good = json.loads(where.read_text(encoding="utf-8"))
    where.write_text(json.dumps(good | {"token": "stolen"}), encoding="utf-8")
    assert "answered HTTP 403" in denial(hook, payload(PREFIX + "log_verification", _verification()))
    where.write_text(json.dumps(good), encoding="utf-8")
    endpoint.decide = lambda *a: 1 / 0
    assert "answered HTTP 500" in denial(hook, payload(PREFIX + "log_verification", _verification()))
    assert "ZeroDivisionError" in denial(hook, payload(PREFIX + "log_verification", _verification()))


class _Fixed(BaseHTTPRequestHandler):
    body = b""

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *a):
        return


@pytest.mark.parametrize("body, why", [
    (b"<html>", "could not be reached or read: JSONDecodeError"),
    (b"[1, 2]", "answer is not an object"),
    (b'{"decision": "maybe"}', "answer has no decision"),
    (b'{"decision": "deny", "reason": ""}', "answer has no decision"),  # a deny needs its reason
])
def test_a_write_is_blocked_on_an_answer_without_a_decision(hook, tmp_path, body, why):
    handler = type("H", (_Fixed,), {"body": body})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        where = tmp_path / "bank-home" / "sessions" / SID / "hook.json"
        where.parent.mkdir(parents=True)
        where.write_text(json.dumps({"port": httpd.server_address[1], "token": "t"}), encoding="utf-8")
        assert why in denial(hook, payload(PREFIX + "log_verification", _verification()))
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.mark.parametrize("raw, why", [
    (b"{not json", "the hook input is not a JSON object"),
    (b"\xff\xfe", "the hook input is not a JSON object"),
    (b'["PreToolUse"]', "the hook input is not a JSON object"),
    (json.dumps(payload(PREFIX + "log_verification", None)).encode(), "tool_input of " + PREFIX + "log_verification"),
])
def test_unreadable_hook_input_is_blocked(hook, raw, why):
    assert why in denial(hook, raw)


def test_a_utf8_bom_is_read(hook, monkeypatch):
    monkeypatch.setattr(hook, "ask_gates", lambda *a: pytest.fail("the gates were asked"))
    assert run_hook(hook, b"\xef\xbb\xbf" + json.dumps(payload("shell", {"command": "ls"})).encode()) == ""


def test_the_session_is_muse_session_id_else_the_payloads(hook, monkeypatch):
    assert hook.session_id({"session_id": "other"}) == SID
    monkeypatch.delenv("MUSE_SESSION_ID")
    assert hook.session_id({"session_id": "other"}) == "other"
    monkeypatch.delenv("GLINT_BANK_HOME")
    monkeypatch.setenv("HOME", "/somewhere")
    assert hook.bank_home() == Path("/somewhere") / ".glint-bank"


# ---- against a real bank server ---------------------------------------------------------------------------------------

def test_the_endpoint_file_names_a_local_port_and_a_token_and_is_removed_on_stop(tmp_path):
    srv = bank.BankServer(home=tmp_path, sid="s/1")
    endpoint = srv.hook_endpoint()
    assert endpoint.path == bank.session_dir(tmp_path, "s/1") / "hook.json" and not endpoint.path.exists()
    with endpoint:
        where = json.loads(endpoint.path.read_text(encoding="utf-8"))
        assert where["port"] == endpoint.port and where["token"] == endpoint.token and len(where["token"]) >= 32
    assert not endpoint.path.exists()
    other = srv.hook_endpoint().start()  # a newer server's file is left alone
    endpoint.path.write_text(json.dumps({"port": 1, "token": "newer"}), encoding="utf-8")
    other.stop()
    assert json.loads(endpoint.path.read_text(encoding="utf-8"))["token"] == "newer"
    third = srv.hook_endpoint().start()
    third.path.unlink()
    third.stop()  # a file already gone is no error


def _post(port: int, body: bytes, route: str = "/decide") -> tuple[int, dict]:
    request = urllib.request.Request(f"http://127.0.0.1:{port}{route}", data=body, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_the_endpoint_refuses_malformed_requests(tmp_path):
    asked = []
    endpoint = hook_endpoint.HookEndpoint(lambda *a: asked.append(a), tmp_path / "hook.json")
    with endpoint:
        ok = {"token": endpoint.token, "tool": "log_verification", "arguments": {}}
        assert _post(endpoint.port, json.dumps(ok).encode(), "/other")[0] == 404
        assert _post(endpoint.port, b"{not json")[0] == 400
        assert _post(endpoint.port, b"x" * (hook_endpoint.MAX_BODY + 1))[0] == 400
        assert _post(endpoint.port, json.dumps(ok | {"tool": 3}).encode())[0] == 400
        assert _post(endpoint.port, json.dumps(ok | {"arguments": "{}"}).encode())[0] == 400
        assert _post(endpoint.port, json.dumps(ok | {"token": "x"}).encode())[0] == 403
        assert asked == []
        assert _post(endpoint.port, json.dumps(ok | {"call_id": "c1"}).encode()) == (200, {"decision": "allow"})
        assert asked == [("log_verification", {}, "c1")]


def test_a_denied_write_is_blocked_with_the_servers_refusal_and_leaves_the_servers_records(hook, served, tmp_path):
    """Before verification a credit limit increase is refused (procedure.verified). The hook's refusal is the text
    the server itself returns for that call on a twin server; the server keeps the refusal, so the same call sent to
    the server afterwards gets the repeat note; the database is unchanged; the gate log has the deny under Muse's
    call id and a hook.denied event."""
    srv, _ = served
    srv.call(None, bank.LOAD, {"task_id": "task_054"})
    srv.call(None, "unlock_discoverable_agent_tool", {"agent_tool_name": rules.SUBMIT})
    twin = bank.BankServer(home=tmp_path / "twin", decision_tools=True, sid="twin")
    twin.call(None, bank.LOAD, {"task_id": "task_054"})
    twin.call(None, "unlock_discoverable_agent_tool", {"agent_tool_name": rules.SUBMIT})
    submit = _hidden(rules.SUBMIT, requested_increase_amount=2500, **SOFIA)
    before, n_events = srv.session().env.get_db_hash(), len(srv.session().events.events())

    reason = denial(hook, payload(PREFIX + "call_discoverable_agent_tool", submit, n=7))
    expected = twin.call(None, "call_discoverable_agent_tool", submit).content[0].text
    assert reason == expected and "Rule: procedure.verified (" in reason
    assert srv.session().env.get_db_hash() == before
    new = srv.session().events.events(after=n_events)
    assert [e["type"] for e in new][-1] == "hook.denied" and "write.attested" not in {e["type"] for e in new}
    assert new[-1]["payload"] == {"tool": "call_discoverable_agent_tool", "session_id": SID, "task_id": "task_054",
                                  "call_id": "call_01a193642e3d7234af2f825eddcaf307"}
    denies = [e["payload"] for e in new if e["type"] == "gate.fired" and e["payload"]["decision"] == "deny"]
    assert [(d["rule_id"], d["call_id"]) for d in denies] == [
        ("procedure.verified", "call_01a193642e3d7234af2f825eddcaf307")]
    again = srv.call(None, "call_discoverable_agent_tool", submit).content[0].text
    assert again == expected + REPEAT_NOTE  # the server saw the hook's refusal
    assert twin.call(None, "call_discoverable_agent_tool", submit).content[0].text == expected + REPEAT_NOTE
    twin.session().events.close()


def test_an_allowed_write_passes_and_logs_nothing_until_it_runs(hook, served):
    srv, _ = served
    srv.call(None, bank.LOAD, {"task_id": "task_054"})
    for name, args in _lookups():
        assert run_hook(hook, payload(PREFIX + name, args)) == ""
        srv.call(None, name, args)
    n_events = len(srv.session().events.events())
    assert run_hook(hook, payload(PREFIX + "log_verification", _verification())) == ""
    assert len(srv.session().events.events()) == n_events  # an allow logs nothing: the real call does
    reply = srv.call(None, "log_verification", _verification())
    assert not reply.is_error
    assert len(srv.session().events.events("write.attested", after=n_events)) == 1  # attested once, when it ran
    # the verification the server recorded is what the hook's next decision sees: within the maximum, the submit passes
    srv.call(None, "unlock_discoverable_agent_tool", {"agent_tool_name": rules.SUBMIT})
    assert run_hook(hook, payload(PREFIX + "call_discoverable_agent_tool", _hidden(
        rules.SUBMIT, requested_increase_amount=2500, **SOFIA))) == ""
    over = denial(hook, payload(PREFIX + "call_discoverable_agent_tool", _hidden(
        rules.SUBMIT, requested_increase_amount=4000, **SOFIA)))
    assert "Rule: banking.cli.submit.reduce_amount (" in over


def test_decide_lets_through_what_the_gates_do_not_judge(served):
    srv, _ = served
    assert srv.decide(None, bank.LOAD, {"task_id": "task_054"}) is None
    assert srv.decide(None, "no_such_tool", {}) is None  # the server's call answers it (a protocol error)
    assert srv.decide(None, "get_user_information_by_name", {"customer_name": "Sofia Papadopoulos"}) is None
    assert srv.decide(None, "call_discoverable_agent_tool", _hidden("no_such_hidden_tool")) is None


def test_the_hook_runs_as_a_script_and_fails_closed_without_a_server(tmp_path):
    """Under the platform's own stdin encoding (no UTF-8 mode), with non-ASCII arguments sent as raw UTF-8."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUTF8", "PYTHONIOENCODING")} | {
        "GLINT_BANK_HOME": str(tmp_path), "MUSE_SESSION_ID": "s"}
    data = json.dumps(payload(PREFIX + "log_verification", {"name": "Zoë Ångström"}), ensure_ascii=False)
    done = subprocess.run([sys.executable, str(HOOK)], input=data.encode("utf-8"), capture_output=True, env=env,
                          timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    reason = json.loads(done.stdout)["hookSpecificOutput"]["permissionDecisionReason"]
    assert "hook.json does not exist" in reason  # it read the input and looked for the endpoint
    done = subprocess.run([sys.executable, str(HOOK)], input=json.dumps(payload("shell", {})).encode(),
                          capture_output=True, env=env, timeout=60, check=False)
    assert done.returncode == 0 and done.stdout == b""
