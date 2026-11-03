"""glint.bank.server: the τ² banking environment over MCP with every write gated, its replays (design V1, V2), and the
Muse plugin bundle. Every server here keeps its gate logs in a test folder (GLINT_BANK_HOME is never the real one);
no model, no network. The in-process MCP client goes through the server's own list and call handlers."""
import json
import os
import subprocess
import sys
from pathlib import Path

import anyio
import pytest
from loguru import logger
from mcp.client.client import Client
from mcp.shared.exceptions import MCPError
from tau2.environment.toolkit import ToolType

from glint import paths
from glint.bank import server as bank
from glint.gates import rules

logger.remove()
ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "glint-bank"
RUNS = ROOT / "data-public" / "wrong-write-runs.json"  # bank_replay.RECS: the recorded wrong-write runs
SOFIA = {"user_id": "584f9c5d00", "credit_card_account_id": "cc_584f9c5d00_gold"}


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("GLINT_BANK_HOME", str(tmp_path / "bank-home"))
    return bank.BankServer(sid="s1")


def _run(fn, *args):
    return anyio.run(fn, *args)


async def _with_client(srv, body):
    async with Client(srv.mcp_server()) as client:
        return await body(client)


def _verification(task_id: str = "task_054") -> dict:
    """The task's gold log_verification arguments: the customer verified today (gates phase 2, procedure.verified:
    every banking write for a user needs that user's verification logged today)."""
    return next(a.arguments for a in bank.tasks()[task_id].evaluation_criteria.actions if a.name == "log_verification")


def _verify(server, sid=None):
    reply = server.call(sid, "log_verification", _verification())
    assert not reply.is_error, reply.content[0].text


def _hidden_args(tool: str, **args) -> dict:
    return {"agent_tool_name": tool, "arguments": json.dumps(args)}


# ---- tool list ----------------------------------------------------------------------------------------------------

def test_tool_list_is_the_toolkits_tools_with_their_schemas_and_annotations(server):
    async def body(client):
        return (await client.list_tools()).tools

    listed = {t.name: t for t in _run(_with_client, server, body)}
    env = bank.build_environment()
    tools = env.tools.get_tools()
    assert set(listed) == set(tools) | {bank.LOAD, bank.POLICY, bank.PLAN_TOOL, bank.RETENTION_TOOL, bank.REQUESTS_TOOL,
                                          bank.DISPUTE_ANSWERS_TOOL} and len(tools) == 16
    for name, tool in tools.items():
        schema = tool.openai_schema["function"]
        assert listed[name].input_schema == schema["parameters"] and listed[name].description == schema["description"]
        assert listed[name].annotations.read_only_hint is (not env.tools.tool_mutates_state(name))
        assert listed[name].annotations.destructive_hint is (env.tools.tool_type(name) == ToolType.WRITE)
    # plan_disputes is always published (the plan gate needs it) and never changes the bank
    plan = listed[bank.PLAN_TOOL]
    assert plan.annotations.read_only_hint and not plan.annotations.destructive_hint
    assert listed["KB_search"].annotations.read_only_hint and not listed["KB_search"].annotations.destructive_hint
    assert listed["call_discoverable_agent_tool"].annotations.destructive_hint
    assert not listed["unlock_discoverable_agent_tool"].annotations.read_only_hint  # mutates, but not a WRITE
    assert not listed["unlock_discoverable_agent_tool"].annotations.destructive_hint
    assert listed[bank.POLICY].annotations.read_only_hint and not listed[bank.LOAD].annotations.read_only_hint


def test_decision_tools_are_published_only_with_the_flag(tmp_path):
    off = {t.name for t in bank.BankServer(tmp_path, sid="a").tool_list()}
    listed = {t.name: t for t in bank.BankServer(tmp_path, decision_tools=True, sid="b").tool_list()}
    on = set(listed)
    assert on - off == bank.DECISION_TOOLS == {"check_credit_limit_increase", "check_card_dispute",
                                               "check_verification", "KB_read"}
    # GENERIC with mutates_state=True only so τ²'s replay rebuilds the verdict record: it never writes the bank
    check = listed["check_credit_limit_increase"].annotations
    assert check.read_only_hint and not check.destructive_hint
    with pytest.raises(MCPError, match="unknown tool"):
        bank.BankServer(tmp_path, sid="c").call(None, "check_credit_limit_increase", SOFIA)


def test_every_tool_name_fits_64_characters_under_the_plugin_prefix(server):
    prefix = "mcp__plugin_glint_bank_b__"  # mcp__plugin_<plugin id, - as _>_<server id>__ (hook-name probe 2026-10-28)
    names = [t.name for t in bank.BankServer(server.home, decision_tools=True, sid="x").tool_list()]
    assert max(len(prefix + n) for n in names) <= 64


# ---- the gated call path ------------------------------------------------------------------------------------------

def test_a_denied_write_is_an_error_result_with_the_gate_text_and_leaves_the_database_unchanged(server):
    async def body(client):
        await client.call_tool(bank.LOAD, {"task_id": "task_054"})
        await client.call_tool("unlock_discoverable_agent_tool", {"agent_tool_name": rules.SUBMIT})
        unverified = await client.call_tool("call_discoverable_agent_tool", _hidden_args(
            rules.SUBMIT, requested_increase_amount=2500, **SOFIA))
        assert not (await client.call_tool("log_verification", _verification())).is_error
        before = server.session().env.get_db_hash()
        denied = await client.call_tool("call_discoverable_agent_tool", _hidden_args(
            rules.SUBMIT, requested_increase_amount=4000, **SOFIA), meta={"callId": "call-7"})
        after_deny = server.session().env.get_db_hash()
        allowed = await client.call_tool("call_discoverable_agent_tool", _hidden_args(
            rules.SUBMIT, requested_increase_amount=2500, **SOFIA))
        return unverified, before, denied, after_deny, allowed

    unverified, before, denied, after_deny, allowed = _run(_with_client, server, body)
    # the hidden tool was unwrapped and gated; only the submit gate's own events are looked at here
    fired = [e["payload"] for e in server.session().events.events("gate.fired")
             if e["payload"]["tool"] == rules.SUBMIT and e["payload"]["gate"] == "banking_cli_submit_gate"]
    # before any verification even a submit within the maximum is blocked (gates phase 2: procedure.verified)
    assert unverified.is_error and "Rule: procedure.verified (" in unverified.content[0].text
    assert denied.is_error and after_deny == before
    text = denied.content[0].text
    assert text.startswith("Error: this write was blocked by a policy gate and was NOT executed. Nothing changed.\n"
                           "Rule: banking.cli.submit.reduce_amount (")
    for part in ("\n- requested increase $4,000.00 exceeds", "\nWhy it matters: Nothing was submitted. Tell the customer the largest increase",
                 "\nWhat to do next: Submit an increase no larger than max_increase", "\nVerdict: {"):
        assert part in text
    # the log carries the session, the task and Muse's call id
    fired = fired[1:]  # the submit gate's allow of the unverified call, which procedure.verified then blocked
    assert fired[0]["decision"] == "deny" and fired[0]["rule_id"] == "banking.cli.submit.reduce_amount"
    assert {k: fired[0][k] for k in ("session_id", "task_id", "call_id")} == {
        "session_id": "s1", "task_id": "task_054", "call_id": "call-7"}
    assert not allowed.is_error and server.session().env.get_db_hash() != before
    assert fired[1]["decision"] == "allow" and fired[1]["call_id"] != "call-7"


def test_a_tau2_tool_error_is_an_error_result_and_an_unknown_tool_a_protocol_error(server):
    async def body(client):
        bad_args = await client.call_tool("get_user_information_by_id", {"wrong": "x"})
        with pytest.raises(MCPError, match="unknown tool"):
            await client.call_tool(rules.SUBMIT, {"requested_increase_amount": 1})  # hidden: only via the wrapper
        return bad_args

    bad_args = _run(_with_client, server, body)
    assert bad_args.is_error and bad_args.content[0].text.startswith("Error:")


def test_reads_are_not_gated_and_tools_work_before_any_scenario(server):
    reply = server.call(None, "get_user_information_by_name", {"customer_name": "Sofia Papadopoulos"})
    assert not reply.is_error and "584f9c5d00" in reply.content[0].text
    assert server.session().events.events("gate.fired") == [] and server.session().task_id == ""


def test_sessions_are_isolated(server):
    server.load_scenario("a", "task_054")
    server.load_scenario("b", "task_054")
    for sid in ("a", "b"):
        _verify(server, sid)
        server.call(sid, "unlock_discoverable_agent_tool", {"agent_tool_name": rules.SUBMIT})
    b_fired = len(server.session("b").events.events("gate.fired"))
    b_before = server.session("b").env.get_db_hash()
    reply = server.call("a", "call_discoverable_agent_tool", _hidden_args(rules.SUBMIT, requested_increase_amount=2500,
                                                                          **SOFIA))
    assert not reply.is_error and server.session("a").env.get_db_hash() != b_before
    assert server.session("b").env.get_db_hash() == b_before
    assert server.session("a").env is not server.session("b").env
    a_log = server.home / "sessions" / "a" / "gates.sqlite"
    assert a_log.exists() and (server.home / "sessions" / "b" / "gates.sqlite").exists()
    a_submits = [e for e in server.session("a").events.events("gate.fired") if e["payload"]["tool"] == rules.SUBMIT]
    assert a_submits and all(e["payload"]["decision"] == "allow" for e in a_submits)
    assert len(server.session("b").events.events("gate.fired")) == b_fired  # a's submit left no event in b's log


def test_load_scenario_resets_and_never_leaks_the_customer_instructions(server):
    server.load_scenario(None, "task_054")
    first = server.session().env
    _verify(server)
    server.call(None, "unlock_discoverable_agent_tool", {"agent_tool_name": rules.SUBMIT})
    submitted = server.call(None, "call_discoverable_agent_tool", _hidden_args(
        rules.SUBMIT, requested_increase_amount=2500, **SOFIA))
    assert not submitted.is_error, submitted.content[0].text
    changed = first.get_db_hash()
    reply = server.call(None, bank.LOAD, {"task_id": "task_054"})
    fresh = bank.build_environment(bank.tasks()["task_054"])
    assert server.session().env is not first and server.session().env.get_db_hash() == fresh.get_db_hash() != changed
    text = reply.content[0].text
    assert not reply.is_error and json.loads(text)["task_id"] == "task_054"
    task = bank.tasks()["task_054"]
    assert "Sofia" not in text and "4,000" not in text and task.user_scenario.instructions[:40] not in text
    # the base database, and an unknown task
    base = server.call(None, bank.LOAD, {})
    assert not base.is_error and server.session().env.get_db_hash() == bank.build_environment().get_db_hash()
    unknown = server.call(None, bank.LOAD, {"task_id": "task_999"})
    assert unknown.is_error and "unknown task_id" in unknown.content[0].text
    loaded = [e["payload"]["task_id"] for e in server.session().events.events("scenario.loaded")]
    assert loaded == ["task_054", "task_054", ""] and server.session().events.verify() > 3


def test_a_session_log_opened_with_another_key_setting_names_the_fix(server, monkeypatch):
    from glint.gates.eventlog import KEY_ENV
    monkeypatch.setenv(KEY_ENV, "operator key")
    server.load_scenario(None, "task_054")
    server.session().events.close()
    monkeypatch.delenv(KEY_ENV)
    restarted = bank.BankServer(sid="s1")  # the same session folder, now without the key
    with pytest.raises(MCPError, match=f"keyed log.*Restart the server with {KEY_ENV}"):
        restarted.call(None, bank.POLICY, {})
    monkeypatch.setenv(KEY_ENV, "operator key")
    assert not bank.BankServer(sid="s1").call(None, bank.POLICY, {}).is_error


def test_get_bank_policy_is_the_environment_policy(server):
    reply = server.call(None, bank.POLICY, {})
    assert not reply.is_error and reply.content[0].text == bank.build_environment().get_policy()
    assert len(reply.content[0].text) > 5000


def test_tau2_prints_go_to_stderr(server, monkeypatch, capsys):
    """τ²'s get_tasks reports a task file that fails to load with print(); on stdio that corrupts the stream."""
    real = bank.get_environment

    def noisy(*args, **kwargs):
        print("τ² warning on stdout")
        return real(*args, **kwargs)

    monkeypatch.setattr(bank, "get_environment", noisy)
    server.load_scenario(None, "task_054")
    out = capsys.readouterr()
    assert out.out == "" and "τ² warning on stdout" in out.err


def test_session_ids_and_home(monkeypatch, tmp_path):
    monkeypatch.setenv("MUSE_SESSION_ID", "01a1-muse")
    monkeypatch.setenv("GLINT_BANK_HOME", str(tmp_path))
    assert bank.default_session_id() == "01a1-muse" and bank.bank_home() == tmp_path
    assert bank.BankServer().sid == "01a1-muse" and bank.BankServer().home == tmp_path
    monkeypatch.delenv("MUSE_SESSION_ID")
    monkeypatch.delenv("GLINT_BANK_HOME")
    assert bank.default_session_id() == f"local-{os.getpid()}"
    monkeypatch.delenv("HOME", raising=False)
    assert bank.bank_home() == Path.home() / ".glint-bank"
    # Muse's cleared server environment (2026-10-28): HOME but no USERPROFILE, where Path.home() raises on Windows
    for var in ("USERPROFILE", "HOMEDRIVE", "HOMEPATH"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    assert bank.bank_home() == tmp_path / "h" / ".glint-bank"
    assert bank.session_dir(tmp_path, "../x y") == tmp_path / "sessions" / ".._x_y"


# ---- stdio and the CLIs -------------------------------------------------------------------------------------------

def test_selftest_round_trips_over_stdio_with_a_clean_stdout(capsys):
    """--selftest starts the server as a stdio subprocess and talks JSON-RPC to it: any stray stdout breaks it."""
    assert bank.main(["--selftest"]) == 0
    report = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert report["ok"] and report["tasks"] == 97 and len(report["tools"]) == 22  # 16 τ², 4 Glint records, 2 server


def test_selftest_with_the_decision_tools_lists_them(capsys):
    """The plugin's server command passes --decision-tools; its self-test starts the server the same way."""
    assert bank.main(["--decision-tools", "--selftest"]) == 0
    report = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert report["ok"] and bank.DECISION_TOOLS <= set(report["tools"]) and len(report["tools"]) == 26


def test_selftest_with_the_decision_tools_fails_when_they_are_missing(monkeypatch, capsys):
    async def gates_only(env, decision_tools):
        return {"tools": [f"t{i}" for i in range(17)] + [bank.LOAD, bank.POLICY], "load_error": False,
                "policy_error": False, "policy_chars": 1, "lookup_error": False}

    monkeypatch.setattr(bank, "_selftest_round_trip", gates_only)
    assert bank.selftest(decision_tools=False) == 0
    assert bank.selftest(decision_tools=True) == 1


def test_every_stdout_line_of_the_stdio_server_is_json_rpc(tmp_path):
    """The raw stdout of a stdio session: initialize, tools/list, load_scenario. The SDK client skips a line it cannot
    parse, so a round trip alone does not show a clean stdout; every line here must be a JSON-RPC message."""
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": bank.LOAD,
                                                                       "arguments": {"task_id": "task_054"}}}]
    env = dict(os.environ) | {"PYTHONPATH": str(ROOT / "src"), "GLINT_BANK_HOME": str(tmp_path),
                              "MUSE_SESSION_ID": "raw"}
    proc = subprocess.Popen([sys.executable, "-X", "utf8", "-m", "glint.bank.server"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, text=True, encoding="utf-8")
    try:
        lines = []
        for request in requests:
            proc.stdin.write(json.dumps(request) + "\n")
            proc.stdin.flush()
            if "id" in request:  # read up to this request's response
                while True:
                    line = proc.stdout.readline()
                    assert line, "the server closed stdout"
                    lines.append(line)
                    if line.lstrip().startswith("{") and json.loads(line).get("id") == request["id"]:
                        break
    finally:
        proc.kill()
        proc.wait(timeout=30)
    messages = [json.loads(line) for line in lines]  # a stray print would raise here
    assert all(m["jsonrpc"] == "2.0" for m in messages)
    assert len(messages[-2]["result"]["tools"]) == 22 and not messages[-1]["result"]["isError"]


def test_selftest_fails_when_the_round_trip_does(monkeypatch, capsys):
    async def broken(env, decision_tools):
        return {"tools": [], "load_error": True, "policy_error": False, "policy_chars": 0, "lookup_error": False}

    monkeypatch.setattr(bank, "_selftest_round_trip", broken)
    assert bank.selftest() == 1


def test_brief_and_log_clis(server, capsys, monkeypatch):
    assert bank.main(["brief", "task_054"]) == 0
    assert "Sofia Papadopoulos" in capsys.readouterr().out
    server.load_scenario(None, "task_054")
    _verify(server)
    server.call(None, "unlock_discoverable_agent_tool", {"agent_tool_name": rules.SUBMIT})
    server.call(None, "call_discoverable_agent_tool", _hidden_args(rules.SUBMIT, requested_increase_amount=4000,
                                                                   **SOFIA))
    server.session().events.close()
    assert bank.main(["log", "s1"]) == 0
    out = capsys.readouterr().out
    assert "scenario.loaded  task_054" in out and "banking.cli.submit.reduce_amount" in out
    assert "gates.manifest" in out and "hash chain intact (sha256)" in out and "head: event " in out
    with pytest.raises(SystemExit, match="no gate log"):
        bank.main(["log", "nobody"])
    with pytest.raises(SystemExit, match="unknown task_id"):
        bank.main(["brief", "task_999"])
    with pytest.raises(SystemExit):
        bank.main(["brief"])


def test_main_serves_stdio(monkeypatch):
    served = []

    async def run_stdio(self):
        served.append(self.decision_tools)

    monkeypatch.setattr(bank.BankServer, "run_stdio", run_stdio)
    assert bank.main(["--decision-tools"]) == 0 and served == [True]


# ---- replays (design V1, V2) --------------------------------------------------------------------------------------

def test_gold_replay_through_mcp_has_no_false_refusal_and_matches_the_ungated_twin():
    """The 77 banking TRAIN tasks (pilot + rest); the 20 test_untouched tasks stay unread (decision 2026-10-26)."""
    import bank_replay
    import gate_replay

    report = _run(bank_replay.gold)
    untouched = set(json.loads(gate_replay.BANKING_SPLIT.read_text(encoding="utf-8"))["test_untouched"])
    assert report["tasks"] == 77 and len(untouched) == 20 and not untouched & set(bank_replay.train_task_ids())
    assert report["false_refusals"] == [] and report["also_rejected"] == []
    # task_005's gold logs a placeholder verification and then changes an unverified user's email (gate_replay lists
    # it apart): its two gold writes are denied, so only its database differs from the twin's
    assert [(p["task_id"], p["rule_id"]) for p in report["placeholder_gold"]] == [
        ("task_005", "banking.verification.user"), ("task_005", "procedure.verified")]
    assert report["gold_writes"] == report["allowed"] + 2 > 250
    assert report["placeholder_hash_differs"] == ["task_005"]
    assert report["hash_mismatches"] == [] and report["result_mismatches"] == []


def test_replay_environments_record_the_hidden_reads_gold_makes_as_taus_runner_does():
    # without τ²'s read log a replay never records a hidden read, so its hash check proves nothing about reads
    import gate_replay

    task = bank.tasks()["task_039"]
    assert "get_user_dispute_history_7291" in bank.build_environment(task).tools._read_log_allowlist
    assert gate_replay._env("banking_knowledge", task).tools._read_log_allowlist == \
        bank.build_environment(task).tools._read_log_allowlist
    assert bank.build_environment().tools._read_log_allowlist == set()


def test_wrong_write_replay_denies_each_with_its_expected_rule():
    import bank_replay

    report = _run(bank_replay.wrong, json.loads(RUNS.read_text(encoding="utf-8")))
    stopped = {r["run"]: r for r in report["stopped"]}
    assert len(stopped) == report["denied_as_expected"] == 24
    assert all(r["ok"] and r["is_error"] and r["db_unchanged"] for r in stopped.values())
    # a credit card dispute needs the dispute-history read first (credit_cards_(general)_015 criterion 4): every run
    # that filed without it in its whole call list stops at that filing, task_054's base runs before their
    # over-maximum submit (that deny is tested on its own: test_a_denied_write_is_an_error_result_with_the_gate_text_...)
    lookups = sorted(run.split(maxsplit=1)[1] for run, r in stopped.items() if r["rule_id"] == "procedure.lookups"
                     and r["tool"] == rules.DISPUTE)
    assert lookups == ["base task_040 trial 0", "base task_040 trial 2", "base task_040 trial 3",
                       "base task_041 trial 0", "base task_054 trial 0", "base task_054 trial 1",
                       "base task_054 trial 2", "tools task_040 trial 0", "tools task_040 trial 1",
                       "tools task_040 trial 2", "tools task_041 trial 0", "tools task_041 trial 3"]
    assert all(stopped[f"banking12 {run}"]["skipped_in_recording"] for run in lookups)
    assert stopped["banking12 tools task_051 trial 0"]["rule_id"] == "banking.cli.approve"
    assert "first failing check: high_utilization" in stopped["banking12 tools task_051 trial 0"]["message"]
    rules_by_task = {}
    for r in stopped.values():
        rules_by_task.setdefault(r["run"].split()[2], []).append(r["rule_id"])
    # every recorded task_040 run is stopped before a wrong flag lands: at a filing with no dispute-history read,
    # by the plan made from its own disputes (replayplan), or at a filing with wrong card digits and no card-digits
    # tool given
    assert sorted(rules_by_task["task_040"]) == ["procedure.card_digits.give", "procedure.dispute_plan.order"] + [
        "procedure.lookups"] * 6
    # gates phase 2 stops runs earlier, on steps the recording shows were skipped: the approve of task_051 base
    # trial 0 came with no dispute-history or pending-order lookup in the whole run (it was not denied by the approve
    # rule: the customer paid down before submitting, so the policy approves on the live database)
    r = stopped["banking12 base task_051 trial 0"]
    assert (r["rule_id"], r["skipped_in_recording"]) == ("procedure.lookups", True)
    # base trials 1 and 2 paid the card from checking after the submit and before any decision
    for trial, write in ((1, 62), (2, 72)):
        r = stopped[f"banking12 base task_051 trial {trial}"]
        assert (r["write"], r["rule_id"], r["skipped_in_recording"]) == (write, "procedure.increase_undecided", True)
    r = stopped["banking12 base task_041 trial 1"]  # no log_verification anywhere before its disputes
    assert (r["rule_id"], r["skipped_in_recording"]) == ("procedure.verified", True)
    assert stopped["banking12 tools task_041 trial 1"]["rule_id"] == "banking.verification"  # wrong time_verified
    assert sorted(rules_by_task["task_041"]) == ["banking.dispute", "banking.verification"] + [
        "procedure.dispute_plan.order"] * 2 + ["procedure.lookups"] * 3 + ["procedure.verified"]
    # base task_051 trial 2's transfer is not reached (its payment before the decision is denied first), nor base
    # task_054 trial 2's (its filing without the history read is)
    assert sum(r.startswith("procedure.transfer.") for r in rules_by_task["task_051"] + rules_by_task["task_054"]) == 1


def test_replay_main_prints_and_exits_by_the_result(monkeypatch, capsys, tmp_path):
    import bank_replay

    real_gold = bank_replay.gold

    async def small_gold():
        return await real_gold(["task_054", "task_040"])

    runs = [r for r in json.loads(RUNS.read_text(encoding="utf-8")) if r["task"] == "task_054" and r["trial"] == 0]
    (tmp_path / "runs.json").write_text(json.dumps(runs), encoding="utf-8")
    monkeypatch.setattr(bank_replay, "gold", small_gold)
    out = tmp_path / "out.json"
    assert bank_replay.main(["--recs", str(tmp_path / "runs.json"), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "0 false refusals" in text and "banking.cli.submit.reduce_amount" in text
    assert json.loads(out.read_text(encoding="utf-8"))["gold"]["tasks"] == 2

    async def refusing_gold():
        return {"tasks": 1, "gold_writes": 1, "allowed": 0, "also_rejected": [], "hash_mismatches": [{"task_id": "t"}],
                "placeholder_gold": [{"task_id": "p", "tool": "y", "rule_id": "q", "why": "placeholder"}],
                "placeholder_hash_differs": ["p"],
                "result_mismatches": [{"task_id": "t", "tool": "x", "text": "boom"}],
                "false_refusals": [{"task_id": "t", "tool": "x", "rule_id": "r", "reason": "why"}]}

    monkeypatch.setattr(bank_replay, "gold", refusing_gold)
    assert bank_replay.main(["--recs", str(tmp_path / "runs.json")]) == 1
    assert "FALSE REFUSAL t x [r]" in capsys.readouterr().out


# ---- the Muse plugin bundle ---------------------------------------------------------------------------------------

def _lf(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def test_bundled_banking_data_is_the_vendored_tau2_data_byte_for_byte():
    bundled = PLUGIN / "data" / "tau2" / "domains" / "banking_knowledge"
    vendored = paths.tau2_data() / "tau2" / "domains" / "banking_knowledge"
    files = sorted(p.relative_to(bundled) for p in bundled.rglob("*") if p.is_file())
    assert len(files) > 700 and Path("db.json") in files
    assert {p.parent for p in files} >= {Path("."), Path("documents"), Path("tasks"), Path("prompts")}
    for rel in files:
        # the bundle holds τ²'s committed (LF) bytes; a Windows checkout of the vendored copy may have CRLF
        assert _lf(bundled / rel) == _lf(vendored / rel), rel
    assert sorted(p.name for p in (vendored / "tasks").glob("task_*.json")) == sorted(
        p.name for p in (bundled / "tasks").glob("task_*.json"))
    assert "MIT License" in (PLUGIN / "data" / "LICENSE").read_text(encoding="utf-8")


def test_manifest_names_files_that_exist_and_the_launcher_pins_its_dependencies(tmp_path):
    manifest = json.loads((PLUGIN / ".muse-plugin" / "plugin.json").read_text(encoding="utf-8"))
    caps = manifest["capabilities"]
    assert manifest["name"] == "glint-bank" and caps["mcpServers"] == [
        {"id": "b", "transport": "stdio", "command": ["uv", "run", "--script", "mcp/glint_bank.py", "--decision-tools"]}]
    for cap in caps["commands"] + caps["reminders"]:
        assert (PLUGIN / cap["path"]).is_file()
    assert {t.removeprefix("mcp__plugin_glint_bank_b__") for t in caps["reminders"][0]["tools"]} <= {
        t.name for t in bank.BankServer(tmp_path, decision_tools=True, sid="m").tool_list()}  # as the plugin starts it
    start = (PLUGIN / "commands" / "start.md").read_text(encoding="utf-8")
    assert "$ARGUMENTS" in start and bank.LOAD in start and bank.POLICY in start
    launcher = (PLUGIN / "mcp" / "glint_bank.py").read_text(encoding="utf-8")
    # stage 1 needs no packages; the server stage's uv script pins them (tests/test_launcher.py checks its metadata)
    assert '# requires-python = ">=3.12,<3.13"\n# dependencies = []\n' in launcher
    assert 'GLINT_REQUIREMENT = "glint @ git+https://github.com/quanticsoul4772/glint@' in launcher


def test_launcher_points_tau2_at_the_bundled_data_and_runs_utf8(tmp_path):
    """The launcher in a fresh interpreter without UTF-8 mode: it restarts itself in UTF-8 mode and τ² reads the
    plugin's data folder (brief prints a task τ² loaded from there). Without UTF-8 mode τ² reads task_100 as cp1252
    on Windows."""
    env = {k: v for k, v in os.environ.items() if k not in ("TAU2_DATA_DIR", "PYTHONUTF8")}
    # the server stage, as uv runs it once the dependencies are installed (stage 1 is in tests/test_launcher.py)
    env |= {"PYTHONPATH": str(ROOT / "src"), "GLINT_BANK_HOME": str(tmp_path), "GLINT_BANK_STAGE": "server"}
    # stdout in UTF-8, so text τ² misread as cp1252 shows as mojibake instead of round-tripping back to the same bytes
    probe = ("import runpy, sys; sys.stdout.reconfigure(encoding='utf-8'); "
             "sys.argv = [sys.argv[0], 'brief', 'task_100']; "
             f"runpy.run_path(r'{PLUGIN / 'mcp' / 'glint_bank.py'}', run_name='__main__')")
    done = subprocess.run([sys.executable, "-c", probe], check=False, env=env, capture_output=True, text=True,
                          encoding="utf-8", timeout=180)
    assert done.returncode == 0, done.stderr[-2000:]
    # task_100 is the one task file with non-ASCII text; read as cp1252 its dash would come out as mojibake
    assert "Tomoko Ishikawa" in done.stdout and "today — don't" in done.stdout
    assert str(PLUGIN / "data") in done.stderr  # τ² logs the data directory it uses
