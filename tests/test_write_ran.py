"""The gate log records outcomes as well as decisions (write.ran: the tool's result status and the database hash after
each allowed write, glint.gates.enforce), and glint-bank keys its session logs by default (a per-install key file,
glint.gates.eventlog.install_key) while older unkeyed logs stay readable and verifiable. Every server here keeps its
files in a test folder."""
import logging
import os
import sqlite3
from types import SimpleNamespace

import pytest
from tau2.data_model.message import ToolMessage
from test_bank import _verify, server  # noqa: F401  (the server fixture)
from test_gates import _rewrite_chain
from test_gates_phase2 import _bank, _hidden, _play

from glint.bank import server as bank
from glint.gates import ChainBroken, Enforcer, EventLog
from glint.gates.eventlog import KEY_ENV, install_key, recorded_scheme


def _ran(events):
    return [e["payload"] for e in events.events("write.ran")]


def test_an_allowed_write_logs_its_result_and_the_database_hash_after_it():
    env, task, events = _bank("task_048")
    log = _play(env, task, "log_credit_card_closure_reason_4521")
    before = env.get_db_hash()
    reply = _hidden(env, "log_credit_card_closure_reason_4521", **log)
    assert "was NOT executed" not in reply.content
    ran = _ran(events)[-1]
    assert ran["tool"] == "log_credit_card_closure_reason_4521" and ran["status"] == "ok" and "error" not in ran
    assert ran["db_hash"] == env.get_db_hash() != before and ran["replay"] is False
    # it follows the write's decision in the chain: its gates' gate.fired events, then write.attested, then write.ran
    types = [e["type"] for e in events.events()]
    assert types[-1] == "write.ran" and types[-2] == "write.attested" and types[-3] == "gate.fired"
    assert events.verify() == len(types)


def test_a_refused_write_logs_no_outcome():
    env, task, events = _bank("task_048")
    log = _play(env, task, "log_credit_card_closure_reason_4521")
    n = len(_ran(events))
    close = {"credit_card_account_id": log["credit_card_account_id"], "user_id": log["user_id"]}
    assert "was NOT executed" in _hidden(env, "close_credit_card_account_7834", **close).content
    assert len(_ran(events)) == n


def test_a_tool_error_is_logged_as_the_status_and_a_logging_failure_is_not_raised(caplog):
    events = EventLog(":memory:")
    enforcer = Enforcer("banking_knowledge", events)
    env = SimpleNamespace(get_db_hash=lambda: "hash-after")
    reply = ToolMessage(id="t", role="tool", content="Error: no such card " + "x" * 300, requestor="assistant")
    enforcer._log_ran(env, "close_debit_card_4721", reply, replay=True)
    ran = _ran(events)[-1]
    assert (ran["status"], ran["db_hash"], ran["replay"]) == ("error", "hash-after", True)
    assert ran["error"].startswith("Error: no such card") and len(ran["error"]) == 200

    def broken():
        raise RuntimeError("database gone")
    with caplog.at_level(logging.ERROR, logger="glint.gates"):
        enforcer._log_ran(SimpleNamespace(get_db_hash=broken), "close_debit_card_4721", reply, replay=False)
    assert "the write.ran event was not logged" in caplog.text and len(_ran(events)) == 1


def test_bank_sessions_are_keyed_by_default_with_a_key_file_outside_the_session(server, monkeypatch):  # noqa: F811
    monkeypatch.delenv(KEY_ENV, raising=False)
    _verify(server)
    s = server.session()
    assert s.events.scheme == "hmac-sha256" and s.events.key == bank.session_key(server.home)
    key_file = bank.key_file(server.home)
    assert key_file == server.home / "log.key" and key_file.exists()
    assert bank.session_dir(server.home, "s1") not in key_file.parents
    assert _ran(s.events)[-1]["tool"] == "log_verification" and _ran(s.events)[-1]["status"] == "ok"
    # GLINT_LOG_KEY wins over the file; GLINT_LOG_KEY_FILE moves the file
    monkeypatch.setenv(KEY_ENV, "operator key")
    assert bank.session_key(server.home) == b"operator key"
    monkeypatch.delenv(KEY_ENV)
    monkeypatch.setenv(bank.KEY_FILE_ENV, str(server.home.parent / "elsewhere" / "k"))
    assert bank.key_file(server.home) == server.home.parent / "elsewhere" / "k"
    assert bank.session_key(server.home) != key_file.read_text().encode()


def test_the_install_key_is_made_once_and_never_replaced(tmp_path):
    path = tmp_path / "keys" / "log.key"
    first = install_key(path)
    assert len(first) == 64 and install_key(path) == first and path.read_text() == first.decode()
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600
    path.write_text("")
    with pytest.raises(ChainBroken, match="is empty"):
        install_key(path)


def test_a_changed_result_in_a_keyed_session_log_fails_the_check(server, monkeypatch, capsys):  # noqa: F811
    """The tamper test: a writer without the key edits a recorded write.ran (an error turned into success) and
    recomputes the chain; the keyed check fails, and `log SESSION_ID` refuses it."""
    monkeypatch.delenv(KEY_ENV, raising=False)
    _verify(server)
    s = server.session()
    s.events.close()
    path = bank.session_dir(server.home, "s1") / "gates.sqlite"
    raw = sqlite3.connect(path)
    rowid, payload = raw.execute("SELECT rowid, payload FROM events WHERE type = 'write.ran' ORDER BY rowid DESC"
                                 ).fetchone()
    raw.close()
    _rewrite_chain(path, rowid, payload.replace('"status": "ok"', '"status": "error"'))
    with pytest.raises(ChainBroken, match="row 1: hash"):
        EventLog(path, key=bank.session_key(server.home)).verify()
    with pytest.raises(ChainBroken):
        bank.main(["log", "s1"])


def test_an_old_unkeyed_session_log_stays_readable_and_verifiable(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(KEY_ENV, raising=False)
    home = tmp_path / "bank-home"
    monkeypatch.setenv("GLINT_BANK_HOME", str(home))
    old = EventLog(bank.session_dir(home, "old") / "gates.sqlite")  # as glint-bank wrote it before keying
    old.emit("gate.fired", {"decision": "allow", "rule_id": "r", "tool": "log_verification"})
    old.close()
    assert recorded_scheme(bank.session_dir(home, "old") / "gates.sqlite") == "sha256"
    assert bank.main(["log", "old"]) == 0
    out = capsys.readouterr().out
    assert "hash chain intact (sha256)" in out and "unkeyed: anyone who can write this file" in out
    # resuming it under the session key is refused with the way out, not silently continued unkeyed
    from mcp.shared.exceptions import MCPError
    with pytest.raises(MCPError, match="records an unkeyed chain.*start a new session"):
        bank.BankServer(sid="old").call(None, bank.POLICY, {})


def test_recorded_scheme_reads_without_writing(tmp_path):
    assert recorded_scheme(tmp_path / "missing.sqlite") is None
    keyed = tmp_path / "k.sqlite"
    EventLog(keyed, key=b"k").close()
    assert recorded_scheme(keyed) == "hmac-sha256"
    empty = tmp_path / "e.sqlite"
    sqlite3.connect(empty).close()
    assert recorded_scheme(empty) is None
    pre_meta = tmp_path / "p.sqlite"
    raw = sqlite3.connect(pre_meta)
    raw.execute("CREATE TABLE events (id TEXT)")
    raw.execute("INSERT INTO events VALUES ('e1')")
    raw.commit()
    raw.close()
    assert recorded_scheme(pre_meta) == "sha256"
