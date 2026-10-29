"""scripts/muse_check.py on bootstrap-log lines in Muse's format (copied from the local-tracing bootstrap logs of
2026-10-28, paths shortened): a server that did not reach capability discovery `ready` fails the check."""
import muse_check

PREFIX = "2026-10-28T23:49:{s}Z INFO tbh.local.mcp fbcode/musecode/build/src/crates/agent-mcp/src/mcp/x.rs:27 "
FAILED_LOG = "\n".join(PREFIX.format(s=s) + rest for s, rest in [
    ("42.79", 'event="mcp.config.resolve" stage="settings" outcome="ready" reason="none" server_count=1'),
    ("45.45", ('event="mcp.transport.startup" server="glint" transport="stdio" mode="optional" outcome="ready" '
              'reason="none" duration_ms=931')),
    ("45.45", ('event="mcp.handshake" server="glint" transport="stdio" mode="optional" outcome="ready" reason="none" '
              'duration_ms=932')),
    ("45.46", ('event="mcp.capability.discovery" server="glint" transport="stdio" mode="optional" outcome="ready" '
              'reason="none" tool_count=7 resource_count=0 duration_ms=3')),
    ("47.65", ('event="mcp.shutdown" server="plugin:glint-bank:b" transport="stdio" outcome="complete" '
              'reason="requested" duration_ms=9')),
    ("49.61", ('event="mcp.transport.startup" server="plugin:glint-bank:b" transport="stdio" mode="optional" '
              'outcome="failed" reason="transport_closed" duration_ms=4147')),
    ("49.61", ('event="mcp.handshake" server="plugin:glint-bank:b" transport="stdio" mode="optional" '
              'outcome="failed" reason="transport_closed" duration_ms=4147')),
    ("50.33", ('event="mcp.capability.discovery" server="plugin:glint-probe:glintp" transport="stdio" '
              'mode="optional" outcome="ready" reason="none" tool_count=7 duration_ms=3')),
]) + "\n2026-10-28T23:50:00Z INFO tbh.local.provider other line without mcp events\n"
READY_LOG = PREFIX.format(s="33.60") + (
    'event="mcp.capability.discovery" server="plugin:glint-bank:b" transport="stdio" mode="optional" '
    'outcome="ready" reason="none" tool_count=18 resource_count=0 duration_ms=321\n')


def test_a_server_that_never_reached_ready_fails_loudly_with_its_reason(tmp_path, capsys):
    log = tmp_path / "cli-x.log"
    log.write_text(FAILED_LOG, encoding="utf-8")
    assert muse_check.main([str(log), "plugin:glint-bank:b", "glint"]) == 1
    captured = capsys.readouterr()
    assert ("FAILED: plugin:glint-bank:b: mcp.handshake outcome=failed reason=transport_closed after 4147 ms; never "
            "reached mcp.capability.discovery ready") in captured.out
    assert "  glint: ready (7 tools)" in captured.out
    assert "(not expected) plugin:glint-probe:glintp: ready (7 tools)" in captured.out
    assert "MUSE CHECK FAILED: 1 of 2 expected MCP servers did not start: plugin:glint-bank:b" in captured.err


def test_every_expected_server_ready_passes(tmp_path, capsys):
    log = tmp_path / "cli-y.log"
    log.write_text(READY_LOG, encoding="utf-8")
    assert muse_check.main([str(log), "plugin:glint-bank:b"]) == 0
    assert "plugin:glint-bank:b: ready (18 tools)" in capsys.readouterr().out


def test_a_server_missing_from_the_log_fails(tmp_path, capsys):
    log = tmp_path / "cli-z.log"
    log.write_text(READY_LOG, encoding="utf-8")
    assert muse_check.main([str(log), "plugin:glint-bank:b", "glint"]) == 1
    assert "FAILED: glint: no startup line in the log" in capsys.readouterr().out


def test_a_phase_that_never_finished_is_named(tmp_path):
    started = PREFIX.format(s="10.00") + ('event="mcp.transport.startup" server="s" transport="stdio" '
                                          'mode="optional" outcome="ready" reason="none" duration_ms=5')
    ok, lines = muse_check.check(started, ["s"])
    assert not ok and lines == [("FAILED: s: mcp.transport.startup outcome=ready reason=none after 5 ms; never reached "
                                "mcp.capability.discovery ready")]


def test_usage_without_servers(capsys):
    assert muse_check.main(["only-a-log.log"]) == 2
    assert "python scripts/muse_check.py LOG SERVER" in capsys.readouterr().err
