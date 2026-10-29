"""Did every expected MCP server start in a Muse run? Muse treats plugin servers as optional: one that fails to start
(2026-10-28: plugin:glint-bank:b, reason transport_closed) is logged once and the run carries on without its tools.

Reads Muse's local-tracing bootstrap log for one run (`<muse data>/local-tracing/bootstrap/cli-<id>.log`; the session
folder's `cli-<id>.log` has the same format) and prints each server's startup: `ready` when the log has its
`event="mcp.capability.discovery" ... outcome="ready"` line, else the last startup phase that failed and its reason.
Exits 1, loudly, when any expected server did not reach ready.

Usage: python scripts/muse_check.py LOG SERVER [SERVER ...]
       (server names as Muse logs them, e.g. plugin:glint-bank:b for the glint-bank plugin's server "b")
"""
import re
import sys
from pathlib import Path

STARTUP_EVENTS = ("mcp.transport.startup", "mcp.handshake", "mcp.capability.discovery")
FIELD = re.compile(r'(\w+)=(?:"((?:[^"\\]|\\.)*)"|(\S+))')


def mcp_events(text: str) -> list[dict]:
    """The startup-phase records of every MCP server in a bootstrap log, in order, as field dicts."""
    events = []
    for line in text.splitlines():
        if 'event="mcp.' not in line:
            continue
        fields = {m.group(1): m.group(2) if m.group(2) is not None else m.group(3) for m in FIELD.finditer(line)}
        if fields.get("event") in STARTUP_EVENTS and "server" in fields:
            events.append(fields)
    return events


def startup(events: list[dict], server: str) -> tuple[bool, str]:
    """(ready, description) for one server."""
    mine = [e for e in events if e["server"] == server]
    ready = [e for e in mine if e["event"] == "mcp.capability.discovery" and e.get("outcome") == "ready"]
    if ready:
        return True, f"ready ({ready[-1].get('tool_count', '?')} tools)"
    if not mine:
        return False, "no startup line in the log (not configured, or not started in this run)"
    failed = [e for e in mine if e.get("outcome") != "ready"]
    last = failed[-1] if failed else mine[-1]
    return False, (f"{last['event']} outcome={last.get('outcome')} reason={last.get('reason')} "
                   f"after {last.get('duration_ms', '?')} ms; never reached mcp.capability.discovery ready")


def check(text: str, expected: list[str]) -> tuple[bool, list[str]]:
    events = mcp_events(text)
    lines, ok = [], True
    for server in expected:
        ready, what = startup(events, server)
        ok &= ready
        lines.append(f"{server}: {what}" if ready else f"FAILED: {server}: {what}")
    others = sorted({e["server"] for e in events} - set(expected))
    for server in others:
        lines.append(f"(not expected) {server}: {startup(events, server)[1]}")
    return ok, lines


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.strip().split("Usage: ")[1], file=sys.stderr)
        return 2
    path, expected = Path(argv[0]), argv[1:]
    ok, lines = check(path.read_text(encoding="utf-8", errors="replace"), expected)
    print(f"Muse MCP startup in {path}:")
    for line in lines:
        print(f"  {line}")
    if not ok:
        failed = [line.split(": ")[1] for line in lines if line.startswith("FAILED")]
        print(f"MUSE CHECK FAILED: {len(failed)} of {len(expected)} expected MCP servers did not start: "
              f"{', '.join(failed)}. Muse carried on without their tools; this run did not use them.", file=sys.stderr)
        return 1
    print(f"all {len(expected)} expected MCP servers ready")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
