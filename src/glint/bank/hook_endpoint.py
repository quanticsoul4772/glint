"""The gate decision endpoint that glint-bank's Muse Code PreToolUse hook asks (plugins/glint-bank/hooks/gate_check.py).

The gates' records (verifications, plans, refusals) live in the bank server's session, so the hook, a separate process
Muse starts for each tool call, asks the server. While the stdio server runs, `HookEndpoint` serves one route on
127.0.0.1 (a free port) and writes `<session dir>/hook.json` = {"port", "token", "pid"}; the hook finds that file by
GLINT_BANK_HOME (else ~/.glint-bank) and MUSE_SESSION_ID, the session key the server uses.

POST /decide with {"token", "tool", "arguments", "call_id"} (tool: the τ² tool name, without Muse's
`mcp__plugin_glint_bank_b__` prefix) answers 200 {"decision": "allow"} or {"decision": "deny", "reason": <the
refusal text the server would return>}, from `BankServer.decide` (Enforcer.precheck: an allow logs nothing, a deny
leaves the log and records a server refusal leaves, plus a hook.denied event). A wrong token is 403, a malformed
body 400, a decision that raised 500 (the hook blocks on anything but a 200 answer). The file is removed when the
endpoint stops, if it is still this endpoint's.
"""
from __future__ import annotations

import hmac
import json
import os
import secrets
import sys
import threading
import traceback
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Self

FILE = "hook.json"
ROUTE = "/decide"
MAX_BODY = 1 << 20  # a tool call's arguments are far smaller


class HookEndpoint:
    """`decide(tool, arguments, call_id)` returns the refusal text, or None to allow."""

    def __init__(self, decide: Callable[[str, dict, str | None], str | None], path: Path):
        self.decide, self.path = decide, Path(path)
        self.token = secrets.token_urlsafe(32)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler(self))
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="glint-bank-hook", daemon=True)

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    def start(self) -> HookEndpoint:
        self.thread.start()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps({"port": self.port, "token": self.token, "pid": os.getpid()}), encoding="utf-8")
        os.replace(tmp, self.path)  # the hook never reads a half-written file
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        try:
            ours = json.loads(self.path.read_text(encoding="utf-8")).get("token") == self.token
        except FileNotFoundError:
            return
        if ours:  # a newer server for the same session may have replaced it
            self.path.unlink()

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def _handler(endpoint: HookEndpoint) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_BODY:
                return self._send(400, {"error": "missing or oversized body"})
            raw = self.rfile.read(length)  # read before any answer: closing on unread data resets the connection
            if self.path != ROUTE:
                return self._send(404, {"error": f"unknown route {self.path}"})
            try:
                body = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return self._send(400, {"error": "the body is not JSON"})
            if not isinstance(body, dict) or not isinstance(body.get("tool"), str):
                return self._send(400, {"error": "the body needs a string `tool`"})
            if not hmac.compare_digest(str(body.get("token", "")).encode("utf-8"), endpoint.token.encode("utf-8")):
                return self._send(403, {"error": "wrong token"})
            arguments = body.get("arguments")
            if not isinstance(arguments, dict):
                return self._send(400, {"error": "`arguments` must be an object"})
            call_id = body.get("call_id") if isinstance(body.get("call_id"), str) else None
            try:
                refusal = endpoint.decide(body["tool"], arguments, call_id)
            except Exception as e:  # noqa: BLE001  (on stderr, and answered as 500: the hook blocks the call)
                traceback.print_exc(file=sys.stderr)
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})
            self._send(200, {"decision": "allow"} if refusal is None else {"decision": "deny", "reason": refusal})

        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args) -> None:
            return  # no access log on the server's stderr (Muse's server log)

    return Handler
