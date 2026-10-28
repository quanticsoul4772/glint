"""M6a: a logging mock of the Meta Model API, to learn the wire protocol Muse Code speaks when pointed at
`--base-url`. It records method, path, non-secret headers and the request body, then answers 503 so Muse stops.
Run: python scripts/mock_meta_server.py 8765   (writes measure/data/m6a-requests.jsonl)
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from glint import paths

OUT = paths.MEASURE / "m6a-requests.jsonl"
SECRET_HEADERS = {"authorization", "x-api-key", "cookie"}


class Handler(BaseHTTPRequestHandler):
    def _log(self):
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length).decode("utf-8", "replace") if length else ""
        rec = {
            "method": self.command, "path": self.path,
            "headers": {k: ("<redacted>" if k.lower() in SECRET_HEADERS else v) for k, v in self.headers.items()},
            "body": body,
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    def _reply(self):
        self._log()
        payload = json.dumps({"error": {"message": "glint M6a mock: protocol capture only", "type": "mock"}}).encode()
        self.send_response(503)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST = _reply

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 8765), Handler).serve_forever()
