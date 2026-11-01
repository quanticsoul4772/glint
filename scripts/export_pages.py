"""Export the Bank Desk's replay mode as a static site (GitHub Pages): the page plus every GET route it reads, each
as a .json file, taken from the desk app itself so the files are exactly what the API serves.

- index.html: the desk page with `window.GLINT_STATIC = true`, so it fetches `<route>.json` with relative URLs and
  works at a subpath (https://<user>.github.io/glint/).
- api/scenarios.json, api/scenarios/<id>.json, api/results.json.
- api/live/status.json: always unavailable. Live mode runs a GPU and is never offered on the static site.

No GPU, no keys, no τ²: the desk imports without it. Usage: python scripts/export_pages.py [OUT_DIR] (default site/)
"""
import json
import re
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from glint.desk.app import HERE, app

STATIC_FLAG = "<script>window.GLINT_STATIC = true;</script>\n"


def export(out: Path) -> list[Path]:
    client = TestClient(app)
    written = []

    def write(rel: str, data) -> None:
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        written.append(path)

    def get(route: str):
        r = client.get(route)
        r.raise_for_status()  # a route that fails must stop the export, never publish a broken site
        return r.json()

    scenarios = get("/api/scenarios")
    write("api/scenarios.json", scenarios)
    for s in scenarios:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", s["id"]):  # the id becomes a file name and a URL path segment
            raise SystemExit(f"scenario id {s['id']!r} is not a safe file name")
        write(f"api/scenarios/{s['id']}.json", get(f"/api/scenarios/{s['id']}"))
    write("api/results.json", get("/api/results"))
    write("api/try.json", get("/api/try"))
    write("api/live/status.json", {"available": False})
    page = (HERE / "static" / "index.html").read_text(encoding="utf-8")
    if "<script>" not in page:
        raise SystemExit("index.html has no <script>: cannot set the static flag")
    (out / "index.html").write_text(page.replace("<script>", STATIC_FLAG + "<script>", 1), encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")  # serve files as they are
    return written + [out / "index.html"]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0]) if argv else Path("site")
    files = export(out)
    print(f"wrote {len(files)} files to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
