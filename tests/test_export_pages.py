"""The static replay site (scripts/export_pages.py): every file is what the desk API serves, the page fetches relative
.json URLs so it works at a subpath, and live mode is never offered."""
import json
import re

import export_pages
from fastapi.testclient import TestClient

from glint.desk.app import app

client = TestClient(app)


def test_every_exported_file_is_what_the_api_serves(tmp_path):
    export_pages.main([str(tmp_path)])
    listing = client.get("/api/scenarios").json()
    assert json.loads((tmp_path / "api" / "scenarios.json").read_text(encoding="utf-8")) == listing
    for s in listing:
        exported = json.loads((tmp_path / "api" / "scenarios" / f"{s['id']}.json").read_text(encoding="utf-8"))
        assert exported == client.get(f"/api/scenarios/{s['id']}").json()
    assert json.loads((tmp_path / "api" / "results.json").read_text(encoding="utf-8")) == client.get(
        "/api/results").json()
    assert json.loads((tmp_path / "api" / "live" / "status.json").read_text(encoding="utf-8")) == {"available": False}
    assert (tmp_path / ".nojekyll").exists()


def test_the_page_sets_static_mode_and_fetches_only_relative_urls(tmp_path):
    export_pages.main([str(tmp_path)])
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert page.index("window.GLINT_STATIC = true") < page.index("const API")
    # an absolute /api/ URL would point at the domain root, not the site's subpath (github.io/glint/)
    assert not re.search(r"""fetch\(\s*["'`]/""", page)
    assert all("API(" in line for line in page.splitlines()
               if "fetch(" in line and "api/live/start" not in line and "api/live/${" not in line)


def test_the_served_page_is_not_in_static_mode():
    assert "window.GLINT_STATIC = true" not in client.get("/").text
