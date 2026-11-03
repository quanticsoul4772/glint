"""Where Glint's working files live, outside the repo.

GLINT_HOME (environment variable) is the workspace that holds run data, the spend ledger, research results and the
local secrets file; it defaults to the folder that contains this repo. The layout under it:
`measure/data` (run data), `measure/meter/ledger.jsonl` (spend ledger), `research/5-measurements` (result files),
`.env` (API keys, never committed), `.modal.toml` (Modal profile), `.cache/hf` (Hugging Face cache).
τ²-bench's own data comes from τ²'s DATA_DIR (its checkout, or TAU2_DATA_DIR).
"""
from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

# the blind clause lists and their reconciliation: package data, so an installed (non-editable) Glint has them. Kept
# here, in a module that imports nothing from Glint, so glint.declare and glint.gates.reminders can both read it at
# import time without an import cycle (declare -> gates -> enforce -> reminders -> declare).
CLAUSE_DIR = Path(str(resources.files("glint") / "clauses"))


def _default_home() -> Path:
    """The folder that contains the repo (this file is <home>/<repo>/src/glint/paths.py); the current folder when
    the package is installed somewhere shallower (a deployed container puts it at /root/glint/paths.py)."""
    parents = Path(__file__).resolve().parents
    return parents[3] if len(parents) > 3 and parents[1].name == "src" else Path.cwd()


HOME = Path(os.environ.get("GLINT_HOME") or _default_home())
MEASURE = HOME / "measure" / "data"
LEDGER = HOME / "measure" / "meter" / "ledger.jsonl"
RESULTS = HOME / "research" / "5-measurements"
ENV_FILE = HOME / ".env"
MODAL_CONFIG = HOME / ".modal.toml"
HF_CACHE = HOME / ".cache" / "hf"


def tau2_data() -> Path:
    """τ²-bench's data folder (domains, tasks, policies, simulations)."""
    from tau2.utils.utils import DATA_DIR

    return Path(DATA_DIR)
