"""Where Glint's working files live, outside the repo.

GLINT_HOME (environment variable) is the workspace that holds run data, the spend ledger, research results and the
local secrets file; it defaults to the folder that contains this repo. The layout under it:
`measure/data` (run data), `measure/meter/ledger.jsonl` (spend ledger), `research/5-measurements` (result files),
`.env` (API keys, never committed), `.modal.toml` (Modal profile), `.cache/hf` (Hugging Face cache).
τ²-bench's own data comes from τ²'s DATA_DIR (its checkout, or TAU2_DATA_DIR).
"""
from __future__ import annotations

import os
from pathlib import Path

HOME = Path(os.environ.get("GLINT_HOME") or Path(__file__).resolve().parents[3])
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
