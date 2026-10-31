"""The task splits every run and replay reads, kept in the repo (`splits/`) so a clean checkout can rerun the gold
replays and the comparisons without the workspace.

- `split-partition.json`: retail and airline tasks split into pilot, design and held-out (scripts/partition.py).
- `split-partition-banking.json`: the banking tasks split into pilot, rest and the 20 untouched test tasks
  (scripts/headroom_pilot.py); the test tasks are read once, as the final check.
"""
from pathlib import Path

DIR = Path(__file__).resolve().parents[1] / "splits"
PARTITION = DIR / "split-partition.json"
BANKING = DIR / "split-partition-banking.json"
