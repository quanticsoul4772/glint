"""Per-run events for τ² batches: one line and one JSONL record as each simulation finishes, and a stall warning.

τ² only shows "N/M complete" while a batch runs; outcomes, error causes and which tasks fail appear when the whole
batch returns, hours later. `RunEvents` watches τ²'s checkpoint file (`results.json`, rewritten as each simulation
finishes) from a background thread and reports every new simulation as it lands.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

CONTEXT_OVERFLOW = "ContextWindowExceeded"


def outcome(sim: dict) -> str:
    """'pass', 'fail', 'context_overflow' (the model's own conversation outgrew the context: a model failure that τ²
    records as an infrastructure error) or 'infrastructure_error' (anything else τ² could not complete)."""
    if "infrastructure" in str(sim.get("termination_reason")).lower():  # enum text in memory, value on disk
        error = str((sim.get("info") or {}).get("error", ""))
        return "context_overflow" if CONTEXT_OVERFLOW in error else "infrastructure_error"
    return "pass" if (sim.get("reward_info") or {}).get("reward") == 1.0 else "fail"


def event(sim: dict, label: str) -> dict:
    return {"ts": time.time(), "label": label, "task_id": sim.get("task_id"), "trial": sim.get("trial"),
            "outcome": outcome(sim), "reward": (sim.get("reward_info") or {}).get("reward"),
            "termination": str(sim.get("termination_reason")).split(".")[-1],
            "error": str((sim.get("info") or {}).get("error", ""))[:300] or None,
            "duration_s": round(sim.get("duration") or 0.0, 1), "messages": len(sim.get("messages") or [])}


class RunEvents:
    """Context manager: watch `results_path` (τ²'s checkpoint) and, for every simulation that appears, print one line
    to stderr and append one JSON record to `events_path`. Warns once per `stall_s` with no new simulation."""

    def __init__(self, results_path: Path, events_path: Path, label: str, poll_s: float = 30.0,
                 stall_s: float = 1200.0):
        self.results_path, self.events_path, self.label = Path(results_path), Path(events_path), label
        self.poll_s, self.stall_s = poll_s, stall_s
        self.seen: set[tuple] = set()
        self.counts: dict[str, int] = {}
        self._done = threading.Event()
        self._last_new = time.time()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def __enter__(self):
        self.seen = {(s.get("task_id"), s.get("trial")) for s in self._read()}  # resumed runs are not re-reported
        if self.seen:
            self._say(f"{len(self.seen)} runs already in the checkpoint (resumed)")
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._done.set()
        self._thread.join(timeout=self.poll_s + 5)
        self.poll()  # whatever landed since the last poll
        self._say(f"done: {self.counts}")

    def _say(self, msg: str) -> None:
        print(f"[runs {time.strftime('%H:%M:%S')} {self.label}] {msg}", file=sys.stderr, flush=True)

    def _read(self) -> list[dict]:
        try:
            return json.loads(self.results_path.read_text(encoding="utf-8")).get("simulations") or []
        except (FileNotFoundError, json.JSONDecodeError):  # not written yet, or caught mid-write: next poll
            return []

    def poll(self) -> list[dict]:
        new = [s for s in self._read() if (s.get("task_id"), s.get("trial")) not in self.seen]
        if new:
            self._last_new = time.time()
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8") as fh:
                for s in new:
                    self.seen.add((s.get("task_id"), s.get("trial")))
                    e = event(s, self.label)
                    self.counts[e["outcome"]] = self.counts.get(e["outcome"], 0) + 1
                    fh.write(json.dumps(e) + "\n")
                    loud = "  <-- " + e["error"][:160] if e["outcome"] not in ("pass", "fail") else ""
                    self._say(f"{e['task_id']} trial {e['trial']}: {e['outcome'].upper()} ({e['termination']}, "
                              f"{e['duration_s']:.0f} s, {e['messages']} msgs){loud}")
            self._say(f"totals {self.counts}")
        elif time.time() - self._last_new > self.stall_s:
            self._say(f"WARNING: no run finished in {self.stall_s / 60:.0f} min ({len(self.seen)} so far)")
            self._last_new = time.time()
        return new

    def _watch(self):
        while not self._done.wait(self.poll_s):
            self.poll()
