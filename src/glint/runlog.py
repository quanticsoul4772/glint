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
import traceback
from collections.abc import Callable
from pathlib import Path

CONTEXT_OVERFLOW = "ContextWindowExceeded"
# τ²'s check on a reply with no text and no tool call (Message.validate); Glimmer returned an empty message twice in
# banking-gated on 2026-10-28. A model failure like an overflow (decision 2026-10-28): a failed run, never excluded
EMPTY_REPLY = "must have either content or tool_calls"
# Glint's own exceptions put their class name first in the message (τ² keeps only str(exception) in info.error)
BUDGET_STOP = "BudgetExceeded:"
GLINT_ERRORS = ("AtemParseError:", "CallLogError:")
# outcomes that are neither a pass nor a fail of the model; every one is counted and reported, never dropped quietly
NOT_RESULTS = ("infrastructure_error", "budget_stop", "eval_error")
# outcomes that count as a failed run of the model
FAILED_RUNS = ("fail", "context_overflow", "empty_reply", "glint_error")


def outcome(sim: dict) -> str:
    """One of:
    - 'pass' / 'fail': the run finished and was evaluated (reward 1 or not);
    - 'context_overflow': the model's own conversation outgrew the context, a model failure that τ² records as an
      infrastructure error;
    - 'empty_reply': the model returned a message with no content and no tool call, which τ² records as an
      infrastructure error; a model failure and a failed run, like context_overflow;
    - 'glint_error': Glint's own code failed inside the run (atem re-parse, call recording): a failed run, charged
      to Glint, never hidden as infrastructure;
    - 'budget_stop': a spend cap stopped the run: not a result, and the arm must not be saved as finished;
    - 'eval_error': the run ended but has no reward (τ²'s evaluation failed): not a result;
    - 'infrastructure_error': anything else τ² could not complete."""
    if "infrastructure" in str(sim.get("termination_reason")).lower():  # enum text in memory, value on disk
        error = str((sim.get("info") or {}).get("error", ""))
        if CONTEXT_OVERFLOW in error:
            return "context_overflow"
        if EMPTY_REPLY in error:
            return "empty_reply"
        if BUDGET_STOP in error:
            return "budget_stop"
        if any(e in error for e in GLINT_ERRORS):
            return "glint_error"
        return "infrastructure_error"
    reward = (sim.get("reward_info") or {}).get("reward")
    if reward is None:
        return "eval_error"
    return "pass" if reward == 1.0 else "fail"


def event(sim: dict, label: str) -> dict:
    return {"ts": time.time(), "label": label, "task_id": sim.get("task_id"), "trial": sim.get("trial"),
            "outcome": outcome(sim), "reward": (sim.get("reward_info") or {}).get("reward"),
            "termination": str(sim.get("termination_reason")).split(".")[-1],
            "error": str((sim.get("info") or {}).get("error", ""))[:300] or None,
            "duration_s": round(sim.get("duration") or 0.0, 1), "messages": len(sim.get("messages") or [])}


# Repeated outcomes worth a loud line while the arm runs: a count reached (then each doubling), or for context
# overflows a share of the finished runs. 2026-10-28, banking-test: 10 runs were lost to one HTTP 303 bug and about
# one run in six overflowed the context, and each showed only as its own line.
ALERT_COUNTS = {"infrastructure_error": 3, "glint_error": 1, "budget_stop": 1, "eval_error": 3}
ALERT_SHARES = {"context_overflow": (0.15, 10), "empty_reply": (0.15, 10)}  # share of finished runs, after at least this many runs


class RunEvents:
    """Context manager: watch `results_path` (τ²'s checkpoint) and, for every simulation that appears, print one line
    to stderr and append one JSON record to `events_path`. Warns once per `stall_s` with no new simulation."""

    def __init__(self, results_path: Path, events_path: Path, label: str, poll_s: float = 30.0,
                 stall_s: float = 1200.0, on_run: Callable[[dict], str | None] | None = None,
                 domain: str | None = None):
        self.results_path, self.events_path, self.label = Path(results_path), Path(events_path), label
        self.on_run = on_run  # called with each new simulation; a returned line is printed (glint.gates.summary)
        # With the run's τ² domain, each failed run is diffed against τ²'s expected actions (glint.inspect): a compact
        # summary goes into its event record and its line, and the arm's end prints a DIFFS line. Imported here, not
        # at the top: glint.inspect loads τ²'s tasks and environments, which most users of this module never need.
        self.diff = None
        if domain is not None:
            from glint import inspect

            self.diff = lambda sim: inspect.compact(inspect.diff_run(domain, sim))
        self.diffs: dict[tuple[str, int], dict] = {}  # (task_id, trial) -> compact diff of each failed run
        self.poll_s, self.stall_s = poll_s, stall_s
        self.seen: set[tuple] = set()
        self.counts: dict[str, int] = {}
        self.tasks_by_outcome: dict[str, dict[str, int]] = {}  # outcome -> task -> runs, for alerts and the summary
        self._next_alert: dict[str, int] = {}
        self.read_errors = 0  # checkpoint reads refused or caught mid-write (retried at the next poll)
        self._bad_reads_in_a_row = 0
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
        self._say(f"done: {self.counts}" + (f"; {self.read_errors} checkpoint reads retried" if self.read_errors
                                            else ""))
        for kind, tasks in sorted(self.tasks_by_outcome.items()):
            if kind not in ("pass", "fail"):
                self._say(f"{kind} by task: {dict(sorted(tasks.items()))}")
        if self.diff is not None:
            from glint import inspect

            self._say(inspect.diffs_line(list(self.diffs.values())))

    def _alert(self, latest: dict | None) -> None:
        """One ALERT line when an outcome repeats past its threshold, again at each doubling, with its tasks and the
        latest cause; so a pattern is one loud line, not a run of separate ones."""
        finished = sum(self.counts.values())
        for kind, n in self.counts.items():
            if kind in ALERT_COUNTS:
                due = n >= self._next_alert.get(kind, ALERT_COUNTS[kind])
            elif kind in ALERT_SHARES:
                share, least = ALERT_SHARES[kind]
                due = finished >= least and n / finished >= share and n >= self._next_alert.get(kind, 1)
            else:
                continue
            if not due:
                continue
            self._next_alert[kind] = 2 * n
            cause = ""
            if latest is not None:
                e = event(latest, self.label)
                cause = f"; latest: {e['error'][:140]}" if e["outcome"] == kind and e["error"] else ""
            self._say(f"ALERT: {n} {kind} runs of {finished} finished ({n / finished:.0%}), tasks "
                      f"{dict(sorted(self.tasks_by_outcome[kind].items()))}{cause}")

    def _say(self, msg: str) -> None:
        print(f"[runs {time.strftime('%H:%M:%S')} {self.label}] {msg}", file=sys.stderr, flush=True)

    def _read(self) -> list[dict]:
        if not self.results_path.exists():
            return []  # τ² has not written its checkpoint yet
        # Windows refuses the read while τ² rewrites the file (PermissionError, 2026-10-27), and a read can land
        # mid-write (JSONDecodeError): both are retried at the next poll, counted, and reported if they persist
        try:
            sims = json.loads(self.results_path.read_text(encoding="utf-8")).get("simulations") or []
        except (OSError, json.JSONDecodeError) as e:
            self.read_errors += 1
            self._bad_reads_in_a_row += 1
            if self._bad_reads_in_a_row in (3, 10, 30):
                self._say(f"WARNING: {self._bad_reads_in_a_row} checkpoint reads in a row failed "
                          f"({type(e).__name__}: {e}); events are not being recorded")
            return []
        self._bad_reads_in_a_row = 0
        return sims

    def poll(self) -> list[dict]:
        new = [s for s in self._read() if (s.get("task_id"), s.get("trial")) not in self.seen]
        if new:
            self._last_new = time.time()
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8") as fh:
                for s in new:
                    self.seen.add((s.get("task_id"), s.get("trial")))
                    e = event(s, self.label)
                    loud = "  <-- " + e["error"][:160] if e["outcome"] not in ("pass", "fail") else ""
                    if self.diff is not None and e["outcome"] in FAILED_RUNS:
                        # a diff that cannot be computed is recorded and printed as such, and the run's event
                        # still lands: the run log must not stop over an analysis error
                        try:
                            e["diff"] = self.diff(s)
                            self.diffs[(str(e["task_id"]), int(e["trial"] or 0))] = e["diff"]
                            loud += "  " + e["diff"]["text"]
                        except Exception as exc:  # noqa: BLE001  (recorded in the event and printed below)
                            e["diff_error"] = f"{type(exc).__name__}: {exc}"[:300]
                            loud += f"  WARNING: no diff ({e['diff_error'][:160]})"
                    self.counts[e["outcome"]] = self.counts.get(e["outcome"], 0) + 1
                    by_task = self.tasks_by_outcome.setdefault(e["outcome"], {})
                    by_task[str(e["task_id"])] = by_task.get(str(e["task_id"]), 0) + 1
                    fh.write(json.dumps(e, default=str) + "\n")
                    kept = (f"{e['duration_s']:.0f} s, {e['messages']} msgs" if e["messages"] else
                            "τ² kept no messages or time")
                    tally = ", ".join(f"{k} {v}" for k, v in sorted(self.counts.items()))
                    self._say(f"{e['task_id']} trial {e['trial']}: {e['outcome'].upper()} (τ²: {e['termination']}, "
                              f"{kept}; so far {tally}){loud}")
                    note = self.on_run(s) if self.on_run is not None else None
                    if note:
                        self._say(note)
            self._alert(new[-1] if new else None)
        elif time.time() - self._last_new > self.stall_s:
            self._say(f"WARNING: no run finished in {self.stall_s / 60:.0f} min ({len(self.seen)} so far)")
            self._last_new = time.time()
        return new

    def _watch(self):
        # An exception here would end the thread and every later event without a word (2026-10-27: a PermissionError
        # stopped events.jsonl for a whole arm); anything unexpected is printed in full and watching goes on
        while not self._done.wait(self.poll_s):
            try:
                self.poll()
            except Exception:  # noqa: BLE001  (printed in full; the next poll retries)
                self._say(f"WARNING: the event watcher hit an error and keeps polling:\n{traceback.format_exc()}")
