"""Per-run events for τ² batches: one line and one JSONL record as each simulation finishes, and a stall warning.

τ² only shows "N/M complete" while a batch runs; outcomes, error causes and which tasks fail appear when the whole
batch returns, hours later. `RunEvents` watches τ²'s checkpoint file (`results.json`, rewritten as each simulation
finishes) from a background thread and reports every new simulation as it lands.
"""
from __future__ import annotations

import json
import re
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
# Fail fast: patterns that cannot be noise stop the arm (RunEvents `on_halt`) instead of only alerting. Runs in a row
# that end with the same outcome and the same cause (the first HALT_CAUSE_CHARS of info.error); context overflows and
# empty replies are model results, not faults, and never stop a run. And HALT_STALLS stall windows in a row with no run
# finished (two 20-minute windows across 12 concurrent runs mean nothing is being served).
HALT_SAME_CAUSE = {"infrastructure_error": 5, "eval_error": 5, "glint_error": 3}
HALT_CAUSE_CHARS = 80


def cause_key(error: str | None) -> str:
    """The part of a run's info.error that is the same when the fault is: quoted text (atem quotes the model's own
    output, 2026-10-29 review) and numbers (ids, counts, ports) are blanked, then the first HALT_CAUSE_CHARS kept."""
    text = re.sub(r"'[^']*'|\"[^\"]*\"", "'…'", str(error or ""))
    return re.sub(r"\d+", "#", text)[:HALT_CAUSE_CHARS]


def run_key(sim: dict) -> tuple:
    """A run's identity in τ²'s checkpoint: its simulation id. τ²'s resume reruns an infrastructure_error or
    budget_stop run under the same (task, trial) and a new id, so keying by (task, trial) hid every rerun of a resumed
    arm (2026-10-29 review); records without an id (tests, older formats) fall back to (task, trial)."""
    return (sim["id"],) if sim.get("id") else (sim.get("task_id"), sim.get("trial"))
HALT_STALLS = 2
LONG_RUN_S = 2700.0  # a run past 45 min since its first model call is warned about (most banking runs take 5-25 min)
HEARTBEAT_S = 600.0  # an ALIVE line every 10 min


class RunEvents:
    """Context manager: watch `results_path` (τ²'s checkpoint) and, for every simulation that appears, print one line
    to stderr and append one JSON record to `events_path`. Warns once per `stall_s` with no new simulation."""

    def __init__(self, results_path: Path, events_path: Path, label: str, poll_s: float = 30.0,
                 stall_s: float = 1200.0, on_run: Callable[[dict], str | None] | None = None,
                 domain: str | None = None, on_halt: Callable[[str], None] | None = None,
                 activity: Callable[[], float] | None = None,
                 running: Callable[[], dict[tuple[str, str], tuple[float, int, int, float]]] | None = None,
                 long_run_s: float = LONG_RUN_S, heartbeat_s: float = HEARTBEAT_S, gated: bool = True):
        self.results_path, self.events_path, self.label = Path(results_path), Path(events_path), label
        # called once, with the reason, when a fail-fast pattern is seen (HALT_SAME_CAUSE, HALT_STALLS); the caller
        # stops the arm. Without it the pattern is still printed as a FAIL FAST line.
        self.on_halt, self.halted = on_halt, None
        self.activity = activity  # time.time() of the model's latest call; a stall with calls still made is not one
        # (task_id, trial) -> (time of its first model call, model calls so far, peak prompt tokens, time of its
        # latest call) for every run that has called the model: a run past long_run_s is warned about once
        # (2026-10-30: task_018 ran 61 min, 151 messages, unflagged), and a run τ² kept nothing of (a context
        # overflow) gets its calls, time and peak from here
        self.running, self.long_run_s, self.heartbeat_s = running, long_run_s, heartbeat_s
        # an arm without Glint's gates and tools (Glimmer alone): its repeated wrong fields are the model's own, not a
        # regression, and it has no gate refusals to audit (2026-10-31: the unseen base arm printed 10 "a regression
        # would show this way" warnings and "0 of 22 passing runs went through a gate refusal")
        self.gated = gated
        self.finished_runs: set[tuple[str, str]] = set()
        self._long_warned: set[tuple[str, str]] = set()
        self._last_beat = time.time()
        self.endings: dict[str, int] = {}  # τ²'s termination reason -> runs (max_steps was counted only as a fail)
        self._streak: tuple[tuple[str, str] | None, int] = (None, 0)  # (outcome, cause) of the latest runs, how many
        self._stalls = 0
        self.on_run = on_run  # called with each new simulation; a returned line is printed (glint.gates.summary)
        # With the run's τ² domain, each failed run is diffed against τ²'s expected actions (glint.inspect): a compact
        # summary goes into its event record and its line, and the arm's end prints a DIFFS line. Imported here, not
        # at the top: glint.inspect loads τ²'s tasks and environments, which most users of this module never need.
        self.diff = self.facts = None
        if domain is not None:
            from glint import inspect

            self.diff = lambda sim: inspect.compact(inspect.diff_run(domain, sim))
            # every run's facts (plan calls, filings, history read first, searches, peak prompt tokens, step-order
            # flags) in its event record and line, so a regression shows on its first run, not after the arm
            def facts(sim: dict) -> tuple[dict, str]:
                f = inspect.run_facts(sim)
                return f, inspect.facts_brief(f)

            self.facts = facts
        self.diffs: dict[tuple[str, int], dict] = {}  # (task_id, trial) -> compact diff of each failed run
        self.field_runs: dict[str, list[str]] = {}  # wrong field -> the failed runs it was wrong in (repeat alert)
        self.passes_audited, self.passes_with_refusals, self.passes_with_claims = 0, [], []
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
        resumed = self._read()
        self.seen = {run_key(s) for s in resumed}  # resumed runs are not re-reported; their reruns are
        self.finished_runs |= {(str(s.get("task_id")), str(s.get("trial"))) for s in resumed}
        if self.seen:
            self._say(f"{len(self.seen)} runs already in the checkpoint (resumed)")
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._done.set()
        self._thread.join(timeout=self.poll_s + 5)
        self.poll()  # whatever landed since the last poll
        self._say(f"done: {self.counts}; endings {dict(sorted(self.endings.items()))}"
                  + (f"; {self.read_errors} checkpoint reads retried" if self.read_errors else ""))
        for kind, tasks in sorted(self.tasks_by_outcome.items()):
            if kind not in ("pass", "fail"):
                self._say(f"{kind} by task: {dict(sorted(tasks.items()))}")
        if self.diff is not None:
            from glint import inspect

            self._say(inspect.diffs_line(list(self.diffs.values())))
        if self.facts is not None and self.passes_audited:
            refusals = (f"{len(self.passes_with_refusals)} of {self.passes_audited} passing runs went through a gate "
                        f"refusal {self.passes_with_refusals or ''}; " if self.gated else
                        f"{self.passes_audited} passing runs (no gates in this arm); ")
            self._say(f"PASS AUDIT: {refusals}{len(self.passes_with_claims)} claimed a write that never ran "
                      f"{self.passes_with_claims or ''}")

    def _repeated_fields(self, e: dict) -> list[str]:
        """A WARNING the second time a field is wrong in a failed run of this arm (once per field): the v6 regression
        (issue_noticed_date) showed on its first runs but surfaced only at the arm's end (2026-10-30)."""
        run = f"{e['task_id']} t{e['trial']}"
        out = []
        for field in e["diff"]["fields"]:
            runs = self.field_runs.setdefault(field, [])
            runs.append(run)
            if len(runs) == 2:
                why = ("a regression would show this way; check before the arm ends" if self.gated else
                       "the model's own repeated error (no Glint tools in this arm), not a regression")
                out.append(f"WARNING: repeated wrong field: {field} in 2 failed runs ({', '.join(runs)}); {why}")
        return out

    def _audit_pass(self, e: dict) -> None:
        """Count passing runs that went through a gate refusal or claimed a write that never ran (a pass that may hide
        a violation; procedure-aware evaluation found 27-78% of τ-bench passes corrupt, arXiv 2603.03116)."""
        f = e.get("facts") or {}
        if e["outcome"] == "pass":
            self.passes_audited += 1
            if f.get("gate_refusals"):
                self.passes_with_refusals.append(f"{e['task_id']} t{e['trial']}")
            if f.get("claims_not_done"):
                self.passes_with_claims.append(f"{e['task_id']} t{e['trial']}")

    def _track_streak(self, e: dict) -> None:
        kind = e["outcome"]
        key = (kind, cause_key(e.get("error"))) if kind in HALT_SAME_CAUSE else None
        n = self._streak[1] + 1 if key is not None and key == self._streak[0] else (1 if key is not None else 0)
        self._streak = (key, n)
        if key is not None and n >= HALT_SAME_CAUSE[kind]:
            self._halt(f"{n} runs in a row ended {kind} with the same cause: {key[1] or 'no error recorded'}")

    def _halt(self, reason: str) -> None:
        if self.halted is not None:
            return
        if self._done.is_set():  # the arm's last poll, after τ² returned: every run already finished
            self._say(f"WARNING: fail-fast pattern at the arm's end ({reason}); the arm had finished, so it is kept")
            return
        self.halted = reason
        self._say(f"FAIL FAST: {reason}; stopping the arm")
        if self.on_halt is not None:
            self.on_halt(reason)

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
        new = [s for s in self._read() if run_key(s) not in self.seen]
        if new:
            self._last_new = time.time()
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8") as fh:
                for s in new:
                    self.seen.add(run_key(s))
                    e = event(s, self.label)
                    loud = "  <-- " + e["error"][:160] if e["outcome"] not in ("pass", "fail") else ""
                    repeated: list[str] = []
                    if not e["messages"] and self.running is not None:
                        clock = self.running().get((str(e["task_id"]), str(e["trial"])))
                        if clock is not None:  # τ² kept no messages or time (a context overflow): the call log's
                            start, calls, peak, last = clock
                            e["from_calls"] = {"model_calls": calls, "seconds": round(last - start, 1),
                                               "peak_prompt_tokens": peak}
                    # a run τ² kept no messages of has nothing to diff: its "missing N" fed the DIFFS line and the
                    # repeated-field warnings with empty conversations (12 of 23 diffs in the unseen base arm)
                    if self.diff is not None and e["outcome"] in FAILED_RUNS and e["messages"]:
                        # a diff that cannot be computed is recorded and printed as such, and the run's event
                        # still lands: the run log must not stop over an analysis error
                        try:
                            e["diff"] = self.diff(s)
                            self.diffs[(str(e["task_id"]), int(e["trial"] or 0))] = e["diff"]
                            loud += "  " + e["diff"]["text"]
                            repeated = self._repeated_fields(e)
                        except Exception as exc:  # noqa: BLE001  (recorded in the event and printed below)
                            e["diff_error"] = f"{type(exc).__name__}: {exc}"[:300]
                            loud += f"  WARNING: no diff ({e['diff_error'][:160]})"
                    # a run τ² kept no messages of has no facts: its line printed "searches 0, peak prompt 0k" beside
                    # the call log's real peak (2026-10-31 review)
                    if self.facts is not None and e["messages"]:
                        try:  # like the diff: a facts error is recorded and printed, and the event still lands
                            e["facts"], brief = self.facts(s)
                            loud += "  " + brief
                        except Exception as exc:  # noqa: BLE001  (recorded in the event and printed below)
                            e["facts_error"] = f"{type(exc).__name__}: {exc}"[:300]
                            loud += f"  WARNING: no facts ({e['facts_error'][:160]})"
                    self.counts[e["outcome"]] = self.counts.get(e["outcome"], 0) + 1
                    self.endings[e["termination"]] = self.endings.get(e["termination"], 0) + 1
                    self.finished_runs.add((str(e["task_id"]), str(e["trial"])))
                    self._track_streak(e)
                    by_task = self.tasks_by_outcome.setdefault(e["outcome"], {})
                    by_task[str(e["task_id"])] = by_task.get(str(e["task_id"]), 0) + 1
                    fh.write(json.dumps(e, default=str) + "\n")
                    calls = e.get("from_calls")
                    kept = (f"{e['duration_s']:.0f} s, {e['messages']} msgs" if e["messages"] else
                            "τ² kept no messages or time" + (
                                f"; call log: {calls['model_calls']} model calls over {calls['seconds']:.0f} s, peak "
                                f"prompt {calls['peak_prompt_tokens']} tokens" if calls else ""))
                    tally = ", ".join(f"{k} {v}" for k, v in sorted(self.counts.items()))
                    self._say(f"{e['task_id']} trial {e['trial']}: {e['outcome'].upper()} (τ²: {e['termination']}, "
                              f"{kept}; so far {tally}){loud}")
                    for line in repeated:
                        self._say(line)
                    self._audit_pass(e)
                    note = self.on_run(s) if self.on_run is not None else None
                    if note:
                        self._say(note)
            self._alert(new[-1] if new else None)
        elif time.time() - self._last_new > self.stall_s:
            # a stall window counts toward a stop only when the model made no call in it either: one long run at an
            # arm's tail can go 40 minutes without finishing while still working (task_053, 2026-10-29)
            idle_s = time.time() - self.activity() if self.activity is not None else None
            if idle_s is not None and idle_s < self.stall_s:
                self._stalls = 0
                self._say(f"WARNING: no run finished in {self.stall_s / 60:.0f} min ({len(self.seen)} so far); model "
                          f"calls are still being made (last {idle_s:.0f} s ago)")
            else:
                self._stalls += 1
                self._say(f"WARNING: no run finished in {self._stalls * self.stall_s / 60:.0f} min ({len(self.seen)} "
                          "so far)" + ("" if idle_s is None else f" and no model call in {idle_s / 60:.0f} min"))
                if self._stalls >= HALT_STALLS:
                    self._halt(f"no run finished in {self._stalls * self.stall_s / 60:.0f} min"
                               + ("" if idle_s is None else f" and no model call in {idle_s / 60:.0f} min"))
            self._last_new = time.time()
        if new:
            self._stalls = 0
        self._long_runs()
        return new

    def _open_runs(self) -> dict[tuple[str, str], tuple[float, int, int, float]]:
        if self.running is None:
            return {}
        return {k: v for k, v in self.running().items() if k not in self.finished_runs}

    def _long_runs(self) -> None:
        """One WARNING per run that has gone past long_run_s since its first model call without finishing: τ²'s
        max_steps ends a long run, so a run far past it is hung or looping (searches, refusals)."""
        now = time.time()
        for (task, trial), (start, calls, *_) in sorted(self._open_runs().items()):
            if now - start > self.long_run_s and (task, trial) not in self._long_warned:
                self._long_warned.add((task, trial))
                self._say(f"WARNING: {task} trial {trial} has run {(now - start) / 60:.0f} min ({calls} model calls) "
                          "without finishing; check it is not hung or looping")

    def _beat(self) -> None:
        """An ALIVE line every heartbeat_s, so a watcher can tell a quiet arm from a dead driver (which prints nothing)."""
        if time.time() - self._last_beat < self.heartbeat_s:
            return
        self._last_beat = time.time()
        tally = ", ".join(f"{k} {v}" for k, v in sorted(self.counts.items())) or "none"
        open_runs = self._open_runs()
        oldest = (f", oldest {(time.time() - min(v[0] for v in open_runs.values())) / 60:.0f} min" if open_runs
                  else "")
        idle = f"; last model call {time.time() - self.activity():.0f} s ago" if self.activity is not None else ""
        self._say(f"ALIVE: {len(self.seen)} finished ({tally}); {len(open_runs)} running{oldest}{idle}")

    def _watch(self):
        # An exception here would end the thread and every later event without a word (2026-10-27: a PermissionError
        # stopped events.jsonl for a whole arm); anything unexpected is printed in full and watching goes on
        while not self._done.wait(self.poll_s):
            try:
                self.poll()
                self._beat()
            except Exception:  # noqa: BLE001  (printed in full; the next poll retries)
                self._say(f"WARNING: the event watcher hit an error and keeps polling:\n{traceback.format_exc()}")
