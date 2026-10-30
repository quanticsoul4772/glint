"""The run-log lines a watcher must see while a run goes, as one grep -E pattern (the run-ab skill's monitor uses it).

Every uppercase prefix Glint prints is classified here: ALERTS must match MONITOR; INFO lines are summaries read after
the run; TRANSCRIPT labels are not log lines. tests/test_alert_patterns.py scans src/ and scripts/ for such prefixes
and fails on one that is not classified, so a new alert cannot be missed silently (2026-10-29: glimmer_serve's
`FAILED:` lines, among them a GPU app still running after every stop attempt, and the per-response `EMPTY REPLY:` and
`TRUNCATED:` lines were not in the hand-kept filter).

Usage: python scripts/alert_patterns.py   (prints MONITOR for `grep -E --line-buffered "$(...)"`)
"""
from __future__ import annotations

import re

# Prefixes (and the run outcomes and markers without one) a watcher acts on while the run goes
ALERTS = (
    "FAILED", "METER FAILED", "SMOKE TEST FAILED", "MUSE CHECK FAILED", "COVERAGE GATE FAILED", "FAIL FAST", "ALERT",
    "WARNING", "CHECKPOINT LOCKED", "GPU CAP", "GPU CAP REACHED", "EMPTY REPLY", "TRUNCATED", "POOL WAIT",
    "UNFOLLOWED REDIRECT", "CLAIMED NOT DONE", "VERDICT NOT FILED",
)
OUTCOMES = ("INFRASTRUCTURE_ERROR", "CONTEXT_OVERFLOW", "EMPTY_REPLY", "GLINT_ERROR", "BUDGET_STOP", "EVAL_ERROR",
            "too_many_errors")
# the ends of a run and of an arm, and Python's own failures
MARKERS = (r"BudgetExceeded", r"Traceback", r"SystemExit", r"still has", r"stopped, 0", r"this arm", r"\] done:",
           r"session start", r"/health -> (5|gave up)", r"^exit ")
# Summaries printed at an arm's end or a resume, read with the results, not acted on mid-run
INFO = ("NOTE", "RESUME", "POOL", "TRUNCATIONS", "MIXED CODE", "GATES", "GATE OUTCOMES", "DIFFS", "PASS AUDIT")
# Labels inside transcripts (the desk, glint.inspect), not run-log lines
TRANSCRIPT = ("USER", "AGENT", "TOOL RESULT", "UNKNOWN")

def ere_escape(text: str) -> str:
    """Escape for grep -E: regex metacharacters only. re.escape also escapes spaces ("\\ "), which GNU grep 3.8 and
    later warn about as a stray backslash."""
    return re.sub(r"([.^$*+?(){}\[\]|\\])", r"\\\1", text)


MONITOR = "|".join([*(ere_escape(a) for a in ALERTS), *OUTCOMES, *MARKERS])


def covered(line: str) -> bool:
    return re.search(MONITOR, line) is not None


if __name__ == "__main__":
    print(MONITOR)
