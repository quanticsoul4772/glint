"""Review one finished run log (run-ab step 7): line kinds by count, the error-like lines the monitor's pattern
(scripts/alert_patterns.py) does not match, and counts of the noise earlier fixes removed, which should stay 0.

A missed line is a candidate for alert_patterns.ALERTS; a frequent line kind nobody acts on is a candidate for the
run log's filters. Reads the log only.

Usage: python scripts/log_review.py LOG
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import alert_patterns

# words that mark a line as possibly an error or a warning; a line matching the monitor or a plain PASS/FAIL run line
# is not reported
SUSPECT = re.compile(r"error|warn|exception|traceback|refused|denied|timeout|timed out|retry|abort|killed|cannot|"
                     r"could not|unable|invalid|critical|fatal|lost|missing|not found|\b5\d\d\b", re.IGNORECASE)
# the summaries alert_patterns classifies as INFO (read after the run), by their prefix
INFO_LINE = re.compile(r"^(?:\[\w+ [^\]]*\] )?(?:" + "|".join(map(re.escape, alert_patterns.INFO)) + r"):")
RUN_LINE = re.compile(r"\] task_\d+ trial \d+: (PASS|FAIL) \(")
JSON_KEY = re.compile(r'^\s+"\w+": ')
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# noise removed on 2026-10-29: each should count 0 in a run log
GONE = {
    "LiteLLM 'Provider List' blocks": "Provider List",
    "τ² 'Running task' start lines": ". Running task ",
    "per-poll 'totals {' lines": "] totals {",
    "τ²'s cause-free 'Simulation loop exited' warnings": "emergency cleanup",
    "run lines claiming '0 s, 0 msgs'": "0 s, 0 msgs",
    "τ² registry dumps": "Registering default components",
    "'Failed to get git hash' errors": "Failed to get git hash",
}


def kind(line: str) -> str:
    """A line's kind: its source and first words, with numbers folded."""
    if not line.strip():
        return "<blank>"
    if "[vllm]" in line:
        return "[serve] [vllm]"
    m = re.match(r"\d{4}-\d\d-\d\d [\d:.]+ \| (\w+)\s*\| ([\w.]+:\w+:\d+)", line)
    if m:
        return f"loguru {m.group(1)} {m.group(2)}"
    m = re.match(r"\[(\w+) [\d:]+(?: \w+ \w+)?\] (\S+(?: \S+)?)", line)
    if m:
        return f"[{m.group(1)}] " + re.sub(r"task_\d+ trial", "task trial", m.group(2))
    m = re.match(r"([A-Z][A-Z ]{2,}?):", line)
    return f"{m.group(1)}:" if m else re.sub(r"\d+", "#", line)[:60]


def review(text: str) -> dict:
    lines = [ANSI.sub("", x) for x in text.splitlines()]
    # the by-task report tools_ab prints at an arm's end is JSON ("infra_errors": 0): data, not error lines
    missed = [x for x in lines if SUSPECT.search(x) and not alert_patterns.covered(x) and not RUN_LINE.search(x)
              and not INFO_LINE.match(x) and not JSON_KEY.match(x)]
    return {"lines": len(lines), "kinds": Counter(map(kind, lines)),
            "missed": Counter(re.sub(r"\d+", "#", x)[:150] for x in missed),
            "gone": {label: sum(needle in x for x in lines) for label, needle in GONE.items()},
            "status_lines": sum(x.startswith("Status:") for x in lines)}


def render(r: dict) -> str:
    out = [f"{r['lines']} lines", "line kinds:"]
    out += [f"  {n:5d}  {k}" for k, n in r["kinds"].most_common(25)]
    out.append(f"error-like lines the monitor does not match: {sum(r['missed'].values())}")
    out += [f"  {n:5d}  {k}" for k, n in r["missed"].most_common(25)]
    out.append("noise that should stay 0:")
    out += [f"  {'ok  ' if n == 0 else 'BACK'} {label}: {n}" for label, n in r["gone"].items()]
    out.append(f"τ² Status lines (one per 10 minutes expected): {r['status_lines']}")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    r = review(Path(argv[0]).read_text(encoding="utf-8", errors="replace"))
    print(render(r))
    return 1 if any(r["gone"].values()) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
