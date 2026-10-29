"""Every uppercase prefix Glint prints is classified in scripts/alert_patterns.py, and every alert matches MONITOR."""
import re
from pathlib import Path

import alert_patterns as ap

ROOT = Path(__file__).resolve().parents[1]
PREFIX = re.compile(r"""f?["']([A-Z][A-Z _]{2,}[A-Z]):""")


def printed_prefixes() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted([*(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").glob("*.py")]):
        if path.name == "alert_patterns.py":
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for m in PREFIX.finditer(line):
                found.setdefault(m.group(1), []).append(f"{path.relative_to(ROOT)}:{n}")
    return found


def test_every_printed_prefix_is_classified():
    known = {*ap.ALERTS, *ap.INFO, *ap.TRANSCRIPT}
    unknown = {p: where[:3] for p, where in printed_prefixes().items() if p not in known}
    assert not unknown, ("new uppercase prefixes; add each to ALERTS (the monitor must see it while the run goes), "
                         f"INFO or TRANSCRIPT in scripts/alert_patterns.py: {unknown}")


def test_every_alert_and_outcome_matches_the_monitor_and_info_lines_do_not():
    for alert in ap.ALERTS:
        assert ap.covered(f"{alert}: something"), alert
    for kind in ap.OUTCOMES:
        assert ap.covered(f"[runs 10:00:00 tools banking] task_1 trial 0: {kind} (τ²: x)"), kind
    lines = {
        "[serve 10:14:54] FAILED: glint-serve still has running containers after 5 stop attempts: [...]": True,
        "EMPTY REPLY: task_039 trial 1: finish_reason stop, prompt 9 tokens, completion 0": True,
        "TRUNCATED: task_039 trial 1 (max_tokens 8192)": True,
        "METER FAILED: metering 'x' failed: KeyError": True,
        "[runs 09:24:09 tools b] task_081 trial 0: FAIL (τ²: too_many_errors, 459 s, 62 msgs; so far fail 1)": True,
        "[runs 09:25:10 tools b] task_031 trial 0: PASS (τ²: user_stop, 509 s, 58 msgs; so far fail 2, pass 2)": False,
        "[runs 09:24:09 tools b] task_052 trial 0: FAIL (τ²: user_stop, 466 s, 56 msgs; so far fail 2, pass 1)": False,
        "POOL: 2725 requests, 4 waited over 30 s, longest 146.9 s": False,
        "NOTE: tools banking: 2 runs beyond k=4 left out": False,
    }
    for line, alert in lines.items():
        assert ap.covered(line) is alert, line


def test_the_script_prints_the_pattern(capsys):
    import runpy

    runpy.run_path(str(ROOT / "scripts" / "alert_patterns.py"), run_name="__main__")
    assert capsys.readouterr().out.rstrip("\r\n") == ap.MONITOR


def test_the_pattern_has_no_escaped_spaces_and_grep_accepts_it():
    import shutil
    import subprocess

    assert r"\ " not in ap.MONITOR and ap.ere_escape("a.b (c)") == r"a\.b \(c\)"
    grep = shutil.which("grep")
    if grep is None:
        return
    lines = "METER FAILED: x\nFAIL FAST: y\n[runs] task_1 trial 0: PASS (τ²: user_stop)\nexit 0\n"
    r = subprocess.run([grep, "-E", ap.MONITOR], input=lines, capture_output=True, text=True, encoding="utf-8",
                       check=False)
    assert r.stdout.splitlines() == ["METER FAILED: x", "FAIL FAST: y", "exit 0"] and "warning" not in r.stderr
