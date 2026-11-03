"""How many gates rest on account records and how many on agent-made records (docs/gate-classes.md), and the live
refusals in gate logs split the same way.

The registry's counts are today's banking_knowledge gates (glint.gates.rules.build_registry). Each gate log
(`gates-<domain>.sqlite`) is opened read-only; its live denies are classed by today's registry (older manifests do not
carry the class). Within the agent-made class, every (gate, rule) is listed, because a gate in that class may also
refuse on the database (banking_replacement_gate's increase-first check, for example).

Usage: python scripts/gate_classes.py [GATE_LOG ...]
"""
import sys
from collections import Counter
from pathlib import Path

from glint.gates import summary
from glint.gates.core import AGENT_RECORDS, CHECKS
from glint.gates.rules import build_registry

DOMAIN = "banking_knowledge"


def registry_line(domain: str = DOMAIN) -> str:
    counts = build_registry(domain).checks_counts()
    return f"GATES {domain}: {sum(counts.values())} registered, " + ", ".join(f"{k} {v}" for k, v in counts.items())


def log_lines(label: str, events: list[dict], checks: dict[str, str]) -> tuple[list[str], Counter]:
    """The refusal line for one log, the agent-made class's (gate, rule) lines, and the counts by class."""
    by = summary.denies_by_checks(events, checks)
    totals = Counter({kind: sum(c.values()) for kind, c in by.items()})
    lines = [refusal_line(label, totals)]
    for (gate, rule), n in sorted(by.get(AGENT_RECORDS, Counter()).items()):
        lines.append(f"  {AGENT_RECORDS}  {n:3}  {gate}  {rule}")
    return lines, totals


def refusal_line(label: str, totals: Counter) -> str:
    kinds = [*CHECKS, *sorted(k for k in totals if k not in CHECKS)]
    return f"REFUSALS {label}: {sum(totals.values())} live, " + ", ".join(f"{k} {totals[k]}" for k in kinds)


def main(argv: list[str]) -> int:
    print(registry_line())
    checks = {g.name: g.checks for g in build_registry(DOMAIN).gates()}
    total = Counter()
    for path in map(Path, argv):
        lines, counts = log_lines(str(path), summary.read_log(path), checks)
        total += counts
        print("\n".join(lines))
    if len(argv) > 1:
        print(refusal_line("total", total))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
