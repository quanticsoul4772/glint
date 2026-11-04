"""How many gates rest on account records and how many on agent-made records (docs/gate-classes.md), what kind of
check each gate is (a policy rule, a generated check, or no rule yet), and the live refusals in gate logs split the
same two ways.

The registry's counts are today's banking_knowledge gates (glint.gates.rules.build_registry). A gate's kind follows
its name (`gate_kind`): `no_rule_<tool>` allows every call and checks nothing; `procedure_duplicate_<tool>`,
`procedure_provenance_<tool>` (Glint conventions, no policy clause) and `constraints_<tool>` (the enums and bounds in
the tool's own docstring) are generated for each tool; every other gate is a policy rule. Each gate log
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
POLICY, GENERATED, NO_RULE = "policy rule", "generated check", "no rule"
KINDS = (POLICY, GENERATED, NO_RULE)
GENERATED_PREFIXES = {"procedure_duplicate_": "duplicate", "procedure_provenance_": "provenance",
                      "constraints_": "constraints"}


def gate_kind(name: str) -> tuple[str, str]:
    """(kind, sub-kind) of a gate by its name: KINDS, and for a generated check which generator made it."""
    if name.startswith("no_rule_"):
        return NO_RULE, ""
    for prefix, sub in GENERATED_PREFIXES.items():
        if name.startswith(prefix):
            return GENERATED, sub
    return POLICY, ""


def registry_line(domain: str = DOMAIN) -> str:
    counts = build_registry(domain).checks_counts()
    return f"GATES {domain}: {sum(counts.values())} registered, " + ", ".join(f"{k} {v}" for k, v in counts.items())


def kinds_line(domain: str = DOMAIN) -> str:
    """The registry's gates by kind: policy rules split by class, generated checks by generator, no-rule gates."""
    gates = build_registry(domain).gates()
    policy = Counter(g.checks for g in gates if gate_kind(g.name)[0] == POLICY)
    generated = Counter(gate_kind(g.name)[1] for g in gates if gate_kind(g.name)[0] == GENERATED)
    no_rule = sum(gate_kind(g.name)[0] == NO_RULE for g in gates)
    return (f"KINDS {domain}: {POLICY} {sum(policy.values())} ("
            + ", ".join(f"{k} {policy[k]}" for k in CHECKS) + f"), {GENERATED} {sum(generated.values())} ("
            + ", ".join(f"{s} {generated[s]}" for s in GENERATED_PREFIXES.values()) + f"), {NO_RULE} {no_rule}")


def log_lines(label: str, events: list[dict], checks: dict[str, str]) -> tuple[list[str], Counter, Counter]:
    """The refusal lines for one log (by class, then by kind), the agent-made class's (gate, rule) lines, and the
    counts by class and by kind."""
    by = summary.denies_by_checks(events, checks)
    totals = Counter({kind: sum(c.values()) for kind, c in by.items()})
    kinds = Counter()
    for cls, c in by.items():
        for (gate, _), n in c.items():  # a deny by no registered gate (unparsed arguments, a stage) stays "other"
            kinds[summary.OTHER_CHECKS if cls == summary.OTHER_CHECKS else gate_kind(gate)[0]] += n
    lines = [refusal_line(label, totals), kind_refusal_line(label, kinds)]
    for (gate, rule), n in sorted(by.get(AGENT_RECORDS, Counter()).items()):
        lines.append(f"  {AGENT_RECORDS}  {n:3}  {gate}  {rule}")
    return lines, totals, kinds


def refusal_line(label: str, totals: Counter) -> str:
    kinds = [*CHECKS, *sorted(k for k in totals if k not in CHECKS)]
    return f"REFUSALS {label}: {sum(totals.values())} live, " + ", ".join(f"{k} {totals[k]}" for k in kinds)


def kind_refusal_line(label: str, kinds: Counter) -> str:
    names = [*KINDS, *sorted(k for k in kinds if k not in KINDS)]
    return f"BY KIND {label}: " + ", ".join(f"{k} {kinds[k]}" for k in names)


def main(argv: list[str]) -> int:
    print(registry_line())
    print(kinds_line())
    checks = {g.name: g.checks for g in build_registry(DOMAIN).gates()}
    total, total_kinds = Counter(), Counter()
    for path in map(Path, argv):
        lines, counts, kinds = log_lines(str(path), summary.read_log(path), checks)
        total += counts
        total_kinds += kinds
        print("\n".join(lines))
    if len(argv) > 1:
        print(refusal_line("total", total))
        print(kind_refusal_line("total", total_kinds))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
