"""Reconcile the two blind clause lists (agenda T8/T10, appendix #86) into data-public/clauses/reconciled.json.

pass_a.json and pass_b.json were written by two separate passes over the banking policy documents alone (tools.py for
tool and argument names), without reading any run, gold action or each other (pass A document-first in file-name
order, pass B tool-first in reverse tool order). glint.declare.reconcile sorts their rules into agreed (identical
encodings), conflicts (the same activities encoded differently: an ambiguity finding each, design §5) and single-pass
rules. The output also keeps both passes' own ambiguity notes, by clause. Deterministic; reads no run and no gold.

Usage: python scripts/reconcile_clauses.py [PASS_A PASS_B OUT]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from glint import declare

DIR = Path(__file__).resolve().parents[1] / "data-public" / "clauses"


def build(pass_a: Path, pass_b: Path) -> dict:
    a, b = declare.load(pass_a), declare.load(pass_b)
    r = declare.reconcile(a, b)
    notes = [{"id": c["id"], "doc": c["doc"], "ambiguity": c["ambiguity"]} for c in a + b if c["ambiguity"]]
    counts = {"pass_a": len(a), "pass_b": len(b), "agreed": len(r["agreed"]), "conflicts": len(r["conflicts"]),
              "a_only": len(r["a_only"]), "b_only": len(r["b_only"]),
              "agreed_checkable": sum(1 for c in r["agreed"] if c["checkable"])}
    return {"source": "two blind passes over the banking policy documents (no runs, no gold read)",
            "counts": counts, **r, "ambiguity_notes": notes}


def main(argv: list[str]) -> int:
    pa, pb, out = (Path(x) for x in argv[:3]) if len(argv) >= 3 else (DIR / "pass_a.json", DIR / "pass_b.json",
                                                                       DIR / "reconciled.json")
    data = build(pa, pb)
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"reconciled: {data['counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
