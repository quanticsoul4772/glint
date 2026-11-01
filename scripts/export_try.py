"""The desk's "Try the gate" data: the real credit-limit gates' decision on task_054's bank, after the customer is
verified, for every requested increase from $100 to $6,000 in $100 steps. Each decision comes from
`Enforcer.check` on the live τ² database (nothing is executed), so the page shows the gate's own words with no model
and no server. Written to src/glint/desk/try_credit_limit.json; `tests/test_desk.py` regenerates it and fails when
the committed file no longer matches the gates.

Run: python -X utf8 scripts/export_try.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src" / "glint" / "desk" / "try_credit_limit.json"
TASK = "task_054"
AMOUNTS = range(100, 6001, 100)


def build() -> dict:
    from glint.bank import server as bank
    from glint.gates import rules

    with tempfile.TemporaryDirectory() as home:
        srv = bank.BankServer(home=Path(home), sid="try")
        srv.load_scenario(None, TASK)
        task = bank.tasks()[TASK]
        verification = next(a.arguments for a in task.evaluation_criteria.actions if a.name == "log_verification")
        reply = srv.call(None, "log_verification", verification)
        if reply.is_error:
            raise SystemExit(f"verification failed: {reply.content[0].text}")
        s = srv.session()
        account = {"user_id": "584f9c5d00", "credit_card_account_id": "cc_584f9c5d00_gold"}
        rows = []
        for amount in AMOUNTS:
            result = s.enforcer.check(rules.SUBMIT, account | {"requested_increase_amount": amount},
                                      s.env.tools.db, toolkit=s.env.tools)
            rows.append({"amount": amount, "allowed": result.allowed, "rule": result.rule_id,
                         "message": "" if result.allowed else result.message(s.enforcer.wording)})
        s.events.close()
    return {"task_id": TASK, "tool": "submit_credit_limit_increase_request", "card": "Gold Rewards Card",
            "current_limit": 5000, "customer": "Sofia Papadopoulos (simulated τ²-bench customer)",
            "note": ("Each decision is the gates' own output on task_054's τ²-bench database after the customer is "
                     "verified, computed by scripts/export_try.py; no write was executed."),
            "decisions": rows}


def main() -> int:
    OUT.write_text(json.dumps(build(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
