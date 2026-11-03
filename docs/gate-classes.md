# What each gate rests on: account records or agent-made records

Every registered gate carries a class (`Gate.checks`, `glint.gates.core.CHECKS`), listed per gate in the `gates.manifest` event with the counts in `checks_counts`.

- **`account_records`**: the decision reads the live database, the write's arguments and the record of tool calls that actually ran in the conversation (the reads done, the writes that ran, the tool results shown). A misbehaving agent cannot make such a gate allow a write without changing what the database or a real tool result shows. Examples: the tier maximum of a credit limit increase, closure eligibility, the debit card wait and fees, the required lookups, the verified customer.
- **`agent_records`**: at least one of the gate's checks is passed by what the agent itself states the customer said, or by repeating a refused call. The gate never sees the conversation. Unless a conversation stage is configured (`glint.gates.stages`, off when no model is set), nothing compares these statements with what the customer said.

## The agent-made class (15 of the 146 banking gates)

| Gate | Passed by |
|---|---|
| `banking_request_<tool>` (12, one per tool in `rules.REQUEST_TOOLS`) | a `record_requests` call naming the action and target, with any `customer_words`. The two reversible tools (`banking.FLAGGED_TOOLS`: the closure reason log and the debit card order) are allowed and flagged even with no record. |
| `banking_retention_answer_gate` | a `record_retention_answer` call recording a declined offer |
| `banking_replacement_gate` | its address check: a `record_address_confirmation` call, or the same order sent again; its expedite check: the same order sent again. Its increase-first check reads the database. |
| `procedure_dispute_plan_gate` | a `plan_disputes` call. The plan's order and values come from the database, but the customer's answers in it (merchant contact, the date the charge was noticed, the card action) are as the agent states them. |

`record_dispute_answers` is not read by any gate: missing or differing answers appear as an open item and do not block a filing.

## The limit, shown in tests

`tests/test_gate_classes.py` shows two writes that run on made-up records:
- task_045: the customer accepts the retention credit and keeps the card in τ²'s gold. One `record_retention_answer` call recording a decline the customer never gave lets the close run.
- task_084: a `record_requests` call naming a debit card the customer never mentioned, with made-up words, unlocks its close.

So "a refused write never runs" holds for every gate. "The agent cannot skip the check" holds for the account-record gates. For the agent-made gates, an agent that makes up the record passes the check.

## Refusals by class

Live refusals in the gate logs (`replay` false), classed by today's registry (`python scripts/gate_classes.py LOG ...`). A gate keeps its class even when it refused on a database check.

| Set (gate log) | Live refusals | account records | agent-made records |
|---|---|---|---|
| v9, the 12 development tasks (`banking-plan-v9-gated`) | 59 | 21 | 38 (replacement address ask 14, expedite ask 9; dispute plan 15) |
| unseen (`banking-unseen-gated`) | 21 | 21 | 0 |
| v10 (`banking-unseen-v10-gated`) | 20 | 20 | 0 |
| pressure (`banking-pressure-gated`) | 13 | 13 | 0 |
| **the four gated banking sets** | **113** | **75** | **38** |
| test split, 2026-11-02 (`banking-test-gated`) | 45 | 21 | 24 (request gates 20, address ask 4) |
| test split, 2026-11-03 (`banking-test-v2-gated`) | 40 | 36 | 4 (request gates 2, address ask 2) |

The v9 logs predate `record_address_confirmation`. At that time the address ask was passed only by a second order.
