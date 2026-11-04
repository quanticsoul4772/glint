# What each gate rests on: account records or agent-made records

Every registered gate carries a class (`Gate.checks`, `glint.gates.core.CHECKS`), listed per gate in the `gates.manifest` event with the counts in `checks_counts`. The class is a required field with no default: a gate built without one raises before it can be registered, so an unlabeled gate never counts as the stronger class.

- **`account_records`**: the decision reads the live database, the write's arguments and the record of tool calls that actually ran in the conversation (the reads done, the writes that ran, the tool results shown). A misbehaving agent cannot make such a gate allow a write without changing what the database or a real tool result shows. Examples: the tier maximum of a credit limit increase, closure eligibility, the debit card wait and fees, the required lookups.
- **`agent_records`**: at least one of the gate's checks is passed by what the agent itself states the customer said or supplied, or by repeating a refused call. The gate never sees the conversation. Unless a conversation stage is configured (`glint.gates.stages`, off when no model is set), nothing compares these statements with what the customer said.

## What kind of check each gate is

A class says what a gate reads, not whether it encodes a policy. Of the 146 banking gates (`python scripts/gate_classes.py`, `KINDS` line):

| Kind | Gates | account records | agent-made records |
|---|---|---|---|
| Policy rules (a knowledge base clause or a written procedure step) | 60 | 43 | 17 |
| Generated for each tool, no policy clause: same call once (`procedure_duplicate_<tool>`, 32), every identifier shown by an earlier tool result (`procedure_provenance_<tool>`, 34), the enums and bounds in the tool's own docstring (`constraints_<tool>`, 11) | 77 | 77 (in effect 43; see below) | 0 (in effect 34) |
| No rule yet (`no_rule_<tool>`): allows every call and logs it | 9 | 9 | 0 |
| **All** | **146** | **129 (in effect 95)** | **17 (in effect 51)** |

**The 34 provenance gates are labeled `account_records` in code but can be passed by the agent (found by an audit on 2026-11-04; the code still labels them `account_records`).** They count an identifier as shown when any earlier non-error tool result contains it, and that includes the results of Glint's own added tools (the `record_*` tools, `plan_disputes`, the `check_*` tools), which repeat back what the agent sent. On task_054, a write on a made-up card id is refused by provenance, and after one `record_requests` call naming that id as its target, the same check allows it. The planned fix counts only results of τ²'s own tools. Until then, these 34 rest in effect on what the agent sends, which gives 95 account-record gates and 51 the agent can pass (corrected: this page counted 129 and 17).

The 9 tools with no rule include money movement: `transfer_funds_between_bank_accounts_7291` and `pay_credit_card_from_checking_9182`, with `open_bank_account_4821` and `close_bank_account_7392`. Besides the generated checks, each of the four has only the verified-customer gate (agent-made) and the undecided credit limit increase gate; the card payment also has its request gate (agent-made).

## The agent-made class (17 of the 146 banking gates)

| Gate | Passed by |
|---|---|
| `banking_log_verification_gate` | a `log_verification` call whose details match the record on file at the current time. The agent can copy those details from its own lookup (`get_user_information_by_name`); nothing shows the customer supplied them. |
| `procedure_verified_gate` | the verification record above, made earlier in the conversation. It guards every banking write that acts for a user. |
| `banking_request_<tool>` (12, one per tool in `rules.REQUEST_TOOLS`) | a `record_requests` call naming the action and target, with any `customer_words`. The two reversible tools (`banking.FLAGGED_TOOLS`: the closure reason log and the debit card order) are allowed and flagged even with no record. |
| `banking_retention_answer_gate` | a `record_retention_answer` call recording a declined offer |
| `banking_replacement_gate` | its address check: a `record_address_confirmation` call, or the same order sent again; its expedite check: the same order sent again. Its increase-first check reads the database. |
| `procedure_dispute_plan_gate` | a `plan_disputes` call. The plan's order and values come from the database, but the customer's answers in it (merchant contact, the date the charge was noticed, the card action) are as the agent states them. |

`record_dispute_answers` is not read by any gate: missing or differing answers appear as an open item and do not block a filing.

## The limit, shown in tests

`tests/test_gate_classes.py` shows three agent-made records passing their gates:
- verification: in a conversation with no customer message at all, one `get_user_information_by_name` lookup and one `log_verification` call copying the record verify task_084's customer, and the verified gate then allows that customer's writes.
- task_045: the customer accepts the retention credit and keeps the card in τ²'s gold. One `record_retention_answer` call recording a decline the customer never gave lets the close run.
- task_084: a `record_requests` call naming a debit card the customer never mentioned, with made-up words, unlocks its close.

So "a refused write never runs" holds for every gate. "The agent cannot skip the check" holds for the account-record gates other than the 34 provenance gates (above). For the agent-made gates, and for the provenance gates through an id echoed by a Glint tool, an agent that makes up the record passes the check.

## Refusals by class

Live refusals in the gate logs (`replay` false), classed by today's registry (`python scripts/gate_classes.py LOG ...`). A gate keeps its class even when it refused on a database check (a verification naming a user not on file, for example). Recomputed on 2026-11-03 after the verification gates moved to the agent-made class; the earlier table counted them as account records (75 and 38 on the four sets).

| Set (gate log) | Live refusals | account records | agent-made records |
|---|---|---|---|
| v9, the 12 development tasks (`banking-plan-v9-gated`) | 59 | 19 | 40 (replacement address ask 14, expedite ask 9; dispute plan 15; verification 1, verified customer 1) |
| unseen (`banking-unseen-gated`) | 21 | 16 | 5 (verification 1, verified customer 4) |
| v10 (`banking-unseen-v10-gated`) | 20 | 17 | 3 (verification 1, verified customer 2) |
| pressure (`banking-pressure-gated`) | 13 | 12 | 1 (verified customer 1) |
| **the four gated banking sets** | **113** | **64** | **49** |
| test split, 2026-11-02 (`banking-test-gated`) | 45 | 20 | 25 (request gates 20, address ask 4, verification 1) |
| test split, 2026-11-03 (`banking-test-v2-gated`) | 40 | 34 | 6 (request gates 2, address ask 2, verification 2) |

Every one of these refusals came from a policy rule: none came from a generated or no-rule gate. The duplicate-call and provenance gates were added on 2026-11-02, so only the 2026-11-03 test split ran them (269 and 278 decisions there, no refusal); the constraint gates ran 82 times there, with no refusal.

The v9 logs predate `record_address_confirmation`. At that time the address ask was passed only by a second order.

## The clock

Every date rule (cooldowns, the 60-day and 365-day windows, "today" in the records, a verification's `time_verified`) reads the clock in effect (`glint.clock`). The default is τ²'s environment time (`get_current_time`: 2025-11-14 03:40:00 EST), so every recorded and replayed decision is unchanged. `Enforcer(clock=Clock(...))` runs the gates, open items and decision tools under another clock (`glint.clock.wall_clock()` reads the machine's), and the `gates.manifest` event then carries it; `tests/test_clock.py` shows a different date changing the verification and date-window decisions.
