# Glint's gates as a LlamaFirewall scanner

`glint.integrations.llamafirewall_scanner` puts Glint's gate decision inside Meta's LlamaFirewall (version 1.0.3) as
a custom scanner on assistant messages, beside Prompt Guard 2 and AlignmentCheck (design 2026-11-02 §9).

## Use

```python
from llamafirewall import LlamaFirewall, Role
from glint.gates import Enforcer, EventLog
from glint.integrations.llamafirewall_scanner import make_scanner

enforcer = Enforcer("banking_knowledge", EventLog("gates.jsonl"))
enforcer.attach(env)              # the gates read the records this wrapper keeps on env.tools
make_scanner(enforcer, env)       # registers the scanner as "glint_gates"
firewall = LlamaFirewall({Role.ASSISTANT: ["glint_gates"]})
result = firewall.scan(assistant_message)   # an AssistantMessage carrying tool_calls
```

Install `llamafirewall==1.0.3` into a separate virtual environment, never the run folder's `.venv`; it pulls in torch,
transformers, codeshield and semgrep. Two known problems, found in the 2026-11-03 run:
- **The run used Glint from source (`PYTHONPATH=<repo>/src`).** An installed copy then failed on the first `Enforcer` because it read `data-public/clauses/reconciled.json` from the source tree. The clause lists now ship inside the package (`src/glint/clauses/`), and CI installs the wheel into a fresh venv and builds an `Enforcer`.
- **Restore `mcp==2.3.0` after installing llamafirewall.** semgrep pins `mcp` 1.29.0, which Glint's bank server cannot use; the `glint[llamafirewall]` extra has the same conflict.

## Mapping

| Glint gate result | LlamaFirewall `ScanDecision` | Reason |
|---|---|---|
| Refusal (any deny) | `BLOCK`, score 1.0 | the refusal the agent would get: rule, what the check found, facts, why it matters, next step |
| Hold: `banking.replacement.expedite_ask` or `banking.replacement.address_ask` | `HUMAN_IN_THE_LOOP_REQUIRED`, score 1.0 | the same refusal message |
| Allow, or a call that is not a gated write | `ALLOW`, score 0.0 | the allowing rule |

- In a message with several tool calls, the first `BLOCK` wins, then a hold, then `ALLOW`. Each call is judged on the database as it is when the message is scanned.
- A tool call whose arguments do not parse to a JSON object is a `BLOCK`.
- Both the OpenAI tool-call shape and a flat `{"name", "arguments"}` shape are read, because LlamaFirewall does not fix the shape of `tool_calls`.
- Treating the two ask rules as holds is a Glint convention. Each refuses an order once so the agent asks the customer first (policy `credit_card_replacements_001`).

The scanner only reads. It records nothing (since PR #43 it logs no `gate.fired` or `write.attested` events), so a
hold repeats until the call runs through the gated environment, and a call the scanner allows is checked and recorded
when it runs through the environment's wrapper.

## What was verified and what was not

Verified:
- **Unit tests with fakes** (`tests/test_llamafirewall_scanner.py`, no llamafirewall installed):
  - the decision mapping;
  - message-level precedence;
  - tool-call parsing;
  - the scanner class, built against a fake `llamafirewall` module that has 1.0.3's names and shapes: `Scanner(scanner_name)`, `async scan(message, past_trace)`, `ScanResult(decision, reason, score)`, `ScanDecision.HUMAN_IN_THE_LOOP_REQUIRED`, and `register_llamafirewall_scanner(name)`. Those names were read from the 1.0.3 wheel's source.
- **One real `Enforcer`** on τ² banking training task_081:
  - a freeze before verification is `BLOCK`;
  - after gold's steps up to the freeze it is `ALLOW`;
  - the card is unchanged either way.

- **A live LlamaFirewall run (2026-11-03, CPU, llamafirewall 1.0.3, Glint at 76f56df).** LlamaFirewall was configured with only the Glint scanner (no PromptGuard or AlignmentCheck model loaded) on τ² banking training task_054:
  - a tool call submitting a $4,000 credit-limit increase was `BLOCK`, score 1.0, with Glint's refusal (`banking.cli.submit.reduce_amount`: the premium-tier maximum is $2,500);
  - the same call for $2,500 was `ALLOW`, score 0.0;
  - the database's credit-limit requests were the same before and after, so neither scan ran the write.

## Recorded Glimmer tool calls through LlamaFirewall (2026-11-03)

`scripts/lf_replay.py` put every recorded tool call of the Glint arm in the four gated banking sets (v9, the unseen set
from its lean copy, v10, the pressure set; 314 runs that count as results) through LlamaFirewall 1.0.3 with only the
`glint_gates` scanner. It ran on CPU with no model and no network, in a separate venv with Glint from source (this
branch, off `6502f11`). Each run was replayed in a fresh environment at the task's initial state, with the tools
arm's decision tools and a current-code Enforcer (stages off). Every call was sent in its recorded order. Before each
gated assistant call ran, LlamaFirewall scanned an `AssistantMessage` carrying that recorded call, so the call was
judged on the database as it was just before it ran live. The database followed the recorded run: a call the live
gate allowed was run, and a call it refused was not.

The live decision is read from the recorded tool reply: a gate refusal is a deny, or a hold under the two ask rules;
anything else is an allow.

| | Gated calls | Agree with the live gate | Writes agree |
|---|---|---|---|
| Current gates | 1,082 (964 writes, 118 transfers or give-tool calls) | 828 (76.5%) | 711 of 964 |
| Current gates, request gate off (`--no-request-gate`) | 1,082 | 995 (92.0%) | 878 of 964 |

- **Every live refusal and hold was matched.** All 90 live denies were `BLOCK` and all 23 live holds were
  `HUMAN_IN_THE_LOOP_REQUIRED`, both scores 1.0. That is the 113 live refusals of the four-set evaluation (109 write
  calls, 4 transfers). Every live `ALLOW` that LlamaFirewall also allowed had score 0.0.
- **Every disagreement is a live allow that LlamaFirewall blocks**, under a rule added or tightened after the runs
  were recorded. No live refusal was allowed. With the current gates there are 254 disagreements in 98 runs of 25
  tasks (v9 8, unseen 82, v10 90, pressure 74):

  | Rule (current code) | Calls |
  |---|---|
  | `banking.request.no_record`: the request gate (research row 6). None of the four sets had it or its `record_requests` tool live | 171 |
  | `banking.savings_credit.amount` | 16 |
  | `banking.retention.no_answer` | 15 |
  | `banking.interest_report.expected_apy` | 12 |
  | `banking.checking_credit.amount_unmatched` | 7 |
  | `debit.dispute.liability_zero` | 6 |
  | `debit.dispute.duplicate.earliest` | 6 |
  | `banking.interest_report.amount` | 5 |
  | `debit.dispute.open_limit` | 4 |
  | `debit.dispute.provisional_credit` | 3 |
  | `banking.verification.once` | 3 |
  | `banking.closure_reason.already_on_record` | 3 |
  | `debit.order.entry_wait` | 2 |
  | `glint.duplicate_write` (a repeated `give_discoverable_user_tool`) | 1 |

  With the request gate off there are 87 disagreements in 52 runs of 15 tasks (v9 1, unseen 30, v10 27, pressure
  29): the same 83 calls under the other rules, plus 4 `constraints.value` refusals that the request gate had
  refused first. The script prints each disagreement with its set, task, trial, message index, tool and both rules.
- **The scanner path matches the gated environment.** On all 1,082 calls, LlamaFirewall's decision equals the
  decision the gated environment made when the same call was sent through it.
- **The replayed state matched the recording.** Of 18,316 other tool calls, 0 replies differ from the recorded
  reply in error status. Of the live-allowed gated calls, 0 differ either. No call the live gate refused ran in the
  replay. 254 live-allowed calls that the current gates refuse (87 with the request gate off) were run through τ²'s
  own tool instead, so the database kept following the recording.

What this shows and what it does not:
- Real Glimmer tool calls, 1,082 of them, went through `LlamaFirewall.scan`, and on every recorded refusal LlamaFirewall
  returned the decision the live gate made. The live-allow disagreements measure rule changes since the runs, not the
  scanner.
- These are scans of recorded calls. No agent loop acted on the decisions, and the host-loop question below stays
  open.

Not verified:
- **Blocking before execution in a host agent loop.** Whether a host stops a single tool call on the scanner's `BLOCK` depends on the host calling `LlamaFirewall.scan` before it runs the call. This is design §9's open question, and it remains open.
