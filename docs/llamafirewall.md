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

Not verified:
- **Blocking before execution in a host agent loop.** Whether a host stops a single tool call on the scanner's `BLOCK` depends on the host calling `LlamaFirewall.scan` before it runs the call. This is design §9's open question, and it remains open.
