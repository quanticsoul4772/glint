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

Install with the optional extra: `uv sync --extra llamafirewall` (or `pip install "glint[llamafirewall]"`). The
extra pulls in torch, transformers and codeshield. Do not install it into the run folder's `.venv`.

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

The scanner only reads. It records nothing, so a hold repeats until the call runs through the gated environment. Each
scan logs its `gate.fired` events like a live check, and a call the scanner allows is checked again when it runs
through the environment's wrapper.

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

Not verified:
- **A live LlamaFirewall run.** The package was never installed or imported, so this integration has not been run inside LlamaFirewall itself.
- **Blocking before execution in a host agent loop.** Whether a host stops a single tool call on the scanner's `BLOCK` depends on the host calling `LlamaFirewall.scan` before it runs the call. This is design §9's open question, and it remains open.
