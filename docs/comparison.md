# How Glint compares in detail

Moved here from the README on 2026-11-03, unchanged except for link paths.

Each cell is from the project's own documentation (checked 2026-11-01; Invariant's guardrail docs were removed on 2026-01-12, so its row uses their last published version). "Not documented" means the docs do not say, not that the feature is absent.

| Project | Mainly guards against | Decides with | Reads the application's current data at decision time | The agent gets on a block | Tamper-evident decision log |
|---|---|---|---|---|---|
| [AWS AgentCore Policy](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/policy.md) | Unauthorized agent tool calls | Deterministic Cedar rules (optional guardrail scores) | The request, identity claims and session history; application data not documented | An error naming the policy denial | Not documented (CloudWatch logs) |
| [Invariant Guardrails](https://github.com/invariantlabs-ai/invariant) | Prompt injection, unsafe tool data flows | Rules plus classifier and LLM detectors | The trace (messages, tool calls, outputs) | The violated rule and where | Not documented |
| [NVIDIA NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails) | Jailbreaks, unsafe or off-topic conversation | Colang flows plus LLM and classifier checks | Through custom Python actions | A refusal message (configurable) | Not documented |
| [Meta LlamaFirewall](https://github.com/meta-llama/PurpleLlama/tree/main/LlamaFirewall) | Prompt injection, goal hijacking, insecure code | Classifier, LLM auditor, static analysis | Messages and the agent's trace | A decision, a reason and a score | Not documented |
| **Glint** | Writes that break a business's written policy | Deterministic Python functions | Yes: τ²'s live database plus the call's arguments | The rule, the facts, the consequence and the next step; the write did not run | Hash chain; HMAC-keyed with `GLINT_LOG_KEY` |

The others mostly decide whether a conversation or a call is safe; Glint decides whether a specific write is allowed by the business's policy given the account's current records, and tells the agent how to fix it. It complements them rather than replacing them: a prompt-injection scanner and a policy gate catch different failures.
