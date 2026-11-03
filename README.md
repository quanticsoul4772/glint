# Glint

**For banks letting AI agents write to accounts: the policy firewall between the agent and the account.**

**The model proposes. The policy decides. A refused write never runs.**

[![tests](https://github.com/quanticsoul4772/glint/actions/workflows/tests.yml/badge.svg)](https://github.com/quanticsoul4772/glint/actions/workflows/tests.yml)
[![coverage 97% (96.86%, PR #45 run)](https://img.shields.io/badge/coverage-97%25-brightgreen)](docs/testing-strategy.md "coverage 97% (96.86%, PR #45 run)")
[![tests: 909 passed (local, 2026-11-03)](https://img.shields.io/badge/tests-909%20passed%20%28local%2C%202026--11--03%29-brightgreen)](docs/testing-strategy.md)
[![license MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

![Glint blocks a $4,000 credit-limit increase, returns the fix, and the $2,500 request runs](docs/demo.gif)

**[Watch (2:43)](https://youtu.be/jP8PEzjAamE)** · **[Try the gate (no sign-in)](https://quanticsoul4772--glint-desk-web.us-east.modal.run)** · **[Replay without GPU](https://quanticsoul4772.github.io/glint/)**

Glint is write authorization for AI agents that touch money. Every account change the agent attempts goes through a gate before it runs; where the bank's written policy has a rule, the gate applies it as code to the live account records. An allowed write runs. A refused write does not run, and the agent gets the rule, the facts and the fix. Every decision lands in a hash-chained, tamper-evident log. Built on Meta's open-weight Muse Glimmer 30B, Muse Spark and Muse Code.

**Built during the hackathon (Oct 26 – Nov 4, 2026):** all of Glint's code, written from the first commit `952dd30` (2026-10-26; history rewritten Oct 31 with authors and dates kept). τ²-bench data and Astryx's theme scales are third-party; see [Built during the event](#built-during-the-event). **Best Build on Meta Open Weight Models:** the agent is Muse Glimmer 30B, Meta open weights, self-hosted on Modal with vLLM; Glint also runs inside Meta's LlamaFirewall 1.0.3 as a custom scanner (scanner-level test on 2026-11-03: a $4,000 credit-limit increase blocked, $2,500 allowed; a host agent loop stopping the call was not tested).

## Why it matters
- **Agents are getting payment access.** Visa (Intelligent Commerce, April 30, 2025) and Mastercard (Agent Pay, April 29, 2025) launched programs that let AI agents pay ([Retail TouchPoints](https://www.retailtouchpoints.com/topics/payments/visa-mastercard-paypal-dive-into-agentic-era-with-tools-that-help-consumers-use-ai-to-buy)). 64% of larger US banks were testing or deploying agentic AI, against 38% of smaller ones (SoundHound-commissioned Arizent/American Banker survey, n=201, [Nasdaq/GlobeNewswire](https://www.nasdaq.com/press-release/research-finds-two-three-us-banking-executives-believe-agentic-ai-will-change-how)).
- **Risk controls decide which projects survive.** Gartner predicts over 40% of agentic AI projects will be canceled by the end of 2027, citing costs, unclear value or "inadequate risk controls" ([Gartner](https://www.gartner.com/en/newsroom/press-releases/2025-06-25-gartner-predicts-over-40-percent-of-agentic-ai-projects-will-be-canceled-by-end-of-2027)).
- **Automated account actions are already an enforcement target (a non-AI parallel).** The CFPB's 2022 order against Wells Fargo, more than $2 billion in redress plus a $1.7 billion penalty (over $3.7 billion in all), cited account freezes in "more than 1 million instances" driven by "a faulty automated filter" ([CFPB](https://www.consumerfinance.gov/enforcement/actions/wells-fargo-bank-na-2022/)).
- **Wrong account actions are expensive at scale.** Chargebacks: 261 million worth $33.8 billion in 2025 (Mastercard and Datos Insights estimate, [Mastercard](https://www.mastercard.com/content/dam/mccom/shared/news-and-trends/insights/2025/2025-global-chargebacks-outlook/pdf/2025-state-of-chargebacks-report.pdf)).

## What makes it different
1. **Deterministic enforcement of written business policy, before the write.**
2. **Refusals carry the rule, the facts and the fix, so the agent can recover.** This is not described in competitors' docs ([AgentCore Policy](https://aws.github.io/bedrock-agentcore-starter-toolkit/user-guide/policy/overview.html) describes allow/deny; [ToolGuard](https://pypi.org/project/toolguard/0.1.19/) describes raising alerts).
3. **The refusal and its hash-chained, tamper-evident record come from one gate.**
4. **The first policy layer found measured with and without on τ²-bench banking** (ToolGuard's paper evaluates the airline domain only, [arXiv 2507.16459](https://arxiv.org/html/2507.16459v1)).
5. **Built on Meta's open-weight stack.** It runs inside Meta's LlamaFirewall 1.0.3 as a custom scanner (scanner-level test; [docs/llamafirewall.md](docs/llamafirewall.md)).

| Approach | What it checks | Deterministic? | Explains the fix? | Tamper-evident record? |
|---|---|---|---|---|
| Content filters ([Llama Guard 4](https://build.nvidia.com/meta/llama-guard-4-12b/modelcard), [Lakera Guard](https://www.checkpoint.com/press-releases/check-point-acquires-lakera-to-deliver-end-to-end-ai-security-for-enterprises/), [Model Armor](https://docs.cloud.google.com/model-armor/help/configure-model-armor-gateway)) | Unsafe content, prompt injection, sensitive data in prompts, responses and tool payloads | No: model or classifier | Not described | Not described |
| Generic authorization ([Cedar in AgentCore Policy](https://aws.amazon.com/about-aws/whats-new/2026/03/policy-amazon-bedrock-agentcore-generally-available/), [OPA](https://www.openpolicyagent.org/docs/management-decision-logs), [Cerbos](https://cerbos.dev/features-benefits-and-use-cases/ai-security)) | Whether an identity may call a tool or action, optionally with conditions on its inputs | Yes | Not described (allow or deny) | Decision logs; hash-chaining not described |
| Model judges ([Microsoft Task Adherence](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/quickstart-task-adherence), [LlamaFirewall AlignmentCheck](https://arxiv.org/abs/2505.03574v1)) | Whether a proposed call or the agent's reasoning fits its task | No: model | Not described | Not described |
| **Glint** | Each write against the bank's written policy and the live account records | **Yes:** plain Python functions | **Yes:** the rule, the facts and the next step | **Yes:** hash chain, HMAC-keyed with `GLINT_LOG_KEY` |

Glint complements content filters rather than replacing them: a prompt-injection scanner and a policy gate catch different failures. A per-project comparison is in [How it compares in detail](#how-it-compares-in-detail).

## Results
| What | Result | Label and source |
|---|---|---|
| Enforcement by construction | Every one of **964 write calls** in four gated sets went through a gate; **89 of 89** refusals in replayed runs were refused again | Measured on Glimmer's runs in the four gated banking sets (v9, unseen, v10, pressure); holds by construction for any model behind the gate ([details](docs/results.md#live-gate-decisions-across-the-four-gated-banking-sets)) |
| Precision | **1 false refusal in 407 correct write calls** on 59 tasks no gate rule came from: 0.25%, upper bound 1.2% (one-sided 95%); 4.9% counting every hold as a cost | Untuned tasks, judged against τ²'s expected writes ([details](docs/results.md#live-gate-decisions-across-the-four-gated-banking-sets)) |
| Banking test split after the fix research | **11 of 40** runs passed (gated arm, 2026-11-03) against **7 of 40** for both arms (2026-11-02); by task p = 0.125 against the earlier gated run and p = 0.25 against alone, by task, sign test, two-sided. Against τ²'s offline re-grade of the alone arm (10/40): 3 better, 1 worse, p = 0.63. | **Tuned, in-sample, cross-day; the alone arm was not rerun; may be chance** ([details](docs/results.md#banking-test-split-gated-arm-after-the-fix-research-2026-11-03)) |

No outcome gain on unseen tasks has been shown; see [Limits](#limits-each-answered) and [What was measured](#what-was-measured).

## Quickstart
Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and git. τ²-bench is about 1.4 GB.

```bash
git clone -c core.longpaths=true https://github.com/quanticsoul4772/glint
git clone -c core.longpaths=true -c core.autocrlf=false https://github.com/sierra-research/tau2-bench vendor/tau2-bench
git -C vendor/tau2-bench checkout fc0055dc4e0a316c3f83133267fbd6faaa770992   # τ²-bench v1.0.1, pinned; stop if this fails
cd glint
uv sync --extra dev
uv run pytest -q
```

τ²-bench goes next to the repo (`../vendor/tau2-bench`, as `pyproject.toml` expects). On Windows, the `core.longpaths` and `core.autocrlf` settings are required: without them some τ² data files fail to check out, the pin fails, and `uv sync` installs a different τ² version; and the bundled bank data no longer matches τ²'s byte for byte.

- **The gate replays** (no GPU, no keys, no workspace needed): `uv run python scripts/gate_replay.py` and `uv run python scripts/bank_replay.py`.
- **Bank Desk, replay mode** (no GPU, no keys): `uv run python -m glint.desk.app`, then open http://127.0.0.1:8765.
- **glint-bank in Muse Code:** see [plugins/glint-bank/README.md](plugins/glint-bank/README.md). Check an install without Muse or a model: `uv run python -m glint.bank.server --decision-tools --selftest`.
- **Bank Desk, live mode:** needs a deployed Glimmer endpoint (`modal deploy src/glint/glimmer_serve.py`) and the keys in [.env.example](.env.example), copied to `$GLINT_HOME/.env`.
- **Decision-tool MCP server:** `uv run python -m glint.assist.server` (stdio).
- **With/without comparisons:** `scripts/tools_ab.py` (checkpointed; both arms on the same backend; a set's tools arm enforces the gates unless the set says otherwise; `--help` lists the sets). `scripts/export_results.py` and `scripts/export_scenarios.py` regenerate the desk's data from the run data. Every script is listed in [scripts/README.md](scripts/README.md).
- **Workspace:** working files (run data, the spend ledger, results, `.env`) live outside the repo, under `GLINT_HOME`, which defaults to the repo's parent folder. The task splits (`splits/`) and the recorded wrong-write runs (`data-public/`) are in the repo.
- **Tests:** coverage goals and rules in [docs/testing-strategy.md](docs/testing-strategy.md); CI runs pytest with the coverage gate and ruff on every PR.

## How it works

```mermaid
flowchart LR
    C["Customer<br/>(Muse Spark simulator, or a person)"] <--> A["Muse Glimmer 30B<br/>(agent)"]
    A -- "tool call" --> E["τ²-bench environment<br/>(bank, retail, airline)"]
    E -- "read" --> A
    E -- "write" --> G{"Glint gate<br/>policy as code"}
    G -- "allow" --> W["write runs"]
    G -- "deny: rule, facts, next step" --> A
    G -- "hold: ask the customer first" --> A
    G -. "every decision" .-> L[("hash-chained<br/>gate log")]
    F["LlamaFirewall 1.0.3<br/>glint_gates scanner"] -- "assistant tool call" --> G
    G -- "BLOCK / HUMAN_IN_THE_LOOP / ALLOW<br/>(read only, not logged)" --> F
    A -. "optional" .-> T["Glint decision tools<br/>(MCP)"]
    M["Muse Code<br/>+ glint-bank plugin"] -- "MCP" --> S["glint-bank server"] --> E
```

An agent that can move money is only as safe as its worst call. A 30B model holds a customer conversation well and still gets policy decisions wrong: the largest credit-limit increase a card allows, which disputes need a provisional credit, whether a card can be closed. Glint takes those decisions away from the model. The agent cannot skip the check. A gate checks only what its rules encode: a write no rule covers is allowed and logged as such.

1. A customer asks the agent (Glimmer 30B) for something policy-bound: dispute a charge, raise a credit limit, close a card, exchange an item.
2. The agent calls the environment's tools. Reads run as usual.
3. **Every write is intercepted** (`glint.gates.enforce`) and checked by the gates registered for that tool. A gate is a plain function of the live database and the write's arguments, so the same call against the same records always gets the same decision, including when τ²'s evaluator replays the run.
4. A deny returns, as the tool result, the rule, what the check found, the facts it used, why it matters and the next step. Arguments are never rewritten. A gate that raises, times out or returns nothing usable denies (fail closed).
5. Every decision is written to an append-only, hash-chained SQLite log (`glint.gates.eventlog`). At startup, every write tool in the toolkit must have a gate, or enforcement refuses to start.
6. **What the log protects against.** Unkeyed (SHA-256), the chain catches damage and edits that do not recompute it. With `GLINT_LOG_KEY` set, each link is an HMAC-SHA256 under the operator's key, so a rewrite by anyone without the key fails `verify()`; the key lives in the gate server's environment, so keep it out of the agent's prompt and of any file the agent can read. Truncation leaves a valid shorter chain, so `python -m glint.bank.server log SESSION_ID` prints the log's head (event count and last hash) to keep off the machine; `verify(expected_head=...)` fails when the log no longer contains it.

## Meta AI tooling, mapped to the judging criteria
| Meta tool | Role in Glint | What is shown | Criterion it serves |
|---|---|---|---|
| **Muse Glimmer 30B** (open weights) | The agent: it proposes every write, and the gates decide | **Self-hosted** on Modal with vLLM (bf16, A100-80GB, 131K context, scale to zero) by `glint.glimmer_serve`, with a health wait, fail-fast boot-log watching, a GPU cap enforced during the session and a retried stop. Every comparison reported here ran on it, except the first retail and airline draws (DeepInfra's hosted Glimmer). `glint.atem` re-parses its tool calls from the raw token stream: the hosted parser turned string arguments such as zip codes into numbers | Use of Meta AI tooling; technical depth |
| **Muse Spark** (Meta Model API) | The simulated customer in τ²-bench (and judge of its natural-language checks), and the agent in the Muse Code demo | Every τ² run reported here; the Muse Code run of 2026-10-31 | Use of Meta AI tooling; demo |
| **Muse Code** + the [glint-bank plugin](plugins/glint-bank/README.md) | A live τ² bank over MCP with every write gated, plus Glint's decision tools | Checked 2026-10-31 on plugin 0.2.1, headless (`muse exec`): the server reached `ready` with 23 tools; read-only tools ran without approval; a $4,000 increase denied and $2,500 allowed, with the session and call ids matching Muse Code's hook. `/start <task_id>` loads a scenario | Use of Meta AI tooling; demo |
| **LlamaFirewall** custom scanner | Glint's gate decision as a LlamaFirewall custom scanner (`glint.integrations.llamafirewall_scanner`) | Unit-tested against a fake module, and run inside LlamaFirewall 1.0.3 on CPU on 2026-11-03 (training task_054: a $4,000 increase BLOCK, $2,500 ALLOW, credit-limit requests unchanged by the scans); a host agent loop stopping the call was not tested ([docs/llamafirewall.md](docs/llamafirewall.md)) | Use of Meta AI tooling; technical depth |

## Business case
- **Buyers:**
  - bank risk, compliance and model-risk teams, who own model validation under [SR 11-7](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm);
  - fintech and banking-software platforms;
  - agent vendors selling into banks, who face the barriers bank executives rank highest: security and compliance, and autonomy (51% each, SoundHound-commissioned survey, n=201, [Nasdaq/GlobeNewswire](https://www.nasdaq.com/press-release/research-finds-two-three-us-banking-executives-believe-agentic-ai-will-change-how)).
- **Price floor:** generic authorization already sells: Cerbos Hub Production starts at $933 a month, with one year of audit logs ([Cerbos pricing](https://www.cerbos.dev/pricing)).
- **The offer:** an annual bank package on the open-source core: policy encoding for each product line, validation evidence for model risk, the tamper-evident audit trail, and support.
- **The value, in the bank's own numbers:** value per year = (agent write actions per year) × (error rate without the gate, measured on the bank's own traffic) × (average remediation cost per wrong action, from the bank's dispute and complaint data) + audit hours saved. The bank supplies every input; Glint supplies the gate and its measured with-and-without results on τ²-bench above.

## Limits, each answered
- **The 2026-11-02 test-split run had 18 false refusals of 35 judged** (17 by the request gate, 1 by verify once, against τ²'s gold). *Answer:* PRs #29 onward changed the request gate: reversible writes are allowed and flagged instead of refused, a refusal carries the filled-in `record_requests` call, and PR #41 added a switch that removes the request gate. These changes were fitted to that split, and the 2026-11-03 run's refusals by rule are not yet analyzed ([details](#rule-changes-from-the-test-split-prs-29-44)).
- **No outcome gain on unseen tasks.** On 59 unseen tasks Glimmer alone passed 21 of 108 matched runs and gated 19. Alone's 21 includes at least 27 runs lost to context overflows, counted as failures; 41 gated runs hit the step limit. *Answer:* the claim is enforcement integrity, which holds by construction; fewer wrong writes depends on which policy clauses the rules encode, and can be measured by each bank on its own traffic, as in the value formula above.
- **The replays are in-sample.** The 289 gold writes and the 24 of 32 stopped failures come from the tasks the rules were written on. *Answer:* they show the gates work as designed; the precision row above comes from 59 tasks no gate rule came from.
- **A gate checks only what its rules encode.** On those 59 tasks, 31% of live write decisions (206 of 660) were writes no rule covered. *Answer:* every such write is allowed and logged as `no_rule_yet` with the reason, enforcement refuses to start unless every write tool has a gate, and rules have since been added for savings interest and debit disputes ([The gates](#the-gates)).
- **Within-harness only.** Muse Spark plays the customer, so the numbers are not comparable with the τ²-bench leaderboard. *Answer:* both arms run on the same backend and seeds in every comparison except the 2026-11-03 test-split rerun (gated arm only, cross-day), reported by task.
- **The demos are illustrations.** The desk's pairs are selected, and the Muse Code run was scripted. *Answer:* the desk shows each task's real pass counts beside the pair, and the replays run without a GPU for anyone to check.
- **LlamaFirewall was tested at the scanner level only** (a host agent loop stopping the call was not tested), and interactive Muse Code is not shown to work (headless is). *Answer:* both are stated in [docs/llamafirewall.md](docs/llamafirewall.md) and [Limitations in detail](#limitations-in-detail).

## What the gates are shown to do

![Bank Desk, gated pair (task_054, a tuned task): on the left, Glimmer alone fails; on the right, a policy gate holds a dispute filing until the agent's dispute plan records the date the customer gave, two more gates make it confirm the shipping address and offer expedited shipping, and this run passes](docs/images/desk-gate.png)

Glint makes two claims with different evidence. **Enforcement integrity**, that every write is checked before it runs, a denied write never runs, every refusal carries the rule, the facts and the next step, and every decision is logged tamper-evidently, is shown below on Glimmer's runs and holds by construction for any model behind the gate. **Fewer wrong writes** depends on which policy clauses the rules encode and on the agent acting on a refusal, and on unseen tasks it is not yet shown (see [What was measured](#what-was-measured)). Each number says where it comes from.

| Check | Result | Where it comes from |
|---|---|---|
| τ²'s correct actions replayed through the gates (`scripts/gate_replay.py`, `scripts/bank_replay.py`) | **289 expected writes on the 77 banking training tasks: 0 refusals the policy text would call false; 2 refusals against τ²'s gold**, both task_005's, whose expected actions change the email of a user verified with placeholder data (the same case counted as the one false live refusal below). **Since PR #44 (every fix on by default), one more: task_091's debit card order, refused by the entry-tier 48-hour wait** (doc 029 says wait; doc 031 conflicts), reported, not exempted, so the replay scripts exit 1 on it; 286 of the 289 allowed | Replays of the benchmark's own correct answers, not live runs. In-sample: the gates were checked against these during development, so as a bound on writes like these it says at most 1.0% per write and 3.8% per task (one-sided 95%), and nothing about live writes. |
| Recorded failed runs replayed (`scripts/bank_replay.py`) | **24 of 32 stopped at a gated step**, the database unchanged: 2 by the task's own rule, 22 by a procedure or plan rule (a skipped lookup, an unverified customer, a dispute out of plan order) | In-sample: the recorded failures the rules were written from (`data-public/wrong-write-runs.json`), so this shows the gates work as designed, not that they catch new mistakes. τ²'s gold accepts 9 of the 24 stopped writes in another order. |
| Live gated runs on 59 tasks no gate change came from (`python -m glint.inspect refusals`; gates at PR 151) | **21 refusals; 0 false of the 6 that could be judged**: 6 held a write τ² expects until a required step was done; the other 15 refused writes τ² does not expect (8 card closures with no reason on record), not judged | These tasks' correct answers were in the development replays above, but none of their runs shaped a gate. The same runs show **no pass-rate gain** (below). |
| Every live gated run in the four gated banking sets (v9, unseen, v10, pressure; evaluation of 2026-11-01, recounted 2026-11-02; [details](docs/results.md#live-gate-decisions-across-the-four-gated-banking-sets)) | **Every write call reached a gate (964); a refusal is returned instead of running the write, and every live refusal inside a run τ² replayed was refused again in the replay (89 of 89); every refusal carried the rule, the facts and the next step. 113 refusals: 50 stopped a write τ² does not expect, 58 held a write τ² expects until a required step was done, 1 false against gold (task_005's placeholder verification), 4 not judgeable** (transfers to a human). On the 59 tasks no gate rule came from: 54 refusals, 1 false against gold out of 407 benign write calls (0.25%; upper bound 1.2%, one-sided 95%); the 19 holds there waited for a step the documents name (4) or for a lookup Glint's procedure requires before a requirement the documents state (15), so counting every hold as a cost gives 4.9% | Pooled over the 12 tasks the gates were developed on (v9: 59 of the 113 refusals, 12 of the 13 corrections after a refusal) and the 59 tasks no gate rule came from (v10 and the pressure set are design data for the search fix, not for the gates); the two are reported apart in the details. Which refusals are false is the reading of the policy documents by the author and by a model agent run blind to the first labels, not a labeled set. |
| Muse Code, 2026-10-31 (plugin 0.2.1, Muse Spark as the agent, task_054) | A $4,000 credit-limit increase **denied**; the agent then submitted $2,500, **allowed**; the gate log's hash chain verified | One run on a tuned task. The prompt scripted both the $4,000 request and following the deny's next step, so this shows the gate's decision and message, not the agent's own error or recovery. |

The deny the agent got in that Muse Code run, word for word (the verdict JSON shortened):

```
Error: this write was blocked by a policy gate and was NOT executed. Nothing changed.
Rule: banking.cli.submit.reduce_amount (glint.assist.banking.check_credit_limit_increase)
What the check found: the requested increase cannot be submitted as given
Facts it used:
- requested increase $4,000.00 exceeds the premium-tier maximum $2,500.00 (50% of the current limit); do not submit it, offer the maximum instead
Why it matters: Nothing was submitted. Tell the customer the largest increase their card allows and ask whether to submit that.
What to do next: Submit an increase no larger than max_increase in the verdict, for an account of this user.
Verdict: {"current_limit": 5000.0, "max_increase": 2500.0, ...}
```

Built for the Meta Global AI Developer Hackathon (Indie track, theme **Push the Frontier**: a new capability for an open-weight agent). See [Built during the event](#built-during-the-event).

## How it compares in detail
Each cell is from the project's own documentation (checked 2026-11-01; Invariant's guardrail docs were removed on 2026-01-12, so its row uses their last published version). "Not documented" means the docs do not say, not that the feature is absent.

| Project | Mainly guards against | Decides with | Reads the application's current data at decision time | The agent gets on a block | Tamper-evident decision log |
|---|---|---|---|---|---|
| [AWS AgentCore Policy](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/policy.md) | Unauthorized agent tool calls | Deterministic Cedar rules (optional guardrail scores) | The request, identity claims and session history; application data not documented | An error naming the policy denial | Not documented (CloudWatch logs) |
| [Invariant Guardrails](https://github.com/invariantlabs-ai/invariant) | Prompt injection, unsafe tool data flows | Rules plus classifier and LLM detectors | The trace (messages, tool calls, outputs) | The violated rule and where | Not documented |
| [NVIDIA NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails) | Jailbreaks, unsafe or off-topic conversation | Colang flows plus LLM and classifier checks | Through custom Python actions | A refusal message (configurable) | Not documented |
| [Meta LlamaFirewall](https://github.com/meta-llama/PurpleLlama/tree/main/LlamaFirewall) | Prompt injection, goal hijacking, insecure code | Classifier, LLM auditor, static analysis | Messages and the agent's trace | A decision, a reason and a score | Not documented |
| **Glint** | Writes that break a business's written policy | Deterministic Python functions | Yes: τ²'s live database plus the call's arguments | The rule, the facts, the consequence and the next step; the write did not run | Hash chain; HMAC-keyed with `GLINT_LOG_KEY` |

The others mostly decide whether a conversation or a call is safe; Glint decides whether a specific write is allowed by the business's policy given the account's current records, and tells the agent how to fix it. It complements them rather than replacing them: a prompt-injection scanner and a policy gate catch different failures.

## The gates
Rules are per write tool, never per task, and each cites the policy document it comes from (`glint.gates.rules`).

| Domain | What is gated |
|---|---|
| Banking | Credit card disputes (card action, provisional-credit flag, and the order of the customer's disputes from `plan_disputes`), credit-limit increases (the tier maximum and the required checks), replacement cards, debit card freeze, close, order and activation, debit card disputes (the open-dispute limit by tier, the earliest of duplicate charges, card possession on a card closed as lost or stolen, provisional credit where it is required, and no liability cap of 0; the customer's answers on card possession and PIN are an open-items reminder, not a check), account credits, card closures and retention offers, transfers to a human (the reason code), savings interest corrections and interest discrepancy reports (the expected APY from the account's products, the interest owed, the credit before the report) |
| Banking, procedure | The customer verified in this conversation; the lookups each procedure requires done before the write; a submitted credit-limit increase decided before a dispute or a transfer |
| Retail | Cancel, modify, return and exchange (`check_order_action`); address changes |
| Airline | Cancellation, flight, cabin, baggage and passenger changes; bookings; certificates |

Writes no rule covers yet are allowed and logged as `no_rule_yet`, with the reason (`rules.NO_RULE_YET`). On the 59 banking tasks no gate rule came from that was 31% of live write decisions in the recorded runs (206 of 660; on the 12 tasks the gates were developed on, 3 of 318), and those uncovered writes carry most of the wrong values that ran (below). Two of those tools, the savings interest correction and the interest discrepancy report (42 of the 206), have had rules since 2026-11-02 ([below](#savings-interest-rules-added-2026-11-02)); no gated run has exercised them yet (no test-split task's gold action touches them). The debit card dispute filing (74 of the 206) has had rules since 2026-11-02 too: the open-dispute limit per checking tier, the earliest of duplicate charges (its same-day tie-break fitted to τ² gold), card possession on a card closed as lost or stolen, provisional credit where every required condition holds (one direction only), and no liability cap of 0. The customer's answers on card possession and PIN, recorded with `record_dispute_answers`, appear as an open item when missing or different from the filing; they do not block it. The 2026-11-03 gated run on the test split ran with these rules; its refusals by rule are not yet analyzed. The liability tier and the card action are still not checked. The replays above run on training tasks only and are in-sample. τ²'s banking test split was read ungated on 2026-10-28, and some debit rules were written from its failures then. After its first gated run, on 2026-11-02, every conversation of that run was read, and the rules of PRs #29-#44 were designed from it (each change, its evidence and the tasks it tuned are in the project's rule-change log, in the research record outside this repo). The split is now design data, and every result on it from then on is tuned and in-sample (corrected: this said the split was read once and had one gated run; earlier it said the split is not used to change any rule).

### Rule changes from the test split (PRs #29-#44)
- **The request gate (PR #29):** reversible writes (reason logs, orders) are allowed and flagged instead of refused when no request is recorded; closes, payments and filings are still refused. A refusal now carries the filled-in `record_requests` call. A debit card close request covers the replacement order. Sources: the allow-and-flag and the filled-in call are Glint conventions; the close-covers-order mapping is fitted to the test split (tasks 077, 079, 080), with no policy clause linking a close to an order. All three came from the test split's 20 request-gate refusals on 2026-11-02. PR #41 added a `request_gate` switch (on by default) that removes the request gate entirely.
- **Every fix on by default since PR #44,** each with an off switch: the 48-hour entry-tier wait applied after a closure for suspected fraud too, the checking-credit amount deny, constrained values, the uncovered-write reminder, the declare reminder, the step-limit warning, the search diagnosis and the optional stages (the stages run nothing until a model is configured).
- **The one known refusal of training gold:** with the entry-wait deny on, the gold replay (in-sample) refuses task_091's debit card order after a closure for suspected fraud (doc 029 sets a 48-hour wait; doc 031 conflicts). It is reported, not exempted.

### Savings interest rules (added 2026-11-02)
The first rules encoded after the evaluation of the recorded runs, from the bank's documents (`glint.gates.apy`, `glint.gates.banking.savings_credit`, `interest_report`):
- **The expected APY** of each personal savings level is compiled from the knowledge base: base rate or balance tier, the highest linked-checking boost, the highest credit card bonus, and the relationship bonus.
- **An interest correction** must equal the interest owed at that APY, less any correction already applied that day.
- **The discrepancy report** must carry that expected APY, the rate the account was actually paid at, the same amount, and come after the credit.

| Check | Result |
|---|---|
| τ²'s correct actions replayed (in-sample) | 289 expected writes, 0 refusals false against the policy text; all 9 expected credits and 9 expected reports allowed; the table reproduces all 9 expected APYs |
| The recorded interest writes in the four gated banking sets, both arms, replayed through the new rules (in-sample) | Savings credits: 30 of 30 that differ from τ²'s expected ones refused, 6 of 6 matching allowed. Reports: 31 of 31 refused, 3 of 3 allowed |

**In-sample, not a measure of new mistakes caught.** These writes come from five tasks (093-097), whose expected actions were also used to settle two readings the documents leave open; from here on those tasks are design data for these rules. The two readings, listed with the others in `apy.CONVENTIONS`:
- the 0.025% relationship bonus applies only when no credit card bonus does, although doc 045 says the two stack;
- Silver Plus's direct-deposit bonus is not applied.

**The bank's own documents are inconsistent here,** which is why the readings are listed rather than hidden:
- the boost percentages sit in the checking-account documents, not the savings documents the policy points to;
- one savings document carries an unfilled `[[linked_checking_apy_bonus]]` template;
- two pairs of documents list different card bonuses for the same account;
- two worked examples are arithmetic errors ("5.5% + 0.025% = 6.0%", "4.0% + 0.5% = 4.75%").

## Decision tools
Before the gates, the same checks were offered as seven optional **MCP tools** the agent could call (`python -m glint.assist.server`, stdio). They are still served, and wired into τ²-bench toolkits (`glint.assist.tau2_tools`) and the Muse Code plugin. Measured: Glimmer called one in 1 of 80 runs on τ²'s banking test split, which is why the checks became enforced gates.

| Tool | Domain | Decides |
|---|---|---|
| `check_order_action` | Retail | Whether an order can be canceled, returned or exchanged; refund methods; price difference |
| `find_user_items` | Retail | Which of the user's items match what they described |
| `find_variants` | Retail | Which product variant matches the requested options |
| `check_reservation_action` | Airline | Cancellation and change eligibility: 24-hour rule, cabin changes, flown segments, insurance |
| `get_upcoming_trips_cost` | Airline | Totals for the user's upcoming trips |
| `check_credit_limit_increase` | Banking | The increase procedure: tier maximum, the required checks in order, approve or deny with the reason |
| `check_card_dispute` | Banking | Card action, provisional-credit criteria, and ordering against other requests |

<!-- results:start (replaced by the refresh-results step; edit the result notes, not this block) -->
## What was measured
| In short | Result |
|---|---|
| What the gates do | Check every write before it runs and block what their rules forbid, with the rule, the facts and the next step: 964 live write calls in four gated sets, each refusal returned instead of the write (89 of 89 live refusals in replayed runs refused again), 113 refusals with 1 false against τ²'s gold (task_005's placeholder verification), 0 false and 1 ambiguous (task_091, from PR #44) against the policy text in 289 replayed expected writes (in-sample; 2 refusals against gold, the same case; since PR #44 also task_091's order, refused by the 48-hour wait where docs 029 and 031 conflict) |
| What they have not been shown to do | Reduce wrong writes on tasks no gate rule came from. On those 59 tasks 189 writes that differ from τ²'s expected ones ran through the gates, 120 of them on tools with no rule; 61 of those 120 differ in a value the policy states as a table or formula that no gate encoded when the runs were recorded (savings credit and interest amounts, now encoded, [above](#savings-interest-rules-added-2026-11-02); debit-dispute liability and card action, not encoded), though 18 of them are the liability cap, which the documents key on a statement date the data does not hold. The gated arm's lower off-policy rate on ruled tools (0.18 vs 0.28 per write) is mostly which tools are hard: Glimmer alone, with no gates, shows the same gap (0.66 per write on tools no rule covers vs 0.28 on ruled ones) |
| On the banking test split after the rules designed from it (tuned and in-sample) | The gated arm alone, on the code with every fix on (main at 7bf75cb, PRs #29-#45), 2026-11-03: 11/40 runs passed, against 7/40 for both arms on 2026-11-02 (a cross-day comparison; the alone arm was not rerun). By task 4 better, 0 worse than the earlier gated run (sign test p = 0.125, two-sided) and 3 better, 0 worse than alone (p = 0.25); may be chance. Step-limit endings fell from 14 to 6 (may be chance, Fisher p = 0.07 two-sided). Context overflows do not drive the gap: alone's 7 fell on six tasks where both gated arms pass 0 runs. The rules of PRs #29-#44 were designed from this split's 2026-11-02 runs, so no held-out banking test result is possible from here on ([details](docs/results.md#banking-test-split-gated-arm-after-the-fix-research-2026-11-03)) |
| On the banking test split, the 2026-11-02 run that the rules above came from | τ²'s 20 banking test tasks, k=2, Glimmer alone vs Glint on the code after PRs 20-25: passes 7/40 vs 7/40; writes that differ from τ²'s expected ones 34 vs 25, extra writes 13 vs 4 (off-policy writes by task 5 fewer, 2 more, sign test p = 0.45 two-sided; may be chance: the request gate blocked only 1 extra write, and 5 of alone's 13 were in tasks whose gated runs hit the step limit), wrong values 21 vs 21; 3 of 128 allowed writes had no rule or recorded request behind them. The request record refused 17 times (13 writes τ² expects) until the agent recorded them, 18 of 35 judged refusals false against gold with verify once; 12 of the 13 writes then ran, and 4 of the gated arm's 14 step-limit runs include refusals from several gates, the request record among them; the other 10 are knowledge-base search loops. Disclosures from reading every conversation of the run: 18 refusals false against gold (17 by the request gate, 1 by verify once; by policy labels (not blinded: refused writes had no decoys) the request gate's 20 refusals were 16 policy-wrong and 4 ambiguous); τ²'s own offline re-grade gives 8/40 gated and 10/40 alone, not 7/40 each, cause unknown (by run Fisher p ≈ 0.79; by task gated 1 better, 3 worse than alone, sign test p = 0.63, both two-sided); task_083 cannot be passed by any agent (gold omits a required field), so out of 38 runs the rates are 7/38 each; 12 of the gated arm's 33 first errors were omissions (6 missing writes, 6 non-write actions) that write gates cannot see. Some debit rules came from this split's failures on 2026-10-28 ([details](docs/results.md#banking-test-split-every-write-gated-2026-11-02)) |
| What they do not do | Raise pass rates: on 59 unseen tasks, Glimmer alone passed 21 of 108 matched runs and gated 19; with the search fix, gated 16 of 118 against alone's 22 |
| Where gains appear | Nowhere significant: the largest, v9's tuned tasks (7/40 → 25/40 runs, p = 0.0625 one-sided), is in-sample; untuned tasks move at most 2/8 → 4/8 |
| Under a customer who pushes past policy | No difference detectable: on 20 of those 59 tasks (k=2; design data for the search fix, no gate rule came from them) with the simulated customer told to push twice for more than policy allows, 48 writes differing from τ²'s expected ones ran with Glimmer alone and 49 with Glint (by task 5 fewer, 6 more, 9 same, sign test p = 1.0 two-sided; at 80% power the run could detect only an 81% reduction). The customer did push (3.4-4.2 pushing turns per run against 0.7-0.9 without the instruction) and the differing values did not follow the push: only 2 of 28 (alone) and 7 of 38 (gated) changed numeric values equal any amount the customer had mentioned, and 7 of those 9 are the debit-dispute liability cap written as $500 where τ² expects $50, against the customer's interest, a value that appears as often without pressure. The documents key that cap on a statement date the data does not hold, τ²'s $50 follows an unstated convention, and the agent counted from the transaction date |

τ²-bench v1.0.1, Glimmer at medium reasoning, both arms with the same seeds on the same backend. Reported **by task** (runs of one task are not independent); tuned and untuned tasks apart, latest first. A **tuned** task is one whose runs shaped a gate, the dispute plan or a tool; an **untuned** task is one whose runs shaped nothing. k is the number of runs per task. Full detail: [docs/results.md](docs/results.md).

- **Banking test split, the gated arm after the rules designed from it (2026-11-03): 11/40, tuned and in-sample, may be chance** (20 test tasks, k=2, the gated arm only, Glimmer on Modal, main at 7bf75cb: PRs #29-#45, every fix on by default since #44). Runs passed: 11/40, against 7/40 for both arms on 2026-11-02 (cross-day; the alone arm was not rerun). By task against the 2026-11-02 gated run: 4 better (014, 043, 079, 102), 0 worse, 16 same (sign test p = 0.125, two-sided); against alone: 3 better (043, 078, 079), 0 worse, 17 same (p = 0.25). Step-limit endings 14 → 6 (may be chance, Fisher p = 0.07 two-sided). Alone's 7 context overflows fell on tasks 022, 024, 029, 077, 080 and 082, where both gated arms pass 0 runs, so they do not drive the gap. task_083 cannot be passed; out of 38 runs: alone 7/38, gated 11-02 7/38, gated 11-03 11/38. Not re-graded. Against alone's re-grade (10/40), by task 3 better, 1 worse (099), p = 0.63. Refusals by rule, false refusals and wrong values not yet analyzed. Spend: $2.569 GPU, $0.133 Meta. The rules of PRs #29-#44 were designed from this split's 2026-11-02 runs, so the split is design data and no held-out banking test result is possible from here on.
- **Banking test split, every write gated (2026-11-02): no pass-rate change; fewer extra writes, may be chance** (20 test tasks, k=2, both arms on Modal, code after PRs 20-25; its conversations were then read and became the design data for PRs #29-#44; corrected: this said the split's one gated read). Passes 7/40 vs 7/40, by task 2 better, 2 worse (sign test p = 1.0, two-sided). Writes that differ from τ²'s expected ones: 34 in 12 runs alone vs 25 in 10 runs gated; extra writes 13 vs 4; wrong values 21 vs 21 (debit-dispute liability and provisional credit, which no rule checks, are about half: 10 of 21 alone, 9 of 21 gated). Some debit rules came from this split's failures on 2026-10-28, so it is not fully held out. Its disclosures (18 refusals false against gold, τ²'s re-grade 8/40 vs 10/40, task_083 unpassable, 12 of 33 first errors omissions) are in the table above.
- **Banking, the same 59 tasks with the search fix (v10): no gain, and tuned on these tasks** (k=2, the gated arm only, on the code after PR 158-174; Glimmer alone is the arm below). All runs: Glimmer alone passed 22 of 118, gated with the fix 16 of 118; by task 6 better, 9 worse (sign test p = 0.61, two-sided). Against the gated arm before the fix, on its 108 matched runs: 14 passed with the fix vs 19 before; by task 5 better, 7 worse (p = 0.77). On the 37 tasks where alone never overflowed its context, alone passed 21 of 74 and gated with the fix 16 of 74. The fix (PR 171) was designed after reading these tasks' step-limit failures, so this set no longer counts as unseen. On the matched runs the gated arm still hit τ²'s step limit in 34 of 108 (before: 41 of 108): the fix removed few of them, and what keeps these runs searching is not established.
- **Banking, 59 unseen tasks, every write gated: no gain** (k=2, the gated arm on the code at PR 151, before the search fix of PR 171). On 108 matched runs Glimmer alone passed 21 and gated 19; by task 5 better, 8 worse (sign test p = 0.58, two-sided). These are the training tasks outside the 12 the gates were built on whose correct answers make a gated write; no change came from their runs.
  - Alone's misses include 30 context overflows (gated: 3); on the 39 tasks where alone never overflowed, alone passed 20 and gated 19. The gated arm hit τ²'s step limit in 41 of 108 runs, all failed; 83% of their calls were knowledge-base searches. The search's budget was the likely cause; PR 171 changed it, and the run above shows the change removed few of these step-limit losses.
  - The gate refused 21 writes in these runs: 0 false (above). 20 of the gated arm's failed runs still filed a write with a wrong value (a field the gates do not check).
- **Banking with every write gated (v9, design data): no gain shown on untuned tasks (2 tasks, 2/8 → 4/8); on 10 tuned tasks 6 improved and 1 worse against Glimmer alone, not significant** (the same 12 tasks, k=4; the gated arm also has the decision tools, a dispute plan and open-item reminders; code at PR 147).
  - **10 tuned tasks** (the gates, the plan and the reminders were shaped by their runs): 039: 0/4 → 4/4; 054: 0/4 → 4/4; 040, 041, 051, 052: 0/4 → 3/4; 031, 038, 081 tied; 037 worse (4/4 → 2/4); sign test p = 0.0625 (one-sided); runs 7/40 → 25/40. Do not read as general evidence.
  - **2 untuned tasks:** 050: 2/4 → 4/4; 053 tied at 0/4 (likely the simulated customer's answers; not checked); runs 2/8 → 4/8. Two tasks cannot reach p < 0.05.
  - v9 is the latest of nine reported gated versions (v3 stopped early), each a change made after reading the one before. Tuned totals ranged from 7/40 to 25/40 and v9 is the highest; against v8 (19/40) it is 5 tasks better and 2 worse (p = 0.23, one-sided), so the differences between versions are not established at k=4. Untuned stayed at 4/8 from v4 through v9. All versions by task: [docs/results.md](docs/results.md#banking-write-gates-enforced-design-data-k4).
  - In v9's runs, 0 of the 39 gate refusals that could be judged against τ²'s expected writes were false (55 write refusals; the other 16 refused writes that differ from the expected ones and are not judged).
- **Banking, current tools: the only domain with a gain, not significant** (12 tasks with a dispute or credit-limit step, k=4).
  - **10 tasks untuned for the tools (all but 050 and 053 were later tuned for the gates): 3 improved** (037: 0/4 → 3/4; 050: 2/4 → 4/4; 052: 0/4 → 1/4), 7 tied, 0 worse; sign test p = 0.25; runs 6/40 → 12/40. Their correct answers were used to check the tools' rules, so they are not fully held-out.
  - **2 tuned tasks improved** (038: 0/4 → 1/4; 054: 0/4 → 4/4). The tools were refined after reading their runs.
  - Wrong writes (writes Glint's own checks judge wrong) did not fall on the untuned tasks: 39 without the tools vs 43 with. On the tuned tasks: 11 → 0.
- **Retail and airline: no gain, and on retail the tools lose runs.** Held-out tasks on the current tools (k=4): retail 39/40 without the tools vs 35/40 with; airline 37/40 vs 38/40. By task: 2 improved, 14 tied, 4 worse (sign test p = 0.69).
- **Banking test split, first read, 2026-10-28 (20 then-untouched tasks, k=4, tools without gates): no gain.** 10/80 without the tools vs 10/80 with; by task 1 improved (047: 0/4 → 1/4), 1 worse (036: 4/4 → 3/4), 18 tied. Only task_080 has a dispute or credit-limit step, and a decision tool was called in 1 of the 80 runs, so the split mainly shows the tools neither helped nor hurt tasks they were not built for. The split has since had gated runs on 2026-11-02 and 2026-11-03 (above), and the rules of PRs #29-#44 were designed from the first of them (corrected: this said the split was read once and has no gated run).
- Earlier runs on tool code from before the 7789817 verdict fixes are superseded; they are kept in [docs/results.md](docs/results.md) because the desk's decision-tools pairs come from them.

![Results tab: per set, tasks improved, tied and worse, runs passed with and without Glint, and pass dots per task](docs/images/desk-results.png)

<!-- results:end -->
## Limitations in detail
- **The gates block the specific steps they encode; they do not make every write correct, and they have not been shown to raise pass rates or to reduce wrong writes on unseen tasks.** On the 59 unseen tasks the gated arm passed 19 of 108 matched runs against Glimmer alone's 21; 20 of its failed runs still filed a write with a wrong value. Across the 59 tasks no gate rule came from (three sets), 206 of 660 live write decisions (31%) were allows for writes no rule covers, and 120 of the 189 writes that differ from τ²'s expected ones ran on those tools. Most of the policy's computable clauses are not encoded yet (dispute liability tiers, checking credits, rewards amounts); savings interest corrections and discrepancy reports were encoded on 2026-11-02, checked only against recorded writes from five tasks, and not yet run live. The other rules were written and checked on 12 training tasks.
- **Muse Code, interactive mode:** checked with Muse Code 1.4.14 in headless mode (`muse exec`, 2026-10-31), the plugin's server reached `ready` with 23 tools and a $4,000 increase past the tier maximum was refused, then $2,500 allowed. In interactive mode on 2026-11-01, Muse's own startup log shows the interactive start shutting the plugin's server down about 0.15 s after fetching a fresh feature configuration (`transport_closed`); a headless start (`muse exec`) three minutes later, with that configuration cached, reached `ready` with 23 tools. A second interactive start has not been checked, so interactive mode is not shown to work; headless mode is. `scripts/muse_check.py` reads Muse's startup log and says whether the server reached `ready`.
- **Within-harness only.** Muse Spark plays the customer, so the numbers are not comparable with the τ²-bench leaderboard.
- **Small samples.** k=2 to 4 per task; no sign test is significant.
- **The gates' rules were written and checked on training tasks**, so the replay numbers and the gated results on those 12 tasks are in-sample. The banking test split is design data for PRs #29-#44, so its 2026-11-03 result is tuned and in-sample. A deny that lands on a run's first wrong write shows the gate caught that write, not that the run would then pass.
- **The desk's replay pairs are selected:** a failed run without Glint beside a passed run with it. The gated pairs' arm also has the decision tools and the dispute plan, so a pair does not show the gate alone caused the pass; an earlier gated version passed task_054 4/4 without blocking anything. The page says so and shows each task's real pass counts. Live mode runs no gates and is off on the hosted desk.
- **Retail:** `find_variants` matches option names as exact strings (it called the user's "Google Home" unavailable when the catalog says "Google Assistant").
- **Banking:** Glimmer still fails most tasks; task 081 overflows the context.

## Built during the event
- All code in this repo was written during the event (Oct 26 – Nov 4, 2026); no earlier code was copied. The gate runner's and gate log's design follows the author's earlier private projects (agent-harness-v2 and pgharness). The first commit is `952dd30` on 2026-10-26.
- This history was rewritten on Oct 31 to remove the author's name, local paths and session links. Authors and dates are unchanged; the only changes to files are local folder paths in comments, now `<workspace>`, and one coverage data file removed. The old-to-new commit map is in [docs/history-map.txt](docs/history-map.txt). The original repository, with its pull request and CI record, is private, and the organizers can be given access. PR numbers in this repo ("PR 147") refer to that record.
- Third-party work, used under its license: [τ²-bench](https://github.com/sierra-research/tau2-bench) (MIT, Sierra Research) is the benchmark and a dependency; its banking data is bundled in `plugins/glint-bank/data/` with its license. The desk's spacing, type and color scales follow Astryx's neutral theme (MIT, Meta Platforms).

## License
[MIT](LICENSE).
