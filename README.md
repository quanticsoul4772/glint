# Glint

**For banks letting AI agents write to accounts: the policy firewall between the agent and the account.**

**The model proposes. The policy decides. A refused write never runs.**

[![tests](https://github.com/quanticsoul4772/glint/actions/workflows/tests.yml/badge.svg)](https://github.com/quanticsoul4772/glint/actions/workflows/tests.yml)
[![coverage 97% (CI on main)](https://img.shields.io/badge/coverage-97%25-brightgreen)](docs/testing-strategy.md "coverage 97% (CI on main)")
[![license MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

### ▶ [Hackathon submission video: Glint demo (2:50)](https://youtu.be/L6PhtP6KedU)
Meta Global AI Developer Hackathon · Indie track · Push the Frontier

![Glint blocks a $4,000 credit-limit increase, returns the fix, and the $2,500 request runs](docs/demo.gif)

**[Try the gate (no sign-in)](https://quanticsoul4772--glint-desk-web.us-east.modal.run/#try)** · **[Replay without GPU](https://quanticsoul4772.github.io/glint/)**

**Your AI agent can talk to customers. Glint decides what it is allowed to change.** Every account change the agent attempts goes through a gate before it runs. Where the bank's written policy has a rule, the gate applies it as code to the live account records:
- **An allowed write runs.**
- **A refused write never runs,** and the agent gets the rule, the facts and the fix, so it can recover.
- **Every decision is on the record:** each decision, and the result of every allowed write, lands in a hash-chained log (tamper-evident when keyed with a key the agent cannot read and its head is kept off the machine; glint-bank keys it by default, but its default key file sits under `~/.glint-bank`, where an agent with file tools can read it, so point `GLINT_LOG_KEY` or `GLINT_LOG_KEY_FILE` elsewhere; the measured runs' logs are unkeyed).

Built on Meta's open-weight Muse Glimmer 30B (the agent, self-hosted), with Muse Spark (Meta Model API) as the simulated customer, a Muse Code plugin with a native blocking hook, and a LlamaFirewall scanner.

**Built during the hackathon (Oct 26 – Nov 4, 2026):** all of Glint's code, written from the first commit `952dd30` (2026-10-26; history rewritten Oct 31 with authors and dates kept). τ²-bench data and Astryx's theme scales are third-party; see [Built during the event](#built-during-the-event). **Best Build on Meta Open Weight Models:** the agent is Muse Glimmer 30B, Meta open weights, self-hosted on Modal with vLLM; Glint also runs inside Meta's LlamaFirewall 1.0.3 as a custom scanner: replayed through it, all 1,082 recorded gated Glimmer calls of the four gated sets returned every live refusal again (a host agent loop stopping the call was not tested).

## Why it matters
- **Agents are getting payment access.** Visa (Intelligent Commerce, April 30, 2025) and Mastercard (Agent Pay, April 29, 2025) launched programs that let AI agents pay merchants for consumers; those programs cover payments, not account servicing, which is where Glint sits ([Retail TouchPoints](https://www.retailtouchpoints.com/topics/payments/visa-mastercard-paypal-dive-into-agentic-era-with-tools-that-help-consumers-use-ai-to-buy)). 64% of larger US banks were testing or deploying agentic AI, against 38% of smaller ones (SoundHound-commissioned Arizent/American Banker survey, n=201, [Nasdaq/GlobeNewswire](https://www.nasdaq.com/press-release/research-finds-two-three-us-banking-executives-believe-agentic-ai-will-change-how)).
- **Risk controls decide which projects survive.** Gartner predicts over 40% of agentic AI projects will be canceled by the end of 2027, citing costs, unclear value or "inadequate risk controls" ([Gartner](https://www.gartner.com/en/newsroom/press-releases/2025-06-25-gartner-predicts-over-40-percent-of-agentic-ai-projects-will-be-canceled-by-end-of-2027)).
- **Automated account actions are already an enforcement target (a non-AI parallel).** The CFPB's 2022 order against Wells Fargo, more than $2 billion in redress plus a $1.7 billion penalty (over $3.7 billion in all), found the bank "froze more than 1 million consumer accounts based on a faulty automated filter's determination" of possible fraud ([CFPB press release](https://www.consumerfinance.gov/archive/newsroom/cfpb-orders-wells-fargo-to-pay-37-billion-for-widespread-mismanagement-of-auto-loans-mortgages-and-deposit-accounts/)).
- **Disputed account activity is already large.** Chargebacks were forecast at 261 million, worth $33.8 billion, in 2025 (a Mastercard and Datos Insights estimate, [Mastercard](https://www.mastercard.com/content/dam/mccom/shared/news-and-trends/insights/2025/2025-global-chargebacks-outlook/pdf/2025-state-of-chargebacks-report.pdf)); it is context for the cost of wrong account actions, not a measure of the agent problem.

## What makes it different
1. **Deterministic enforcement of written business policy, before the write.**
2. **Refusals carry the rule, the facts and the fix, so the agent can recover.** This is not described in competitors' docs ([AgentCore Policy](https://aws.github.io/bedrock-agentcore-starter-toolkit/user-guide/policy/overview.html) describes allow/deny; [ToolGuard](https://pypi.org/project/toolguard/0.1.19/) describes raising alerts).
3. **The refusal and its hash-chained, tamper-evident record come from one gate.**
4. **Measured with and without on τ²-bench banking.** Published write gates on τ-bench and τ²-bench test other domains: a gate compiled from τ²'s airline policy raised a 235B agent's pass^1 from 0.39 to 0.54 ([arXiv 2609.25686](https://arxiv.org/abs/2609.25686)); Z3-checked tool calls on τ-bench ([arXiv 2603.20449](https://arxiv.org/abs/2603.20449)); ToolGuard on airline ([arXiv 2507.16459](https://arxiv.org/html/2507.16459v1)). No published gate on τ²'s banking domain was found. [FraudBench](https://arxiv.org/abs/2608.18136), an adversarial benchmark built on the same banking environment, is the next test.
5. **Built on Meta's open-weight stack.** It runs inside Meta's LlamaFirewall 1.0.3 as a custom scanner (scanner-level test; [docs/llamafirewall.md](docs/llamafirewall.md)).

| Approach | What it checks | Deterministic? | Explains the fix? | Tamper-evident record? |
|---|---|---|---|---|
| Content filters ([Llama Guard 4](https://build.nvidia.com/meta/llama-guard-4-12b/modelcard), [Lakera Guard](https://www.checkpoint.com/press-releases/check-point-acquires-lakera-to-deliver-end-to-end-ai-security-for-enterprises/), [Model Armor](https://docs.cloud.google.com/model-armor/help/configure-model-armor-gateway)) | Unsafe content, prompt injection, sensitive data in prompts, responses and tool payloads | No: model or classifier | Not described | Not described |
| Generic authorization ([Cedar in AgentCore Policy](https://aws.amazon.com/about-aws/whats-new/2026/03/policy-amazon-bedrock-agentcore-generally-available/), [OPA](https://www.openpolicyagent.org/docs/management-decision-logs), [Cerbos](https://cerbos.dev/features-benefits-and-use-cases/ai-security)) | Whether an identity may call a tool or action, with conditions on its inputs and on application data the caller passes in | Yes | Not described (allow or deny) | Decision logs; hash-chaining not described |
| Model judges ([Microsoft Task Adherence](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/quickstart-task-adherence), [LlamaFirewall AlignmentCheck](https://arxiv.org/abs/2505.03574v1)) | Whether a proposed call or the agent's reasoning fits its task | No: model | Not described | Not described |
| **Glint** | Each write against the bank's written policy and the live account records | **Yes:** plain Python functions | **Yes:** the rule, the facts and the next step | **Yes:** hash chain, HMAC-keyed with `GLINT_LOG_KEY` or a per-install key file (glint-bank's default) |

Glint complements content filters rather than replacing them: a prompt-injection scanner and a policy gate catch different failures. A per-project comparison is in [How it compares in detail](docs/comparison.md).

## Results
| What | Result | Label and source |
|---|---|---|
| Enforcement by construction | Every one of **964 write calls** in four gated sets went through a gate; **89 of 89** refusals in replayed runs were refused again | Measured on Glimmer's runs in the four gated banking sets (v9, unseen, v10, pressure); holds by construction for any model behind the gate ([details](docs/results.md#live-gate-decisions-across-the-four-gated-banking-sets)) |
| Refuses off-policy writes before they run | On 59 tasks no gate rule came from, **34 writes τ²'s correct answer does not make were refused before they ran**; in 16 of those cases the agent later ran a write that still differed (among the 189 differing writes that ran, most on tools no rule covered yet) (corrected: an earlier breakdown counted one refusal twice) | Untuned tasks, gates at PR 151 (unseen), #5 (v10), #11 (pressure) ([details](docs/results.md#live-gate-decisions-across-the-four-gated-banking-sets)) |
| Precision | **1 false refusal in 407 correct write calls** on 59 tasks no gate rule came from: 0.25%, upper bound 1.2% (one-sided 95%); 4.9% counting every hold as a cost | Untuned tasks, gates at PR 151 (unseen), #5 (v10), #11 (pressure), judged against τ²'s expected writes; 2 more refusals (v10 task_045 trial 0) count as holds by the author's reading, and as false without it: 3 in 407, 0.74%, upper bound 1.9% ([details](docs/results.md#live-gate-decisions-across-the-four-gated-banking-sets)) |
| Banking test split after the fix research | **No held-out result:** the gated arm passed 11 of 40 runs (2026-11-03) against Glimmer alone's 7 of 40 (2026-11-02; τ²'s offline re-grade: 10 of 40). By task against alone: 3 better, 0 worse, p = 0.25; against the re-grade: 3 better, 1 worse, p = 0.63 (sign test, two-sided). 4 of its 25 judged refusals were false against gold (gates at #45; later merges changed the clock, gate classes and logging, not gate decisions). | **Tuned on this split, in-sample, cross-day; the alone arm was not rerun; may be chance** ([details](docs/results.md#banking-test-split-gated-arm-after-the-fix-research-2026-11-03)) |

No outcome gain on unseen tasks has been shown; see [Limits](#limits-each-answered) and [What was measured](docs/evidence.md#what-was-measured).

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

- **The gate replays** (no GPU, no keys, no workspace needed): `uv run python scripts/gate_replay.py` and `uv run python scripts/bank_replay.py`. Each prints task_091's entry-wait refusal as "KNOWN (decided)" and exits 0; any other false refusal exits 1. bank_replay's last lines are the gold summary, the recorded-failure count (24 stopped: 2 by the task's own policy rule, 22 earlier by a procedure or plan rule) and a `verdict:` line.
- **Bank Desk, replay mode** (no GPU, no keys): `uv run python -m glint.desk.app`, then open http://127.0.0.1:8765.
- **glint-bank in Muse Code:** see [plugins/glint-bank/README.md](plugins/glint-bank/README.md). Check an install without Muse or a model: `uv run python -m glint.bank.server --decision-tools --selftest`. It prints one JSON line: the tool list, then on task_054 a $4,000 credit-limit increase denied (`banking.cli.submit.reduce_amount`) and $2,500 allowed.
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

An agent that can move money is only as safe as its worst call. A 30B model holds a customer conversation well and still gets policy decisions wrong: the largest credit-limit increase a card allows, which disputes need a provisional credit, whether a card can be closed. Glint takes those decisions away from the model. The agent cannot skip the checks on account records: 95 of the 146 banking gates read only the database, the arguments and tool calls that ran, but only 43 of those are policy rules; 43 are checks generated for every tool (repeat calls, the tool's own argument bounds) and 9 allow every call because no rule covers them yet, among them funds transfers and card payments. The other 51 can be passed by what the agent itself sends: 17 accept records the agent makes, customer verification among them, and the 34 checks that an identifier was shown by an earlier result also count an id that one of Glint's own record or check tools echoed back ([docs/gate-classes.md](docs/gate-classes.md); corrected 2026-11-04: this counted those 34 among the account-record checks). A gate checks only what its rules encode: a write no rule covers is allowed and logged as such.

1. A customer asks the agent (Glimmer 30B) for something policy-bound: dispute a charge, raise a credit limit, close a card, exchange an item.
2. The agent calls the environment's tools. Reads run as usual.
3. **Every write is intercepted** (`glint.gates.enforce`) and checked by the gates registered for that tool. A gate is a plain function of the live database and the write's arguments, so the same call against the same records always gets the same decision, including when τ²'s evaluator replays the run.
4. A deny returns, as the tool result, the rule, what the check found, the facts it used, why it matters and the next step. Arguments are never rewritten. A gate that raises, times out or returns nothing usable denies (fail closed).
5. Every decision, and after each allowed write its result and a hash of the database, is written to an append-only, hash-chained SQLite log (`glint.gates.eventlog`). At startup, every write tool in the toolkit must have a gate, or enforcement refuses to start.
6. **What the log protects against.** Unkeyed (SHA-256), the chain catches damage and edits that do not recompute it. With `GLINT_LOG_KEY` set, or the per-install key file glint-bank creates by default, each link is an HMAC-SHA256 under the operator's key, so a rewrite by anyone without the key fails `verify()`; the key lives in the gate server's environment, so keep it out of the agent's prompt and of any file the agent can read. glint-bank's default key file (`~/.glint-bank/log.key`) is in the agent's reach when the agent has file tools in the home folder, as Muse Code's does, so set `GLINT_LOG_KEY` or `GLINT_LOG_KEY_FILE` to a place those tools cannot read. Truncation leaves a valid shorter chain, so `python -m glint.bank.server log SESSION_ID` prints the log's head (event count and last hash) to keep off the machine; `verify(expected_head=...)` fails when the log no longer contains it.

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

## Meta AI tooling, mapped to the judging criteria
| Meta tool | Role in Glint | What is shown | Criterion it serves |
|---|---|---|---|
| **Muse Glimmer 30B** (open weights) | The agent: it proposes every write, and the gates decide | **Self-hosted** on Modal with vLLM (bf16, A100-80GB, 131K context, scale to zero) by `glint.glimmer_serve`, with a health wait, fail-fast boot-log watching, a GPU cap enforced during the session and a retried stop. Every comparison reported here ran on it, except the first retail and airline draws (DeepInfra's hosted Glimmer). `glint.atem` re-parses its tool calls from the raw token stream: the hosted parser turned string arguments such as zip codes into numbers | Use of Meta AI tooling; technical depth |
| **Muse Spark** (Meta Model API) | The simulated customer in τ²-bench (and judge of its natural-language checks), the pushing customer of the pressure set (told to push twice for more than policy allows), and the agent in the Muse Code demo | Every τ² run reported here; the pressure set ([What was measured](docs/evidence.md#what-was-measured)); the Muse Code run of 2026-10-31 | Use of Meta AI tooling; demo |
| **Muse Code** + the [glint-bank plugin](plugins/glint-bank/README.md) | A live τ² bank over MCP with every write gated, plus Glint's decision tools; since plugin 0.4.0 also a Muse Code `PreToolUse` hook that sends each glint-bank write to Glint's gates and blocks it on a deny (the hook script denies when it cannot get a decision; if the hook process itself times out or crashes, Muse decides, and the server still gates every call) | **Checked live 2026-11-03 on plugin 0.4.0** (installed from tag `glint-bank-v0.4.0`), headless (`muse exec`, Muse Spark as the agent, task_054): the server reached `ready` with 27 tools; Muse Code's `PreToolUse` hook blocked the $4,000 increase before it reached the server (`banking.cli.submit.reduce_amount`, a `hook.denied` event, no `write.ran`), then $2,500 was allowed and ran; the keyed gate log verified; session and call ids matched Muse Code's hook log. Earlier check: 2026-10-31 on plugin 0.2.1. `/start <task_id>` loads a scenario | Use of Meta AI tooling; demo |
| **LlamaFirewall** custom scanner | Glint's gate decision as a LlamaFirewall custom scanner (`glint.integrations.llamafirewall_scanner`) | Unit-tested against a fake module; run inside LlamaFirewall 1.0.3 on CPU on 2026-11-03: task_054 ($4,000 BLOCK, $2,500 ALLOW), and all 1,082 recorded Glimmer gated calls of the four gated sets (964 writes), each scanned on the database as recorded: every live refusal returned again (90 BLOCK, 23 HUMAN_IN_THE_LOOP_REQUIRED); 828 of 1,082 agree with the live gate, and all 254 disagreements are live allows that rules added since block (171 the request gate; 995 of 1,082 with it off); a host agent loop stopping the call was not tested ([docs/llamafirewall.md](docs/llamafirewall.md), `scripts/lf_replay.py`) | Use of Meta AI tooling; technical depth |

**Which parts are open weights:** the agent, Muse Glimmer 30B, self-hosted on Modal. Muse Spark (the simulated customer and judge) runs through Meta's Model API.

**Why Muse Glimmer, specifically:**
- **The bank owns the model.** Open weights let a bank run the agent on its own hardware, so customer data need not leave its control, next to a gate it also owns.
- **Self-hosting surfaced a real defect.** vLLM's Glimmer tool-call parser turned string arguments into numbers (a zip code "02134" loses its leading zero); `glint.atem` re-parses tool calls from the raw token stream to fix it.
- **Glimmer acts on refusals.** All 13 corrections of an off-policy write after a refusal followed a next step that named the corrected value or the question to ask; 12 were in v9, on three tuned tasks.

## Business case
- **Buyers:**
  - bank risk, compliance and model-risk teams, who own model validation under [SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm) (April 2026; it supersedes SR 11-7);
  - fintech and banking-software platforms;
  - agent vendors selling into banks, who face the barriers bank executives rank highest: security and compliance, and autonomy (51% each, SoundHound-commissioned survey, n=201, [Nasdaq/GlobeNewswire](https://www.nasdaq.com/press-release/research-finds-two-three-us-banking-executives-believe-agentic-ai-will-change-how)).
- **Price floor:** generic authorization already sells: Cerbos Hub Production starts at $933 a month, with one year of audit logs ([Cerbos pricing](https://www.cerbos.dev/pricing)).
- **The offer:** an annual bank package on the open-source core: policy encoding for each product line, validation evidence for model risk, the tamper-evident audit trail, and support.
- **The encoding cost, stated:** one bank's banking policy took 146 gates, 60 of them policy rules; the whole gates package (banking, retail, airline and the enforcement code) is about 5,800 lines. Each product line is a similar body of rules, which is the work the package sells.
- **The value, in the bank's own numbers:** value per year = (agent write actions per year) × (error rate without the gate, measured on the bank's own traffic) × (average remediation cost per wrong action, from the bank's dispute and complaint data) + audit hours saved. The bank supplies every input; Glint supplies the gate and its measured with-and-without results on τ²-bench above.
- **The path into a bank** (a plan; the log-only mode in step 1 is not built yet):
  1. Log-only first: the gates run on the bank's own agent traffic and record what they would refuse, without blocking.
  2. One rule per policy clause, each citing its document, reviewed by the bank's policy owners; enforcement turns on clause by clause.
  3. The `no_rule_yet` log is the backlog: every write no rule covers is listed with its tool and reason, so coverage grows from real traffic.

## Limits, each answered
- **The 2026-11-02 test-split run had 18 false refusals of 35 judged** (17 by the request gate, 1 by verify once, against τ²'s gold). *Answer:* PRs #29 onward changed the request gate: reversible writes are allowed and flagged instead of refused, a refusal carries the filled-in `record_requests` call, and PR #41 added a switch that removes the request gate. These changes were fitted to that split. The 2026-11-03 run still had 4 refusals false against gold out of 25 judged (of 40): task_077 trial 0 (`request.not_requested`), task_077 trial 1 and task_080 trial 0 (`verification.once`), task_043 trial 1 (`closure_reason.ineligible`) ([details](docs/rule-changes.md)).
- **No outcome gain on unseen tasks.** On 59 unseen tasks Glimmer alone passed 21 of 108 matched runs and gated 19. Alone's 87 failed matched runs include 30 context overflows, counted as failures; 41 gated runs hit the step limit. *Answer:* the claim is enforcement integrity, which holds by construction; fewer wrong writes depends on which policy clauses the rules encode, and can be measured by each bank on its own traffic, as in the value formula above. A preprint finds the same pattern: enforcement "pays when failures are state-decidable and frequent, and hurts when the gate's judgement is wrong" ([arXiv 2609.25686](https://arxiv.org/abs/2609.25686)).
- **The replays are in-sample.** The 289 gold writes and the 24 of 32 stopped failures come from the tasks the rules were written on. *Answer:* they show the gates work as designed; the precision row above comes from 59 tasks no gate rule came from.
- **A gate checks only what its rules encode.** On those 59 tasks, 31% of live write decisions (206 of 660) were writes no rule covered. *Answer:* every such write is allowed and logged as `no_rule_yet` with the reason, enforcement refuses to start unless every write tool has a gate, and rules have since been added for savings interest and debit disputes ([The gates](docs/gates.md)).
- **Nothing checks the amount of a funds transfer or a card payment yet.** The bank's two money-moving tools, `transfer_funds_between_bank_accounts_7291` and `pay_credit_card_from_checking_9182`, pass the procedure gates: the customer verified in this conversation, a pending credit-limit decision settled first, no repeated call, identifiers from an earlier result, and for payments the customer's recorded request and the tool's own argument bounds. No rule checks the amount, the balance or a documented limit, so a transfer that passes those gates runs and is logged as `no_rule_yet`. *Answer:* the amount rules are the next to encode, from the four policy documents that describe these tools; until then every such write is on the hash-chained record for review.
- **Within-harness only.** Muse Spark plays the customer, so the numbers are not comparable with the τ²-bench leaderboard. *Answer:* both arms run on the same backend and seeds in every comparison except the 2026-11-03 test-split rerun (gated arm only, cross-day), reported by task.
- **The demos are illustrations.** The desk's pairs are selected, and the Muse Code run was scripted. *Answer:* the desk shows each task's real pass counts beside the pair, and the replays run without a GPU for anyone to check.
- **LlamaFirewall was tested at the scanner level only** (a host agent loop stopping the call was not tested), and interactive Muse Code is not shown to work (headless is). *Answer:* both are stated in [docs/llamafirewall.md](docs/llamafirewall.md) and [Limitations in detail](docs/evidence.md#limitations-in-detail).

## More detail
- [docs/evidence.md](docs/evidence.md): what the gates are shown to do, every measured result by task, and the limitations in detail.
- [docs/results.md](docs/results.md): per-run detail behind every result.
- [docs/gates.md](docs/gates.md): what each domain's gates check, the savings interest rules, and the decision tools that came before the gates.
- [docs/gate-classes.md](docs/gate-classes.md): gates on account records vs gates on records the agent makes.
- [docs/rule-changes.md](docs/rule-changes.md): the rules designed from the banking test split (PRs #29-#44).
- [docs/comparison.md](docs/comparison.md): Glint against AgentCore Policy, Invariant, NeMo Guardrails and LlamaFirewall, cell by cell.
- [docs/llamafirewall.md](docs/llamafirewall.md), [plugins/glint-bank/README.md](plugins/glint-bank/README.md), [scripts/README.md](scripts/README.md), [docs/testing-strategy.md](docs/testing-strategy.md).

## Built during the event
- All code in this repo was written during the event (Oct 26 – Nov 4, 2026); no earlier code was copied. The gate runner's and gate log's design follows ideas from the author's earlier private work; no code from it was copied. The first commit is `952dd30` on 2026-10-26.
- This history was rewritten on Oct 31 to remove the author's name, local paths and session links. Authors and dates are unchanged; the only changes to files are local folder paths in comments, now `<workspace>`, and one coverage data file removed. The old-to-new commit map is in [docs/history-map.txt](docs/history-map.txt). The original repository, with its pull request and CI record, is private, and the organizers can be given access. PR numbers written without # ("PR 147") refer to that private record (glint-private); numbers with # ("#45") are this repository's.
- Third-party work, used under its license: [τ²-bench](https://github.com/sierra-research/tau2-bench) (MIT, Sierra Research) is the benchmark and a dependency; its banking data is bundled in `plugins/glint-bank/data/` with its license. The desk's spacing, type and color scales follow Astryx's neutral theme (MIT, Meta Platforms).

## License
[MIT](LICENSE).
