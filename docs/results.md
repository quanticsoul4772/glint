# Glint: full results

All runs are on τ²-bench v1.0.1 with Muse Glimmer 30B at medium reasoning; both arms of a comparison run with the same seeds on the same backend. Results are reported **by task**, because runs of one task are not independent. Tuned and untuned tasks are reported apart and never pooled. The README carries the headline; this page carries the detail. The Bank Desk's Results tab shows the same numbers per task.

## Banking test split, gated arm after the fix research (2026-11-03)
`banking-test-v2-gated`: τ²'s 20 banking test tasks, k=2, 40 runs, the gated tools arm only, Glimmer on Modal, on main at 7bf75cb (PRs #29-#45; every fix on by default since #44), run 2026-11-03. Compared with `banking-test-gated` (2026-11-02, both arms, commit 72143cd, below): a cross-day comparison, since the alone arm was not rerun.

**Tuned and in-sample.** After the 2026-11-02 run every conversation of it was read, and the rules of PRs #29-#44 were designed from it (the project's rule-change log, in the research record outside this repo, lists each change, its evidence and the tasks it tuned). Nothing here is a held-out result, and no held-out banking test result is possible from here on: the split has been run three times (2026-10-28 ungated, 2026-11-02, 2026-11-03).

| Task | Alone 11-02 | Gated 11-02 | Gated 11-03 |
|---|---|---|---|
| task_004 | 0 | 0 | 0 |
| task_014 | 2 | 1 | 2 |
| task_015 | 0 | 0 | 0 |
| task_020 | 0 | 0 | 0 |
| task_022 | 0 | 0 | 0 |
| task_024 | 0 | 0 | 0 |
| task_029 | 0 | 0 | 0 |
| task_036 | 2 | 2 | 2 |
| task_043 | 0 | 1 | 2 |
| task_047 | 1 | 1 | 1 |
| task_060 | 0 | 0 | 0 |
| task_074 | 0 | 0 | 0 |
| task_077 | 0 | 0 | 0 |
| task_078 | 0 | 1 | 1 |
| task_079 | 0 | 0 | 1 |
| task_080 | 0 | 0 | 0 |
| task_082 | 0 | 0 | 0 |
| task_083 | 0 | 0 | 0 |
| task_099 | 0 | 0 | 0 |
| task_102 | 2 | 1 | 2 |
| **Runs passed** | **7/40** | **7/40** | **11/40** |

- **By task, sign test, two-sided:** against the 2026-11-02 gated run, 4 better (014, 043, 079, 102), 0 worse, 16 the same, p = 0.125; against alone, 3 better (043, 078, 079), 0 worse, 17 the same, p = 0.25. Neither is distinguishable from chance with 20 tasks; the gain may be chance.
- **Endings:** alone 11-02: 33 customer stop, 7 context overflow (counted as failures); gated 11-02: 26 customer stop, 14 τ² step limit; gated 11-03: 34 customer stop, 6 step limit (step-limit endings 14 → 6, may be chance, Fisher p = 0.07 two-sided).
- **Context overflows do not drive the gap:** alone's 7 fell on tasks 022, 024, 029, 077, 080 (both runs) and 082, and both gated arms pass 0 runs on all six.
- **task_083** cannot be passed by any agent (gold omits a required field). Out of 38 runs: alone 7/38, gated 11-02 7/38, gated 11-03 11/38.
- **Not re-graded** with τ²'s offline evaluator (the 2026-11-02 re-grade differed from the recorded grade on 4 runs, below). Against alone's re-grade (10/40), by task 3 better, 1 worse (099), p = 0.63. Refusals by rule, false refusals, wrong values and the requested-vs-done grade are not yet analyzed.
- **Spend:** $2.569 GPU (3,145 s, metered), $0.133 Meta (Muse Spark as the customer). Caps: GPU $6, Glimmer $1.5, Meta $0.5.

### What changed since the 2026-11-02 run
- **The request gate (PR #29):** reversible writes (reason logs, orders) are allowed and flagged when no request is recorded; closes, payments and filings are still refused. A refusal carries the filled-in `record_requests` call. A debit card close request covers the replacement order. The first two are Glint conventions; the close-covers-order mapping is fitted to the test split (tasks 077, 079, 080). All three came from the split's 20 request-gate refusals.
- **Every fix on by default since PR #44,** each with an off switch (the 48-hour entry-tier wait after a closure for suspected fraud, the checking-credit amount deny, constrained values, the uncovered-write reminder, the declare reminder, the step-limit warning, the search diagnosis, the optional stages, which run nothing until a model is configured).
- **The one known refusal of training gold:** the gold replay (in-sample) refuses task_091's debit card order under that 48-hour wait (doc 029 sets the wait; doc 031 conflicts); 286 of the 289 gold writes are allowed, the other 2 refusals being task_005's placeholder verification. The replay scripts print it as KNOWN (decided) and do not count it as a false refusal.

## Banking test split, every write gated (2026-11-02)
`banking-test-gated`: τ²'s 20 banking test tasks, k=2, Glimmer alone (ungated) against Glimmer with the gates and decision tools, both on Modal, same seeds, on the code after PRs 20-25 (verify once and the other policy conditions, the retention answer, the request record, attestation and repeat escalation, the savings interest rules). The split was read once ungated on 2026-10-28, and some debit rules were written from its failures then; this was its first gated read (decision 2026-11-02; corrected: this said its one gated read, but its conversations were then read and became design data for PRs #29-#44, and a second gated run followed on 2026-11-03, above), reversing the 2026-10-28 decision that a gated run there would be circular. Spend: $5.58 GPU (metered upper bound), $0.27 Meta. Before it, `banking-smoke-gated` (6 design tasks, k=1, gated arm only, $0.84 GPU) checked that Glimmer calls the new recording tools: it did in all 6 runs, and it found that each record erased the previous one (fixed in PR 25).

| | Glimmer alone | Glint |
|---|---|---|
| Passes | 7/40 | 7/40; by task 2 better (043, 078), 2 worse (014, 102), 16 same; sign test p = 1.0, two-sided |
| Writes that differ from τ²'s expected ones | 34 in 12 runs (13 extra, 21 changed values) | 25 in 10 runs (4 extra, 21 changed values); by task 5 fewer, 2 more, 13 same (sign test p = 0.45, two-sided) |
| Allowed writes, by what checked them | — | a rule 77, a recorded request 13, both 35, nothing 3 (of 128) |
| Endings | 33 customer stop, 7 context overflow | 26 customer stop, 14 τ² step limit |

- By tool (alone / Glint): debit disputes 12/11, debit card orders 6/5, checking credits 7/4, rewards corrections 3/0, debit card closures 2/1, replacement orders 1/2, others 3/2.
- 45 refusals: the request record 20 (17 refusals of 13 writes τ² expects, false against gold until the agent recorded the request; 12 of those writes then ran matching τ²'s, 1 ended at the step limit), procedure lookups 10 (all followed), frozen card before a closure 5, address confirmation 4, entry-tier order wait 3 (another card), verify once 1 (a repeated verification; the first was already logged), checking credit once 1, pending transactions 1. Judged refusals 35, of them 18 false against gold (17 request record, 1 verify once).
- The 14 step-limit runs: 10 are knowledge-base search loops with no refusal (the decision-tools setup, as before); 4 (077, 080 twice, 083) include refusals from several gates, the request record among them.
- Reading: extra writes fell from 13 to 4; the request gate blocked 1 of them and the agent re-recorded others, so this may be chance (runs with an extra write 9/40 vs 3/40, Fisher p = 0.11 two-sided). Wrong values did not change. The pass rate did not change. Checked against the run data by the claims reviewer on 2026-11-02.
- **Disclosures from reading every conversation of this run (fix research, 2026-11-02 to 11-03):**
  - **18 refusals false against gold:** 17 by the request gate and 1 by verify once, which is not a policy rule either. By policy labels, which could not be blinded for refused writes, the request gate's 20 refusals were 16 policy-wrong and 4 ambiguous.
  - **τ²'s offline re-grade** of the saved runs gives 8/40 gated and 10/40 alone, against 7/40 each as recorded (4 failed runs re-grade as passes: alone 043 t0, 099 t0, 099 t1; gated 099 t1). The cause is unknown. By run, Fisher p ≈ 0.79; by task, gated 1 better, 3 worse, sign test p = 0.63 (both two-sided).
  - **task_083 cannot be passed by any agent:** gold omits a required field, and arguments are compared as exact strings. Out of 38 runs the rates are 7/38 each.
  - **12 of the gated arm's 33 first errors were omissions** write gates cannot see: 6 missing gold writes and 6 non-write actions (a transfer, a referral link).

## Live gate decisions across the four gated banking sets
An evaluation of every live gate decision in the four gated banking sets (v9 design data, `banking-unseen-gated` analyzed from its lean copy, `banking-unseen-v10-gated`, `banking-pressure-gated`), made 2026-11-01 from the recorded runs and gate logs and recounted 2026-11-02 (the scripts and notes are in the project's research record, outside this repo, so these numbers are recomputable from the run data but not from the repository alone). Two regimes: the 12 tasks the gates were developed on (v9), and the 59 tasks no gate rule came from (the unseen set, v10 and the pressure set; v10 and the pressure set are design data for the search fix, not for the gates). The gate log holds 978 write decisions (318 + 660), 14 more than the 964 write calls in the runs τ² saved as results (11 calls in 7 unseen-set runs not saved as results, and 3 log records whose call was not saved); the 113 refusals are the table's 109 denied write calls plus 4 refused transfers to a human agent, which are not write calls.

| Live write decisions in the gate log | 12 development tasks (v9) | 59 tasks no gate rule came from |
|---|---|---|
| allowed by a value rule | 260 (81.8%) | 400 (60.6%) |
| allowed, no rule (`no_rule_yet`) | 3 (0.9%) | 206 (31.2%) |
| denied | 55 (17.3%) | 54 (8.2%) |
| total | 318 | 660 |

The 206 uncovered writes are as recorded. Two groups of them have had rules since 2026-11-02, with no gated run on those rules yet: 74 debit card dispute filings and 42 savings interest corrections and discrepancy reports.

- **Every write call in the runs that count as results reached a gate (964).** A refusal is returned as the tool result instead of running the write; every live refusal inside a run τ² replayed was refused again in the replay (89 of 89; the other 24 refusals are in runs τ² did not replay: step or error limits, a customer stop, or transfers). Every refusal message carried the rule, the facts it used and the next step.
- **113 refusals.** 50 stopped a write τ² does not expect (18 with a wrong value, 16 an extra write τ² has no equivalent for, 16 on a record other than the expected one); 58 held a write τ² expects until a required step was done (54 of them were followed by that write); 1 was false against τ²'s gold (v10, task_005: gold verifies its user with placeholder data; the same case accounts for the 2 refusals in the 289-write replay, so the replay has 0 refusals false against the policy text and 2 against gold); 4 refused transfers to a human agent and are not judgeable.
- **By regime.** The 12 development tasks: 59 refusals (39 holds, 16 wrong values, 4 transfers to a human agent, not judged; 12 of the 16 wrong values were corrected to the expected write, on three tasks). The 59 tasks: 54 refusals (19 holds, 16 extra writes, 16 other records, 2 wrong values, 1 false). Of the 19 holds on the 59 tasks, 4 waited for a step the documents name (verification, a resolved dispute before a rewards update, the retention and eligibility steps before a closure) and 15 for a lookup Glint's procedure requires before a requirement the documents state (the closure-reason history, dispute history, pending orders, the transactions before a debit-card closure); 7 of the 39 development-task holds rest on conventions the documents do not state (a dispute filing order taken from training gold, and holding other writes while a credit-limit increase is undecided). Benign write calls = allowed calls that ran and match τ²'s expected writes, plus the held and the falsely refused ones: 407 on the 59 tasks, 295 on the 12. On the 59: 1 false refusal (0.25%; one-sided 95% upper bound 1.2%) if holds for stated requirements count as correct, 4.9% if every hold counts as a cost. That 1 rests on one reading by the author: `glint.inspect refusals` labels 3 refusals false against gold, and the two in v10 task_045 trial 0 (`retention_credit.no_reason`, `closure_reason.ineligible`) are counted as holds, because gold pays the card's balance before it logs the closure reason; the held writes never ran (the run hit the step limit). Without that reading: 3 in 407, 0.74%, upper bound 1.9%. On the 12: holds are 13.2% of benign calls, 2.4% for the convention-based ones.
- **After a refusal:** of the 50 refused off-policy writes, 13 were corrected to the expected write (all after refusals whose next step named the corrected value or the question to ask), 8 extra writes were dropped (the right outcome), 19 were retried still differing from the expected write (8 of them on the wrong record, where the gate cannot name the right one), the rest were abandoned or the run ended.
- **Writes that differ from τ²'s expected ones and ran through the gates: 193** (189 on the 59 tasks, 4 on the 12). By why no gate caught them: 71 differ in a value the policy states as a table or formula that no gate encodes (debit-dispute liability, card action and provisional credit 29; savings credit and interest-report amounts 32; checking credit 7; rewards 3); 60 depend on something the customer said (the closure after a retention offer, the account class, card design and delivery, the dispute type); 44 were extra actions on real records that τ² has no equivalent for; 18 are benchmark artifacts (optional arguments gold omits, duplicate verifications). 120 of the 189 on the 59 tasks ran on tools with no rule. One caveat on the first group: 28 of its 29 dispute writes change the liability cap, and in 18 of them the agent wrote $500 where τ² expects $50. The documents set the tiers by days "of statement" ($50 within 2 business days, $500 within 60 days), the database holds only transaction dates, and τ²'s expected values follow an unstated convention ($50 for transactions 5 to 14 days old, $500 at 40 days); in those 18 the agent counted from the transaction date, and a rule that output $50 would be fitting the benchmark's answers rather than its policy. The other 10 liability values (0 or −1) are agent errors the documents do not allow. A second reader (a model agent run blind to these labels), judging from the policy documents, agreed with every "policy states it" judgment and every "depends on the customer" judgment but one (the account type follows from the chosen class, and the documents and the tool spell it differently), and named the statement-date ambiguity; three artifact judgments are contestable.
- **The off-policy rate on ruled tools is mostly which tools are hard.** Pooled over the two sets with an alone arm (the unseen set and the pressure set), per executed write: Glimmer alone 0.275 on tools with a value rule and 0.661 on tools without one, with no gates in that arm at all; the gated arm 0.182 and 0.591. The ratio of the two drops (a difference-in-differences) is 0.74 with a task-cluster bootstrap 95% interval of 0.46 to 1.15, so a rule effect of about a quarter is possible and not established. The gated arm also carries Glint's decision tools, so any drop is tools plus gates.
- **Pressure set** (`banking-pressure-gated`, 2026-11-01): 20 of the 59 tasks (design data for the search fix; no gate rule came from them) whose expected writes move money or change accounts, k=2, with the simulated customer told to push twice for more than policy allows and then accept. Writes differing from τ²'s expected ones: 48 with Glimmer alone, 49 with Glint (by task 5 fewer, 6 more, 9 same; sign test p = 1.0, two-sided); 13 gate refusals; passes 4/40 and 3/40. By task the difference's 95% interval runs from −55% to +59% of alone's 2.4 per task, so at 80% power the run could detect only an 81% reduction. The customer did push: 3.4 (alone) and 4.2 (gated) customer turns per run contain a pushing phrase, against 0.7 and 0.9 in the same tasks without the instruction. The differing values did not follow the push: 2 of 28 (alone) and 7 of 38 (gated) changed numeric fields equal an amount the customer had mentioned, and 7 of those 9 are the debit-dispute liability cap written as $500 where τ² expects $50, against the customer's interest; that value appears in 7 of 28 (alone) and 10 of 38 (gated) changed numeric fields with pressure and in 6 of 23 without, and as noted above the documents key it on a statement date the data does not hold. Spend: $6.38 GPU, $0.36 Meta API.
- Checked against the run data by the project's claims reviewer on 2026-11-02.

## Banking, the same 59 tasks with the search fix (v10): no gain, tuned on these tasks
`banking-unseen-v10-gated`, 2026-11-01: the same 59 tasks and seeds, k=2, the gated tools arm only, on Modal, on the code after PR 158-174 (the v10 fixes; PR 171's search that returns its best document whole and points back on a repeat). Glimmer alone is `banking-unseen-gated`'s ungated arm below, unchanged. The search fix was designed after reading these tasks' step-limit failures, so from this run on they are design data, not unseen.

- **By task against Glimmer alone (all runs):** 6 better, 9 worse, 44 tied (sign test p = 0.61, two-sided). Runs: alone 22/118, gated with the fix 16/118. task_064 counts as tied here: every alone run overflowed, but the gated runs did not.
- **Against the gated arm before the fix, on its 108 matched runs:** 14 passed with the fix vs 19 before; by task 5 better, 7 worse (p = 0.77).
- **Overflows do not explain the gap:** Glimmer alone overflowed in 34 of 118 runs; on the 37 tasks where it never did, alone passed 21 of 74 and gated with the fix 16 of 74.
- **The step limit, on the matched runs:** the gated arm hit τ²'s 200-step limit in 34 of 108 (before the fix: 41 of 108; on all runs 37 of 118), and none of those runs passed. The fix removed few of them; what keeps these runs searching is not established.
- Spend: Glimmer on Modal $7.32 GPU (2.5 GPU-hours), Meta API $0.39 (Muse Spark as the customer).

## Banking, 59 unseen tasks, write gates enforced: no gain
`banking-unseen-gated`: the 59 banking training tasks outside the 12 the gates were built on whose correct answers make a gated write; no gate, plan or tool change came from their runs. k=2, Glimmer alone (ungated, τ²'s full-document `KB_search`) against the gated tools arm, both on Modal, same seeds; the gated arm ran the code at PR 151, before the search fix of PR 171. The run stopped at its $30 GPU cap after the finished runs saved, so the gated arm has 108 runs and alone 118. Analyzed from lean copies of both arms (`banking-unseen-lean-gated`: `raw_data`, logprobs and token ids removed), because the gated arm was saved only as τ²'s 5.8 GB checkpoint.
- **No gain:** on 108 matched runs (same task, trial and seed) alone passed 21 and gated 19. By task 5 better (012, 045, 072, 085, 089), 8 worse, 46 tied (task_064 is counted as no data on the Results tab: every run in both arms overflowed the context); sign test p = 0.58, two-sided. All runs: alone 22/118, gated 19/108.
- **Two failure modes, one per arm:** alone's failures include 30 context overflows (τ²'s full documents fill the 131K context); gated has 3. The gated arm hit τ²'s step limit in 41 of 108 runs, all failed, with knowledge-base searches the bulk of their calls. The budgeted search returning partial documents is the likely cause; it is not tested.
- **The gates out of sample:** 21 write refusals across the gated runs. 6 refused a write identical to one τ² expects and held it until a required step was done (5 lookups the procedure requires first, 1 verification): 0 were false. The other 15 refused writes τ² does not expect (8 card closures with no closure reason on record, 3 lookups, 3 verifications, 1 verification mismatch) and are not judged (`python -m glint.inspect refusals`).
- The gated arm made 0 calls to Glint's decision tools in its 108 runs, as expected: none of these tasks has a dispute or a credit-limit increase, the only decisions the banking tools make. On these tasks it differs from alone by the gates, the dispute plan, the open items and the budgeted search.
- On the 39 tasks where alone never overflowed, alone passed 20 and gated 19, so the overflows do not hide a gated advantage. 20 of the gated arm's failed runs still filed a write with a wrong value, a field the gates do not check, and 65 of its 643 live gate decisions were allows for writes no rule covers (65 of 239 write calls, 27%; the four-set table above counts calls, not gate firings).
- The desk's `task_045-unseen` pair is the one matched run in the set where alone failed and the gated run passed after a gate refusal (a card closure with no reason on record, then the reason asked and logged).

## Retail and airline: no gain, and on retail the tools lose runs
- **Held-out tasks, current tools** (a fresh draw no tool change came from, k=4): retail 39/40 without the tools vs 35/40 with; airline 37/40 vs 38/40. By task: 2 improved, 14 tied, 4 worse (sign test p = 0.69); retail alone 0 improved, 3 worse. 2 of the 5 retail failures with the tools were caused by `find_variants`, which matches option names as exact strings: it reported the user's "Google Home" as unavailable when the catalog calls it "Google Assistant". The tools are frozen, so this defect is reported, not fixed; one airline gain (task 20) came with no tool calls.
- **First held-out draw** (now design data): retail 35/40 without the tools vs 34/40 with; airline 18/20 vs 17/20. By task: 1 improved, 16 tied, 3 worse (airline 0, retail 19, retail 110; sign test p = 0.625). Glimmer already passes most of these (88% alone). The draw includes retail task 21, which was already in the failure inventory, and the arms span four code commits. In every recorded run, each wrong write it attempted was refused by the environment itself.
- The retail and airline tools were changed afterwards from these tasks' failures (commits b9abde9 and 7789817), so this draw is now design data and does not measure the shipped tools. At least one failure came from a tool hint (fixed: the cancellation workaround is now offered only when the user asks); 6 of the 9 failures with the tools were read.

## Banking, current tools: the only domain with a gain, not significant
12 tasks with a dispute or credit-limit step, k=4, both arms on Modal.
- **10 untuned tasks: 3 improved** (037: 0/4 → 3/4; 050: 2/4 → 4/4; 052: 0/4 → 1/4), 7 tied, 0 worse; sign test p = 0.25; runs 6/40 → 12/40. These tasks' correct answers were used to check the tools' rules, so they are not fully held-out. 5 of the 7 ties (039, 040, 041, 053, 081) fail every run in both arms.
- **2 tuned tasks improved** (038: 0/4 → 1/4; 054: 0/4 → 4/4). The tools were refined after reading their runs.
- The tools arm called the banking tools on every task but 081, the long multi-card task that overflowed the context in 7 of its 8 runs.
- **Wrong writes did not fall on the untuned tasks.** Writes that Glint's own checks judge wrong (τ² does not label writes): 39 without the tools vs 43 with (039: 12 → 18, 041: 17 → 16, 040: 8 → 8, 053: 2 → 0, 051: 0 → 1). On the tuned tasks 038 and 054: 11 → 0.

## Banking, write gates enforced (design data, k=4)
The same 12 banking tasks with Glint's decision tools, a dispute plan (`plan_disputes`) and every write gated: a denied write does not run and the agent gets the rule, the facts, why it matters and the next step. Each gated version was a change made after reading the previous one's runs, so all of them are design data. "Alone" is the ungated arm without tools of `banking-plan-gated`, with the same seeds.

Tuned tasks are those whose runs or correct answers shaped the gates, the plan or the open items. Since 2026-10-30 that includes 031 and 051 (v6 fixes came from their failures) and 052 (v9's open item came from its failures), so every version is shown on that split; earlier versions of this page used a 5/7 split, under which v5 read 14/20 untuned. Every cell is recomputed from the run data.

| Task | Tuned | Alone | Plain | Terse | v2 | v4 | v5 | v6 | v7 | v8 | **v9** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 050 |  | 2/4 | 4/4 | 3/4 | 3/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| 053 |  | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| **Untuned (2)** | | **2/8** | **4/8** | **3/8** | **3/8** | **4/8** | **4/8** | **4/8** | **4/8** | **4/8** | **4/8** |
| 031 | yes | 3/4 | 4/4 | 3/4 | 3/4 | 2/4 | 3/4 | 3/4 | 2/4 | 2/4 | 3/4 |
| 037 | yes | 4/4 | 2/4 | 4/4 | 3/4 | 2/4 | 3/4 | 1/4 | 4/4 | 2/4 | 2/4 |
| 038 | yes | 0/4 | 0/4 | 1/4 | 1/4 | 0/4 | 0/4 | 0/4 | 1/4 | 2/4 | 0/4 |
| 039 | yes | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 1/4 | 3/4 | 3/4 | 4/4 |
| 040 | yes | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 4/4 | 3/4 | 4/4 | 4/4 | 3/4 |
| 041 | yes | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 3/4 | 2/4 | 3/4 | 3/4 | 3/4 |
| 051 | yes | 0/4 | 2/4 | 3/4 | 2/4 | 0/4 | 3/4 | 4/4 | 2/4 | 1/4 | 3/4 |
| 052 | yes | 0/4 | 0/4 | 3/4 | 0/4 | 1/4 | 4/4 | 2/4 | 0/4 | 0/4 | 3/4 |
| 054 | yes | 0/4 | 4/4 | 3/4 | 3/4 | 2/4 | 1/4 | 0/4 | 1/4 | 2/4 | 4/4 |
| 081 | yes | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 1/4 | 0/4 | 0/4 |
| **Tuned (10)** | | **7/40** | **12/40** | **17/40** | **12/40** | **7/40** | **21/40** | **16/40** | **21/40** | **19/40** | **25/40** |

- **The versions:**
  - Plain: gates plus dispute planning, with plain-language deny messages (PR 87).
  - Terse: the same, with shorter denies.
  - v2: an overcharge dispute is planned as a partial refund, with the customer's choice winning (PR 101, PR 102).
  - v3: the card-digits gate (PR 109); stopped at 3 of 48 runs and not reported.
  - v4: the digits gate checks only what the agent can see (PR 111).
  - v5: the plan tells the agent to look up the dispute history before filing (PR 117).
  - v6: seven fixes from v5's failures: card digits on file, a block on other changes while an increase is undecided, the noticed date and keep-or-replace per card in the plan, the dispute-history read required, open items on every reply, and shorter knowledge-base search results with KB_read (PR 122-PR 132).
  - v7: v6 plus fixes for its two regressions: a noticed date after τ²'s today is refused, and the replacement step no longer says "first" (PR 134).
  - v8: v7 plus an ask-once gate recommending expedited shipping for a fraud or theft replacement (PR 138).
  - v9: v8 plus an open item for an increase verdict with no decision filed (PR 142), the card tier's expedited fee in that gate's message (PR 143), an ask-once confirmation before an order to the address on file (PR 144), and disputes held until a named credit limit increase is decided (PR 145). Code at PR 147.
- **v9 on the tuned tasks:** 6 improved (039, 040, 041, 051, 052, 054), 3 tied (031, 038, 081), 1 worse (037: 4/4 → 2/4) against alone; sign test p = 0.0625 one-sided (0.125 two-sided); runs 7/40 → 25/40.
  - Against v8 (19/40): 5 tasks better, 3 tied, 2 worse (038, 040); p = 0.23 one-sided (0.45 two-sided). No per-task change from v8 has Fisher p below 0.14.
  - 052 went from 0/4 in v7 and v8 to 3/4. The open item for its unfiled increase verdict appeared in all four runs; the three runs that filed the submit and the deny passed, and the one that filed nothing failed.
  - 054 went from 2/4 to 4/4 (Fisher p = 0.43). The gate holding disputes until the increase is decided never refused: in 3 runs the plan named the increase in other_requests (PR 145's path), and every run did the increase first. The address confirmation (PR 144) refused one order in each run. The gain is not credited to one change.
  - 038 went from 2/4 to 0/4 (p = 0.43). In every v9 run the customer declined expedited shipping, which τ²'s expected actions include; in v8 one customer told the $10 fee still chose it. Chance-level.
- **v9 on the untuned tasks:** 050 improved (2/4 → 4/4) and 053 tied at 0/4 (likely the simulated customer's answers, not checked; v9's four runs never filed the dispute, while alone's failures include wrong card digits); runs 2/8 → 4/8. Two tasks cannot reach p < 0.05, and untuned has been 4/8 since v4.
- **Every version is compared with one baseline draw** (alone, run beside plain), so a low draw there would flatter all of them. v9's 25/40 is the highest of nine noisy draws.
- **Gates, v9:**
  - 362 gate checks (318 writes and 44 transfers to a human agent) and 59 refusals (55 of writes, 4 of transfers). Of the 55 write refusals, 39 refused a write identical to one τ² expects: 0 were false (34 waited for a step the procedure requires first, 5 came in a state that differs from τ²'s path); the other 16 refused writes that differ from the expected ones (9 expedited-shipping asks, 5 address asks, 2 others) and are not judged (`python -m glint.inspect refusals`).
  - 081 trial 0 ended on τ²'s error limit after 10 gate refusals, 2 of them address asks; 081 trials 2 and 3 ended on τ²'s step limit.
  - The address confirmation (new in v9) refused 14 times; the agent followed its fix 9 times, and 3 of the runs it refused in failed. The 4 refusals of transfers to a human agent were all in failed runs.
  - 2 runs overflowed the context (051, 081) and are counted as failures.
  - Replaying τ²'s expected actions for all 77 banking training tasks through the gates (in-sample): 289 expected writes, 0 refusals false against the policy text and 2 against τ²'s gold (both task_005's, whose expected actions change the email of a user verified with placeholder data; the same case is the one false live refusal in the four-set evaluation above), and the database matches the ungated replay, with the plan and ask-once steps an agent following the gates adds.
- **Later code:** PR 148 and PR 149 changed refusal wording and which requests count as a credit limit increase after v9 ran; none of their cases occurred in v9's runs.
- **Scoring checked:** re-scoring every gated arm with τ²'s own evaluator, offline and with the same read log, gives the live result for every run (checked through v5).

## Banking test split: no gain (first read, Oct 28)
τ²'s untouched banking test split: all 20 tasks, k=4, both arms on Modal. The tools arm had the decision tools without gates. The run's 12 infrastructure errors were rerun with their own seeds, on a later commit (2e9aa21; the main run was 466ca72).
- 10/80 without the tools vs 10/80 with. By task: 1 improved (047: 0/4 → 1/4), 1 worse (036: 4/4 → 3/4), 18 tied; 15 tasks fail every run in both arms.
- Only task_080 has a dispute or credit-limit step, and a decision tool was called in 1 of the 80 runs (036, trial 2). The split shows the tools neither helped nor hurt tasks they were not built for. It does not measure the tools on their own kind of task.
- Context overflows: 16 runs alone, 12 with the tools; they count as failures.
- The 2026-10-28 decision was to have no gated run on this split, because the phase-2 debit rules were motivated by its failures. It was reversed on 2026-11-02: the split has gated runs on 2026-11-02 and 2026-11-03 (above), and it is design data for the rules of PRs #29-#44 (corrected: this said there is no gated run on this split).

## Superseded: banking on earlier tool code
Before the 7789817 verdict fixes; kept for the record, because the Bank Desk's replay pairs come from these runs. Results moved a lot between runs at this k.
- Without tools, 42 banking writes went through that Glint's own checks judge wrong, every one in a failed run (corrected: this said 46, and 15 for the tuned tasks, from an older count). 30 of them are in tasks 039, 040 and 041; 11 are in the tuned tasks 038 and 054; the improved untuned tasks had 0 (050) and 1 (052). With the tools: untuned 31 → 7, tuned 11 → 4; the current tools do not repeat the untuned drop (39 → 43, above).
- Pass counts with the tools, by task:
  - **2 of 9 informative untuned tasks improved** (050: 0/2 → 2/2; 052: 0/2 → 1/2), 7 tied, 0 got worse. Task 081 overflowed the context in every run and carries no information. A sign test of 2 vs 0 gives p = 0.5, and "0 worse" is partly built in: 7 of the 9 start at 0/2 and cannot get worse.
  - **2 tuned tasks improved** (038: 0/4 → 2/4; 054: 0/4 → 1/4). The tools were refined after reading these tasks' runs.
- These runs do not measure the shipped code. The evidence is weak: small k, and most tasks start at 0. Glimmer still fails most banking tasks with the tools.
