# Glint: full results

All runs are on τ²-bench v1.0.1 with Muse Glimmer 30B at medium reasoning; both arms of a comparison run with the same seeds on the same backend. Results are reported **by task**, because runs of one task are not independent. Tuned and untuned tasks are reported apart and never pooled. The README carries the headline; this page carries the detail. The Bank Desk's Results tab shows the same numbers per task.

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
- On the 39 tasks where alone never overflowed, alone passed 20 and gated 19, so the overflows do not hide a gated advantage. 20 of the gated arm's failed runs still filed a write with a wrong value, a field the gates do not check, and 65 of its 643 live gate decisions were allows for writes no rule covers.
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
  - 362 writes checked and 59 refusals (55 of writes, 4 of transfers to a human agent). Of the 55 write refusals, 39 refused a write identical to one τ² expects: 0 were false (34 waited for a step the procedure requires first, 5 came in a state that differs from τ²'s path); the other 16 refused writes that differ from the expected ones (9 expedited-shipping asks, 5 address asks, 2 others) and are not judged (`python -m glint.inspect refusals`).
  - 081 trial 0 ended on τ²'s error limit after 10 gate refusals, 2 of them address asks; 081 trials 2 and 3 ended on τ²'s step limit.
  - The address confirmation (new in v9) refused 14 times; the agent followed its fix 9 times, and 3 of the runs it refused in failed. The 4 refusals of transfers to a human agent were all in failed runs.
  - 2 runs overflowed the context (051, 081) and are counted as failures.
  - Replaying τ²'s expected actions for all 77 banking training tasks through the gates: 289 expected writes, 0 false refusals, and the database matches the ungated replay, with the plan and ask-once steps an agent following the gates adds. The 2 denied writes are task_005's: its expected actions change the email of a user verified with placeholder data.
- **Later code:** PR 148 and PR 149 changed refusal wording and which requests count as a credit limit increase after v9 ran; none of their cases occurred in v9's runs.
- **Scoring checked:** re-scoring every gated arm with τ²'s own evaluator, offline and with the same read log, gives the live result for every run (checked through v5).

## Banking test split: no gain (read once, Oct 28)
τ²'s untouched banking test split: all 20 tasks, k=4, both arms on Modal. The tools arm had the decision tools without gates. The run's 12 infrastructure errors were rerun with their own seeds, on a later commit (2e9aa21; the main run was 466ca72).
- 10/80 without the tools vs 10/80 with. By task: 1 improved (047: 0/4 → 1/4), 1 worse (036: 4/4 → 3/4), 18 tied; 15 tasks fail every run in both arms.
- Only task_080 has a dispute or credit-limit step, and a decision tool was called in 1 of the 80 runs (036, trial 2). The split shows the tools neither helped nor hurt tasks they were not built for. It does not measure the tools on their own kind of task.
- Context overflows: 16 runs alone, 12 with the tools; they count as failures.
- There is no gated run on this split (decision 2026-10-28): the phase-2 debit rules were motivated by its failures, so a gated run there would be circular.

## Superseded: banking on earlier tool code
Before the 7789817 verdict fixes; kept for the record, because the Bank Desk's replay pairs come from these runs. Results moved a lot between runs at this k.
- Without tools, 42 banking writes went through that Glint's own checks judge wrong, every one in a failed run (corrected: this said 46, and 15 for the tuned tasks, from an older count). 30 of them are in tasks 039, 040 and 041; 11 are in the tuned tasks 038 and 054; the improved untuned tasks had 0 (050) and 1 (052). With the tools: untuned 31 → 7, tuned 11 → 4; the current tools do not repeat the untuned drop (39 → 43, above).
- Pass counts with the tools, by task:
  - **2 of 9 informative untuned tasks improved** (050: 0/2 → 2/2; 052: 0/2 → 1/2), 7 tied, 0 got worse. Task 081 overflowed the context in every run and carries no information. A sign test of 2 vs 0 gives p = 0.5, and "0 worse" is partly built in: 7 of the 9 start at 0/2 and cannot get worse.
  - **2 tuned tasks improved** (038: 0/4 → 2/4; 054: 0/4 → 1/4). The tools were refined after reading these tasks' runs.
- These runs do not measure the shipped code. The evidence is weak: small k, and most tasks start at 0. Glimmer still fails most banking tasks with the tools.
