# Glint: full results

All runs are on τ²-bench v1.0.1 with Muse Glimmer 30B at medium reasoning; both arms of a comparison run with the same seeds on the same backend. Results are reported **by task**, because runs of one task are not independent. Tuned and untuned tasks are reported apart and never pooled. The README carries the headline; this page carries the detail. The Bank Desk's Results tab shows the same numbers per task.

## Retail and airline: no gain, and on retail the tools lose runs
- **Held-out tasks, current tools** (a fresh draw no tool change came from, k=4): retail 39/40 without the tools vs 35/40 with; airline 37/40 vs 38/40. By task: 2 improved, 14 tied, 4 worse (sign test p = 0.69); retail alone 0 improved, 3 worse. 2 of the 5 retail failures with the tools were caused by `find_variants`, which matches option names as exact strings: it reported the user's "Google Home" as unavailable when the catalog calls it "Google Assistant". The tools are frozen, so this defect is reported, not fixed; one airline gain (task 20) came with no tool calls.
- **First held-out draw** (now design data): retail 35/40 without the tools vs 34/40 with; airline 18/20 vs 17/20. By task: 1 improved, 16 tied, 3 worse (airline 0, retail 19, retail 110; sign test p = 0.625). Glimmer already passes most of these (88% alone). The draw includes retail task 21, which was already in the failure inventory, and the arms span four code commits. In every recorded run, each wrong write it attempted was refused by the environment itself.
- The retail and airline tools were changed afterwards from these tasks' failures (commits aa1dadc and ce7b242), so this draw is now design data and does not measure the shipped tools. At least one failure came from a tool hint (fixed: the cancellation workaround is now offered only when the user asks); 6 of the 9 failures with the tools were read.

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
  - Plain: gates plus dispute planning, with plain-language deny messages (#87).
  - Terse: the same, with shorter denies.
  - v2: an overcharge dispute is planned as a partial refund, with the customer's choice winning (#101, #102).
  - v3: the card-digits gate (#109); stopped at 3 of 48 runs and not reported.
  - v4: the digits gate checks only what the agent can see (#111).
  - v5: the plan tells the agent to look up the dispute history before filing (#117).
  - v6: seven fixes from v5's failures: card digits on file, a block on other changes while an increase is undecided, the noticed date and keep-or-replace per card in the plan, the dispute-history read required, open items on every reply, and shorter knowledge-base search results with KB_read (#123-#132).
  - v7: v6 plus fixes for its two regressions: a noticed date after τ²'s today is refused, and the replacement step no longer says "first" (#134).
  - v8: v7 plus an ask-once gate recommending expedited shipping for a fraud or theft replacement (#138).
  - v9: v8 plus an open item for an increase verdict with no decision filed (#142), the card tier's expedited fee in that gate's message (#143), an ask-once confirmation before an order to the address on file (#144), and disputes held until a named credit limit increase is decided (#145). Code at #147.
- **v9 on the tuned tasks:** 6 improved (039, 040, 041, 051, 052, 054), 3 tied (031, 038, 081), 1 worse (037: 4/4 → 2/4) against alone; sign test p = 0.0625; runs 7/40 → 25/40.
  - Against v8 (19/40): 5 tasks better, 3 tied, 2 worse (038, 040); p = 0.23. No per-task change from v8 has Fisher p below 0.14.
  - 052 went from 0/4 in v7 and v8 to 3/4. The open item for its unfiled increase verdict appeared in all four runs; the three runs that filed the submit and the deny passed, and the one that filed nothing failed.
  - 054 went from 2/4 to 4/4, but the gate holding disputes until the increase is decided never fired: every run did the increase first on its own. The gain is not credited to a gate (Fisher p = 0.43).
  - 038 went from 2/4 to 0/4 (p = 0.43). In every v9 run the customer declined expedited shipping, which τ²'s expected actions include; in v8 one customer told the $10 fee still chose it. Chance-level.
- **v9 on the untuned tasks:** 050 improved (2/4 → 4/4) and 053 tied at 0/4; runs 2/8 → 4/8. Two tasks cannot reach p < 0.05, and untuned has been 4/8 since v4.
- **Every version is compared with one baseline draw** (alone, run beside plain), so a low draw there would flatter all of them. v9's 25/40 is the highest of nine noisy draws.
- **Gates, v9:**
  - 362 writes checked and 59 refusals (55 of writes, 4 of transfers to a human agent). Of the 55 write refusals, 39 refused a write identical to one τ² expects: 0 were false (34 waited for a step the procedure requires first, 5 came in a state that differs from τ²'s path); the other 16 refused writes that differ from the expected ones (9 expedited-shipping asks, 5 address asks, 2 others) and are not judged (`python -m glint.inspect refusals`).
  - 081 trial 0 ended on τ²'s error limit after 10 gate refusals, 2 of them address asks.
  - Replaying τ²'s expected actions for all 77 banking training tasks through the gates: 289 expected writes, 0 false refusals, and the database matches the ungated replay, with the plan and ask-once steps an agent following the gates adds. The 2 denied writes are task_005's: its expected actions change the email of a user verified with placeholder data.
- **Later code:** #148 and #149 changed refusal wording and which requests count as a credit limit increase after v9 ran; none of their cases occurred in v9's runs.
- **Scoring checked:** re-scoring every gated arm with τ²'s own evaluator, offline and with the same read log, gives the live result for every run (checked through v5).

## Banking test split: no gain (read once, Oct 28)
τ²'s untouched banking test split: all 20 tasks, k=4, both arms on Modal. The tools arm had the decision tools without gates. The run's 12 infrastructure errors were rerun with their own seeds, on a later commit (70af7d7; the main run was a2e9da7).
- 10/80 without the tools vs 10/80 with. By task: 1 improved (047: 0/4 → 1/4), 1 worse (036: 4/4 → 3/4), 18 tied; 15 tasks fail every run in both arms.
- Only task_080 has a dispute or credit-limit step, and a decision tool was called in 1 of the 80 runs (036, trial 2). The split shows the tools neither helped nor hurt tasks they were not built for. It does not measure the tools on their own kind of task.
- Context overflows: 16 runs alone, 12 with the tools; they count as failures.
- There is no gated run on this split (decision 2026-10-28): the phase-2 debit rules were motivated by its failures, so a gated run there would be circular.

## Superseded: banking on earlier tool code
Before the ce7b242 verdict fixes; kept for the record, because the Bank Desk's replay pairs come from these runs. Results moved a lot between runs at this k.
- Without tools, 42 banking writes went through that Glint's own checks judge wrong, every one in a failed run (corrected: this said 46, and 15 for the tuned tasks, from an older count). 30 of them are in tasks 039, 040 and 041; 11 are in the tuned tasks 038 and 054; the improved untuned tasks had 0 (050) and 1 (052). With the tools: untuned 31 → 7, tuned 11 → 4; the current tools do not repeat the untuned drop (39 → 43, above).
- Pass counts with the tools, by task:
  - **2 of 9 informative untuned tasks improved** (050: 0/2 → 2/2; 052: 0/2 → 1/2), 7 tied, 0 got worse. Task 081 overflowed the context in every run and carries no information. A sign test of 2 vs 0 gives p = 0.5, and "0 worse" is partly built in: 7 of the 9 start at 0/2 and cannot get worse.
  - **2 tuned tasks improved** (038: 0/4 → 2/4; 054: 0/4 → 1/4). The tools were refined after reading these tasks' runs.
- These runs do not measure the shipped code. The evidence is weak: small k, and most tasks start at 0. Glimmer still fails most banking tasks with the tools.
