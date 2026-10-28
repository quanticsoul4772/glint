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

## Banking test split (in progress)
A run on τ²'s untouched banking test split (all 20 tasks, k=4). Only one of them (task_080) has a dispute or credit-limit step; the other 19 show whether the tools hurt tasks they were not built for. Results will be added when it ends.

## Superseded: banking on earlier tool code
Before the ce7b242 verdict fixes; kept for the record, because the Bank Desk's replay pairs come from these runs. Results moved a lot between runs at this k.
- Without tools, 42 banking writes went through that Glint's own checks judge wrong, every one in a failed run (corrected: this said 46, and 15 for the tuned tasks, from an older count). 30 of them are in tasks 039, 040 and 041; 11 are in the tuned tasks 038 and 054; the improved untuned tasks had 0 (050) and 1 (052). With the tools: untuned 31 → 7, tuned 11 → 4; the current tools do not repeat the untuned drop (39 → 43, above).
- Pass counts with the tools, by task:
  - **2 of 9 informative untuned tasks improved** (050: 0/2 → 2/2; 052: 0/2 → 1/2), 7 tied, 0 got worse. Task 081 overflowed the context in every run and carries no information. A sign test of 2 vs 0 gives p = 0.5, and "0 worse" is partly built in: 7 of the 9 start at 0/2 and cannot get worse.
  - **2 tuned tasks improved** (038: 0/4 → 2/4; 054: 0/4 → 1/4). The tools were refined after reading these tasks' runs.
- These runs do not measure the shipped code. The evidence is weak: small k, and most tasks start at 0. Glimmer still fails most banking tasks with the tools.
