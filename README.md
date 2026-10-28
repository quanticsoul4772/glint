# Glint

**Deterministic decision tools that take over the analytical decisions of Meta's open-weight Muse Glimmer 30B.**

A 30B model can hold a customer conversation, but it often gets the decision wrong. Examples: whether a policy allows a cancellation, which variant matches what the customer described, whether a credit-limit increase must come before a dispute. Glint moves those decisions out of the model into small, deterministic tools (policy as code). The model calls a tool, gets an authoritative verdict with its reasons, and acts on it.

Built for the Meta Global AI Developer Hackathon (Indie track, Oct 26 – Nov 4, 2026). All code in this repo was written during the event; every commit is dated Oct 26 or later.

## What is here
| Part | What it does |
|---|---|
| `glint.assist`: decision tools | **Retail:** order eligibility (status, refund methods, price difference, return vs exchange), item finder, variant chooser. **Airline:** reservation eligibility (24-hour rule, cabin changes, flown segments, insurance), upcoming-trip totals. **Banking:** credit-limit increase procedure (tier maximum, the seven required checks in order, approve or deny with the reason) and card disputes (`card_action`, provisional-credit criteria). No model inside. |
| `glint.assist.server` | The same tools as an **MCP server** (`python -m glint.assist.server`, stdio), usable from any MCP client |
| `glint.assist.tau2_tools` | The tools inside [τ²-bench](https://github.com/sierra-research/tau2-bench) toolkits, for measurement |
| `glint.glimmer_serve` | Glimmer 30B **self-hosted** on Modal with vLLM (bf16, A100-80GB, 131K context, scale to zero). Includes a health wait, fail-fast boot log watching, a GPU cap enforced during the session, and a retried stop |
| `glint.atem` | Re-parses Glimmer's tool calls from its raw token stream. The hosted parser turned string arguments such as zip codes into numbers |
| `glint.desk`: **Bank Desk** | A web page: the same banking customer, handled by Glimmer alone and by Glimmer with the Glint tools, side by side. **Replay mode** plays recorded runs (the pairs are selected, and the page shows each task's real pass counts). **Live mode** runs a task now with Muse Spark as the customer, with a daily GPU cap |
| `glint.decisions` | Labels each recorded write action "would a Glint tool have overruled this?" |
| `glint.meter`, `runlog`, `calllog` | Spend ledger and caps, per-run events, exact token-id recording |
| `scripts/tools_ab.py` | With/without-tools comparisons on τ² (both arms on the same backend, checkpointed) |

**Meta models used:**
- **Muse Glimmer 30B** (open weights, self-hosted) is the agent;
- **Muse Spark** (Meta Model API) plays the customer and judges natural-language checks.

## What we measured
On τ²-bench, Glimmer at medium reasoning, both arms with the same seeds. Results are reported **by task**, because runs of one task are not independent.

- **Retail and airline:** no gain, and on retail the tools lose runs.
  - **Held-out tasks, current tools** (a fresh draw no tool change came from, k=4): retail 39/40 without the tools vs 35/40 with; airline 37/40 vs 38/40. By task: 2 improved, 14 tied, 4 worse (sign test p = 0.69); retail alone 0 improved, 3 worse. 2 of the 5 retail failures with the tools were caused by `find_variants`, which matches option names as exact strings: it reported the user's "Google Home" as unavailable when the catalog calls it "Google Assistant". The tools are frozen, so this defect is reported, not fixed; one airline gain (task 20) came with no tool calls.
  - **First held-out draw** (now design data): retail 35/40 without the tools vs 34/40 with; airline 18/20 vs 17/20. By task: 1 improved, 16 tied, 3 worse (airline 0, retail 19, retail 110; sign test p = 0.625). Glimmer already passes most of these (88% alone). The draw includes retail task 21, which was already in the failure inventory, and the arms span four code commits. In every recorded run, each wrong write it attempted was refused by the environment itself.
  - The retail and airline tools were changed afterwards from these tasks' failures (commits aa1dadc and ce7b242), so this draw is now design data and does not measure the shipped tools. At least one failure came from a tool hint (fixed: the cancellation workaround is now offered only when the user asks); 6 of the 9 failures with the tools were read.
- **Banking: where the tools may help.** Without tools, 46 banking writes went through that Glint's own checks judge wrong (τ² does not label writes), every one in a failed run. 30 of them are in tasks 039, 040 and 041, which did not improve with the tools; 15 are in the tuned tasks 038 and 054. The improved untuned tasks had 0 (050) and 1 (052) such writes, so these writes do not explain the gains. The labels use the checks from before commit ce7b242, and there is no with-tools count yet. With the tools, by task:
  - **2 of 9 informative untuned tasks improved** (050: 0/2 → 2/2; 052: 0/2 → 1/2), 7 tied, 0 got worse. Task 081 overflowed the context in every run and carries no information. A sign test of 2 vs 0 gives p = 0.5, and "0 worse" is partly built in: 7 of the 9 start at 0/2 and cannot get worse.
  - **2 tuned tasks improved** (038: 0/4 → 2/4; 054: 0/4 → 1/4). The tools were refined after reading these tasks' runs.
  - These banking runs used the tools before a round of verdict fixes (commit ce7b242, 2026-10-27 16:59), so they do not measure the shipped code. The 12-task banking run queued tonight does.
  - The evidence is weak: small k, and most tasks start at 0. Glimmer still fails most banking tasks with the tools.
- **In progress (2026-10-27):** a 12-task banking run at k=4 on the current tools. A check on τ²'s untouched banking test split is planned, not yet run.

The full notes, including corrections, are kept with the project's research files.

## Run it
Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/sierra-research/tau2-bench ../vendor/tau2-bench
git -C ../vendor/tau2-bench checkout fc0055dc4e0a316c3f83133267fbd6faaa770992   # tau2-bench v1.0.1, pinned
uv sync --extra dev
uv run pytest -q
```

- **Bank Desk, replay mode** (no GPU, no keys): `uv run python -m glint.desk.app`, then open http://127.0.0.1:8765.
- **Bank Desk, live mode:** needs a deployed Glimmer endpoint (`modal deploy src/glint/glimmer_serve.py`) and `GLINT_GLIMMER_URL`, `MODAL_PROXY_KEY`, `MODAL_PROXY_SECRET` and `MODEL_API_KEY` set.
- **MCP server:** `uv run python -m glint.assist.server` (stdio).
- **Workspace:** working files (run data, ledger, results, `.env`) live outside the repo, under `GLINT_HOME`, which defaults to the repo's parent folder.

## License
MIT. τ²-bench (MIT, Sierra Research) is a dependency, and its banking tasks appear in the Bank Desk replay data.
