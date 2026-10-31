"""Glimmer with vs without the decision tools. Same tasks and seeds in both arms: arm "base" runs first with τ²'s own
tools; arm "tools" then installs glint.assist (retail: check_order_action, find_user_items, find_variants; airline:
check_reservation_action, get_upcoming_trips_cost; banking: check_credit_limit_increase, check_card_dispute).
Glimmer at medium strength through glint.atem; contributor Spark user simulator and NL judge. One meter logger.

Backends:
- DeepInfra (default): hosted Glimmer, paid per token.
- `--modal`: Glimmer served by glint.glimmer_serve on Modal's free credit. The whole run is one `session()`
  (deploy, wait for /health with the log watch, stop and meter the GPU time at the end), and both arms run there.
  Runs from the two backends are never mixed in one comparison (data under `<set>_modal`).

Sets:
- `design`: the tasks the tools were built from (research/3-project/tool-offload-research.md): retail 20, 93, 82 and
  airline 7; k=8; caps Glimmer $2, Meta $0.50.
- `heldout`: 10 retail + 10 airline train tasks drawn at random (seed 2026) after excluding every task used to design
  the tools or classified in the failure inventory; test-40 is never used; caps Glimmer $3.50, Meta $0.50 (size
  chosen with parallax `decide`). Retail k=4; airline k=2, because about $2 was left on DeepInfra: trials 0-1 of the
  base arm were already done, and τ² derives trial seeds as a prefix-stable sequence, so k=2 reuses them unchanged.
- `banking`: the banking tasks whose gold has a credit card dispute or a credit limit increase, outside the untouched
  test split (`5-measurements/split-partition-banking.json`); 038 and 054 are the recorded failures. These are also
  the tasks whose gold calls the banking tools are checked against (tests/test_assist.py), so they are design tasks,
  not held-out. k=4, retrieval bm25_grep, output capped at 8,192 tokens; caps Meta $1, GPU $6.

Runs go domain by domain (base then tools), so a domain that finishes has both arms.

- `banking-failures`: 038 and 054 only, run first; shares `banking`'s stamp and checkpoint.
- `heldout2`: a fresh draw of 10 retail + 10 airline train tasks (seed 2027) excluding the design tasks, the whole
  failure inventory (twin-B 21, 44, 78 included) and every task of the first held-out draw; k=4.
- `banking-other`: the other 10 banking tasks of `banking` (not tuned on), k=2.
- `gate-banking`: activation-gate training data, no-tools arm only: the 12 banking tasks x 25 trials, concurrency 12.
- `banking-high-alone`: Glimmer alone at high reasoning on the banking12 tasks, k=4, base arm only, Modal only. A
  set's `strength` (low, medium or high; default medium) sets Glimmer's reasoning for every arm of that set.
- `banking-p2-gated`: `banking-gated` again on the phase-1 + phase-2 gates, in its own folder (`banking-gated` ran
  on phase 1 only).
- `banking-plan-gated`: `banking-p2-gated` again after dispute planning (#85, #86) and the plain-language deny
  messages (#87), in its own folder. It measures the two changes together, not either one alone.
- `banking-plan-terse-gated`: the tools arm of `banking-plan-gated` only, with `deny_wording` "terse": the same gates
  and dispute planning, with each deny in #86's layout (Reason, Purpose, Fix, Verdict; the facts joined into the
  Reason line, no consequence line).
  Against `banking-plan-gated`'s tools arm it separates the messages from the planning (decision 2026-10-29). Set
  key `deny_wording` (default "plain") is how the tools arm's denies read (glint.gates.core.WORDINGS).
- `banking-plan-v2-gated`: the tools arm of `banking-plan-gated` only, after the incorrect_amount dispute became a
  planned partial refund of the overcharge with the customer's choice winning and neutral asking (#101, #102). It is
  compared with `banking-plan-gated`'s arms from the same day (its base arm runs no gates, so these changes cannot
  touch it).
- `banking-plan-v3-gated`: the same tools arm after the card-digits gate (#109): a credit card dispute carries the
  card's real last 4 digits, from the account or from the customer's own tool after it was given. Stopped at 3 of 48
  runs: #109 checked the customer-side digits with a copy of τ²'s tool code and printed them in its block.
- `banking-plan-v4-gated`: the same tools arm on #111: the card-digits gate checks only what the agent can see (the
  customer was given the tool; the digits on the account record), and no block names a digit.
- `banking-plan-v5-gated`: the same tools arm on #117: plan_disputes tells the agent to look up the dispute history
  before filing (τ² scores that read where its gold makes it; 12 failed v4 runs skipped only it).
- `banking-plan-v6-gated`: the same tools arm on the v6 fixes from the 23 failed v5 runs (#121-#129, research/
  v6-plan.md): card digits as 4 numerals and from the record when on file; other changes wait for an undecided
  increase; the plan asks the noticed date and keep-or-replace per card; the dispute-history read is required;
  replies list the open items; and KB_search/grep return excerpts within a context budget, with KB_read. The
  shorter search results reach this arm only (the compared base arm keeps τ²'s whole documents).
- `banking-plan-v7-gated`: the same tools arm on v6 plus the two fixes for v6's regressions (#134): the plan refuses
  a noticed date after τ²'s today and names today in its question, and its replacement step no longer says "first"
  (research/5-measurements/v6-diagnosis-2026-10-30.md).
- `banking-plan-v8-gated`: the v7 tools arm plus the ask-first gate on expedited shipping (a fraud or theft
  replacement without it is refused once with credit_card_replacements_001's recommendation; the repeat is allowed),
  from v7's largest remaining cause (research/5-measurements/v7-diagnosis-2026-10-30.md).
- `banking-plan-v9-gated`: the v8 tools arm plus four fixes from v8's failures
  (research/5-measurements/v8-diagnosis-2026-10-30.md): an open item for an increase verdict with no decision filed
  (#142), the card tier's fee in the expedite refusal (#143), an ask-once confirmation before an order to the address
  on file (#144), and plan_disputes' other_requests holding disputes until a named increase is decided (#145). 052 is
  tuned from here on (the open item came from its runs), leaving 050 and 053 untuned.
- `banking-plan-v10-gated`: the v9 tools arm plus #148, #149 and four fixes from v9's failures
  (research/5-measurements/v9-next-causes-2026-10-30.md): a ready check_card_dispute names the filing call
  (#153), the open-items footer names the filing tool (#154), the card-digits text names the give call (#155),
  and check_verification (#156). 053 is tuned from here on (fix A came partly from its runs), leaving 050
  untuned.
- `banking-unseen-gated`: the 59 banking train tasks outside banking12 whose gold makes a gated write, k=2, Glimmer
  alone (base, ungated) against the gated tools arm, Modal only. No live run of these tasks drove a fix, so this is
  the evidence on tasks the loop did not tune (their gold is in the gold replays and some gate tests).
- `banking-gated`, `heldout2-gated`: the same tasks, k and caps as `banking` and `heldout2`, with the tools arm ENFORCING the write gates (glint.gates, decision 2026-10-28): every assistant
  write is checked before it runs and a denied write does not run. Modal only. The base arm stays ungated. Each
  domain's gate decisions go to `gates-<domain>.sqlite` in the set's data folder (hash-chained; one file per run).
  There is deliberately no gated set on the banking test split: it was read once, as the final check, and the gate
  rules were shaped partly by its failures (decision 2026-10-28).

Set key `gates` (default true): the tools arm enforces the write gates. The sets recorded before the gates existed set
it to false, so a resume of one of them stays comparable with its recorded runs.

Optional set key `connections`: the size of LiteLLM's shared HTTP connection pool. When present, `generate()`
replaces τ²'s shared clients (sync and async; 10 connections) with ones of that size before any call, still following
Modal's redirects and carrying the logging hooks (`glimmer_serve.use_connections`). τ² runs `concurrency` simulations,
plus the user simulator's calls, on that one pool, so a smaller pool queues requests silently (a POOL WAIT line in the
log reports it). Sets without the key keep τ²'s client unchanged, so their runs stay comparable; no existing set has
it.

Usage: python tools_ab.py [design|heldout|heldout2|banking|banking-failures|banking-other|gate-banking|banking-gated|
                           banking-p2-gated|banking-plan-gated|banking-plan-terse-gated|banking-plan-v2-gated|
                           banking-plan-v3-gated|banking-plan-v4-gated|banking-plan-v5-gated|
                           banking-plan-v6-gated|banking-plan-v7-gated|banking-plan-v8-gated|banking-plan-v9-gated|
                           banking-plan-v10-gated|
                           banking-unseen-gated|
                           heldout2-gated|
                           banking-high-alone]
       [--modal] [--analyze] [--allow-mixed-code] [--rerun-infrastructure-errors]

`--rerun-infrastructure-errors` (needs `--allow-mixed-code` when the code changed): the runs that ended in an
infrastructure error are taken out of τ²'s checkpoint, the arm file and events.jsonl, and kept in
`released-<arm>-<domain>-<time>.json` with the commit that releases them; the resume then reruns exactly those
(task, trial) pairs with their own seeds.
"""
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from contextlib import ExitStack
from contextvars import ContextVar
from pathlib import Path

import quiet_logs  # noqa: F401  (must stay the first import that can load loguru: τ²'s registry logs at import)

# isort: split
import litellm
import tau2.evaluator.evaluator_nl_assertions as nl
from m2_pilot import GLIMMER, META, SPARK, glimmer_args
from tau2.data_model.simulation import TextRunConfig
from tau2.data_model.tasks import Task
from tau2.registry import registry
from tau2.runner.batch import run_domain, run_tasks
from tau2.utils import llm_utils

from glint import glimmer_serve, inspect, paths
from glint.assist.tau2_tools import install
from glint.atem import reparsing
from glint.calllog import recording
from glint.gates import Enforcer, EventLog
from glint.gates import summary as gate_summary
from glint.gates.rules import REQUIRED_GATES
from glint.meter import BudgetExceeded, Caps, Meter, guarded, litellm_callback, register_litellm_prices
from glint.ratelimit import RateLimiter
from glint.runlog import RunEvents, outcome

TOOL_NAMES = {"check_order_action", "find_user_items", "find_variants", "check_reservation_action",
              "get_upcoming_trips_cost", "check_credit_limit_increase", "check_card_dispute"}
# Tasks the tools were designed from, or that are in the failure inventory
DESIGN_TASKS = {"retail": {"20", "93", "82", "52", "98", "103", "104"}, "airline": {"7"}}
MEASURE = paths.MEASURE
RESULTS = paths.RESULTS
BANKING_TASKS = paths.tau2_data() / "tau2" / "domains" / "banking_knowledge" / "tasks.json"
BANKING_SPLIT = RESULTS / "split-partition-banking.json"
BANKING_WRITES = ("file_credit_card_transaction_dispute_4829", "submit_credit_limit_increase_request_7392")
MODAL_CLI = str(Path(__file__).resolve().parents[1] / ".venv" / "Scripts" / "modal.exe")


def heldout_tasks(n: int = 10, seed: int = 2026, exclude: dict[str, set[str]] | None = None) -> dict[str, list[str]]:
    """The first held-out draw used DESIGN_TASKS only, which misses the twin-B tasks of the failure inventory (21, 44,
    78); task 21 was drawn. Kept as it was so the recorded draw reproduces."""
    rng = random.Random(seed)
    out = {}
    for domain in ("retail", "airline"):
        skip = DESIGN_TASKS[domain] | (exclude or {}).get(domain, set())
        ids = [t.id for t in registry.get_tasks_loader(domain)("train") if t.id not in skip]
        out[domain] = sorted(rng.sample(ids, n), key=int)
    return out


# Tasks the tools or their fixes have been tuned on, or that are in the failure inventory: the design tasks, the twin-B
# tasks, and the first held-out draw (its failures produced the airline-hint and return/exchange fixes)
INVENTORY = {"retail": {"21", "44", "78"}, "airline": set()}


def heldout2_tasks(n: int = 10, seed: int = 2027) -> dict[str, list[str]]:
    first = heldout_tasks()
    return heldout_tasks(n, seed, {d: INVENTORY[d] | set(first[d]) for d in ("retail", "airline")})


def banking_test_tasks() -> dict[str, list[str]]:
    """τ²'s untouched banking test split (all 20 tasks): the final check, never used to change the tools."""
    return {"banking_knowledge": sorted(json.loads(BANKING_SPLIT.read_text(encoding="utf-8"))["test_untouched"])}


def banking_tasks() -> dict[str, list[str]]:
    untouched = set(json.loads(BANKING_SPLIT.read_text(encoding="utf-8"))["test_untouched"])
    ids = [t["id"] for t in json.loads(BANKING_TASKS.read_text(encoding="utf-8"))
           if t["id"] not in untouched and any(a["name"] == "call_discoverable_agent_tool"
                                               and a["arguments"].get("agent_tool_name") in BANKING_WRITES
                                               for a in t["evaluation_criteria"].get("actions") or [])]
    return {"banking_knowledge": sorted(ids)}


def banking_unseen_tasks() -> dict[str, list[str]]:
    """The banking train tasks outside banking12 whose gold makes a gated write (59): no live run of these drove a
    tool, gate or plan change. Their gold is in the gold replays, and some write-gate tests use it (decision
    2026-10-30). The test split stays unread."""
    untouched = set(json.loads(BANKING_SPLIT.read_text(encoding="utf-8"))["test_untouched"])
    seen, gated = set(banking_tasks()["banking_knowledge"]), set(REQUIRED_GATES["banking_knowledge"])

    def writes(task: dict) -> set[str]:
        return {a["arguments"].get("agent_tool_name") if a["name"] == "call_discoverable_agent_tool" else a["name"]
                for a in task["evaluation_criteria"].get("actions") or []}

    return {"banking_knowledge": sorted(t["id"] for t in json.loads(BANKING_TASKS.read_text(encoding="utf-8"))
                                        if t["id"] not in untouched | seen and gated & writes(t))}


SETS = {
    "design": {"tasks": lambda: {"retail": ["20", "93", "82"], "airline": ["7"]}, "trials": 8, "glimmer_usd": 2.0,
               "meta_usd": 0.5, "gpu_usd": 4.0, "data": MEASURE / "tools_ab", "out": RESULTS / "tools-ab.json",
               "save": "tools-ab"},
    "heldout": {"tasks": heldout_tasks, "trials": 4, "airline_trials": 2, "glimmer_usd": 3.5, "meta_usd": 0.5,
                "gpu_usd": 6.0, "data": MEASURE / "tools_ab_heldout", "out": RESULTS / "tools-ab-heldout.json",
                "save": "tools-ab-heldout"},
    "heldout2": {"tasks": heldout2_tasks, "trials": 4, "concurrency": 12, "glimmer_usd": 3.5, "meta_usd": 1.0,
                 "gpu_usd": 8.0,
                 "data": MEASURE / "tools_ab_heldout2", "out": RESULTS / "tools-ab-heldout2.json",
                 "save": "tools-ab-heldout2"},
    # fresh folders (2026-10-28): the earlier banking stamp folder holds runs from before the checkpoint fix
    "banking": {"tasks": banking_tasks, "trials": 4, "concurrency": 12, "glimmer_usd": 3.0, "meta_usd": 1.5,
                "gpu_usd": 12.0, "data": MEASURE / "tools_ab_banking12", "out": RESULTS / "tools-ab-banking12.json",
                "save": "tools-ab-banking12"},
    # The final check on τ²'s untouched banking test split; the tools are frozen
    "banking-test": {"tasks": banking_test_tasks, "trials": 4, "concurrency": 12, "glimmer_usd": 3.0, "meta_usd": 2.0,
                     "gpu_usd": 20.0, "data": MEASURE / "tools_ab_banking_test",
                     "out": RESULTS / "tools-ab-banking-test.json", "save": "tools-ab-banking-test"},
    # The 10 banking tasks not tuned on (the banking tools were changed after reading 038 and 054 runs only); k=2 to
    # keep Modal credit for activation capture
    "banking-other": {"tasks": lambda: {"banking_knowledge": [t for t in banking_tasks()["banking_knowledge"]
                                                              if t not in ("task_038", "task_054")]},
                      "trials": 2, "glimmer_usd": 3.0, "meta_usd": 1.0, "gpu_usd": 5.0,
                      "data": MEASURE / "tools_ab_banking_other", "out": RESULTS / "tools-ab-banking-other.json",
                      "save": "tools-ab-banking-other"},
    # Activation-gate data (research/3-project/activation-gate-design.md): no-tools runs only, labelled afterwards by
    # glint.decisions; 12 tasks x 25 trials
    "gate-banking": {"tasks": banking_tasks, "trials": 25, "arms": ("base",), "concurrency": 12, "glimmer_usd": 3.0,
                     "meta_usd": 3.0, "gpu_usd": 30.0, "data": MEASURE / "gate_banking",
                     "out": RESULTS / "gate-banking-runs.json", "save": "gate-banking"},
    # The two recorded failures first. Same stamp and τ² save_to as `banking`, so the full set later resumes these
    # simulations instead of running them again
    "banking-failures": {"tasks": lambda: {"banking_knowledge": ["task_038", "task_054"]}, "trials": 4,
                         "glimmer_usd": 3.0, "meta_usd": 1.0, "gpu_usd": 6.0,
                         "data": MEASURE / "tools_ab_banking_failures", "stamp_dir": MEASURE / "tools_ab_banking",
                         "out": RESULTS / "tools-ab-banking-failures.json", "save": "tools-ab-banking"},
    # Enforced write gates (decision 2026-10-28): the measurement run is banking-gated (the banking12 tasks, k=4)
    "banking-gated": {"tasks": banking_tasks, "trials": 4, "concurrency": 12, "glimmer_usd": 3.0, "meta_usd": 1.5,
                      "gpu_usd": 12.0, "gates": True, "requires_modal": True,
                      "data": MEASURE / "tools_ab_banking12_gated", "out": RESULTS / "tools-ab-banking12-gated.json",
                      "save": "tools-ab-banking12-gated"},
    # Glimmer alone at high reasoning on the banking12 tasks (decision 2026-10-28): the comparison point for "medium
    # behind the gates" vs "high alone" (Meta reports τ³-Banking at high and xhigh only). Base arm only, Modal only.
    "banking-high-alone": {"tasks": banking_tasks, "trials": 4, "arms": ("base",), "strength": "high",
                           "concurrency": 12, "glimmer_usd": 3.0, "meta_usd": 1.0, "gpu_usd": 8.0,
                           "gates": False, "requires_modal": True, "data": MEASURE / "tools_ab_banking12_high",
                           "out": RESULTS / "tools-ab-banking12-high.json", "save": "tools-ab-banking12-high"},
    # The same measurement on the phase-1 + phase-2 gates (decision 2026-10-28): banking-gated ran on phase 1 only
    # (started before phase 2 merged); a fresh folder keeps the two apart instead of resuming across gate versions
    "banking-p2-gated": {"tasks": banking_tasks, "trials": 4, "concurrency": 12, "glimmer_usd": 3.0, "meta_usd": 1.5,
                         "gpu_usd": 12.0, "gates": True, "requires_modal": True,
                         "data": MEASURE / "tools_ab_banking12_p2_gated",
                         "out": RESULTS / "tools-ab-banking12-gated-p2.json", "save": "tools-ab-banking12-gated-p2"},
    # The same measurement after dispute planning and the plain-language deny messages (decision 2026-10-29)
    "banking-plan-gated": {"tasks": banking_tasks, "trials": 4, "concurrency": 12, "glimmer_usd": 3.0,
                           "meta_usd": 1.5, "gpu_usd": 12.0, "gates": True, "requires_modal": True,
                           "data": MEASURE / "tools_ab_banking12_plan_gated",
                           "out": RESULTS / "tools-ab-banking12-gated-plan.json",
                           "save": "tools-ab-banking12-gated-plan"},
    "banking-plan-terse-gated": {"tasks": banking_tasks, "trials": 4, "arms": ("tools",), "concurrency": 12,
                                 "glimmer_usd": 3.0, "meta_usd": 1.0, "gpu_usd": 8.0, "gates": True,
                                 "deny_wording": "terse", "requires_modal": True,
                                 "data": MEASURE / "tools_ab_banking12_plan_terse_gated",
                                 "out": RESULTS / "tools-ab-banking12-gated-plan-terse.json",
                                 "save": "tools-ab-banking12-gated-plan-terse"},
    # The overcharge partial refund with customer choice and neutral asking (#101, #102; decision 2026-10-29)
    "banking-plan-v2-gated": {"tasks": banking_tasks, "trials": 4, "arms": ("tools",), "concurrency": 12,
                              "glimmer_usd": 3.0, "meta_usd": 1.0, "gpu_usd": 8.0, "gates": True,
                              "requires_modal": True, "data": MEASURE / "tools_ab_banking12_plan_v2_gated",
                              "out": RESULTS / "tools-ab-banking12-gated-plan-v2.json",
                              "save": "tools-ab-banking12-gated-plan-v2"},
    # The card-digits gate (#109; decision 2026-10-29)
    "banking-plan-v3-gated": {"tasks": banking_tasks, "trials": 4, "arms": ("tools",), "concurrency": 12,
                              "glimmer_usd": 3.0, "meta_usd": 1.0, "gpu_usd": 8.0, "gates": True,
                              "requires_modal": True, "data": MEASURE / "tools_ab_banking12_plan_v3_gated",
                              "out": RESULTS / "tools-ab-banking12-gated-plan-v3.json",
                              "save": "tools-ab-banking12-gated-plan-v3"},
    # The card-digits gate without the hidden check (#111; decision 2026-10-29)
    "banking-plan-v4-gated": {"tasks": banking_tasks, "trials": 4, "arms": ("tools",), "concurrency": 12,
                              "glimmer_usd": 3.0, "meta_usd": 1.0, "gpu_usd": 8.0, "gates": True,
                              "requires_modal": True, "data": MEASURE / "tools_ab_banking12_plan_v4_gated",
                              "out": RESULTS / "tools-ab-banking12-gated-plan-v4.json",
                              "save": "tools-ab-banking12-gated-plan-v4"},
    # The plan's dispute-history hint (#117; decision 2026-10-29)
    "banking-plan-v5-gated": {"tasks": banking_tasks, "trials": 4, "arms": ("tools",), "concurrency": 12,
                              "glimmer_usd": 3.0, "meta_usd": 1.0, "gpu_usd": 8.0, "gates": True,
                              "requires_modal": True, "data": MEASURE / "tools_ab_banking12_plan_v5_gated",
                              "out": RESULTS / "tools-ab-banking12-gated-plan-v5.json",
                              "save": "tools-ab-banking12-gated-plan-v5"},
    # The v6 fixes from the v5 failure diagnosis (#121-#129; decision 2026-10-30)
    "banking-plan-v6-gated": {"tasks": banking_tasks, "trials": 4, "arms": ("tools",), "concurrency": 12,
                              "glimmer_usd": 3.0, "meta_usd": 1.0, "gpu_usd": 8.0, "gates": True,
                              "requires_modal": True, "data": MEASURE / "tools_ab_banking12_plan_v6_gated",
                              "out": RESULTS / "tools-ab-banking12-gated-plan-v6.json",
                              "save": "tools-ab-banking12-gated-plan-v6"},
    # v6 plus the fixes for its two regressions (#134; decision 2026-10-30)
    "banking-plan-v7-gated": {"tasks": banking_tasks, "trials": 4, "arms": ("tools",), "concurrency": 12,
                              "glimmer_usd": 3.0, "meta_usd": 1.0, "gpu_usd": 8.0, "gates": True,
                              "requires_modal": True, "data": MEASURE / "tools_ab_banking12_plan_v7_gated",
                              "out": RESULTS / "tools-ab-banking12-gated-plan-v7.json",
                              "save": "tools-ab-banking12-gated-plan-v7"},
    # v7 plus the ask-first gate on expedited shipping (decision 2026-10-30)
    "banking-plan-v8-gated": {"tasks": banking_tasks, "trials": 4, "arms": ("tools",), "concurrency": 12,
                              "glimmer_usd": 3.0, "meta_usd": 1.0, "gpu_usd": 8.0, "gates": True,
                              "requires_modal": True, "data": MEASURE / "tools_ab_banking12_plan_v8_gated",
                              "out": RESULTS / "tools-ab-banking12-gated-plan-v8.json",
                              "save": "tools-ab-banking12-gated-plan-v8"},
    # v8 plus the four fixes from its failures (#142-#145; decision 2026-10-30)
    "banking-plan-v9-gated": {"tasks": banking_tasks, "trials": 4, "arms": ("tools",), "concurrency": 12,
                              "glimmer_usd": 3.0, "meta_usd": 1.0, "gpu_usd": 8.0, "gates": True,
                              "requires_modal": True, "data": MEASURE / "tools_ab_banking12_plan_v9_gated",
                              "out": RESULTS / "tools-ab-banking12-gated-plan-v9.json",
                              "save": "tools-ab-banking12-gated-plan-v9"},
    # v9 plus #148, #149 and the four fixes from its failures (#153-#156; decision 2026-10-30)
    "banking-plan-v10-gated": {"tasks": banking_tasks, "trials": 4, "arms": ("tools",), "concurrency": 12,
                               "glimmer_usd": 3.0, "meta_usd": 1.0, "gpu_usd": 8.0, "gates": True,
                               "requires_modal": True, "data": MEASURE / "tools_ab_banking12_plan_v10_gated",
                               "out": RESULTS / "tools-ab-banking12-gated-plan-v10.json",
                               "save": "tools-ab-banking12-gated-plan-v10"},
    # the broad run on tasks no fix came from: Glimmer alone (ungated) vs the gated tools arm, k=2 (decision
    # 2026-10-30, v9 plan step 7)
    "banking-unseen-gated": {"tasks": banking_unseen_tasks, "trials": 2, "concurrency": 12, "glimmer_usd": 3.0,
                             "meta_usd": 1.5, "gpu_usd": 30.0, "gates": True, "requires_modal": True,
                             "data": MEASURE / "tools_ab_banking_unseen_gated",
                             "out": RESULTS / "tools-ab-banking-unseen-gated.json",
                             "save": "tools-ab-banking-unseen-gated"},
    "heldout2-gated": {"tasks": heldout2_tasks, "trials": 4, "concurrency": 12, "glimmer_usd": 3.5, "meta_usd": 1.0,
                       "gpu_usd": 8.0, "gates": True, "requires_modal": True,
                       "data": MEASURE / "tools_ab_heldout2_gated", "out": RESULTS / "tools-ab-heldout2-gated.json",
                       "save": "tools-ab-heldout2-gated"},
}
# Recorded before the gates existed: their tools arm stays ungated so a resume matches the recorded runs
for _name in ("design", "heldout", "heldout2", "banking", "banking-test", "banking-other", "gate-banking",
              "banking-failures"):
    SETS[_name]["gates"] = False


def for_backend(cfg: dict, modal: bool) -> dict:
    if cfg.get("requires_modal") and not modal:
        raise SystemExit("this set runs on Modal only: pass --modal")
    if not modal:
        return cfg
    stamp_dir = cfg.get("stamp_dir", cfg["data"])
    return cfg | {"modal": True, "data": cfg["data"].with_name(cfg["data"].name + "_modal"),
                  "stamp_dir": stamp_dir.with_name(stamp_dir.name + "_modal"),
                  "out": cfg["out"].with_name(cfg["out"].stem + "-modal.json"), "save": cfg["save"] + "-modal"}


def _trials(cfg: dict, domain: str) -> int:
    return cfg.get(f"{domain}_trials", cfg["trials"])


def _agent(cfg: dict, domain: str, endpoint: str | None) -> tuple[str, dict]:
    if endpoint:
        args = glimmer_serve.litellm_args(endpoint, os.environ["MODAL_PROXY_KEY"], os.environ["MODAL_PROXY_SECRET"],
                                          cfg.get("strength", "medium"))
        model = f"openai/{glimmer_serve.MODEL_NAME}"
    else:
        model, args = GLIMMER, glimmer_args(cfg.get("strength", "medium"))
    if domain == "banking_knowledge":
        args = {**args, "max_tokens": 8192}  # without the cap the default 65,536 overflowed the context
    return model, args


def _save_dir(cfg: dict, arm: str, domain: str, stamp: str) -> Path:
    """τ²'s checkpoint folder for one arm, the same path run_domain derives from save_to."""
    from tau2.utils.utils import DATA_DIR

    return DATA_DIR / "simulations" / f"glint-{cfg['save']}-{arm}-{domain}-{stamp}"


def _run(cfg: dict, domain: str, ids: list[str], arm: str, stamp: str, endpoint: str | None):
    model, args = _agent(cfg, domain, endpoint)
    common = {"domain": domain, "agent": "llm_agent", "llm_agent": model, "llm_args_agent": args,
              "user": "user_simulator", "llm_user": SPARK, "llm_args_user": META, "num_trials": _trials(cfg, domain),
              "max_concurrency": cfg.get("concurrency", 8) if endpoint else 4, "seed": 5000,
              "save_to": f"glint-{cfg['save']}-{arm}-{domain}-{stamp}", "auto_review": False, "max_retries": 0,
              "auto_resume": True,
              # the run log gets only τ²'s warnings and errors; each simulation's full debug log goes to its own
              # artifacts/task_<id>/sim_<id>/task.log in the checkpoint folder (95% of the run log was τ² debug)
              "log_level": "WARNING", "verbose_logs": True}
    if domain == "banking_knowledge":
        by_id = {t["id"]: t for t in json.loads(BANKING_TASKS.read_text(encoding="utf-8"))}
        config = TextRunConfig(**common, retrieval_config="bm25_grep")
        # run_tasks persists nothing without save_path (its docstring), so banking runs had no checkpoint: a crash
        # lost every finished run and resume could not work. Pass the paths run_domain would derive.
        save_dir = _save_dir(cfg, arm, domain, stamp)
        return run_tasks(config, [Task.model_validate(by_id[i]) for i in ids], save_path=save_dir / "results.json",
                         save_dir=save_dir)
    return run_domain(TextRunConfig(**common, task_split_name="train", task_ids=ids))


def _commit() -> str:
    import subprocess

    # check=True and a non-empty commit: an empty commit would switch the mixed-code guard off for good
    repo = str(Path(__file__).resolve().parents[1])
    out = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                         check=True).stdout.strip()
    if not out:
        raise SystemExit(f"git rev-parse gave no commit for {repo}; the run cannot record which code it measures")
    dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain", "src"], capture_output=True, text=True,
                           check=True).stdout.strip()
    return out + ("-dirty" if dirty else "")


KEPT_ON_RESUME = ("context_overflow", "empty_reply", "glint_error")  # failed runs: results, never rerun


def keep_failed_runs_on_resume() -> None:
    """τ²'s resume (tau2.runner.checkpoint.try_resume) reruns every simulation whose termination reason is
    infrastructure_error, and rewrites the checkpoint without them. Glint splits that reason (glint.runlog.outcome):
    a context overflow, an empty reply or a Glint error is a failed run, a result, and must not be rerun until it
    passes. This wraps
    the resume τ²'s batch runner calls so those runs stay as recorded, on disk and in the runs it skips; only
    Glint's infrastructure_error and budget_stop runs (not results) are rerun."""
    from tau2.data_model.simulation import Results
    from tau2.runner import batch

    if getattr(batch.try_resume, "glint_keeps_failures", False):
        return
    resume = batch.try_resume

    def wrapped(save_path, simulation_results, tasks, num_trials, auto_resume=False, results_format="json"):
        save_path = Path(save_path)
        given = tasks
        before = Results.load(save_path).simulations if save_path.exists() else []
        if save_path.exists() and Results._detect_format(save_path) != "json":
            raise RuntimeError(f"{save_path}: τ²'s dir checkpoint format; the failed-run keeper handles json only")
        keep = [s for s in before if outcome(s.model_dump(mode="json")) in KEPT_ON_RESUME]
        results, done, tasks = resume(save_path, simulation_results, tasks, num_trials, auto_resume, results_format)
        present = {s.id for s in results.simulations}
        back = [s for s in keep if s.id not in present]
        if back:
            results.simulations.extend(back)
            done |= {(s.trial, s.task_id, s.seed) for s in back}
            save_path.write_text(results.model_dump_json(indent=2), encoding="utf-8")
            print(f"KEPT {len(back)} failed runs (context overflow, empty reply or Glint error) as recorded; τ²'s "
                  f"resume would have rerun them: {sorted((s.task_id, s.trial) for s in back)}", flush=True)
        if before:  # τ²'s own "Resuming run from N runs. M runs remaining." does not count the runs kept above
            have = {(task_id, trial) for trial, task_id, _ in done}
            pairs = [(t.id, trial) for t in given for trial in range(num_trials)]
            rerun = sorted(p for p in pairs if p not in have)
            print(f"RESUME: {len(pairs) - len(rerun)} runs kept, rerunning {len(rerun)}: {rerun}", flush=True)
        return results, done, tasks

    wrapped.glint_keeps_failures = True
    batch.try_resume = wrapped


# τ²'s two ERROR records for a failed simulation, (logger name, function, line): the failed model call and the failed
# run. The `[runs]` line (glint.runlog.RunEvents) carries the same outcome and cause once. A record from any other place
# still prints; if τ² moves these lines, they print again rather than anything else being dropped.
# (the orchestrator's "Simulation loop exited with an exception" names no cause; the run's own line, with info.error,
# follows when the run finishes)
RUN_LOG_DROPPED = {("tau2.utils.llm_utils", "generate", 417), ("tau2.runner.progress", "run_with_retry", 101),
                   ("tau2.orchestrator.orchestrator", "run", 287)}
# τ² console lines left out of the run log: a resume prints one per finished run (the RESUME line lists what reruns),
# and a failed run's error is printed again, wrapped, after its two ERROR records
# τ²'s "N/M (trial t/k). Running task X, trial t" start line is dropped too: the run's own line reports its end
QUIET_LINES = (re.compile(r"Skipping task \S+, trial \d+ because it has already been run\."),
               re.compile(r"  Task \S+ failed permanently after \d+ attempts: .*", re.DOTALL),
               re.compile(r".*\. Running task \S+, trial \d+"))
# τ²'s "Status: n/N complete ..." progress line prints every few seconds while runs are open (313 lines in a 96-run
# log); one every STATUS_EVERY_S is kept, so the log still shows what is running
STATUS = re.compile(r"Status: \d+/\d+ complete\..*", re.DOTALL)
STATUS_EVERY_S = 600.0
# The (task id, trial) of the simulation the current thread runs, set around τ²'s run_with_retry
CURRENT_RUN: ContextVar[tuple[str, int] | None] = ContextVar("glint_current_run", default=None)


def run_log_filter(record) -> bool:
    """loguru filter for the run log: False for the records RUN_LOG_DROPPED names by their source line only (τ²'s two
    duplicate ERROR records of a failed run, and its cause-free simulation loop WARNING)."""
    return (record["name"], record["function"], record["line"]) not in RUN_LOG_DROPPED


class _RunLogLogger:
    """τ²'s batch module logger with `run_log_filter` on the console handler run_tasks adds (it removes every handler
    first, so a filter added before it would be lost). The per-simulation task.log handler, a file with its own
    filter, is added unchanged, so each simulation's own log keeps every record."""

    def __init__(self, real):
        self._real = real

    def add(self, sink, **kwargs):
        if callable(sink) and kwargs.get("filter") is None:
            kwargs["filter"] = run_log_filter
        return self._real.add(sink, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


class _QuietConsole:
    """τ²'s shared rich console without QUIET_LINES, and with one Status line every STATUS_EVERY_S."""

    def __init__(self, real, clock=time.monotonic):
        self._real, self._clock, self._last_status = real, clock, None

    def print(self, *objects, **kwargs):
        from rich.text import Text

        if len(objects) == 1 and isinstance(objects[0], Text):
            plain = objects[0].plain
            if any(p.fullmatch(plain) for p in QUIET_LINES):
                return None
            if STATUS.fullmatch(plain):
                now = self._clock()
                if self._last_status is not None and now - self._last_status < STATUS_EVERY_S:
                    return None
                self._last_status = now
        # one log line per message: rich wraps at the console width (80 in a run log), so one Status line was 4
        kwargs.setdefault("soft_wrap", True)
        return self._real.print(*objects, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


def quiet_run_log() -> None:
    """Keep τ²'s per-run noise out of the run log; each simulation's own artifacts keep it (results.json holds the
    run the evaluation table summarized; artifacts/task_<id>/sim_<id>/task.log holds every log record).
    - the evaluation table after each simulation: run_with_retry is called with console_display=False;
    - the "Skipping task ..." lines of a resume and the wrapped repeat of a failed run's error (QUIET_LINES);
    - τ²'s two ERROR records for a failed simulation (RUN_LOG_DROPPED);
    - LiteLLM's "Provider List" / "Give Feedback / Get Help" boilerplate printed after every error."""
    from tau2.runner import batch
    from tau2.utils.display import ConsoleDisplay

    litellm.suppress_debug_info = True
    if not isinstance(ConsoleDisplay.console, _QuietConsole):
        ConsoleDisplay.console = _QuietConsole(ConsoleDisplay.console)
    if not isinstance(batch.logger, _RunLogLogger):
        batch.logger = _RunLogLogger(batch.logger)
    if getattr(batch.run_with_retry, "glint_quiet", False):
        return
    run_with_retry = batch.run_with_retry

    def quiet(run_fn, *, task, trial, **kwargs):
        token = CURRENT_RUN.set((task.id, trial))
        try:
            return run_with_retry(run_fn, task=task, trial=trial, **(kwargs | {"console_display": False}))
        finally:
            CURRENT_RUN.reset(token)

    quiet.glint_quiet = True
    batch.run_with_retry = quiet


def current_run() -> tuple[str, str]:
    """(task id, trial) of the simulation this thread runs, from the run_with_retry wrapper, else the task from τ²'s
    llm_log_dir (artifacts/task_<id>/sim_<id>/llm_debug); "unknown" for what cannot be found."""
    run = CURRENT_RUN.get()
    if run is not None:
        return run[0], str(run[1])
    log_dir = llm_utils.llm_log_dir.get()
    task_dir = Path(log_dir).parent.parent.name if log_dir is not None else ""
    return (task_dir[len("task_"):] if task_dir.startswith("task_") else "unknown"), "unknown"


def run_context() -> dict:
    """The run a gate event or a model call belongs to: task_id, trial (a number, as in events.jsonl and τ²'s
    results) and simulation_id, τ²'s id for this attempt, from its per-run log folder (artifacts/task_<id>/
    sim_<id>/llm_debug; tools_ab runs with verbose_logs). A resumed rerun keeps (task, trial) and gets a new
    simulation id, so its gate events no longer mix with the first attempt's (2026-10-30 review)."""
    task_id, trial = current_run()
    log_dir = llm_utils.llm_log_dir.get()
    sim_dir = Path(log_dir).parent.name if log_dir is not None else ""
    return {"task_id": task_id, "trial": int(trial) if trial.isdigit() else trial,
            "simulation_id": sim_dir[len("sim_"):] if sim_dir.startswith("sim_") else None}


SAVE_TRIES, SAVE_WAIT_S = 6, 0.5


def retry_locked_checkpoint_saves() -> None:
    """τ² saves its checkpoint by renaming a temp file over results.json, once. On Windows the rename fails with
    "[WinError 5] Access is denied" while another process has results.json open (the event watcher reads it every
    30 s; antivirus and indexers also open files), and τ² then drops the finished run as an infrastructure error
    (2026-10-28, banking-test task_022 trial 3). This wraps the save τ²'s batch runner uses: a refused save is
    retried a few times, each refusal printed; if the lock outlasts the retries the error is raised as before."""
    from tau2.runner import batch

    if getattr(batch.create_checkpoint_fns, "glint_retries_saves", False):
        return
    create = batch.create_checkpoint_fns

    def retrying(fn, what):
        def call(*args, **kwargs):
            for attempt in range(1, SAVE_TRIES + 1):
                try:
                    return fn(*args, **kwargs)
                except PermissionError as e:
                    print(f"CHECKPOINT LOCKED: τ²'s {what} was refused (attempt {attempt}/{SAVE_TRIES}): {e}",
                          flush=True)
                    if attempt == SAVE_TRIES:
                        raise
                    time.sleep(SAVE_WAIT_S)
        return call

    def wrapped(save_path, lock):
        save_fn, replace_fn = create(save_path, lock)
        return retrying(save_fn, "checkpoint save"), retrying(replace_fn, "checkpoint replace")

    wrapped.glint_retries_saves = True
    batch.create_checkpoint_fns = wrapped


def release_infrastructure_errors(cfg: dict, stamp: str, commit: str) -> int:
    """Take the infrastructure-error runs out of each arm's checkpoint so a resume reruns them (see the docstring
    above). Nothing is deleted: the released runs are written to a file first. Returns how many were released."""
    data, released = cfg["data"], 0
    tasks = json.loads((data / "tasks.json").read_text(encoding="utf-8")) if (data / "tasks.json").exists() else {}
    for domain in tasks:
        for arm in cfg.get("arms", ("base", "tools")):
            checkpoint = _save_dir(cfg, arm, domain, stamp) / "results.json"
            if not checkpoint.exists():
                continue
            results = json.loads(checkpoint.read_text(encoding="utf-8"))
            drop = [s for s in results["simulations"] if outcome(s) == "infrastructure_error"]
            if not drop:
                continue
            keys = {(s["task_id"], s["trial"]) for s in drop}
            side = data / f"released-{arm}-{domain}-{time.strftime('%Y%m%d-%H%M%S')}.json"
            side.write_text(json.dumps({"released_on_commit": commit, "simulations": drop}), encoding="utf-8")
            results["simulations"] = [s for s in results["simulations"] if (s["task_id"], s["trial"]) not in keys]
            checkpoint.write_text(json.dumps(results), encoding="utf-8")
            (data / f"{arm}-{domain}.json").unlink(missing_ok=True)  # the arm runs again; τ² skips the kept runs
            events = data / "events.jsonl"
            if events.exists():
                rows = [json.loads(x) for x in events.read_text(encoding="utf-8").splitlines()]
                events.write_text("".join(json.dumps(r) + "\n" for r in rows if not (
                    r["label"] == f"{arm} {domain}" and r["outcome"] == "infrastructure_error"
                    and (r["task_id"], r["trial"]) in keys)), encoding="utf-8")
            with (cfg.get("stamp_dir", data) / "reruns.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.time(), "commit": commit, "arm": arm, "domain": domain,
                                    "runs": sorted(keys)}) + "\n")
            print(f"RELEASED {len(drop)} infrastructure-error runs of {arm} {domain} for a rerun on {commit}: "
                  f"{sorted(keys)}; kept in {side.name}", flush=True)
            released += len(drop)
    return released


def empty_reply_line(response) -> str | None:
    """An `EMPTY REPLY:` line for a Glimmer response with no text and no tool call (τ² then ends the run with "must have
    either content or tool_calls", outcome empty_reply), with its finish reason and token counts; else None."""
    choice = response.choices[0]
    message = choice.message
    if (message.content or "").strip() or message.tool_calls:
        return None
    task_id, trial = current_run()
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None)
    tokens = (f"prompt {usage.prompt_tokens} tokens, completion {usage.completion_tokens}"
              + (f" (reasoning {reasoning})" if reasoning is not None else "")) if usage else "no token counts"
    return f"EMPTY REPLY: {task_id} trial {trial}: finish_reason {choice.finish_reason}, {tokens}"


def _enforcer(cfg: dict, domain: str, data: Path) -> Enforcer | None:
    """The write gates for the tools arm (None when the set has `gates` false); decisions go to one log per run."""
    if not cfg.get("gates", True):
        return None
    return Enforcer(domain, EventLog(data / f"gates-{domain}.sqlite"),
                    context=run_context,
                    wording=cfg.get("deny_wording", "plain"))


def rerun_command(argv: list[str] | None = None) -> str:
    """The launch.sh command that reruns this set's infrastructure errors (the guard refuses a bare tools_ab.py)."""
    args = [a for a in (sys.argv[1:] if argv is None else argv)
            if a not in ("--rerun-infrastructure-errors", "--allow-mixed-code")]
    return (f"bash scripts/launch.sh LOG -- python -X utf8 scripts/tools_ab.py {' '.join(args)} "
            "--rerun-infrastructure-errors (add --allow-mixed-code if the code changed)")


def generate(cfg: dict) -> None:
    # One stamp per comparison: a resumed run keeps the same meter phases (so the caps count earlier spend) and the
    # same τ² save_to (so finished simulations are resumed, not paid for again)
    data = cfg["data"]
    data.mkdir(parents=True, exist_ok=True)
    stamp_dir = cfg.get("stamp_dir", data)
    stamp_dir.mkdir(parents=True, exist_ok=True)
    stamp_file = stamp_dir / "stamp.txt"
    commit_file = stamp_dir / "commit.txt"
    commit = _commit()
    if not stamp_file.exists():
        stamp_file.write_text(time.strftime("%Y%m%d-%H%M%S"), encoding="utf-8")
        commit_file.write_text(commit, encoding="utf-8")
    stamp = stamp_file.read_text(encoding="utf-8").strip()
    # A resume on different code would merge runs of two code versions into one comparison (the first held-out
    # arms span four commits)
    started = commit_file.read_text(encoding="utf-8").strip() if commit_file.exists() else None
    if started == "" and "--allow-mixed-code" not in sys.argv:
        raise SystemExit(f"{commit_file} is empty: which code this comparison started on is unknown. Write the "
                         "starting commit into it, or pass --allow-mixed-code to resume knowingly")
    if started and started != commit and "--allow-mixed-code" not in sys.argv:
        raise SystemExit(f"this comparison started on commit {started}, the code is now {commit}; resuming would mix "
                         "versions. Start a fresh stamp, or pass --allow-mixed-code.")
    if started and started != commit:
        print(f"MIXED CODE: this comparison started on {started}, the code is now {commit}; τ²'s \"Run config has "
              "changed\" warning below is expected", flush=True)
    keep_failed_runs_on_resume()
    retry_locked_checkpoint_saves()
    if "--rerun-infrastructure-errors" in sys.argv and not release_infrastructure_errors(cfg, stamp, commit):
        raise SystemExit("--rerun-infrastructure-errors: no infrastructure-error runs to rerun")
    tasks = cfg["tasks"]()
    (data / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
    pending = [(d, ids, arm) for d, ids in tasks.items() for arm in cfg.get("arms", ("base", "tools"))
               if not (data / f"{arm}-{d}.json").exists()]
    if not pending:
        return
    if cfg.get("connections"):  # before any call: the pool size every call of the run shares (see the docstring)
        glimmer_serve.use_connections(cfg["connections"])
    glim = Meter(Caps(meta_usd=cfg["glimmer_usd"], gpu_usd=0.0), phase=f"{cfg['save']}-glimmer-{stamp}")
    meta = Meter(Caps(meta_usd=cfg["meta_usd"], gpu_usd=0.0), phase=f"{cfg['save']}-meta-{stamp}")
    gpu = Meter(Caps(meta_usd=0.0, gpu_usd=cfg["gpu_usd"]), phase=f"{cfg['save']}-gpu-{stamp}")
    register_litellm_prices()
    quiet_run_log()
    litellm.callbacks = [litellm_callback((glim, ("Muse-Glimmer",)), (meta, ("muse-spark", "muse-wonder")))]
    base, glimmer = llm_utils.completion, reparsing(llm_utils.completion)

    halt = threading.Event()  # set by glimmer_serve.session() when the GPU cap trips or its watcher fails
    stopped: dict[str, str | None] = {"reason": None}  # set by fail_fast (runlog: a pattern that cannot be noise)
    modal_env: dict[str, dict | None] = {"env": None}  # the Modal CLI environment once the app is deployed

    def fail_fast(reason: str) -> None:
        stopped["reason"] = reason
        halt.set()  # Glimmer calls are refused from here on
        # Stop the app now, as a cap trip does: the session's own stop waits for τ²'s batch, and a hung server's
        # in-flight calls would keep the GPU billing until they returned (2026-10-29 review). The session's exit
        # stops it again; a second stop of a stopped app is harmless.
        if modal_env["env"] is not None:
            print("FAIL FAST: stopping the Modal app now", flush=True)
            threading.Thread(target=glimmer_serve.stop_until_stopped, args=(MODAL_CLI, modal_env["env"]),
                             daemon=True, name="glint-fail-fast-stop").start()
    truncated, truncated_lock = [0], threading.Lock()  # Glimmer responses cut at max_tokens in the current arm

    last_call = [time.time()]  # when Glimmer last answered: RunEvents tells a slow run from a stalled server by it
    run_clock: dict[tuple[str, str], tuple[float, int, int, float]] = {}
    clock_lock = threading.Lock()

    def route(*args, **kwargs):
        if "Muse-Glimmer" not in kwargs.get("model", ""):
            return base(*args, **kwargs)
        if stopped["reason"]:
            raise BudgetExceeded(f"stopped early (fail fast): {stopped['reason']}")
        if halt.is_set():
            raise BudgetExceeded("the Modal session was halted (GPU cap reached or the GPU watcher failed)")
        response = glimmer(*args, **kwargs)
        last_call[0] = time.time()
        # each run's first call, call count, peak prompt tokens and latest call, for RunEvents: the long-run warning,
        # and the record of a run τ² keeps nothing of (a context overflow)
        prompt = int(getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0)
        with clock_lock:
            first, calls, peak, _ = run_clock.get(current_run(), (last_call[0], 0, 0, 0.0))
            run_clock[current_run()] = (first, calls + 1, max(peak, prompt), last_call[0])
        if response.choices[0].finish_reason == "length":  # τ²'s own warning names no task
            task_id, trial = current_run()
            with truncated_lock:
                truncated[0] += 1
            print(f"TRUNCATED: {task_id} trial {trial} (max_tokens {kwargs.get('max_tokens', 'unset')})", flush=True)
        empty = empty_reply_line(response)
        if empty:
            print(empty, flush=True)
        return response

    # every Glimmer call's token counts and timing, tied to its task and trial (the gate log's run context)
    recorded = recording(route, data / "calls.jsonl", context=lambda: tuple(run_context().values()))
    llm_utils.completion = guarded(guarded(recorded, glim, only=("Muse-Glimmer",)), meta, RateLimiter(per_minute=120),
                                   only=("muse-spark", "muse-wonder"))
    nl.DEFAULT_LLM_NL_ASSERTIONS, nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = SPARK, {**META, "temperature": 0.0}
    with ExitStack() as stack:
        endpoint = None
        if cfg.get("modal"):
            env = os.environ | {"MODAL_CONFIG_PATH": str(paths.MODAL_CONFIG), "MODAL_PROFILE": "quanticsoul4772",
                                "PYTHONIOENCODING": "utf-8"}
            endpoint = glimmer_serve.deploy(MODAL_CLI, env)
            modal_env["env"] = env
            halt = stack.enter_context(glimmer_serve.session(gpu, MODAL_CLI, env, tag=cfg["save"]))
            log = stack.enter_context(glimmer_serve.LogWatch(MODAL_CLI, env))
            glimmer_serve.wait_ready(endpoint, os.environ["MODAL_PROXY_KEY"], os.environ["MODAL_PROXY_SECRET"],
                                     failed=log.failure, log_lines=lambda: len(log.lines))
        for domain, ids, arm in pending:
            enforcer = _enforcer(cfg, domain, data) if arm == "tools" else None
            if arm == "tools":
                install(domain, enforcer=enforcer)
            glim_before, meta_before = glim.meta_usd, meta.meta_usd
            events = RunEvents(_save_dir(cfg, arm, domain, stamp) / "results.json", data / "events.jsonl",
                               label=f"{arm} {domain}",
                               on_run=gate_summary.ErrorLimitAlerts() if enforcer is not None else None,
                               domain=domain, on_halt=fail_fast, activity=lambda: last_call[0],
                               running=lambda: dict(run_clock), gated=enforcer is not None)
            truncated[0] = 0
            with clock_lock:
                run_clock.clear()  # the arms share (task, trial) keys: the last arm's runs are not this arm's
            glimmer_serve.reset_pool_stats()
            with events:
                results = _run(cfg, domain, ids, arm, stamp, endpoint)
            pool = glimmer_serve.pool_stats()
            print(f"TRUNCATIONS: {arm} {domain}: {truncated[0]} Glimmer responses stopped at max_tokens", flush=True)
            print(f"POOL: {pool['requests']} requests, {pool['waited']} waited over "
                  f"{glimmer_serve.POOL_WAIT_WARN_S:.0f} s, longest {pool['longest_s']:.1f} s", flush=True)
            if enforcer is not None:  # the gate decisions live only in the gate log; one line puts them in the run log
                sims = [s.model_dump(mode="json") for s in results.simulations]
                gate_events = enforcer.events.events()
                print(gate_summary.line(f"{arm} {domain}", gate_summary.summarize(gate_events, sims)), flush=True)
                for line in gate_summary.outcome_lines(f"{arm} {domain}", gate_summary.rule_outcomes(
                        gate_events, sims, inspect.arm_diffs(domain, sims))):
                    print(line, flush=True)
            if stopped["reason"]:  # as for a cap stop: no arm file, so a resumed run starts this arm again
                raise BudgetExceeded(f"{arm} {domain}: stopped early (fail fast): {stopped['reason']}; the arm file is "
                                     "not written and a resumed run starts this arm again")
            if halt.is_set():  # the drained runs are not results: no arm file, so a resume reruns this arm
                raise BudgetExceeded(f"{arm} {domain}: the Modal session was halted mid-arm; the arm file is not "
                                     "written and a resumed run starts this arm again")
            kinds = Counter(outcome(s.model_dump(mode="json")) for s in results.simulations)
            if kinds["budget_stop"]:  # cap-stopped runs are not results: no arm file, so a resume reruns this arm
                raise BudgetExceeded(f"{arm} {domain}: {kinds['budget_stop']} runs were stopped by a spend cap; the "
                                     "arm file is not written and a resumed run starts this arm again")
            expected = len(ids) * _trials(cfg, domain)
            if len(results.simulations) < expected:
                print(f"WARNING: {arm} {domain}: {len(results.simulations)} simulations, expected {expected} "
                      f"({len(ids)} tasks x k={_trials(cfg, domain)}); the missing runs are not in this arm",
                      flush=True)
            (data / f"{arm}-{domain}.json").write_text(results.model_dump_json(), encoding="utf-8")
            if kinds["glint_error"]:
                print(f"WARNING: {arm} {domain}: {kinds['glint_error']} of {len(results.simulations)} runs failed in "
                      "Glint's own code (atem or call recording); they count as failed runs. Causes are in "
                      "info.error and events.jsonl", flush=True)
            if kinds["eval_error"]:
                print(f"WARNING: {arm} {domain}: {kinds['eval_error']} of {len(results.simulations)} runs have no "
                      "reward (τ²'s evaluation failed); they are excluded from the analysis", flush=True)
            if kinds["infrastructure_error"]:
                print(f"WARNING: {arm} {domain}: {kinds['infrastructure_error']} of {len(results.simulations)} runs "
                      "ended in an infrastructure error; they are excluded from the analysis and are not retried. "
                      f"Rerun them with: {rerun_command()}", flush=True)
            if kinds["context_overflow"]:
                print(f"WARNING: {arm} {domain}: {kinds['context_overflow']} of {len(results.simulations)} runs "
                      "overflowed the context; they count as failed runs", flush=True)
            if kinds["empty_reply"]:
                print(f"WARNING: {arm} {domain}: {kinds['empty_reply']} of {len(results.simulations)} runs ended on an "
                      "empty Glimmer reply (no content, no tool call); they count as failed runs", flush=True)
            print(f"{arm} {domain}: {len(results.simulations)} sims; this arm Glimmer "
                  f"${glim.meta_usd - glim_before:.3f}, Meta ${meta.meta_usd - meta_before:.3f}; phase totals Glimmer "
                  f"${glim.meta_usd:.3f}, "
                  f"Meta ${meta.meta_usd:.3f}", flush=True)


def analyze(cfg: dict) -> None:
    report = {}
    # Domains come from the result files themselves (the design run predates tasks.json)
    # (arm files only: the released-<arm>-<domain>-<time>.json side files of a rerun are not a domain)
    domains = sorted({p.stem.split("-", 1)[1] for arm in ("base", "tools") for p in cfg["data"].glob(f"{arm}-*.json")})
    if not domains:
        raise FileNotFoundError(f"no results in {cfg['data']}")
    planned = tuple(cfg.get("arms", ("base", "tools")))
    if len(planned) == 1:  # a one-arm set is compared with another set's arm (its docstring says which)
        print(f"NOTE: this set runs the {planned[0]} arm only; its tasks are not compared within the set", flush=True)
    for arm in ("base", "tools"):
        for domain in domains:
            path = cfg["data"] / f"{arm}-{domain}.json"
            if not path.exists():
                # an arm the set never runs is not missing (13 such WARNINGs ended each one-arm run, 2026-10-29)
                if arm in planned:
                    print(f"WARNING: no {arm} arm for {domain} ({path.name} missing): its tasks have one arm only "
                          "and are left out of the comparison", flush=True)
                continue
            beyond_k = 0
            for s in json.loads(path.read_text(encoding="utf-8"))["simulations"]:
                if s["trial"] >= _trials(cfg, domain):  # a resumed checkpoint can hold trials beyond this k
                    beyond_k += 1
                    continue
                key = f"{domain}-{s['task_id']}"
                row = report.setdefault(key, {}).setdefault(arm, {"passes": 0, "runs": 0, "tool_calls": Counter(),
                                                                   "infra_errors": 0, "context_overflows": 0,
                                                                   "empty_replies": 0, "glint_errors": 0, "eval_errors": 0,
                                                                   "budget_stops": 0})
                kind = outcome(s)
                if kind in ("infrastructure_error", "eval_error", "budget_stop"):  # not results; counted below
                    row[{"infrastructure_error": "infra_errors", "eval_error": "eval_errors",
                         "budget_stop": "budget_stops"}[kind]] += 1
                    continue
                # a context overflow is the model's own conversation outgrowing the context, an empty reply is the
                # model returning nothing (decision 2026-10-28), and a Glint error is Glint's own failure: all are
                # failed runs (τ² files them as infrastructure errors; the cause is in info.error)
                row["context_overflows"] += kind == "context_overflow"
                row["empty_replies"] += kind == "empty_reply"
                row["glint_errors"] += kind == "glint_error"
                row["runs"] += 1
                row["passes"] += kind == "pass"
                for m in s["messages"]:
                    for tc in (m.get("tool_calls") or []) if m["role"] == "assistant" else []:
                        if tc["name"] in TOOL_NAMES:
                            row["tool_calls"][tc["name"]] += 1
            if beyond_k:
                print(f"NOTE: {arm} {domain}: {beyond_k} runs beyond k={_trials(cfg, domain)} left out (a resumed "
                      "checkpoint)", flush=True)
    for arms in report.values():
        for row in arms.values():
            row["tool_calls"] = dict(row["tool_calls"])
            row["pass_rate"] = round(row["passes"] / row["runs"], 3) if row["runs"] else None
    cfg["out"].write_text(json.dumps(report, indent=1), encoding="utf-8")
    # a tools arm that never called a Glint tool most likely ran without them (an install that did not take)
    for domain in domains:
        rows = [arms["tools"] for key, arms in report.items() if key.startswith(f"{domain}-") and "tools" in arms]
        if rows and not any(row["tool_calls"] for row in rows):
            print(f"WARNING: the tools arm for {domain} made 0 Glint tool calls in {sum(r['runs'] for r in rows)} "
                  "runs: check that the tools were installed", flush=True)
    print(json.dumps(report, indent=1))
    # Loud summary: excluded runs (infrastructure, evaluation, cap) can leave arms with different run counts, a task
    # can have one arm only, and Glint errors count against the tools
    for key, arms in sorted(report.items()):
        if len(planned) > 1 and set(arms) != set(planned):
            print(f"WARNING: {key}: only the {', '.join(arms)} arm has runs; the task is not compared", flush=True)
        runs = {arm: row["runs"] for arm, row in arms.items()}
        excluded = {arm: {k: row[k] for k in ("infra_errors", "eval_errors", "budget_stops") if row[k]}
                    for arm, row in arms.items()}
        excluded = {arm: e for arm, e in excluded.items() if e}
        glint = {arm: row["glint_errors"] for arm, row in arms.items() if row["glint_errors"]}
        if excluded or glint or len(set(runs.values())) > 1:
            print(f"WARNING: {key}: runs per arm {runs}, excluded {excluded or 'none'}, Glint errors (failed runs) "
                  f"{glint or 'none'}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    cfg = for_backend(SETS[next((a for a in args if a in SETS), "design")], "--modal" in args)
    if "--analyze" not in args:
        generate(cfg)
    analyze(cfg)
