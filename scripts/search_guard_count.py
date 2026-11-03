"""How often the search-loop guard (glint.assist.tau2_tools, SEARCH_GUARD) would have fired on recorded runs.

The guard counts knowledge-base calls (KB_search, grep, KB_read) since the last write that ran and, every N of them,
appends the customer's open items to the answer. This replays that count over the messages of recorded τ² arm files
(`<arm>-banking_knowledge.json` in each folder): a write that ran is an assistant call of a WRITE tool (a hidden one
unwrapped from call_discoverable_agent_tool) whose result is not an error, as the guard sees it. Runs of the 20
banking test-split tasks are skipped (never design data here); the tuned tasks (by_task.TUNED) are reported apart.
Reads only the simulations' messages; no model, no network.

Usage: python scripts/search_guard_count.py DATA_DIR [DATA_DIR ...] [--n 10,15,20,30]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from by_task import TUNED  # scripts/ is on the path when this runs as a script or under the tests

import splits
from glint.runlog import NOT_RESULTS, outcome

KB_TOOLS = ("KB_search", "grep", "KB_read")
WRAPPER = "call_discoverable_agent_tool"
DOMAIN = "banking_knowledge"


def write_tools() -> set[str]:
    """Every WRITE tool of τ²'s banking toolkit, hidden ones included (the wrapper itself is not one)."""
    from tau2.domains.banking_knowledge.tools import KnowledgeTools
    from tau2.environment.toolkit import TOOL_TYPE_ATTR, ToolType

    return {n for n in dir(KnowledgeTools) if not n.startswith("_") and n != WRAPPER
            and getattr(getattr(KnowledgeTools, n), TOOL_TYPE_ATTR, None) == ToolType.WRITE}


def firings(messages: list[dict], writes: set[str], n: int) -> int:
    """How many times the guard fires in one conversation: every n-th knowledge-base call since the last write that
    ran."""
    replies = {str(m.get("id")): m for m in messages if m.get("role") == "tool"}
    since, fired = 0, 0
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for call in m.get("tool_calls") or []:
            name, args = call.get("name"), call.get("arguments") or {}
            if name in KB_TOOLS:
                since += 1
                fired += since % n == 0
                continue
            tool = str(args.get("agent_tool_name", "")) if name == WRAPPER else name
            reply = replies.get(str(call.get("id"))) or {}
            if tool in writes and not reply.get("error") and not str(reply.get("content", "")).startswith("Error"):
                since = 0
    return fired


def count(folders: list[Path], ns: list[int], writes: set[str]) -> dict:
    """(arm column, tuned or untuned, n) -> Counter of runs, runs fired, firings, passing runs fired, failing runs
    fired."""
    test = set(json.loads(splits.BANKING.read_text(encoding="utf-8"))["test_untouched"])
    out: dict = {}
    for folder in folders:
        for arm_file in sorted(folder.glob(f"*-{DOMAIN}.json")):
            col = f"{folder.name}:{arm_file.name.split('-', 1)[0]}"
            for sim in json.loads(arm_file.read_text(encoding="utf-8"))["simulations"]:
                task = str(sim["task_id"])
                if task in test or outcome(sim) in NOT_RESULTS:
                    continue
                group = "tuned" if task in TUNED else "untuned"
                passed = outcome(sim) == "pass"
                for n in ns:
                    f = firings(sim.get("messages") or [], writes, n)
                    c = out.setdefault((col, group, n), Counter())
                    c["runs"] += 1
                    c["firings"] += f
                    c["runs_fired"] += f > 0
                    c["pass_fired" if passed else "fail_fired"] += f > 0
    return out


def main(argv: list[str]) -> int:
    ns = [10, 15, 20, 30]
    if "--n" in argv:
        k = argv.index("--n")
        ns, argv = [int(x) for x in argv[k + 1].split(",")], argv[:k] + argv[k + 2:]
    folders = [Path(a) for a in argv]
    if not folders:
        print(__doc__)
        return 2
    rows = count(folders, ns, write_tools())
    print("arm | tasks | N | runs | runs fired (passing / failing) | firings")
    for (col, group, n), c in sorted(rows.items()):
        print(f"{col} | {group} | {n} | {c['runs']} | {c['runs_fired']} ({c['pass_fired']} / {c['fail_fired']}) | "
              f"{c['firings']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
