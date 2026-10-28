"""Where every with/without run set stands: progress per arm, outcomes, error causes, spend, Modal, ETA.

Reads what the runs already write: each set's events.jsonl (glint.runlog: one record per finished simulation),
tasks.json, stamp.txt and the finished arm files, plus the meter ledger, and (with --modal) `modal app list`.
Usage: python status.py [SET ...] [--modal]        (no SET: every set that has data)
"""
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from loguru import logger

from glint import paths

# τ² logs its whole registry at DEBUG when imported; this command prints only its own report
logger.remove()
logger.add(sys.stderr, level="WARNING")

import tools_ab  # after the log setup above

LEDGER = paths.LEDGER
MODAL = str(Path(__file__).resolve().parents[1] / ".venv" / "Scripts" / "modal.exe")


def _spend(save: str, stamp: str) -> dict[str, float]:
    out = defaultdict(float)
    if LEDGER.exists():
        for line in LEDGER.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("phase", "").startswith(save + "-") and r["phase"].endswith(stamp):
                out["gpu" if r["kind"] == "gpu" else r["phase"][len(save) + 1:].split("-")[0]] += r["usd"]
    return dict(out)


def _modal_state() -> str:
    env = os.environ | {"MODAL_CONFIG_PATH": str(paths.MODAL_CONFIG), "MODAL_PROFILE": "quanticsoul4772"}
    try:
        out = subprocess.run([MODAL, "app", "list", "--json"], env=env, capture_output=True, text=True, timeout=60,
                             check=False)
        apps = [a for a in json.loads(out.stdout) if a.get("description") == "glint-serve"]
    except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError) as e:
        return f"unknown ({type(e).__name__})"
    live = [a for a in apps if a.get("state") != "stopped"]
    return ", ".join(f"{a['state']} ({a['tasks']} containers)" for a in live) or "stopped (0 containers)"


def report(name: str, cfg: dict) -> list[str]:
    data, lines = cfg["data"], []
    stamp_dir = cfg.get("stamp_dir", data)
    if not data.exists():
        return lines
    tasks = json.loads((data / "tasks.json").read_text(encoding="utf-8")) if (data / "tasks.json").exists() else {}
    stamp = (stamp_dir / "stamp.txt").read_text(encoding="utf-8").strip() if (stamp_dir / "stamp.txt").exists() else "?"
    events = [json.loads(x) for x in (data / "events.jsonl").read_text(encoding="utf-8").splitlines()] \
        if (data / "events.jsonl").exists() else []
    by_arm = defaultdict(list)
    for e in events:
        by_arm[e["label"]].append(e)
    lines.append(f"== {name}  ({data.name}, stamp {stamp})")
    for domain, ids in tasks.items():
        for arm in cfg.get("arms", ("base", "tools")):
            expected = len(ids) * tools_ab._trials(cfg, domain)
            evs = by_arm.get(f"{arm} {domain}", [])
            finished = (data / f"{arm}-{domain}.json").exists()
            counts = Counter(e["outcome"] for e in evs)
            line = f"  {arm:5} {domain:17} {len(evs):4}/{expected:<4} {dict(counts) or '-'}"
            if finished:
                line += "  [arm done]"
            elif evs:
                mean = sum(e["duration_s"] for e in evs) / len(evs)
                left = (expected - len(evs)) * mean / max(1, cfg.get("concurrency", 8))
                line += f"  last {time.strftime('%H:%M', time.localtime(evs[-1]['ts']))}, ~{left / 60:.0f} min left"
            lines.append(line)
            for e in evs:
                if e["outcome"] in ("infrastructure_error", "context_overflow"):
                    lines.append(f"      {e['outcome']}: {e['task_id']} trial {e['trial']}: {(e['error'] or '')[:110]}")
    spend = _spend(cfg["save"], stamp)
    if spend:
        lines.append("  spend: " + ", ".join(f"{k} ${v:.3f}" for k, v in sorted(spend.items())))
    return lines


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    names = args or list(tools_ab.SETS)
    for name in names:
        for modal in (False, True):
            for line in report(f"{name}{' --modal' if modal else ''}", tools_ab.for_backend(tools_ab.SETS[name], modal)):
                print(line)
    if "--modal" in sys.argv:
        print(f"Modal glint-serve: {_modal_state()}")


if __name__ == "__main__":
    main()
