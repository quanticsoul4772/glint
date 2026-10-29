"""The GATES line tools_ab prints at a gated arm's end, for a set's data folder that already exists (finished or not).

For every gate log `gates-<domain>.sqlite` in the folder, the arm that wrote it is `tools-<domain>.json` (only the tools
arm is gated). Both are opened read-only; nothing is written into the folder.

Usage: python scripts/gate_summary.py DATA_FOLDER
"""
import json
import sys
from pathlib import Path

from glint.gates import summary


def folder_lines(data: Path) -> list[str]:
    logs = sorted(data.glob("gates-*.sqlite"))
    if not logs:
        raise FileNotFoundError(f"no gate log (gates-<domain>.sqlite) in {data}: the set ran without gates")
    lines = []
    for log in logs:
        domain = log.stem.removeprefix("gates-")
        arm = data / f"tools-{domain}.json"
        if not arm.exists():
            raise FileNotFoundError(f"{arm} is missing: the tools arm of {domain} has not finished, so its runs are "
                                    "not saved in the data folder")
        sims = json.loads(arm.read_text(encoding="utf-8"))["simulations"]
        lines.append(summary.line(f"tools {domain}", summary.summarize(summary.read_log(log), sims)))
    return lines


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    for line in folder_lines(Path(argv[0])):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
