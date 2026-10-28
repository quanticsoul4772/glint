# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "glint @ git+https://github.com/quanticsoul4772/glint@glint-bank-v0.1.0",
#     "tau2[knowledge] @ git+https://github.com/sierra-research/tau2-bench@fc0055dc4e0a316c3f83133267fbd6faaa770992",
# ]
# ///
"""glint-bank launcher: `uv run --script mcp/glint_bank.py [--selftest | --decision-tools | brief TASK | log SID]`.

Runs glint.bank.server over stdio. τ² reads its data from TAU2_DATA_DIR, and a git install of τ² ships no data, so
the launcher points it at the banking data bundled in this plugin (data/, MIT, see data/LICENSE) unless the variable
is already set. τ² opens its task and document files with the platform's default encoding; on Windows that is not
UTF-8, so the launcher restarts itself in UTF-8 mode with the same stdin and stdout.

The direct tau2 reference is needed because Glint's own `tool.uv.sources` path entry does not apply when Glint is
installed as a dependency.
"""
import os
import subprocess
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    if not sys.flags.utf8_mode:
        return subprocess.call([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), *sys.argv[1:]])
    os.environ.setdefault("TAU2_DATA_DIR", str(PLUGIN_ROOT / "data"))
    from glint.bank.server import main as serve

    return serve(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
