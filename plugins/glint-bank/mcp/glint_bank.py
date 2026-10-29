# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""glint-bank launcher: `uv run --script mcp/glint_bank.py [--selftest | --decision-tools | brief TASK | log SID]`.

Two stages, so that every failure is reported:
1. This script, run by the plugin's command with no dependencies, writes the server stage as a uv script into
   GLINT_BANK_HOME and runs it with `uv run --script`. Its inline metadata pins Glint to its git tag
   (GLINT_REQUIREMENT), or to a local checkout when GLINT_BANK_SOURCE names one (an editable install, so it can be
   tested before the repository is public), and τ² to one commit (TAU2_REQUIREMENT, also as an override: uv applies
   Glint's own `tool.uv.sources` entry, a path to a sibling τ² checkout, when it builds Glint, and that path does not
   exist in the git source).
2. The server stage (GLINT_BANK_STAGE=server) runs glint.bank.server over stdio, sharing this process's stdin and
   stdout. τ² reads its data from TAU2_DATA_DIR, and a git install of τ² ships no data, so it is pointed at the
   banking data bundled in this plugin (data/, MIT, see data/LICENSE) unless the variable is already set. τ² opens its
   files with the platform's default encoding, so the server stage runs in UTF-8 mode.

Muse treats a plugin server as optional and carries on when it does not start (2026-10-28: transport_closed and
nothing else). So whatever ends the launcher with an error (uv missing, a failed install, a missing dependency, an
exception, a nonzero exit) is written to stderr and appended to `$GLINT_BANK_HOME/launcher.log` (default
~/.glint-bank), and the launcher exits nonzero. Python stays below 3.13: τ² imports audioop, which 3.13 removed.
"""
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
GLINT_REQUIREMENT = "glint @ git+https://github.com/quanticsoul4772/glint@glint-bank-v0.1.0"
TAU2_REQUIREMENT = ("tau2[knowledge] @ git+https://github.com/sierra-research/tau2-bench"
                    "@fc0055dc4e0a316c3f83133267fbd6faaa770992")
PYTHON = ">=3.12,<3.13"
STAGE = "GLINT_BANK_STAGE"
TAIL_LINES = 40  # the end of the server stage's stderr that goes into the log when it fails


def bank_home() -> Path:
    """GLINT_BANK_HOME, else ~/.glint-bank, as glint.bank.server.bank_home finds it (Muse keeps HOME, not USERPROFILE)."""
    if os.environ.get("GLINT_BANK_HOME"):
        return Path(os.environ["GLINT_BANK_HOME"])
    return (Path(os.environ["HOME"]) if os.environ.get("HOME") else Path.home()) / ".glint-bank"


def report(why: str) -> None:
    """Say why the launcher stops, on stderr (Muse's server log) and in launcher.log."""
    text = f"{time.strftime('%Y-%m-%d %H:%M:%S')} glint-bank launcher (pid {os.getpid()}): {why.rstrip()}\n"
    sys.stderr.write(text)
    sys.stderr.flush()
    home = bank_home()
    home.mkdir(parents=True, exist_ok=True)
    with (home / "launcher.log").open("a", encoding="utf-8") as fh:
        fh.write(text)


def stage_script(source: str | None) -> str:
    """The server stage as a uv script: its inline metadata pins the dependencies, and its body runs this launcher."""
    if source:
        if not (Path(source) / "pyproject.toml").is_file():
            raise FileNotFoundError(f"GLINT_BANK_SOURCE={source} is not a Glint checkout (no pyproject.toml there)")
        glint = "glint"
        sources = ["# [tool.uv.sources]",
                   f'# glint = {{ path = "{Path(source).resolve().as_posix()}", editable = true }}']
    else:
        glint, sources = GLINT_REQUIREMENT, []
    return "\n".join([
        "# /// script", f'# requires-python = "{PYTHON}"', "# dependencies = [", f'#     "{glint}",',
        f'#     "{TAU2_REQUIREMENT}",', "# ]", "# [tool.uv]", "# override-dependencies = [",
        f'#     "{TAU2_REQUIREMENT}",', "# ]", *sources, "# ///",
        "import runpy", f"runpy.run_path({str(Path(__file__).resolve())!r}, run_name='__main__')", ""])


def uv_command(uv: str, args: list[str]) -> list[str]:
    """The uv command that runs the server stage with the pinned dependencies. The stage script is written to
    GLINT_BANK_HOME (uv keeps one environment per script content)."""
    source = os.environ.get("GLINT_BANK_SOURCE")
    script = bank_home() / ("server_stage_local.py" if source else "server_stage.py")
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(stage_script(source), encoding="utf-8")
    return [uv, "run", "--script", str(script), *args]


def launch(args: list[str]) -> int:
    """Stage 1: run the server stage through uv; its stderr passes through, and its end is kept for the log."""
    uv = shutil.which("uv")
    if uv is None:
        report("uv is not on PATH; it installs the server's dependencies (https://docs.astral.sh/uv/)")
        return 1
    try:
        command = uv_command(uv, args)
    except OSError as e:
        report(f"{type(e).__name__}: {e}")
        return 1
    tail: deque[str] = deque(maxlen=TAIL_LINES)
    proc = subprocess.Popen(command, env=os.environ | {STAGE: "server"}, stderr=subprocess.PIPE)

    def relay():
        for raw in proc.stderr:
            line = raw.decode("utf-8", errors="replace")
            tail.append(line)
            sys.stderr.write(line)
            sys.stderr.flush()

    relay_thread = threading.Thread(target=relay, daemon=True)
    relay_thread.start()
    code = proc.wait()
    relay_thread.join(timeout=5)
    if code != 0:
        source = os.environ.get("GLINT_BANK_SOURCE") or GLINT_REQUIREMENT
        report(f"the server stage exited with code {code} (uv install or server failure; Glint from {source}). "
               f"Last stderr lines:\n" + "".join(tail))
    return code


def serve(args: list[str]) -> int:
    """Stage 2: the server itself, inside uv's environment."""
    os.environ.setdefault("TAU2_DATA_DIR", str(PLUGIN_ROOT / "data"))
    try:
        from glint.bank.server import main as server_main
    except ImportError as e:
        report(f"missing dependency: {type(e).__name__}: {e}")
        return 1
    try:
        code = server_main(args)
    except Exception:  # noqa: BLE001  (reported in full, then the launcher exits nonzero)
        report("the server raised:\n" + traceback.format_exc())
        return 1
    if code:
        report(f"the server exited with code {code}")
    return code


def main() -> int:
    args = sys.argv[1:]
    if os.environ.get(STAGE) != "server":
        return launch(args)
    if not sys.flags.utf8_mode:
        return subprocess.call([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), *args])
    return serve(args)


if __name__ == "__main__":
    sys.exit(main())
