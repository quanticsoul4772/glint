"""The glint-bank launcher (plugins/glint-bank/mcp/glint_bank.py): every way it stops with an error is written to
stderr and to $GLINT_BANK_HOME/launcher.log with a nonzero exit, and GLINT_BANK_SOURCE swaps the git tag for a local
checkout. uv is never run here: the server stage is replaced by a local command."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "plugins" / "glint-bank" / "mcp" / "glint_bank.py"
TAU2 = "tau2[knowledge] @ git+https://github.com/sierra-research/tau2-bench@fc0055dc4e0a316c3f83133267fbd6faaa770992"


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    monkeypatch.setenv("GLINT_BANK_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("GLINT_BANK_SOURCE", raising=False)
    monkeypatch.delenv("GLINT_BANK_STAGE", raising=False)
    spec = importlib.util.spec_from_file_location("glint_bank_launcher", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _log(tmp_path) -> str:
    return (tmp_path / "home" / "launcher.log").read_text(encoding="utf-8")


def test_the_stage_script_pins_glint_to_the_tag_and_tau2_to_one_commit(launcher):
    script = launcher.stage_script(None)
    assert '# requires-python = ">=3.12,<3.13"' in script  # τ² imports audioop, which Python 3.13 removed
    assert f'#     "{launcher.GLINT_REQUIREMENT}",' in script and "glint@glint-bank-v0." in script
    # in the dependencies, and as the override of Glint's own tau2 source (a sibling path absent from Glint's git tree)
    assert script.count(TAU2) == 2 and "# [tool.uv]\n# override-dependencies = [\n" in script
    assert "tool.uv.sources" not in script and script.endswith(f"run_path({str(LAUNCHER)!r}, run_name='__main__')\n")


def test_glint_bank_source_installs_the_local_checkout_instead_of_the_tag(launcher, monkeypatch):
    monkeypatch.setenv("GLINT_BANK_SOURCE", str(ROOT))
    command = launcher.uv_command("uv", ["--selftest"])
    assert command[:3] == ["uv", "run", "--script"] and command[-1] == "--selftest"
    script = Path(command[3]).read_text(encoding="utf-8")
    assert Path(command[3]).name == "server_stage_local.py"
    assert f'# glint = {{ path = "{ROOT.as_posix()}", editable = true }}' in script
    assert '#     "glint",' in script and launcher.GLINT_REQUIREMENT not in script and script.count(TAU2) == 2


def test_a_source_that_is_not_a_checkout_is_reported(launcher, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("GLINT_BANK_SOURCE", str(tmp_path / "nowhere"))
    assert launcher.launch([]) == 1
    assert "is not a Glint checkout" in capsys.readouterr().err and "is not a Glint checkout" in _log(tmp_path)


def test_no_uv_is_reported(launcher, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(launcher.shutil, "which", lambda name: None)
    assert launcher.launch([]) == 1
    assert "uv is not on PATH" in capsys.readouterr().err and "uv is not on PATH" in _log(tmp_path)


def test_a_failed_server_stage_is_reported_with_its_stderr(launcher, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "uv")
    fail = [sys.executable, "-c", "import sys; sys.stderr.write('error: no solution for glint\\n'); sys.exit(3)"]
    monkeypatch.setattr(launcher, "uv_command", lambda uv, args: fail)
    assert launcher.launch([]) == 3
    log = _log(tmp_path)
    assert "the server stage exited with code 3" in log and "error: no solution for glint" in log
    assert "exited with code 3" in capsys.readouterr().err


def test_a_clean_server_stage_exit_writes_no_log(launcher, monkeypatch, tmp_path):
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "uv")
    monkeypatch.setattr(launcher, "uv_command", lambda uv, args: [sys.executable, "-c", "pass"])
    assert launcher.launch([]) == 0
    assert not (tmp_path / "home" / "launcher.log").exists()


def test_the_server_stage_reports_a_missing_dependency_an_exception_and_a_nonzero_exit(launcher, monkeypatch,
                                                                                       tmp_path):
    monkeypatch.setitem(sys.modules, "glint.bank.server", None)  # import fails as if the package were missing
    assert launcher.serve([]) == 1
    assert "missing dependency: ModuleNotFoundError" in _log(tmp_path)
    monkeypatch.setitem(sys.modules, "glint.bank.server", type(sys)("glint.bank.server"))

    def boom(args):
        raise RuntimeError("τ² data not found")

    sys.modules["glint.bank.server"].main = boom
    assert launcher.serve([]) == 1
    assert "the server raised:" in _log(tmp_path) and "RuntimeError: τ² data not found" in _log(tmp_path)
    sys.modules["glint.bank.server"].main = lambda args: 4
    assert launcher.serve([]) == 4 and "the server exited with code 4" in _log(tmp_path)
    sys.modules["glint.bank.server"].main = lambda args: 0
    before = _log(tmp_path)
    assert launcher.serve([]) == 0 and _log(tmp_path) == before


def test_main_runs_stage_one_unless_the_stage_is_set(launcher, monkeypatch):
    calls = []
    monkeypatch.setattr(launcher, "launch", lambda args: calls.append(("launch", args)) or 0)
    monkeypatch.setattr(launcher, "serve", lambda args: calls.append(("serve", args)) or 0)
    monkeypatch.setattr(sys, "argv", ["glint_bank.py", "--selftest"])
    launcher.main()
    monkeypatch.setenv("GLINT_BANK_STAGE", "server")
    if sys.flags.utf8_mode:
        launcher.main()
        assert calls == [("launch", ["--selftest"]), ("serve", ["--selftest"])]
    else:
        assert calls == [("launch", ["--selftest"])]
