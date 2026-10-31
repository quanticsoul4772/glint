# glint-bank (Muse plugin)

A live τ²-bench bank (banking_knowledge) as MCP tools, with every write checked by Glint's policy gates before it runs. `/start <task_id>` loads a scenario.

## Launcher

The plugin's server command is `uv run --script mcp/glint_bank.py --decision-tools`. The launcher installs the pinned dependencies through uv (Glint from its git tag, τ² from one commit) and runs `glint.bank.server` over stdio. `--decision-tools` adds Glint's banking decision tools to the gated τ² tools: the budgeted `KB_search`, `KB_read`, `check_verification`, `check_credit_limit_increase` and `check_card_dispute`.

Muse treats a plugin server as optional: when it does not start, Muse logs `transport_closed` and carries on without the tools. The launcher therefore writes why it stopped (uv missing, a failed install, a missing dependency, an exception, a nonzero exit) to stderr and to `$GLINT_BANK_HOME/launcher.log`, and exits nonzero. `scripts/muse_check.py` reads Muse's bootstrap log for a run and fails when an expected server did not reach `ready`.

## Environment

- `GLINT_BANK_HOME`: gate logs, the launcher log and the uv stage script. Default `~/.glint-bank`.
- `GLINT_BANK_SOURCE`: optional path to a local Glint checkout. When set, the launcher installs Glint from that checkout (editable) instead of the git tag, so the plugin can be tested before the repository is public. τ² stays pinned to the same commit.
- `TAU2_DATA_DIR`: τ²'s data. Default: the banking data bundled in `data/` (MIT, see `data/LICENSE`).

Check an install without Muse or a model:

```
GLINT_BANK_SOURCE=/path/to/glint uv run --script mcp/glint_bank.py --decision-tools --selftest
```

It prints `{"ok": true, ...}` on stderr and exits 0 when the server lists its tools (the decision tools among them), loads task_054, reads the policy and runs one lookup.
