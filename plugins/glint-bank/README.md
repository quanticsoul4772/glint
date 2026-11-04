# glint-bank (Muse plugin)

A live τ²-bench bank (banking_knowledge) as MCP tools, with every write checked by Glint's policy gates before it runs. `/start <task_id>` loads a scenario.

## Install in Muse Code

Requires [uv](https://docs.astral.sh/uv/). From a clone of this repository:

```
muse plugins install plugins/glint-bank --scope user
muse plugins approve glint-bank
```

The first start installs Glint (its git tag) and τ² (one pinned commit of its git repository) through uv, so it takes a few minutes; the bank data comes from this plugin's `data/`. Checked with Muse Code 1.4.14 in headless mode (`muse exec`): on 2026-11-03 plugin 0.4.0, installed from its tag, reached `ready` with 27 tools, and its PreToolUse hook blocked a $4,000 credit-limit increase before the server ran it, then allowed $2,500 (earlier check: 2026-10-31, plugin 0.2.1); interactive mode is not shown to work: on 2026-11-01 Muse's interactive start stopped the server about 0.15 s after fetching a fresh feature configuration, while a headless start three minutes later reached `ready` with 23 tools (Muse's bootstrap logs). Use `muse exec` until an interactive start is checked.

## Launcher

The plugin's server command is `uv run --script mcp/glint_bank.py --decision-tools`. The launcher installs the pinned dependencies through uv (Glint from its git tag, τ² from one commit) and runs `glint.bank.server` over stdio. `--decision-tools` adds Glint's banking decision tools to the gated τ² tools: the budgeted `KB_search`, `KB_read`, `check_verification`, `check_credit_limit_increase` and `check_card_dispute`.

Muse treats a plugin server as optional: when it does not start, Muse logs `transport_closed` and carries on without the tools. The launcher therefore writes why it stopped (uv missing, a failed install, a missing dependency, an exception, a nonzero exit) to stderr and to `$GLINT_BANK_HOME/launcher.log`, and exits nonzero. `scripts/muse_check.py` reads Muse's bootstrap log for a run and fails when an expected server did not reach `ready`.

## PreToolUse hook

The plugin also declares a `PreToolUse` hook (`hooks/gate_check.py`, run as `uv run --script hooks/gate_check.py`; standard library only). It puts Glint's gates in Muse Code's own permission path, in front of the server's checks:

- Muse runs it before every tool call. Plugin hooks have no matcher, so it sees every tool. Anything that is not one of glint-bank's write tools (`log_verification`, `change_user_email`, `call_discoverable_agent_tool`, `transfer_to_human_agents`, `give_discoverable_user_tool`, under `mcp__plugin_glint_bank_b__`) goes ahead: the hook prints nothing.
- A glint-bank write is sent to the running server's gates. While the server runs, it serves its session's decisions on `127.0.0.1` and names the port and a per-start token in `$GLINT_BANK_HOME/sessions/<MUSE_SESSION_ID>/hook.json`. The server runs the same gates on the same records and runs nothing (`Enforcer.precheck`). A hidden read sent through `call_discoverable_agent_tool` is allowed.
- On a deny the hook returns `{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": <the gate's refusal>}}`, which blocks the call. The agent sees the same refusal text the server would return. The server keeps the refusal as if it had refused the call itself: the same `gate.fired` events, a `hook.denied` event with Muse's call id, and the repeat note when the same call is sent again.
- An allow logs nothing. The call is checked, logged and attested when it reaches the server.
- Fail closed: when the hook cannot get a decision, it denies the write and says why. That covers no `hook.json`, a server that cannot be reached or does not answer within 8 s, a wrong token, a non-200 answer, or unreadable input. It does not cover the hook process itself failing: if it exceeds Muse's 10 s hook timeout (including `uv run --script` startup) or crashes, Muse's own handling applies, and the server still gates every call it receives.
- The server-side gating is unchanged. The server still checks every call it receives, so a write is checked even where the hook does not run (another MCP client, or a hook not approved).

Muse asks for approval of each plugin capability on its own. Approve the hook along with the server (`muse plugins approve glint-bank`, or `muse plugins approve glint-bank:hook:gate-check`). The hook's unit tests (`tests/test_bank_hook.py`) use the PreToolUse input Muse Code 1.4.14 sent in the 2026-10-28 probes. They run against a real bank server and need neither Muse nor a model. Under `muse exec` on 2026-11-03 (plugin 0.4.0, Muse Spark, task_054) the hook blocked the $4,000 increase with the gate's refusal (`tool blocked by hook`, a `hook.denied` event, no `write.ran`) and an unlooked-up `log_verification` (`glint.identifier_provenance`), then allowed both writes once the agent had done what the refusal asked.

## Environment

- `GLINT_BANK_HOME`: gate logs, the launcher log and the uv stage script. Default `~/.glint-bank`.
- `GLINT_BANK_SOURCE`: optional path to a local Glint checkout. When set, the launcher installs Glint from that checkout (editable) instead of the git tag, so local changes can be tested in Muse Code. τ² stays pinned to the same commit.
- `TAU2_DATA_DIR`: τ²'s data. Default: the banking data bundled in `data/` (MIT, see `data/LICENSE`).

Check an install without Muse or a model:

```
uv run --script plugins/glint-bank/mcp/glint_bank.py --decision-tools --selftest
```

It prints `{"ok": true, ...}` on stderr and exits 0 when the server lists its tools (the decision tools among them), loads task_054, reads the policy and runs one lookup.
