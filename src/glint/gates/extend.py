"""Extension points of the enforcer (glint.gates.enforce): registered options and ordered hooks.

Each feature of the enforcer registers what it adds next to its own code, so two features never edit the same lines
of enforce.py (Glint convention, 2026-11-03: every feature branch used to edit `Enforcer.__init__`, the one
`gates.manifest` call and the allowed-write branch of the `get_response` wrapper, and open PRs collided there).

Options. `option(name, default, doc, ...)` registers one keyword of `Enforcer(...)`: its default, a one-line doc,
an optional `setup(enforcer, value)` (validates the value and sets the enforcer's attributes; without one the value
is stored as `enforcer.<name>`) and an optional `manifest(enforcer, value) -> dict` (the keys it adds to the
`gates.manifest` event; without one it adds none). `Enforcer.__init__` sets up every registered option in `order`
and builds the manifest from them. An unknown keyword is a TypeError, as for any Python call.

Hooks. `hook(phase, order)` registers a function run at one point of the wrapper. Hooks of a phase run sorted by
(order, name), never by import order, so the order is the same in every process and in τ²'s replay. A hook must be a
function of what the replay rebuilds (the database, the call, its reply and the env.tools records kept by hooks),
which is what keeps every replayed decision and reply equal to the live one. The phases:

- "attach" `fn(enforcer, env)`: once per gated environment, after the core records exist; creates a feature's
  env.tools records.
- "before_call" `fn(call)`: at the start of every call through the wrapper.
- "after_read" `fn(call, reply)`: after a call that is not a gated write ran (reads, generic tools, ungated calls).
- "before_write" `fn(call)`: after every gate allowed a write, before it runs (`call.scratch` carries state forward).
- "after_write" `fn(call, reply)`: after an allowed write ran, whether or not its reply is an error.
- "reply" `fn(call, reply)`: on every reply that is not a refusal, read or allowed write, last.
- "open_items" `fn(call) -> list[str]`: lines added after the core open items (enforce._open_items) on banking.

An "after_read", "after_write" or "reply" hook returns a new ToolMessage to replace the reply, or None to keep it.

A new feature registers an option and hooks in its own module (imported by enforce.py) instead of editing
`Enforcer.__init__`, the manifest call or `get_response`.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

PHASES = ("attach", "before_call", "after_read", "before_write", "after_write", "reply", "open_items")
REPLY_PHASES = ("after_read", "after_write", "reply")  # hooks that may replace the reply


@dataclass(frozen=True)
class Option:
    name: str
    default: Any
    doc: str
    order: int
    owner: str
    setup: Callable[[Any, Any], None] | None = None
    manifest: Callable[[Any, Any], dict] | None = None


@dataclass(frozen=True)
class Hook:
    phase: str
    order: int
    name: str
    fn: Callable


@dataclass
class Call:
    """One call through the gated wrapper, as every hook sees it."""
    enforcer: Any
    env: Any
    message: Any  # the tau2 ToolCall
    replay: bool
    write: tuple[str, dict | None] | None = None  # (tool, arguments) for a gated write, else None
    result: Any = None  # the allowing GateResult for a gated write
    scratch: dict = field(default_factory=dict)

    @property
    def tools(self):
        return self.env.tools


OPTIONS: dict[str, Option] = {}
HOOKS: dict[str, dict[str, Hook]] = {p: {} for p in PHASES}


def ran(reply) -> bool:
    """The call ran: its reply is not an error (τ² returns some errors as content starting with "Error")."""
    return not reply.error and not str(reply.content).startswith("Error")


def _owner(fn) -> str:
    return f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', repr(fn))}"


def option(name: str, default: Any, doc: str, *, order: int, owner: str,
           setup: Callable[[Any, Any], None] | None = None,
           manifest: Callable[[Any, Any], dict] | None = None) -> Option:
    """Register an Enforcer keyword. `owner` is the registering module's __name__; a second registration of the name
    from another module is refused, one from the same module (a reload) replaces the first."""
    if not doc or "\n" in doc:
        raise ValueError(f"option {name!r}: give a one-line doc")
    if name in OPTIONS and OPTIONS[name].owner != owner:
        raise ValueError(f"option {name!r} is already registered by {OPTIONS[name].owner}")
    OPTIONS[name] = opt = Option(name, default, doc, order, owner, setup, manifest)
    return opt


def options() -> list[Option]:
    """Every registered option, in setup order (order, then name)."""
    return sorted(OPTIONS.values(), key=lambda o: (o.order, o.name))


def hook(phase: str, order: int, name: str | None = None) -> Callable[[Callable], Callable]:
    """Decorator: register `fn` for `phase` at `order` under `name` (default: its module and qualified name). A second
    registration of the same name (a reload) replaces the first."""
    if phase not in PHASES:
        raise ValueError(f"unknown hook phase {phase!r}; one of {PHASES}")

    def register(fn: Callable) -> Callable:
        key = name or _owner(fn)
        HOOKS[phase][key] = Hook(phase, order, key, fn)
        return fn
    return register


def hooks(phase: str) -> list[Hook]:
    """The hooks of `phase` in run order: (order, name)."""
    return sorted(HOOKS[phase].values(), key=lambda h: (h.order, h.name))


def run(phase: str, *args) -> None:
    """Run the hooks of a phase that returns nothing ("attach", "before_call", "before_write")."""
    for h in hooks(phase):
        h.fn(*args)


def run_reply(phase: str, call: Call, reply):
    """Run the hooks of a reply phase in order, each on the reply the one before returned; the final reply."""
    for h in hooks(phase):
        new = h.fn(call, reply)
        reply = reply if new is None else new
    return reply


def open_items(call: Call) -> list[str]:
    """The "open_items" hooks' lines, in order."""
    return [line for h in hooks("open_items") for line in h.fn(call)]
