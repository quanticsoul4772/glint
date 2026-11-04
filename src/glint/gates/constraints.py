"""Constrained values for the banking write tools: the enums and numeric bounds τ²'s own tool docstrings state, as a
check on a call's arguments and as a JSON schema for decode-time constrained decoding (appendix #39 of
research/fix-research-2026-11-02.md, "Decode-time numeric ranges and enums").

Source: each tool's docstring in τ²'s banking toolkit (`tau2.domains.banking_knowledge.tools.KnowledgeTools`), read
at import: an argument line "name (type): ... Must be one of: 'a', 'b'" gives an enum; "(must be greater than 0)" or
"Must be a positive number" gives a lower bound. τ²'s tools reject a value outside these themselves, so the check
moves the rejection before the call and gives it the gate's deny format; the schema keeps the value from being
generated at all.

The check is ON by default (ENABLED True since the owner's decision of 2026-11-03, every fix on by default): the
banking registry gets one constraint gate per constrained tool (`register`), which changes what an agent sees on such
a call. The off switch is Enforcer(constraints=False) (glint.gates.module_switches). The decode-time path
(`guided_decoding`) returns
the request field a vLLM server takes for guided JSON; no serving code calls it, and checking its effect needs a paid
run (D§8: no deny reaches the log on that path).
"""
from __future__ import annotations

import inspect
import re
from functools import lru_cache

from glint.gates.core import ACCOUNT_RECORDS, Gate, GateRegistry, GateResult, allow, deny

ENABLED = True
SOURCE = "τ² banking tool docstrings (tau2.domains.banking_knowledge.tools)"
_ARG = re.compile(r"^\s*(\w+) \((\w+)\):(.*)$")
_ENUM = re.compile(r"[Mm]ust be one of:\s*(.*)$")
_QUOTED = re.compile(r"'([^']+)'")
_POSITIVE = re.compile(r"must be greater than 0|[Mm]ust be a positive number")
JSON_TYPES = {"string": "string", "number": "number", "float": "number", "integer": "integer", "int": "integer",
              "boolean": "boolean", "bool": "boolean"}


def parse_docstring(doc: str) -> dict[str, dict]:
    """{argument: {"type", "enum"?, "exclusiveMinimum"?}} for the arguments a docstring constrains."""
    out: dict[str, dict] = {}
    for line in (doc or "").splitlines():
        m = _ARG.match(line)
        if not m:
            continue
        name, typ, text = m.groups()
        spec: dict = {}
        if (e := _ENUM.search(text)) and (values := _QUOTED.findall(e.group(1))):
            spec["enum"] = values
        if _POSITIVE.search(text):
            spec["exclusiveMinimum"] = 0
        if spec:
            out[name] = {"type": JSON_TYPES.get(typ.lower(), "string")} | spec
    return out


@lru_cache(maxsize=1)
def tool_constraints() -> dict[str, dict[str, dict]]:
    """{tool: {argument: constraint}} for every banking tool whose docstring constrains an argument."""
    from tau2.domains.banking_knowledge.tools import KnowledgeTools
    table = {}
    for name, fn in inspect.getmembers(KnowledgeTools, inspect.isfunction):
        if not name.startswith("_") and (spec := parse_docstring(inspect.getdoc(fn) or "")):
            table[name] = spec
    return table


def violations(tool: str, args: dict, table: dict | None = None) -> list[str]:
    """Each constrained argument of the call whose value is outside its enum or bound. A missing argument is not judged
    (the tool rejects it)."""
    out = []
    for arg, spec in (table if table is not None else tool_constraints()).get(tool, {}).items():
        if arg not in args or args[arg] is None:
            continue
        value = args[arg]
        if "enum" in spec and value not in spec["enum"]:
            out.append(f"{arg} {value!r} is not one of {spec['enum']}")
        if "exclusiveMinimum" in spec:
            try:
                number = float(value)
            except (TypeError, ValueError):
                out.append(f"{arg} {value!r} is not a number")
                continue
            if number <= spec["exclusiveMinimum"]:
                out.append(f"{arg} {value!r} must be greater than {spec['exclusiveMinimum']}")
    return out


def json_schema(tool: str, table: dict | None = None) -> dict:
    """The JSON schema of the tool's constrained arguments (other arguments are left open)."""
    props = (table if table is not None else tool_constraints()).get(tool, {})
    return {"type": "object", "properties": {k: dict(v) for k, v in props.items()}, "additionalProperties": True}


def guided_decoding(tool: str, table: dict | None = None) -> dict:
    """The extra request body a vLLM OpenAI-compatible server takes to constrain the hidden tool's `arguments` JSON
    (`guided_json`). Not called by the serving code: the decode-time path is built, not wired."""
    return {"guided_json": json_schema(tool, table)}


def gate(tool: str):
    def check(db, a: dict) -> GateResult:
        bad = violations(tool, a)
        if bad:
            return deny("constraints.value", SOURCE, "an argument is outside the values the tool accepts",
                        "Send the call again with each named argument set to one of its allowed values.",
                        {"violations": bad}, consequence="Nothing ran; the tool itself would reject these values.",
                        facts=bad)
        return allow("constraints.value", SOURCE)
    return check


def register(reg: GateRegistry, domain: str) -> None:
    """One constraint gate per constrained banking tool; a no-op unless ENABLED."""
    if not ENABLED or domain != "banking_knowledge":
        return
    for tool in sorted(tool_constraints()):
        reg.register(Gate(f"constraints_{tool}", (tool,), "The enums and bounds in the tool's own docstring.",
                          gate(tool), checks=ACCOUNT_RECORDS))
