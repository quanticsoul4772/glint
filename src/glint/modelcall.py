"""The pluggable model call shared by the LLM-stage harnesses (glint.extract, glint.second_sim, glint.denystate,
glint.draft_rules). Every harness is off by default: with no model call configured it raises ModelOff and calls
nothing. A model call is any function `prompt (str) -> reply text (str)`, named on the command line as
"module:function" (resolve). This repo ships no function that calls a paid service; whoever runs a harness supplies
one, and a run that costs money needs the owner's yes and glint.meter caps (project rules).
"""
from __future__ import annotations

import importlib
import json
import re
from collections.abc import Callable

ModelCall = Callable[[str], str]
OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class ModelOff(RuntimeError):
    """No model call is configured: the LLM stage is off, and nothing was called."""


def resolve(spec: str | None) -> ModelCall | None:
    """The function a "module:function" spec names, or None for no spec (the default: off)."""
    if not spec:
        return None
    module, _, name = spec.partition(":")
    if not module or not name:
        raise ValueError(f"a model call is named as module:function, got {spec!r}")
    fn = getattr(importlib.import_module(module), name)
    if not callable(fn):
        raise TypeError(f"{spec} is not callable")
    return fn


def require(model: ModelCall | None, stage: str) -> ModelCall:
    if model is None:
        raise ModelOff(f"{stage}: no model call is configured, so this LLM stage is off (pass module:function)")
    return model


def parse_object(text: str) -> dict | None:
    """The first JSON object in a model's reply (fenced or bare), or None when there is none."""
    m = OBJECT.search(str(text or ""))
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return v if isinstance(v, dict) else None
