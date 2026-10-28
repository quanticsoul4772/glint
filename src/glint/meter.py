"""Spend meter and hard kill switch (plan: Ground rules / Spend guards; design tension D7).

Costs are computed here from token usage and wall-clock GPU time. Vendor cost fields are never trusted
(AOR lesson: zeros on paid runs, credits as strings, cache tokens uncounted -> 4.3x overrun).
Every charge is appended to a JSONL ledger on D:, so totals survive restarts and can be reconciled
against the Meta billing page and the Modal usage page.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from glint import paths

# $ per 1M tokens (research/2-platform/platform-pricing-limits.md, Meta pricing page 2026-10-26).
META_PRICES = {
    "muse-spark-1.3": {"input": 1.25, "cached": 0.15, "output": 4.25},
    "muse-spark-1.2": {"input": 1.25, "cached": 0.15, "output": 4.25},
    "muse-spark-1.1": {"input": 1.25, "cached": 0.15, "output": 4.25},
    "muse-spark-1.3-contributor": {"input": 0.10, "cached": 0.002, "output": 0.20},
    "muse-spark-1.2-contributor": {"input": 0.10, "cached": 0.002, "output": 0.20},
    "muse-wonder-1.0": {"input": 4.00, "cached": 0.20, "output": 20.00},
    # Spark Open on Baseten (hackathon credit; baseten.co/library/muse-spark-open, 2026-10-26).
    "muse-spark-open-1026": {"input": 1.25, "cached": 0.15, "output": 4.25},
    # Glimmer on DeepInfra (bf16; matches the `estimated_cost` DeepInfra returned on 2026-10-26).
    "Muse-Glimmer-30B": {"input": 0.30, "cached": 0.04, "output": 1.20},
    # Glimmer served on Modal (glint.glimmer_serve): no per-token price; its cost is GPU time, metered by session()
    "Muse-Glimmer-30B-modal": {"input": 0.0, "cached": 0.0, "output": 0.0},
}

DEFAULT_LEDGER = paths.LEDGER


def register_litellm_prices() -> None:
    """Give LiteLLM the per-token prices of the models in META_PRICES. Without them every call logs "This model isn't
    mapped yet" at ERROR level (one line per call in a run log) and τ²'s cost fields read 0. The meter does not use
    LiteLLM's cost; this is for clean logs and τ²'s own bookkeeping."""
    import litellm

    entries = {}
    for name, p in META_PRICES.items():
        entry = {"input_cost_per_token": p["input"] / 1e6, "output_cost_per_token": p["output"] / 1e6,
                 "cache_read_input_token_cost": p["cached"] / 1e6, "litellm_provider": "openai", "mode": "chat"}
        entries[name] = entries[f"openai/{name}"] = entries[f"openai/meta-models/{name}"] = entry
    litellm.register_model(entries)


class BudgetExceeded(RuntimeError):
    # τ² keeps only str(exception) of a failed run: the class name goes first so runlog.outcome can tell a cap stop
    # from an infrastructure error
    def __str__(self) -> str:
        return f"BudgetExceeded: {super().__str__()}"


@dataclass(frozen=True)
class Caps:
    meta_usd: float
    gpu_usd: float


def meta_cost(model: str, prompt_tokens: int, cached_tokens: int, completion_tokens: int) -> float:
    """Cost of one Meta Model API call. Unknown models raise, so an unpriced model can never run free."""
    name = model.split("/")[-1]
    if name not in META_PRICES:
        raise KeyError(f"no price for model {model!r}; add it to META_PRICES before calling it")
    p = META_PRICES[name]
    uncached = max(prompt_tokens - cached_tokens, 0)
    return (uncached * p["input"] + cached_tokens * p["cached"] + completion_tokens * p["output"]) / 1e6


class Meter:
    """Thread-safe running totals backed by an append-only ledger. `check()` raises once a cap is hit."""

    def __init__(self, caps: Caps, ledger: Path = DEFAULT_LEDGER, phase: str = "measure"):
        self.caps, self.ledger, self.phase = caps, ledger, phase
        self._lock = threading.Lock()
        self.meta_usd = 0.0
        self.gpu_usd = 0.0
        ledger.parent.mkdir(parents=True, exist_ok=True)
        if ledger.exists():
            for line in ledger.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if rec.get("phase") != phase:
                    continue
                if rec["kind"] == "meta":
                    self.meta_usd += rec["usd"]
                elif rec["kind"] == "gpu":
                    self.gpu_usd += rec["usd"]

    def _append(self, rec: dict) -> None:
        rec |= {"ts": time.time(), "phase": self.phase}
        with self.ledger.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    def record_meta(self, model: str, prompt_tokens: int, cached_tokens: int, completion_tokens: int,
                    reasoning_tokens: int = 0, tag: str = "") -> float:
        usd = meta_cost(model, prompt_tokens, cached_tokens, completion_tokens)
        with self._lock:
            self.meta_usd += usd
            self._append({"kind": "meta", "model": model, "prompt": prompt_tokens, "cached": cached_tokens,
                          "completion": completion_tokens, "reasoning": reasoning_tokens, "usd": usd, "tag": tag})
        return usd

    def record_gpu(self, seconds: float, usd_per_hour: float, gpu: str, tag: str = "") -> float:
        """GPU + CPU + RAM: pass the all-in container rate. Count load, run *and* idle time."""
        usd = seconds * usd_per_hour / 3600
        with self._lock:
            self.gpu_usd += usd
            self._append({"kind": "gpu", "gpu": gpu, "seconds": seconds, "rate": usd_per_hour, "usd": usd,
                          "tag": tag})
        return usd

    def check_meta(self) -> None:
        if self.meta_usd >= self.caps.meta_usd:
            raise BudgetExceeded(f"Meta API spend ${self.meta_usd:.4f} reached cap ${self.caps.meta_usd}")

    def check_gpu(self) -> None:
        if self.gpu_usd >= self.caps.gpu_usd:
            raise BudgetExceeded(f"GPU spend ${self.gpu_usd:.2f} reached cap ${self.caps.gpu_usd}")

    def check(self) -> None:
        self.check_meta()
        self.check_gpu()


def _matches(model: str, only: tuple[str, ...] | None) -> bool:
    return only is None or any(s in model for s in only)


def guarded(completion_fn, meter: Meter, limiter=None, only: tuple[str, ...] | None = None):
    """Wrap a completion function so the Meta cap and the rate limiter are enforced before every call.

    This cannot live in a LiteLLM callback: LiteLLM catches exceptions raised in `log_pre_api_call` and sends
    the request anyway. Install with e.g. `tau2.utils.llm_utils.completion = guarded(completion, meter, limiter)`.
    With `only`, the cap and limiter apply only to models whose name contains one of the substrings, so separate
    credit pools can be capped by stacking guards.
    """
    def _call(*args, **kwargs):
        if _matches(kwargs.get("model", ""), only):
            meter.check_meta()
            if limiter is not None:
                limiter.acquire()
        return completion_fn(*args, **kwargs)

    return _call


def litellm_callback(*pools):
    """One LiteLLM CustomLogger that meters usage after each call (enforcement is `guarded`).

    Each pool is a Meter, or (Meter, only) to record just the models whose name contains one of the substrings.
    All pools must go through this single logger: LiteLLM keeps only one instance per logger class in
    `litellm.callbacks`, so a second `litellm_callback(...)` would be silently dropped and its pool never metered.
    Aborted / failed calls are metered too when the provider returned usage.
    """
    from litellm.integrations.custom_logger import CustomLogger

    routed = [(p, None) if isinstance(p, Meter) else p for p in pools]
    if not routed:
        raise ValueError("litellm_callback needs at least one meter")

    class _MeterLogger(CustomLogger):
        def __init__(self):
            super().__init__()
            # LiteLLM de-duplicates loggers by class name + public str/bool/int attributes. Without a unique public
            # attribute, a second meter logger (in the same list, or registered later in the same process) is
            # silently dropped and its pool is never metered.
            self.glint_logger_id = uuid.uuid4().hex

        def _record(self, kwargs, response_obj):
            usage = getattr(response_obj, "usage", None)
            if usage is None:
                return
            model = kwargs.get("model", "")
            details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) or 0
            out_details = getattr(usage, "completion_tokens_details", None)
            reasoning = getattr(out_details, "reasoning_tokens", 0) or 0
            tag = str((kwargs.get("litellm_params") or {}).get("metadata", {}).get("tag", ""))
            for meter, only in routed:
                if _matches(model, only):
                    meter.record_meta(model, usage.prompt_tokens or 0, cached, usage.completion_tokens or 0,
                                      reasoning, tag=tag)

        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            self._record(kwargs, response_obj)

        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            self._record(kwargs, response_obj)

    return _MeterLogger()
