"""Conformal bounds for a calibrated verifier cascade (appendix #26; reader 4 K1-K3; design §8): the tooling for
"cascade to a calibrated verifier; unsure cases asked".

A verifier gives each write a score (higher: more likely off-policy). On a calibration set of labeled writes this
module picks thresholds with finite-sample guarantees, assuming the calibration writes and the new writes are
exchangeable (any rule or prompt change voids the bound: reader 4 K2, #25):
- `block_threshold(benign, alpha)`: split conformal. Blocking only scores above the threshold blocks a new benign
  write with probability at most alpha.
- `crc_threshold(losses, alpha, bound)`: conformal risk control (Angelopoulos et al. 2022) for a loss that falls as
  the parameter grows: the smallest parameter whose adjusted calibration risk (n R + B) / (n + 1) is at most alpha.
- `cascade(score, ask_at, block_at)`: allow below ask_at, block above block_at, ask the customer between.
- `ask_threshold(offpolicy, beta)`: the threshold under which at most beta of off-policy writes fall (misses).

The verifier itself is an LLM stage (on the "not before Nov 4" list): this module takes scores from any scorer and
calls none. The deterministic gates' own false-block bound is glint.measures.clopper_pearson_upper (N5).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence


def block_threshold(benign: Sequence[float], alpha: float) -> float:
    """The ceil((n + 1)(1 - alpha))-th smallest benign score; +inf when n is too small for alpha (block nothing)."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    n = len(benign)
    k = math.ceil((n + 1) * (1 - alpha))
    return sorted(benign)[k - 1] if k <= n else math.inf


def ask_threshold(offpolicy: Sequence[float], beta: float) -> float:
    """The floor((n + 1) beta)-th smallest off-policy score: writes scoring below it are allowed, and a new off-policy
    write falls there with probability at most beta. -inf when n is too small (allow nothing unasked)."""
    if not 0 < beta < 1:
        raise ValueError("beta must be in (0, 1)")
    k = math.floor((len(offpolicy) + 1) * beta)
    return sorted(offpolicy)[k - 1] if k >= 1 else -math.inf


def crc_threshold(losses: Callable[[float], Sequence[float]], grid: Sequence[float], alpha: float,
                  bound: float = 1.0) -> float | None:
    """The smallest parameter on `grid` (ascending) whose calibration losses (each in [0, bound], non-increasing in
    the parameter) satisfy (sum + bound) / (n + 1) <= alpha; None when no grid point does."""
    for lam in sorted(grid):
        ls = list(losses(lam))
        if any(not 0 <= x <= bound for x in ls):
            raise ValueError(f"losses must lie in [0, {bound}]")
        if (sum(ls) + bound) / (len(ls) + 1) <= alpha:
            return lam
    return None


def cascade(score: float, ask_at: float, block_at: float) -> str:
    """allow (score < ask_at), block (score > block_at) or ask (between) for one score."""
    if ask_at > block_at:
        raise ValueError("ask_at must not exceed block_at")
    if score > block_at:
        return "block"
    return "ask" if score >= ask_at else "allow"


def calibrate(benign: Sequence[float], offpolicy: Sequence[float], alpha: float, beta: float) -> dict:
    """Both thresholds from one calibration set, with the empirical rates on that set (in-sample, for reading only)."""
    block_at = block_threshold(benign, alpha)
    ask_at = min(ask_threshold(offpolicy, beta), block_at)
    return {"block_at": block_at, "ask_at": ask_at, "alpha": alpha, "beta": beta, "n_benign": len(benign),
            "n_offpolicy": len(offpolicy),
            "benign_blocked": sum(1 for s in benign if s > block_at),
            "benign_asked": sum(1 for s in benign if ask_at <= s <= block_at),
            "offpolicy_allowed": sum(1 for s in offpolicy if s < ask_at)}
