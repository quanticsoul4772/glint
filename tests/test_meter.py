import time

import pytest

from glint.meter import BudgetExceeded, Caps, Meter, guarded, meta_cost
from glint.ratelimit import RateLimiter


def test_meta_cost_hand_math():
    # contributor: 1,000,000 prompt of which 400,000 cached, 100,000 out
    # = 600k*0.10 + 400k*0.002 + 100k*0.20 per 1M = 0.06 + 0.0008 + 0.02
    assert meta_cost("openai/muse-spark-1.3-contributor", 1_000_000, 400_000, 100_000) == pytest.approx(0.0808)
    assert meta_cost("muse-wonder-1.0", 10_000, 0, 2_000) == pytest.approx(0.04 + 0.04)


def test_unknown_model_refuses():
    with pytest.raises(KeyError):
        meta_cost("muse-image-1.0", 1, 0, 1)


def test_cap_trips_and_ledger_persists(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    m = Meter(Caps(meta_usd=0.05, gpu_usd=1.0), ledger)
    m.record_meta("muse-wonder-1.0", 10_000, 0, 2_000)  # $0.08
    with pytest.raises(BudgetExceeded):
        m.check()
    reloaded = Meter(Caps(meta_usd=1.0, gpu_usd=1.0), ledger)
    assert reloaded.meta_usd == pytest.approx(0.08)
    reloaded.record_gpu(3600, 3.26, "A100-80GB")
    assert reloaded.gpu_usd == pytest.approx(3.26)
    with pytest.raises(BudgetExceeded):
        Meter(Caps(meta_usd=1.0, gpu_usd=3.0), ledger).check()


def test_zero_gpu_cap_does_not_block_meta_calls(tmp_path):
    m = Meter(Caps(meta_usd=1.0, gpu_usd=0.0), tmp_path / "ledger.jsonl")
    m.check_meta()
    with pytest.raises(BudgetExceeded):
        m.check_gpu()


def test_guarded_blocks_the_call_over_cap(tmp_path):
    m = Meter(Caps(meta_usd=0.05, gpu_usd=0.0), tmp_path / "ledger.jsonl")
    calls = []
    fn = guarded(lambda **kw: calls.append(kw) or "ok", m)
    assert fn(model="x") == "ok"
    m.record_meta("muse-wonder-1.0", 10_000, 0, 2_000)  # $0.08
    with pytest.raises(BudgetExceeded):
        fn(model="x")
    assert len(calls) == 1


def test_rate_limiter_paces():
    rl = RateLimiter(per_minute=600, burst=1)  # 10/s
    t0 = time.monotonic()
    for _ in range(6):
        rl.acquire()
    assert time.monotonic() - t0 >= 0.45
