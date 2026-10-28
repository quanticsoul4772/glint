import json
import time

import pytest

from glint.meter import BudgetExceeded, Caps, Meter, guarded, litellm_callback, meta_cost
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


def test_stacked_guards_cap_each_pool_separately(tmp_path):
    glimmer = Meter(Caps(meta_usd=0.05, gpu_usd=0.0), tmp_path / "a.jsonl", phase="g")
    meta = Meter(Caps(meta_usd=1.0, gpu_usd=0.0), tmp_path / "b.jsonl", phase="m")
    calls = []
    fn = guarded(guarded(lambda **kw: calls.append(kw["model"]) or "ok", glimmer, only=("Glimmer",)),
                 meta, only=("muse-spark", "muse-wonder"))
    glimmer.record_meta("muse-wonder-1.0", 10_000, 0, 2_000)  # $0.08 recorded in the Glimmer pool: over its cap
    with pytest.raises(BudgetExceeded):
        fn(model="openai/meta-models/Muse-Glimmer-30B")
    assert fn(model="openai/muse-wonder-1.0") == "ok"  # the Meta pool is still under its cap
    assert calls == ["openai/muse-wonder-1.0"]


def _fake_openai_server():
    """A localhost OpenAI-compatible endpoint returning one fixed completion with usage (no network, no cost)."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.server.requests += 1  # counted, so a test can prove a blocked call never reached the endpoint
            self.rfile.read(int(self.headers.get("content-length") or 0))
            body = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                               "choices": [{"index": 0, "finish_reason": "stop",
                                            "message": {"role": "assistant", "content": "ok"}}],
                               "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}})
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.requests = 0
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_every_pool_records_through_litellm_registration(tmp_path):
    """Regression: two litellm_callback() instances in litellm.callbacks left the second pool unmetered (LiteLLM
    keeps one logger per class), so the Meta cap never fired and Wonder spent $55.88 against a $20 cap.
    Goes through litellm.callbacks and a real HTTP completion, the path the scripts use."""
    import litellm

    server = _fake_openai_server()
    base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    glim = Meter(Caps(meta_usd=5.0, gpu_usd=0.0), tmp_path / "ledger.jsonl", phase="glimmer")
    meta = Meter(Caps(meta_usd=5.0, gpu_usd=0.0), tmp_path / "ledger.jsonl", phase="meta")
    saved = litellm.callbacks
    litellm.callbacks = [litellm_callback((glim, ("Muse-Glimmer",)), (meta, ("muse-spark", "muse-wonder")))]
    try:
        for model in ("openai/meta-models/Muse-Glimmer-30B", "openai/muse-wonder-1.0"):
            litellm.completion(model=model, messages=[{"role": "user", "content": "hi"}], api_base=base,
                               api_key="test")
        deadline = time.monotonic() + 10  # LiteLLM runs sync success callbacks on a worker thread
        while (glim.meta_usd == 0 or meta.meta_usd == 0) and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        litellm.callbacks = saved
        server.shutdown()
    assert glim.meta_usd == pytest.approx(meta_cost("Muse-Glimmer-30B", 1000, 0, 100))
    assert meta.meta_usd == pytest.approx(meta_cost("muse-wonder-1.0", 1000, 0, 100))


def _wait_for(meter, usd, timeout=10):
    """LiteLLM records success callbacks on a worker thread; wait until the meter shows `usd`."""
    deadline = time.monotonic() + timeout
    while meter.meta_usd < usd - 1e-12 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert meter.meta_usd == pytest.approx(usd)


def test_cap_blocks_the_call_before_it_reaches_the_endpoint(tmp_path):
    """The path that failed: litellm.callbacks + stacked guards + real HTTP. Once a pool reaches its cap, the next
    call to that pool raises BudgetExceeded and the endpoint receives no request; the other pool keeps working
    until its own cap."""
    import litellm

    server = _fake_openai_server()
    base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    per_glimmer = meta_cost("Muse-Glimmer-30B", 1000, 0, 100)   # $0.00042 per fake call
    per_wonder = meta_cost("muse-wonder-1.0", 1000, 0, 100)     # $0.006 per fake call
    glim = Meter(Caps(meta_usd=2 * per_glimmer, gpu_usd=0.0), tmp_path / "ledger.jsonl", phase="glimmer")
    meta = Meter(Caps(meta_usd=1 * per_wonder, gpu_usd=0.0), tmp_path / "ledger.jsonl", phase="meta")
    saved = litellm.callbacks
    litellm.callbacks = [litellm_callback((glim, ("Muse-Glimmer",)), (meta, ("muse-spark", "muse-wonder")))]
    call = guarded(guarded(litellm.completion, glim, only=("Muse-Glimmer",)), meta,
                   only=("muse-spark", "muse-wonder"))
    msgs = [{"role": "user", "content": "hi"}]
    glimmer, wonder = "openai/meta-models/Muse-Glimmer-30B", "openai/muse-wonder-1.0"
    try:
        call(model=glimmer, messages=msgs, api_base=base, api_key="test")
        _wait_for(glim, per_glimmer)
        call(model=glimmer, messages=msgs, api_base=base, api_key="test")
        _wait_for(glim, 2 * per_glimmer)                       # Glimmer pool now at its cap
        assert server.requests == 2
        with pytest.raises(BudgetExceeded):
            call(model=glimmer, messages=msgs, api_base=base, api_key="test")
        assert server.requests == 2                            # the blocked call never left the process

        call(model=wonder, messages=msgs, api_base=base, api_key="test")   # Meta pool still open
        _wait_for(meta, per_wonder)                            # Meta pool now at its cap
        assert server.requests == 3
        with pytest.raises(BudgetExceeded):
            call(model=wonder, messages=msgs, api_base=base, api_key="test")
        assert server.requests == 3
    finally:
        litellm.callbacks = saved
        server.shutdown()


def test_litellm_knows_the_meta_prices():
    # Without these, every Spark call logged "This model isn't mapped yet" at ERROR level
    import litellm

    from glint.meter import meta_cost, register_litellm_prices

    register_litellm_prices()
    for model in ("openai/muse-spark-1.3-contributor", "openai/meta-models/Muse-Glimmer-30B"):
        p, c = litellm.cost_per_token(model=model, prompt_tokens=1000, completion_tokens=500)
        assert abs((p + c) - meta_cost(model, 1000, 0, 500)) < 1e-12


def test_a_metering_error_closes_the_cap_instead_of_running_free(tmp_path):
    # LiteLLM swallows logger exceptions; an unmetered call must close the pool, not leave the cap open
    import types

    import pytest

    from glint.meter import BudgetExceeded, Caps, Meter, litellm_callback

    meter = Meter(Caps(meta_usd=1.0, gpu_usd=0.0), ledger=tmp_path / "ledger.jsonl", phase="fail-test")
    logger = litellm_callback((meter, ("muse-spark",)))
    usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    # an unpriced model raises inside record_meta
    logger.log_success_event({"model": "muse-spark-unpriced"}, types.SimpleNamespace(usage=usage), 0, 0)
    with pytest.raises(BudgetExceeded, match="meter for phase fail-test is closed"):
        meter.check_meta()


def test_a_successful_call_without_usage_closes_the_cap(tmp_path):
    import types

    import pytest

    from glint.meter import BudgetExceeded, Caps, Meter, litellm_callback

    meter = Meter(Caps(meta_usd=1.0, gpu_usd=0.0), ledger=tmp_path / "ledger.jsonl", phase="no-usage")
    logger = litellm_callback((meter, ("muse-spark",)))
    logger.log_failure_event({"model": "muse-spark-1.3-contributor"}, types.SimpleNamespace(usage=None), 0, 0)
    meter.check_meta()  # a failed call with no usage was not billed
    logger.log_success_event({"model": "muse-spark-1.3-contributor"}, types.SimpleNamespace(usage=None), 0, 0)
    with pytest.raises(BudgetExceeded, match="carried no usage"):
        meter.check_meta()


def test_a_broken_ledger_line_names_the_file_and_line(tmp_path):
    import pytest

    from glint.meter import Caps, Meter

    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text('{"kind": "meta", "usd": 0.1, "phase": "p"}\n{"kind": "me', encoding="utf-8")
    with pytest.raises(RuntimeError, match="line 2 is not valid JSON"):
        Meter(Caps(meta_usd=1.0, gpu_usd=0.0), ledger=ledger, phase="p")


def test_the_ledger_reload_counts_only_the_meters_own_phase(tmp_path):
    # one ledger holds every phase; a resumed run must see its own earlier spend and nothing of another run's
    ledger = tmp_path / "ledger.jsonl"
    rows = [{"kind": "meta", "usd": 0.25, "phase": "a"}, {"kind": "gpu", "usd": 1.5, "phase": "a"},
            {"kind": "meta", "usd": 4.0, "phase": "b"}, {"kind": "gpu", "usd": 9.0, "phase": "b"}]
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    a = Meter(Caps(meta_usd=1.0, gpu_usd=2.0), ledger, phase="a")
    assert a.meta_usd == pytest.approx(0.25) and a.gpu_usd == pytest.approx(1.5)
    a.check()  # phase b's spend is past both caps; it must not stop phase a
    b = Meter(Caps(meta_usd=1.0, gpu_usd=2.0), ledger, phase="b")
    assert b.meta_usd == pytest.approx(4.0) and b.gpu_usd == pytest.approx(9.0)
    with pytest.raises(BudgetExceeded):
        b.check()


def test_guarded_takes_a_rate_limiter_slot_only_for_its_own_pool_and_only_under_the_cap(tmp_path):
    class Limiter:
        taken = 0

        def acquire(self):
            self.taken += 1

    m = Meter(Caps(meta_usd=0.05, gpu_usd=0.0), tmp_path / "ledger.jsonl")
    limiter = Limiter()
    fn = guarded(lambda **kw: "ok", m, limiter, only=("muse-spark",))
    assert fn(model="openai/muse-spark-1.3-contributor") == "ok"
    assert limiter.taken == 1
    assert fn(model="openai/Muse-Glimmer-30B") == "ok"  # another pool's model is neither capped nor paced here
    assert limiter.taken == 1
    m.record_meta("muse-wonder-1.0", 10_000, 0, 2_000)  # $0.08, over the cap
    with pytest.raises(BudgetExceeded):
        fn(model="openai/muse-spark-1.3-contributor")
    assert limiter.taken == 1  # a refused call does not wait for a slot


def test_guarded_paces_calls_through_a_real_rate_limiter(tmp_path):
    m = Meter(Caps(meta_usd=1.0, gpu_usd=0.0), tmp_path / "ledger.jsonl")
    fn = guarded(lambda **kw: "ok", m, RateLimiter(per_minute=600, burst=1))  # 10/s
    t0 = time.monotonic()
    for _ in range(6):
        fn(model="muse-spark-1.3-contributor")
    assert time.monotonic() - t0 >= 0.45


# --- Mutation-testing gaps (cosmic-ray on main, 2026-10-28: 86 of 334 mutants survived) -----------------------------
def test_every_model_price_is_the_published_one():
    # $ per 1M tokens: Meta's pricing page and Baseten's Spark Open page (2026-10-26), DeepInfra's estimated_cost for
    # Glimmer. The caps add up these numbers, so a changed price must be a deliberate edit here too.
    from glint.meter import META_PRICES

    spark = {"input": 1.25, "cached": 0.15, "output": 4.25}
    contributor = {"input": 0.10, "cached": 0.002, "output": 0.20}
    assert META_PRICES == {
        "muse-spark-1.3": spark, "muse-spark-1.2": spark, "muse-spark-1.1": spark,
        "muse-spark-1.3-contributor": contributor, "muse-spark-1.2-contributor": contributor,
        "muse-wonder-1.0": {"input": 4.00, "cached": 0.20, "output": 20.00},
        "muse-spark-open-1026": spark,
        "Muse-Glimmer-30B": {"input": 0.30, "cached": 0.04, "output": 1.20},
        "Muse-Glimmer-30B-modal": {"input": 0.0, "cached": 0.0, "output": 0.0}}
    # cached tokens are billed at the cached price, the rest of the prompt at the input price
    assert meta_cost("muse-spark-1.3", 1_000_000, 400_000, 0) == pytest.approx(0.6 * 1.25 + 0.4 * 0.15)
    assert meta_cost("openai/meta-models/Muse-Glimmer-30B-modal", 10**6, 10**5, 10**6) == 0.0


def test_litellm_gets_the_cached_price_per_token():
    import litellm

    from glint.meter import register_litellm_prices

    register_litellm_prices()
    info = litellm.get_model_info("openai/muse-wonder-1.0")
    assert (info["input_cost_per_token"], info["cache_read_input_token_cost"], info["output_cost_per_token"]) == (
        pytest.approx(4.00e-6), pytest.approx(0.20e-6), pytest.approx(20.00e-6))


def test_a_response_with_missing_token_counts_records_zero_not_one(tmp_path):
    from types import SimpleNamespace

    meter = Meter(Caps(meta_usd=5.0, gpu_usd=0.0), tmp_path / "ledger.jsonl")
    logger = litellm_callback((meter, ("muse-spark",)))
    usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=None)  # no details objects at all
    logger._meter_call({}, SimpleNamespace(usage=usage), "muse-spark-1.3", True)
    (rec,) = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert (rec["prompt"], rec["cached"], rec["completion"], rec["reasoning"]) == (1000, 0, 0, 0)
    assert rec["usd"] == pytest.approx(meta_cost("muse-spark-1.3", 1000, 0, 0))
