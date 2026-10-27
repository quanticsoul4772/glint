"""Plan B step 1 (measurement plan v1 §M2): all 74 retail train tasks x k=16, Glimmer on DeepInfra at medium strength,
contributor Spark as user simulator and NL judge. Runs in batches of k=4 per split so a crash loses at most one batch;
each batch has its own tau2 seed, and its global trial index is batch*4 + trial. The cert_reserve split is generated
but sealed: never labeled or analyzed here. Spend cap $20 (DeepInfra + Meta).
Usage: python planb_k16.py
"""
import json
import time
from pathlib import Path

from m2_pilot import GLIMMER, META, SPARK, glimmer_args, install
from tau2.data_model.simulation import TextRunConfig
from tau2.runner.batch import run_domain

from glint.meter import Caps, Meter

PARTITION = json.loads(Path(r"<workspace>\research\5-measurements\split-partition.json").read_text())
DATA = Path(r"<workspace>\measure\data\planb")
SPLITS = ("pilot", "fit", "cert_reserve")
BATCHES, K = 4, 4

meter = Meter(Caps(meta_usd=20.0, gpu_usd=0.0), phase="planb-k16")
install(meter)
for split in SPLITS:
    (DATA / split).mkdir(parents=True, exist_ok=True)
    for batch in range(BATCHES):
        out = DATA / split / f"batch{batch}.json"
        if out.exists():
            continue
        t0 = time.time()
        config = TextRunConfig(
            domain="retail", task_split_name="train", task_ids=PARTITION["retail"][split], agent="llm_agent",
            llm_agent=GLIMMER, llm_args_agent=glimmer_args("medium"), user="user_simulator", llm_user=SPARK,
            llm_args_user=META, num_trials=K, max_concurrency=25, seed=1000 + batch,
            save_to=f"glint-planb-{split}-b{batch}", auto_review=False,
        )
        results = run_domain(config)
        out.write_text(results.model_dump_json(), encoding="utf-8")
        print(f"{split} batch {batch}: {len(results.simulations)} sims in {time.time() - t0:.0f}s; "
              f"spend so far ${meter.meta_usd:.2f}", flush=True)
