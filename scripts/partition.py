"""One-time partition of the τ² train splits for the measurement phase (plan: Ground rules / Splits).

Retail train (74) -> pilot 10 / fit 44 / cert_reserve 20. Airline train (30) -> pilot 5 (pass rate only).
The test splits are never read. Deterministic: tasks are ordered by sha256(SEED + task_id).
"""
import hashlib
import json

import splits
from glint import paths

SEED = "glint-measurement-2026-10-26"
DOMAINS = paths.tau2_data() / "tau2" / "domains"
OUT = splits.PARTITION


def ordered(ids):
    return sorted(ids, key=lambda i: hashlib.sha256(f"{SEED}:{i}".encode()).hexdigest())


def split_of(domain):
    raw = (DOMAINS / domain / "split_tasks.json").read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


retail, retail_sha = split_of("retail")
airline, airline_sha = split_of("airline")
r = ordered(retail["train"])
a = ordered(airline["train"])
partition = {
    "seed": SEED,
    "source": {"retail_split_sha256": retail_sha, "airline_split_sha256": airline_sha,
               "tau2_commit": "fc0055dc4e0a316c3f83133267fbd6faaa770992"},
    "retail": {"pilot": r[:10], "fit": r[10:54], "cert_reserve": r[54:74]},
    "airline": {"pilot": a[:5]},
    "test_untouched": {"retail": len(retail["test"]), "airline": len(airline["test"])},
}
assert len(partition["retail"]["fit"]) == 44 and len(partition["retail"]["cert_reserve"]) == 20
assert not set(r) & set(retail["test"])
OUT.write_text(json.dumps(partition, indent=1))
print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in partition["retail"].items()}),
      "airline pilot", partition["airline"]["pilot"])
