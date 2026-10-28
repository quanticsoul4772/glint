"""Glimmer activation capture on Modal at bf16 (Transformers), for the teacher-forced replay.

`download` (CPU) puts the pinned Glimmer weights on a Modal Volume once. `capture` (A100-80GB) runs job lines built
by scripts/build_jobs.py: {"traj", "ops": [{"append": ids, "read": [[rel, label]]} | {"truncate": abs_pos}]}.
Each layer's output (the residual stream) is saved at the read positions; the last layer is skipped to match the
llama.cpp layout. The cache is a config-less DynamicCache, so sliding-window layers keep every position and can be
rolled back with `crop`; the sliding window itself is still applied by the attention mask. Prefill runs in chunks so
attention never materialises a full-length score matrix.

Run: MODAL_CONFIG_PATH=... modal run src/glint/glimmer_modal.py --jobs JOBS.jsonl --run NAME
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

from glint import paths

REPO, REVISION = "meta-models/Muse-Glimmer-30B", "cec94fcfd6512f2cfe708138c32533b346ea4f69"
WEIGHTS = "/vol/weights"
GPU, GPU_USD_PER_HOUR = "A100-80GB", 2.94  # A100-80GB $2.50/h + ~$0.44/h for 4 cores and 32 GiB
CHUNK = 2048

app = modal.App("glint-capture")
vol = modal.Volume.from_name("glint-glimmer", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.14.1", "transformers==5.19.0", "accelerate==1.15.0", "huggingface_hub==2.2.0", "hf_transfer==0.1.9",
    "safetensors", "numpy",
).env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})


@app.function(image=image, volumes={"/vol": vol}, cpu=4, memory=8192, timeout=3600, max_containers=1)
def download() -> str:
    from huggingface_hub import snapshot_download

    marker = Path(WEIGHTS) / f".complete-{REVISION}"
    if marker.exists():
        return "present"
    snapshot_download(REPO, revision=REVISION, local_dir=WEIGHTS,
                      allow_patterns=["*.safetensors", "*.json", "*.jinja"])
    marker.touch()
    vol.commit()
    return "downloaded"


@app.function(image=image, volumes={"/vol": vol}, gpu=GPU, cpu=4, memory=32768, timeout=3600, max_containers=1)
def capture(jobs_text: str, run: str, max_seconds: float) -> dict:
    import numpy as np
    import torch
    from transformers import DynamicCache, MuseGlimmerForConditionalGeneration

    t0 = time.time()
    if not (Path(WEIGHTS) / f".complete-{REVISION}").exists():
        raise RuntimeError("weights missing on the volume; run download first")
    model = MuseGlimmerForConditionalGeneration.from_pretrained(
        WEIGHTS, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    model.eval()
    text_model = model.model.language_model
    layers = text_model.layers
    n_save = len(layers) - 1
    load_s = time.time() - t0

    state = {"start": 0, "want": {}, "buf": None}

    def hook(layer_idx):
        def _h(_module, _inputs, output):
            for abs_pos, rows in state["want"].items():
                col = abs_pos - state["start"]
                if 0 <= col < output.shape[1]:
                    vec = output[0, col].to(torch.float16).cpu().numpy()
                    for row in rows:
                        state["buf"][row, layer_idx] = vec
        return _h

    handles = [layers[i].register_forward_hook(hook(i)) for i in range(n_save)]
    out_dir = Path("/vol/out") / run
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.jsonl").unlink(missing_ok=True)
    report, total_tokens, stopped_early = [], 0, False
    t_compute = time.time()
    with torch.inference_mode():
        for line in filter(None, jobs_text.splitlines()):
            if time.time() - t0 > max_seconds:
                stopped_early = True  # the GPU budget is spent; keep what is saved
                break
            job = json.loads(line)
            traj, ops = job["traj"], job["ops"]
            reads, op_reads, pos = [], [], 0
            for op in ops:
                op_reads.append({})
                if "truncate" in op:
                    pos = op["truncate"]
                    continue
                for rel, label in op.get("read", []):
                    # two reads can share a position (P2 = P4 when a turn ends at its reasoning <|eom|>)
                    op_reads[-1].setdefault(pos + rel, []).append(len(reads))
                    reads.append((pos + rel, label))
                pos += len(op["append"])
            state["buf"] = np.zeros((len(reads), n_save, text_model.config.hidden_size), dtype=np.float16)
            filled = np.zeros(len(reads), dtype=bool)
            cache, n_past, t_traj, traj_tokens = DynamicCache(), 0, time.time(), 0
            for op, wanted in zip(ops, op_reads):
                if "truncate" in op:
                    cache.crop(op["truncate"] - n_past)
                    n_past = op["truncate"]
                    continue
                ids = op["append"]
                for off in range(0, len(ids), CHUNK):
                    chunk = ids[off:off + CHUNK]
                    state["start"] = n_past
                    state["want"] = {p: r for p, r in wanted.items() if n_past <= p < n_past + len(chunk)}
                    text_model(input_ids=torch.tensor([chunk], device="cuda"), past_key_values=cache, use_cache=True)
                    for rows in state["want"].values():
                        filled[rows] = True
                    if cache.get_seq_length() != n_past + len(chunk):
                        raise RuntimeError(f"{traj}: cache length {cache.get_seq_length()} != {n_past + len(chunk)}")
                    n_past += len(chunk)
                    traj_tokens += len(chunk)
            if not filled.all():
                raise RuntimeError(f"{traj}: {int((~filled).sum())} reads never captured")
            fname = traj.replace(":", "_").replace("/", "_") + ".npy"
            np.save(out_dir / fname, state["buf"])
            rows = [{"traj": traj, "row": i, "abs_pos": p, "label": lab, "file": fname} for i, (p, lab) in enumerate(reads)]
            # written per trajectory so a later failure keeps everything already computed
            with (out_dir / "index.jsonl").open("a") as fh:
                fh.write("".join(json.dumps(r) + "\n" for r in rows))
            vol.commit()
            a = state["buf"].astype(np.float32)
            report.append({"traj": traj, "tokens": traj_tokens, "reads": len(reads),
                           "seconds": round(time.time() - t_traj, 2), "finite": bool(np.isfinite(a).all()),
                           "norm_by_layer_first_read": [round(float(x), 1) for x in np.linalg.norm(a[0], axis=1)]})
            total_tokens += traj_tokens
    for h in handles:
        h.remove()
    compute_s = time.time() - t_compute
    # plain types only: the result is unpickled locally, where torch is not installed
    result = {"run": run, "load_s": round(load_s, 1), "compute_s": round(compute_s, 1), "tokens": total_tokens,
              "tok_per_s": round(total_tokens / compute_s, 1), "wall_s": round(time.time() - t0, 1),
              "torch": str(torch.__version__), "gpu": str(torch.cuda.get_device_name()),
              "stopped_early": stopped_early, "trajectories": report}
    (out_dir / "report.json").write_text(json.dumps(result, indent=1))
    vol.commit()
    return result


@app.local_entrypoint()
def main(jobs: str, run: str, gpu_cap_usd: float = 1.0):
    from glint.meter import Caps, Meter

    meter = Meter(Caps(meta_usd=0.0, gpu_usd=gpu_cap_usd), phase=f"modal-{run}")
    meter.check_gpu()
    print("download:", download.remote())
    # remaining budget as container seconds, less one trajectory's worth of margin (~60 s)
    max_seconds = (gpu_cap_usd - meter.gpu_usd) / GPU_USD_PER_HOUR * 3600 - 60
    t0 = time.time()
    result = None
    try:
        result = capture.remote(Path(jobs).read_text(encoding="utf-8"), run, max_seconds)
    finally:
        # recorded even when capture raises or times out, so the next run's cap check sees the spend;
        # billed container time is at least the remote wall time, so take the larger clock
        elapsed = time.time() - t0
        meter.record_gpu(max(elapsed, result["wall_s"]) if result else elapsed, GPU_USD_PER_HOUR, GPU, tag=run)
    result["gpu_usd_metered"] = round(meter.gpu_usd, 4)
    out = paths.RESULTS / f"M1-modal-{run}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "trajectories"}, indent=1))
