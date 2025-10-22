#!/usr/bin/env python3
"""
Optimized shard runner for Kaggle T4 x2.

This script:
- detects GPUs (torch.cuda.device_count())
- groups shards (part-xxxx) roughly equally per GPU
- launches one worker process per GPU; each worker runs run_single_dataset.py sequentially
  for the shards assigned to that GPU (so model+GPU context reused within the worker process)
- sets CUDA_VISIBLE_DEVICES per worker so processes do not fight for GPUs
- passes increased --batch_size and --num_workers to speed up GPU utilization
- sets some env vars (OMP_NUM_THREADS, MKL_NUM_THREADS) to avoid CPU oversubscribe

Adjust BATCH_SIZE and NUM_WORKERS to tune performance on your environment.
"""
import os
import sys
import time
from glob import glob
import multiprocessing
import subprocess

# ---------------- CONFIG ----------------
REPO = "/kaggle/working/clip_repo"   # change if repo path differs
SHARDS_ROOT = os.path.join(REPO, "data", "smoking_datasets_split")
#mở theo datasets mình muốn chạy
#SHARDS_ROOT = os.path.join(REPO, "data", "cars_split")
#SHARDS_ROOT = os.path.join(REPO, "data", "cottonweed_split")
OUT_BASE = os.path.join(REPO, "TTC_shard_results")
WEIGHTS_DIR = os.path.join(REPO, "AFT_model_weights")
DATASET_NAME = "smoking"

# Tuning knobs (change as needed)
BATCH_SIZE = 256         # try 64, 128 if GPU memory allows
NUM_WORKERS = 4         # DataLoader num_workers (per run_single_dataset invocation)
RUNS = 1
# ----------------------------------------

PY = sys.executable
MAIN_SCRIPT = os.path.join(REPO, "scripts", "run_single_dataset.py")

def list_shards(root):
    shards = sorted(glob(os.path.join(root, "part-*")))
    return shards

def run_shard_process(shard_path, gpu_id):
    """Run one shard by calling run_single_dataset.py as subprocess (inherits CUDA_VISIBLE_DEVICES)."""
    shard_name = os.path.basename(shard_path)
    outdir = os.path.join(OUT_BASE, DATASET_NAME, shard_name)
    os.makedirs(outdir, exist_ok=True)

    cmd = [
        PY, MAIN_SCRIPT,
        "--dataset", DATASET_NAME,
        "--root", shard_path,
        "--outdir", outdir,
        "--weights_dir", WEIGHTS_DIR,
        "--batch_size", str(BATCH_SIZE),
        "--num_workers", str(NUM_WORKERS),
        "--runs", str(RUNS),
        "--auto_parse"
        # do not add --quiet so we can see logs; add it if you want silence
    ]

    env = os.environ.copy()
    # assign only the single GPU index to this process (so nested subprocesses inherit it)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # reduce thread oversubscription in native libs (tune if you want)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    print(f"[{time.strftime('%H:%M:%S')}] Starting shard {shard_name} on GPU {gpu_id} (cmd len={len(cmd)})")
    start = time.time()
    p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    duration = time.time() - start
    print(f"[{shard_name}] exit {p.returncode} (elapsed {duration:.1f}s) -- GPU {gpu_id}")
    if p.stdout:
        print(f"--- STDOUT {shard_name} (truncated) ---\n{p.stdout[:2000]}")
    if p.stderr:
        print(f"--- STDERR {shard_name} (truncated) ---\n{p.stderr[:4000]}")
    return p.returncode

def worker_proc(shard_list, gpu_id):
    """Worker function: run shards sequentially on assigned GPU."""
    # optionally print GPU info
    try:
        import torch
        if torch.cuda.is_available():
            print(f"Worker for GPU {gpu_id}: device name: {torch.cuda.get_device_name(0)}")
            print(f"  torch.cuda.device_count() = {torch.cuda.device_count()}")
    except Exception:
        pass

    for s in shard_list:
        rc = run_shard_process(s, gpu_id)
        if rc != 0:
            print(f"Warning: shard {s} returned non-zero exit code {rc}. Continue to next shard.")

def main():
    shards = list_shards(SHARDS_ROOT)
    if not shards:
        print("No shard found in", SHARDS_ROOT)
        return

    # detect GPUs
    ngpu = 0
    try:
        import torch
        ngpu = torch.cuda.device_count()
    except Exception:
        ngpu = 0

    if ngpu <= 0:
        print("No GPUs detected. Running sequentially on CPU (may be slow).")
        for s in shards:
            run_shard_process(s, 0)
        return

    print(f"Detected {ngpu} GPUs. Total shards: {len(shards)}")
    # distribute shards to GPUs (round-robin)
    groups = [[] for _ in range(ngpu)]
    for i, s in enumerate(shards):
        groups[i % ngpu].append(s)

    procs = []
    for gpu_id, g in enumerate(groups):
        if not g:
            continue
        p = multiprocessing.Process(target=worker_proc, args=(g, gpu_id))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("All shards finished.")

if __name__ == "__main__":
    main()
