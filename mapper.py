#!/usr/bin/env python3
import os
import sys
import subprocess
import re
import json
import multiprocessing as mp

ROOT_PATH = "/kaggle/working/clip_repo/CLIP-TTCFixedGpu/data"
SCRIPT_PATH = "/kaggle/working/clip_repo/CLIP-TTCFixedGpu/scripts/run_single_dataset.py"
OUTDIR = "/kaggle/working/TTC_mapreduce_results"

# datasets
DATASET_NAME = "smoking"
#DATASET_NAME = "cars"
#DATASET_NAME = "cottonweed"

# Đường dẫn thư mục symlink sẽ thay đổi cho từng part
SMOKING_PATH = os.path.join(ROOT_PATH, DATASET_NAME)


def run_part(part_path: str):
    part_path = part_path.strip()
    if not part_path:
        return None

    part_name = os.path.basename(part_path)

    try:
        idx = int(re.search(r"\d+$", part_name).group())
    except:
        idx = 0
    gpu_id = idx % 2

    # Cập nhật symlink /data/smoking -> part-XXXX
    if os.path.islink(SMOKING_PATH) or os.path.exists(SMOKING_PATH):
        os.unlink(SMOKING_PATH)
    os.symlink(part_path, SMOKING_PATH)

    #chạy TTC
    cmd = [
        "python", SCRIPT_PATH,
        "--dataset", DATASET_NAME,
        "--root", ROOT_PATH,
        "--outdir", OUTDIR,
        "--weights_dir", "/kaggle/working/AFT_model_weights",
        "--batch_size", "32",
        "--runs", "1",
        "--auto_parse",
        "--quiet"
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

    if result.stderr.strip():
        print(f"[{part_name}] stderr:\n{result.stderr}", file=sys.stderr, flush=True)

    metrics = parse_metrics(result.stdout)
    if metrics:
        return f"{part_name}\t{json.dumps(metrics)}"
    else:
        return f"{part_name}\tERROR"


def parse_metrics(stdout_text: str):
    """
    Parse stdout để lấy clean acc & robust acc nếu có.
    Ví dụ:
    - clean acc.  85.05 (ttc: 81.22)
    """
    lines = stdout_text.splitlines()
    acc_line = next((l for l in lines if "clean acc." in l), None)
    if acc_line:
        acc_match = re.search(r"clean acc\.\s+([\d.]+)", acc_line)
        robust_match = re.search(r"ttc:\s*([\d.]+)", acc_line)
        metrics = {}
        if acc_match:
            metrics["acc"] = float(acc_match.group(1))
        if robust_match:
            metrics["robust_acc"] = float(robust_match.group(1))
        return metrics if metrics else None
    return None


def main():
    parts = [l.strip() for l in sys.stdin if l.strip()]
    if not parts:
        return

    # ⚡ Pool song song 2 GPU
    with mp.Pool(processes=2) as pool:
        for output in pool.imap_unordered(run_part, parts):
            if output:
                print(output)
                sys.stdout.flush()


if __name__ == "__main__":
    main()
