"""
Inspect experiment result directories and log files to report which methods and datasets are present.
Run this on the machine where your results live (e.g., Kaggle):
python scripts/debug_inspect_results.py --results_dir /kaggle/working/TTC_incremental_results

It will print per-experiment which top-level run folders exist and parse one log file per run to show the dataset keys found inside logs.
"""
import argparse
from pathlib import Path
import re

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', required=True)
args = parser.parse_args()

results_dir = Path(args.results_dir)
experiments = [d for d in results_dir.iterdir() if d.is_dir()]

for exp in sorted(experiments):
    print(f"Experiment: {exp.name}")
    runs = [p for p in exp.iterdir() if p.is_dir()]
    if not runs:
        print("  (no top-level runs)")
        continue
    for r in sorted(runs):
        print(f"  Run folder: {r.name}")
        # list nested log files (first found)
        logs = list(r.rglob('*.log'))
        print(f"    log files found: {len(logs)}")
        if logs:
            lf = logs[0]
            print(f"    Example log: {lf.relative_to(exp)}")
            # parse dataset keys from the log header lines
            try:
                text = lf.read_text(encoding='utf-8', errors='ignore')
                # find lines like 'minidata:' at start of a block
                keys = set(re.findall(r"^([A-Za-z0-9_\-]+):\n\t- clean acc\\.", text, flags=re.M))
                if keys:
                    print(f"    Parsed dataset keys in log: {sorted(keys)}")
                else:
                    # fallback: look for 'Loading <dataset> from' lines
                    match = re.search(r"Loading\s+([A-Za-z0-9_\-/]+)\s+from", text)
                    if match:
                        print(f"    Loading hint: {match.group(1)}")
            except Exception as e:
                print(f"    Failed to read/parse log: {e}")
    print()

print('Done.')
