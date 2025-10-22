"""
Helper to run a quick TTC evaluation on the local `data/minidata` folder.
Usage (PowerShell):
    python run_minidata.py
This will invoke `code/test_time_counterattack.py` with conservative settings (small batch, few steps)
and write logs to `TTC_results`.
"""
import subprocess
from pathlib import Path
import os

ROOT = Path(__file__).parent
PY = "python"
MAIN = ROOT / "code" / "test_time_counterattack.py"

cmd = [
    PY, str(MAIN),
    "--dataset", "minidata",
    "--root", str(ROOT / "data"),
    "--test_set", "minidata",
    "--batch_size", "8",
    "--num_workers", "0",
    "--test_attack_type", "pgd",
    "--test_eps", "8",
    "--test_numsteps", "1",
    "--test_stepsize", "1",
    "--ttc_eps", "4",
    "--ttc_numsteps", "1",
    "--ttc_stepsize", "1",
    "--evaluate", "True",
]

print("Running:", " ".join(cmd))
ret = subprocess.run(cmd)
print("returncode:", ret.returncode)

# find latest log
outdir = Path("TTC_results") / f"pgd_eps_{8/255.}_numsteps_{1}"
print("Logs are written under TTC_results (see printed log file path in the script output).")


