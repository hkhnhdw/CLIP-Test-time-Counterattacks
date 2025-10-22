"""Run evaluations for baseline CLIP and available AFT checkpoints (FARE, TeCoA, PMG_AFT).
Usage:
    python scripts/run_evals_with_weights.py --root <data_root> --outdir <outdir> [--keep-download]

Notes:
- This script will call download_weights.py to fetch weights into ./AFT_model_weights if they are missing (requires internet).
- It then runs code/test_time_counterattack.py for baseline (no victim_resume) and for each checkpoint found in ./AFT_model_weights.
- Logs are written under the specified outdir (default /kaggle/working/TTC_results).
- After runs, it calls scripts/parse_logs.py and scripts/make_latex_table.py to produce CSV and LaTeX table.

This is a convenience helper; runs may take time depending on dataset size and device.
"""
import subprocess
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

parser = argparse.ArgumentParser()
parser.add_argument('--root', default=str(ROOT / 'data'), help='dataset root')
parser.add_argument('--outdir', default='/kaggle/working/TTC_results', help='writable outdir')
parser.add_argument('--keep-download', action='store_true', help='skip download step if weights missing')
parser.add_argument('--dataset', default='minidata', help='dataset to test (default minidata)')
parser.add_argument('--datasets', nargs='*', help='optional list of datasets to run (overrides --dataset)')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--test_eps', type=float, default=8.0)
parser.add_argument('--test_numsteps', type=int, default=1)
parser.add_argument('--test_stepsize', type=int, default=1)
parser.add_argument('--ttc_eps', type=float, default=4.0)
parser.add_argument('--ttc_numsteps', type=int, default=1)
parser.add_argument('--ttc_stepsize', type=int, default=1)
parser.add_argument('--weights_dir', default=None, help='path to directory containing AFT weights (optional)')
parser.add_argument('--quiet', action='store_true', help='suppress subprocess output; keep only progress prints')
args = parser.parse_args()

quiet = args.quiet

def run_cmd(cmd, check=False):
    if quiet:
        return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=check)
    return subprocess.run(cmd, check=check)

# Resolve weights directory: prefer explicit --weights_dir, otherwise prefer cwd/AFT_model_weights then repo/AFT_model_weights
if args.weights_dir:
    weights_dir = Path(args.weights_dir)
    if not weights_dir.exists():
        print('Specified weights_dir does not exist, creating:', weights_dir)
        weights_dir.mkdir(parents=True, exist_ok=True)
else:
    weights_dir_repo = ROOT / 'AFT_model_weights'
    weights_dir_repo.mkdir(exist_ok=True)
    weights_dir_cwd = Path.cwd() / 'AFT_model_weights'
    weights_dir_cwd.mkdir(exist_ok=True)
    # prefer weights in cwd (where download_weights.py writes by default), fall back to repo weights
    weights_dir = weights_dir_cwd if any(weights_dir_cwd.iterdir()) else weights_dir_repo

print('Using weights_dir:', weights_dir)

# Step 1: download weights if not present
if not args.keep_download:
    missing = False
    expected = ['FARE.pth.tar', 'TeCoA.pth.tar', 'PMG_AFT.pth.tar']
    for fn in expected:
        if not (weights_dir / fn).exists():
            missing = True
            break
    if missing:
        print('Some AFT weights missing; running download_weights.py (requires internet)')
        r = run_cmd([PY, str(ROOT / 'download_weights.py')])
        if r.returncode != 0:
            print('download_weights.py failed; please run it manually or place weights in', weights_dir)

runs = []
# baseline
runs.append({'name': 'CLIP_baseline', 'victim_resume': None})
# checkpoints found in weights_dir
found = sorted(weights_dir.glob('*.pth.tar'))
if not found:
    print('No checkpoint files found in', weights_dir)
else:
    for p in found:
        key = p.stem
        runs.append({'name': key, 'victim_resume': str(p)})

print('Will run experiments:', [r['name'] for r in runs])

# Determine datasets to run
datasets = args.datasets if args.datasets and len(args.datasets) > 0 else [args.dataset]

# Run experiments per dataset
main_script = ROOT / 'code' / 'test_time_counterattack.py'
for ds in datasets:
    print('\n=== Running dataset:', ds, '===')
    for rinfo in runs:
        name = rinfo['name']
        victim = rinfo['victim_resume']
        # place runs under per-dataset subfolder so parser can aggregate across methods
        run_outdir = Path(args.outdir) / f"{ds}" / f"{name}_pgd_eps_{args.test_eps}_numsteps_{args.test_numsteps}"
        run_outdir.mkdir(parents=True, exist_ok=True)
        cmd = [PY, str(main_script),
               '--dataset', ds,
               '--root', args.root,
               '--test_set', ds,
               '--batch_size', str(args.batch_size),
               '--num_workers', str(args.num_workers),
               '--test_attack_type', 'pgd',
               '--test_eps', str(args.test_eps),
               '--test_numsteps', str(args.test_numsteps),
               '--test_stepsize', str(args.test_stepsize),
               '--ttc_eps', str(args.ttc_eps),
               '--ttc_numsteps', str(args.ttc_numsteps),
               '--ttc_stepsize', str(args.ttc_stepsize),
               '--evaluate', 'True',
               '--outdir', str(run_outdir)
               ]
        if victim is not None:
            cmd += ['--victim_resume', victim]
        # run and respect quiet mode
        print('\nRunning:', ' '.join(cmd))
        r = run_cmd(cmd)
        print('Return code', r.returncode)
        if r.returncode != 0:
            print('Run failed for', name, 'on dataset', ds, 'continuing to next')
    # after running all methods for this dataset, parse the dataset folder
    try:
        dataset_folder = Path(args.outdir) / ds
        parser_cmd = [PY, str(ROOT / 'scripts' / 'parse_logs.py'), str(dataset_folder)]
        print('Parsing logs for dataset', ds, parser_cmd)
        run_cmd(parser_cmd)
        summary_csv = dataset_folder / 'summary_parsed.csv'
        if summary_csv.exists():
            run_cmd([PY, str(ROOT / 'scripts' / 'make_latex_table.py'), str(summary_csv)])
        else:
            print('No summary_parsed.csv produced for dataset', ds)
    except Exception as e:
        print('Parser or LaTeX step failed for dataset', ds, ':', e)

# After runs, try to parse the newest run folder and make LaTeX
print('\nRuns finished. Attempting to parse logs and create LaTeX table.')
# run parser on outdir root
try:
    run_cmd([PY, str(ROOT / 'scripts' / 'parse_logs.py'), str(args.outdir)])
    run_cmd([PY, str(ROOT / 'scripts' / 'make_latex_table.py'), str(Path(args.outdir) / 'summary_parsed.csv')])
except Exception as e:
    print('Parser or LaTeX step failed:', e)

print('All done. Look in', args.outdir)
