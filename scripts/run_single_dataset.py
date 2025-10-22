"""
Single dataset experiment runner - for incremental table building.
This script allows you to run experiments for one dataset at a time and incrementally
build up the results for all 6 tables.

Usage:
    python scripts/run_single_dataset.py --dataset minidata --root <data_root> --outdir <outdir> --weights_dir <weights_dir>
    
Then after each dataset, run:
    python scripts/parse_all_experiments.py --results_dir <outdir>
    python scripts/generate_all_tables.py --results_dir <outdir>

This will update the CSV files and LaTeX tables with the new dataset results.
"""
import subprocess
import argparse
from pathlib import Path
import sys
import itertools

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='single dataset to run experiments for')
parser.add_argument('--root', required=True, help='dataset root')
parser.add_argument('--outdir', required=True, help='writable outdir')
parser.add_argument('--weights_dir', required=True, help='path to AFT weights directory')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--runs', type=int, default=3, help='number of runs for test-time methods')
parser.add_argument('--quick', action='store_true', help='run only one experiment (table1) for quick testing')
parser.add_argument('--auto_parse', action='store_true', help='automatically run parser and table generators after the dataset run')
parser.add_argument('--quiet', action='store_true', help='suppress subprocess output; keep only progress prints')
args = parser.parse_args()

print(f'Running experiments for dataset: {args.dataset}')
print(f'Using weights_dir: {args.weights_dir}')

# quiet mode controls whether subprocess output is shown
quiet = args.quiet

def run_cmd(cmd, check=False):
    """Run subprocess and optionally suppress stdout/stderr when quiet."""
    if quiet:
        return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=check)
    return subprocess.run(cmd, check=check)

# Define experiment configurations for each table
if args.quick:
    # Quick mode: only run Table 1
    experiments = [
        {
            'name': 'table1_pgd_eps1',
            'attack_type': 'pgd',
            'test_eps': 1.0,
            'test_numsteps': 10,
            'test_stepsize': 1,
            'ttc_eps': 4.0,
            'ttc_numsteps': 1,
            'ttc_stepsize': 1,
            'methods': ['baseline', 'fare', 'tecoa', 'pmg_aft'],
            'runs': args.runs
        }
    ]
else:
    # Full mode: run Tables 1, 3, 4, 5 (Table 2 is intentionally omitted per user request)
    experiments = [
        {
            'name': 'table1_pgd_eps1',
            'attack_type': 'pgd',
            'test_eps': 1.0,
            'test_numsteps': 10,
            'test_stepsize': 1,
            'ttc_eps': 4.0,
            'ttc_numsteps': 1,
            'ttc_stepsize': 1,
            'methods': ['baseline', 'fare', 'tecoa', 'pmg_aft'],
            'runs': args.runs
        },
        {
            'name': 'table3_ttc_on_aft',
            'attack_type': 'pgd',
            'test_eps': 1.0,
            'test_numsteps': 10,
            'test_stepsize': 1,
            'ttc_eps': 4.0,
            'ttc_numsteps': 1,
            'ttc_stepsize': 1,
            'methods': ['fare', 'tecoa', 'pmg_aft'],
            'runs': args.runs
        },
        {
            'name': 'table4_cw_eps1',
            'attack_type': 'CW',
            'test_eps': 1.0,
            'test_numsteps': 10,
            'test_stepsize': 1,
            'ttc_eps': 4.0,
            'ttc_numsteps': 1,
            'ttc_stepsize': 1,
            'methods': ['baseline', 'fare', 'tecoa', 'pmg_aft'],
            'runs': args.runs
        },
        {
            'name': 'table5_pgd_eps4_aft',
            'attack_type': 'pgd',
            'test_eps': 4.0,
            'test_numsteps': 10,
            'test_stepsize': 1,
            'ttc_eps': 4.0,
            'ttc_numsteps': 1,
            'ttc_stepsize': 1,
            'methods': ['baseline', 'fare', 'tecoa', 'pmg_aft'],
            'runs': args.runs
        }
    ]
    
    # Add hyperparameter study for current dataset only
    experiments.append({
        'name': 'table6_hyperparam',
        'attack_type': 'pgd',
        'test_eps': 1.0,
        'test_numsteps': 10,
        'test_stepsize': 1,
        'ttc_eps': 4.0,
        'ttc_numsteps': 2,
        'ttc_stepsize': 1,
        'methods': ['baseline'],
        'runs': 1,
        'hyperparam_study': True
    })

# Weight file mapping
weight_files = {
    'baseline': None,
    'fare': str(Path(args.weights_dir) / 'FARE.pth.tar'),
    'tecoa': str(Path(args.weights_dir) / 'TeCoA.pth.tar'),
    'pmg_aft': str(Path(args.weights_dir) / 'PMG_AFT.pth.tar')
}

main_script = ROOT / 'code' / 'test_time_counterattack.py'

# Run experiments for the single dataset
for exp in experiments:
    print(f"\n{'='*60}")
    print(f"Running {exp['name']} for dataset {args.dataset}")
    print(f"{'='*60}")
    
    exp_outdir = Path(args.outdir) / exp['name']
    exp_outdir.mkdir(parents=True, exist_ok=True)
    print(f"{'='*60}")

    # (Auto-parse moved to run AFTER completing this experiment so partial results are saved.)
    # For hyperparameter study, iterate over tau_thres and beta values
    if exp.get('hyperparam_study', False):
        # Use the same ordered list of 13 (tau_thres, beta) pairs as in run_comprehensive_experiments.py
        pairs = [
            (0.2, 2.0),
            (0.05, 2.0),
            (0.1, 2.0),
            (0.15, 2.0),
            (0.25, 2.0),
            (0.3, 2.0),
            (0.35, 2.0),
            (0.4, 2.0),
            (0.2, 0.5),
            (0.2, 1.0),
            (0.2, 1.5),
            (0.2, 2.5),
            (0.2, 3.0)
        ]

        for tau_thres, beta in pairs:
            run_name = f"tau{tau_thres}_beta{beta}_{args.dataset}"
            run_outdir = exp_outdir / run_name
            run_outdir.mkdir(parents=True, exist_ok=True)

            cmd = [PY, str(main_script),
                   '--dataset', args.dataset,
                   '--root', args.root,
                   '--test_set', args.dataset,
                   '--batch_size', str(args.batch_size),
                   '--num_workers', str(args.num_workers),
                   '--test_attack_type', exp['attack_type'],
                   '--test_eps', str(exp['test_eps']),
                   '--test_numsteps', str(exp['test_numsteps']),
                   '--test_stepsize', str(exp['test_stepsize']),
                   '--ttc_eps', str(exp['ttc_eps']),
                   '--ttc_numsteps', str(exp['ttc_numsteps']),
                   '--ttc_stepsize', str(exp['ttc_stepsize']),
                   '--tau_thres', str(tau_thres),
                   '--beta', str(beta),
                   '--evaluate', 'True',
                   '--outdir', str(run_outdir)]

            print(f'\nRunning hyperparameter: tau_thres={tau_thres}, beta={beta}')
            r = run_cmd(cmd)
            if r.returncode != 0:
                print(f'Failed: tau_thres={tau_thres}, beta={beta}')
    else:
        # Standard experiments for Tables 1-5
        for method in exp['methods']:
            for run_idx in range(exp['runs']):
                run_name = f"{method}_{args.dataset}_run{run_idx}"
                run_outdir = exp_outdir / run_name
                run_outdir.mkdir(parents=True, exist_ok=True)
                
                cmd = [PY, str(main_script),
                       '--dataset', args.dataset,
                       '--root', args.root,
                       '--test_set', args.dataset,
                       '--batch_size', str(args.batch_size),
                       '--num_workers', str(args.num_workers),
                       '--test_attack_type', exp['attack_type'],
                       '--test_eps', str(exp['test_eps']),
                       '--test_numsteps', str(exp['test_numsteps']),
                       '--test_stepsize', str(exp['test_stepsize']),
                       '--ttc_eps', str(exp['ttc_eps']),
                       '--ttc_numsteps', str(exp['ttc_numsteps']),
                       '--ttc_stepsize', str(exp['ttc_stepsize']),
                       '--evaluate', 'True',
                       '--outdir', str(run_outdir)]
                
                # Add victim_resume for AFT methods
                if weight_files[method] is not None:
                    cmd += ['--victim_resume', weight_files[method]]
                
                print(f'\nRunning: {exp["name"]} - {method} - run {run_idx}')
                r = run_cmd(cmd)
                if r.returncode != 0:
                    print(f'Failed: {method} - run {run_idx}')

    # After finishing this experiment's runs, optionally parse and export CSVs
    if args.auto_parse:
        print('\nAuto-parse enabled: exporting results for completed experiment...')
        try:
            run_cmd([PY, str(ROOT / 'scripts' / 'parse_all_experiments.py'), '--results_dir', str(args.outdir)], check=True)
            run_cmd([PY, str(ROOT / 'scripts' / 'generate_all_tables.py'), '--results_dir', str(args.outdir)], check=True)
            run_cmd([PY, str(ROOT / 'scripts' / 'export_tables_csv.py'), '--results_dir', str(args.outdir)], check=True)
            print('Auto-parse and table generation completed for', exp['name'])
        except subprocess.CalledProcessError as e:
            print('Auto-parse or table generation failed for', exp['name'], e)

print(f"\n{'='*60}")
print(f"Experiments completed for dataset: {args.dataset}")
print("Next steps:")
print(f"1. python scripts/parse_all_experiments.py --results_dir {args.outdir}")
print(f"2. python scripts/generate_all_tables.py --results_dir {args.outdir}")
print("3. Check the updated CSV and LaTeX files")
print(f"{'='*60}")