"""
Comprehensive experiment runner for all 6 tables in the TTC paper.
This script runs all necessary experiments to generate data for Tables 1-6:
- Table 1: PGD eps=1/255 (baseline + AFT + TTC)
- Table 2: PGD eps=4/255 
- Table 3: TTC on AFT models
- Table 4: CW attack eps=1/255
- Table 5: PGD eps=4/255 (with AFT superscripts)
- Table 6: Hyperparameter study (tau_thres, beta)

Usage:
    python scripts/run_comprehensive_experiments.py --root <data_root> --outdir <outdir> --weights_dir <weights_dir>
"""
import subprocess
import argparse
from pathlib import Path
import sys
import itertools

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True, help='dataset root')
parser.add_argument('--outdir', required=True, help='writable outdir')
parser.add_argument('--weights_dir', required=True, help='path to AFT weights directory')
parser.add_argument('--datasets', nargs='*', default=['minidata'], help='datasets to test')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--runs', type=int, default=3, help='number of runs for test-time methods')
parser.add_argument('--auto_parse', action='store_true', help='after each experiment, run parser and export CSVs to save partial results')
parser.add_argument('--quiet', action='store_true', help='suppress subprocess output; keep only progress prints')
args = parser.parse_args()

quiet = args.quiet

def run_cmd(cmd, check=False):
    if quiet:
        return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=check)
    return subprocess.run(cmd, check=check)

print('Using weights_dir:', args.weights_dir)

# Define experiment configurations for each table
experiments = [
    # Table 1: PGD eps=1/255, 10 steps
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
    # Table 2 removed per user request: PGD eps=4/255 experiments are not scheduled here.
    # Table 3: TTC on AFT models (eps=1/255)
    {
        'name': 'table3_ttc_on_aft',
        'attack_type': 'pgd',
        'test_eps': 1.0,
        'test_numsteps': 10,
        'test_stepsize': 1,
        'ttc_eps': 4.0,
        'ttc_numsteps': 1,
        'ttc_stepsize': 1,
        'methods': ['fare', 'tecoa', 'pmg_aft'],  # only AFT methods with TTC
        'runs': args.runs
    },
    # Table 4: CW attack eps=1/255
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
    # Table 5: PGD eps=4/255 (same as Table 2 but different analysis)
    {
        'name': 'table5_pgd_eps4_aft',
        'attack_type': 'pgd',
        'test_eps': 4.0,
        'test_numsteps': 10,
        'test_stepsize': 1,
        'ttc_eps': 4.0,
        'ttc_numsteps': 1,
        'ttc_stepsize': 1,
        'methods': ['baseline', 'fare_eps1', 'fare_eps4', 'tecoa_eps1', 'tecoa_eps4', 'pmg_aft_eps1', 'pmg_aft_eps4'],
        'runs': args.runs
    },
    # Table 6: Hyperparameter study
    {
        'name': 'table6_hyperparam',
        'attack_type': 'pgd',
        'test_eps': 1.0,
        'test_numsteps': 10,
        'test_stepsize': 1,
        'ttc_eps': 4.0,
        'ttc_numsteps': 2,  # Fixed N=2
        'ttc_stepsize': 1,
        'methods': ['baseline'],  # Only baseline for hyperparameter study
        'runs': 1,  # Single run for hyperparameter study
        'hyperparam_study': True
    }
]

# Weight file mapping
weight_files = {
    'baseline': None,
    'fare': str(Path(args.weights_dir) / 'FARE.pth.tar'),
    'tecoa': str(Path(args.weights_dir) / 'TeCoA.pth.tar'),
    'pmg_aft': str(Path(args.weights_dir) / 'PMG_AFT.pth.tar')
}


# Support explicit finetuned variant keys. If you have separate checkpoints for eps1/eps4,
# name them like Fare_eps1.pth.tar or TeCoA_eps4.pth.tar in the weights dir. Otherwise
# these keys will fall back to the base AFT checkpoints above.
for k, v in list(weight_files.items()):
    if k in ['fare', 'tecoa', 'pmg_aft']:
        weight_files[f"{k}_eps1"] = str(Path(args.weights_dir) / f"{k.upper()}_eps1.pth.tar") if (Path(args.weights_dir) / f"{k.upper()}_eps1.pth.tar").exists() else v
        weight_files[f"{k}_eps4"] = str(Path(args.weights_dir) / f"{k.upper()}_eps4.pth.tar") if (Path(args.weights_dir) / f"{k.upper()}_eps4.pth.tar").exists() else v

main_script = ROOT / 'code' / 'test_time_counterattack.py'

# Run experiments
for exp in experiments:
    print(f"\n{'='*60}")
    print(f"Running {exp['name']}")
    print(f"{'='*60}")
    
    exp_outdir = Path(args.outdir) / exp['name']
    exp_outdir.mkdir(parents=True, exist_ok=True)
    
    # For hyperparameter study, iterate over tau_thres and beta values
    if exp.get('hyperparam_study', False):
        # Use exact ordered pairs (tau_thres, beta) as provided by user
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
            for dataset in args.datasets:
                run_name = f"tau{tau_thres}_beta{beta}_{dataset}"
                run_outdir = exp_outdir / run_name
                run_outdir.mkdir(parents=True, exist_ok=True)
                
                cmd = [PY, str(main_script),
                       '--dataset', dataset,
                       '--root', args.root,
                       '--test_set', dataset,
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
                
                print(f'\nRunning hyperparameter study: tau_thres={tau_thres}, beta={beta}, dataset={dataset}')
                r = run_cmd(cmd)
                if r.returncode != 0:
                    print(f'Failed: tau_thres={tau_thres}, beta={beta}, dataset={dataset}')
    # After each experiment (table) completes, optionally parse and export CSVs so partial
    # results are saved (useful under time limits like Kaggle 30h runs).
    if args.auto_parse:
        print(f"Auto-parse enabled: exporting results after experiment {exp['name']}")
        try:
            run_cmd([PY, str(ROOT / 'scripts' / 'parse_all_experiments.py'), '--results_dir', str(args.outdir)], check=True)
            run_cmd([PY, str(ROOT / 'scripts' / 'generate_all_tables.py'), '--results_dir', str(args.outdir)], check=True)
            run_cmd([PY, str(ROOT / 'scripts' / 'export_tables_csv.py'), '--results_dir', str(args.outdir)], check=True)
            print(f"Exported partial results after {exp['name']}")
        except subprocess.CalledProcessError as e:
            print('Auto-parse or table generation failed for', exp['name'], e)
    else:
        # Standard experiments for Tables 1-5
        for dataset in args.datasets:
            for method in exp['methods']:
                for run_idx in range(exp['runs']):
                    run_name = f"{method}_{dataset}_run{run_idx}"
                    run_outdir = exp_outdir / run_name
                    run_outdir.mkdir(parents=True, exist_ok=True)
                    
                    cmd = [PY, str(main_script),
                           '--dataset', dataset,
                           '--root', args.root,
                           '--test_set', dataset,
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
                    victim_resume = None
                    if weight_files.get(method) is not None:
                        victim_resume = weight_files[method]
                        cmd += ['--victim_resume', victim_resume]

                    # Before running, write a small run_info.txt so we can trace which checkpoint was used
                    run_info_path = run_outdir / 'run_info.txt'
                    with run_info_path.open('w', encoding='utf-8') as infof:
                        infof.write(f"method={method}\n")
                        infof.write(f"victim_resume={victim_resume}\n")

                    # Warn if user requested a finetuned variant but fallback to base checkpoint occurred
                    if ('_eps' in method) and victim_resume in [None, '']:
                        print(f"Warning: method={method} requested but no per-variant checkpoint found; running without victim_resume (will fallback to base if available)")
                    
                    print(f'\nRunning: {exp["name"]} - {method} - {dataset} - run {run_idx}')
                    r = subprocess.run(cmd)
                    if r.returncode != 0:
                        print(f'Failed: {method} - {dataset} - run {run_idx}')

        # After running standard methods for this experiment, run CLIP-FT separately
        # if a CLIP-FT checkpoint is present. This keeps fine-tuned CLIP evaluations
        # out of the main method loops and makes the output clearer.



print(f"\n{'='*60}")
print("All experiments completed!")
print(f"Results saved in: {args.outdir}")
print(f"{'='*60}")