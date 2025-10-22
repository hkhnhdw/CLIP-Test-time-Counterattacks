"""
Parse results from comprehensive experiments and generate CSV files for each table.
This script processes the output from run_comprehensive_experiments.py and creates
separate CSV files for Tables 1-6.

Usage:
    python scripts/parse_all_experiments.py --results_dir <results_dir>
"""
import argparse
import re
from pathlib import Path
import csv
from collections import defaultdict
import statistics
import math

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', required=True, help='directory containing experiment results')
args = parser.parse_args()

results_dir = Path(args.results_dir)

def tofloat(x):
    try:
        return float(x)
    except:
        return float('nan')

def parse_log_file(log_path):
    """Parse a single log file and extract results"""
    if not log_path.exists():
        return {}
    
    text = log_path.read_text(encoding='utf-8', errors='ignore')
    
    # Pattern to match dataset results
    pattern = re.compile(r"(?m)^(?:[0-9\-:\., ]+\s*-\s*INFO\s*-\s*)?(.+?):\n\t- clean acc\.\s*([0-9.+-]+|nan|NaN) \(ttc:\s*([0-9.+-]+|nan|NaN)\)\n\t- robust acc\.\s*([0-9.+-]+|nan|NaN) \(ttc:\s*([0-9.+-]+|nan|NaN)\)")
    
    results = {}
    for m in pattern.findall(text):
        dataset, clean, ttc_clean, adv, ttc_adv = m
        # Clean dataset name
        dataset = dataset.strip()
        dataset = re.sub(r'^\s*[0-9\-:\., ]+\s*-\s*INFO\s*-\s*', '', dataset)
        dataset_match = re.search(r'([A-Za-z0-9_\-]+)$', dataset)
        if dataset_match:
            dataset = dataset_match.group(1)
        
        results[dataset] = {
            'clean_acc': tofloat(clean),
            'clean_ttc_acc': tofloat(ttc_clean),
            'adv_acc': tofloat(adv),
            'adv_ttc_acc': tofloat(ttc_adv)
        }
    
    return results

def collect_experiment_results(exp_dir, exp_type):
    """Collect results from all runs of an experiment"""
    results = defaultdict(lambda: defaultdict(list))
    
    if not exp_dir.exists():
        print(f"Warning: {exp_dir} does not exist")
        return results
    
    # Find all log files recursively
    log_files = list(exp_dir.rglob('*.log'))
    
    for log_file in log_files:
        # Parse the directory structure to determine method, dataset, run
        rel_path = log_file.relative_to(exp_dir)
        path_parts = str(rel_path).split('/')
        
        # Extract method, dataset, run from path
        if exp_type == 'hyperparam':
            # For hyperparameter study: tau{val}_beta{val}_{dataset}
            run_name = path_parts[0]  # e.g., "tau0.2_beta2.0_minidata"
            log_results = parse_log_file(log_file)
            if not log_results:
                continue

            # Try to detect dataset by matching parsed log keys (handles underscores)
            base = run_name
            dataset_found = None
            for candidate in log_results.keys():
                if base == candidate or base.endswith('_' + candidate):
                    dataset_found = candidate
                    break

            if dataset_found:
                # method part is the prefix before _{dataset}
                prefix = base[:- (len(dataset_found) + 1)] if base != dataset_found else ''
                # prefix should be like 'tau0.2_beta2.0'
                parts = prefix.split('_') if prefix else []
                if len(parts) >= 2:
                    tau_part = parts[0]
                    beta_part = parts[1]
                    tau_val = tau_part.replace('tau', '')
                    beta_val = beta_part.replace('beta', '')
                    method_key = f"tau{tau_val}_beta{beta_val}"
                    results[dataset_found][method_key].append(log_results[dataset_found])
            else:
                # fallback to original parsing if no match
                parts = run_name.split('_')
                if len(parts) >= 3:
                    tau_part = parts[0]
                    beta_part = parts[1]
                    dataset = '_'.join(parts[2:])
                    tau_val = tau_part.replace('tau', '')
                    beta_val = beta_part.replace('beta', '')
                    method_key = f"tau{tau_val}_beta{beta_val}"
                    if dataset in log_results:
                        results[dataset][method_key].append(log_results[dataset])
        else:
            # For regular experiments: {method}_{dataset}_run{idx}
            run_name = path_parts[0]  # e.g., "baseline_minidata_run0"
            log_results = parse_log_file(log_file)
            if not log_results:
                continue

            # Remove trailing _runN if present and detect dataset by matching parsed log keys
            base = re.sub(r'_run\d+$', '', run_name)
            dataset_found = None
            for candidate in log_results.keys():
                if base == candidate or base.endswith('_' + candidate):
                    dataset_found = candidate
                    break

            if dataset_found:
                # method is the prefix before _{dataset}
                if base == dataset_found:
                    # attempt to infer method from run_name by removing the suffix _{dataset}
                    if ('_' + dataset_found) in run_name:
                        method = run_name[:-(len(dataset_found) + 1)]
                    else:
                        method = run_name
                else:
                    method = base[:-(len(dataset_found) + 1)]

                # normalize method string; ensure it's not empty
                method = method.strip('_')
                if not method:
                    # fallback: try removing trailing _runN from run_name
                    method = re.sub(r'_run\d+$', '', run_name)

                results[dataset_found][method].append(log_results[dataset_found])
            else:
                # fallback to original parsing
                parts = run_name.split('_')
                if len(parts) >= 3:
                    method = parts[0]
                    dataset = '_'.join(parts[1:-1])
                    if dataset in log_results:
                        results[dataset][method].append(log_results[dataset])
    
    return results

def compute_statistics(values):
    """Compute mean and std from list of values"""
    if not values:
        return {'mean': float('nan'), 'std': float('nan')}
    
    valid_values = [v for v in values if not math.isnan(v)]
    if not valid_values:
        return {'mean': float('nan'), 'std': float('nan')}
    
    mean_val = statistics.mean(valid_values)
    std_val = statistics.stdev(valid_values) if len(valid_values) > 1 else 0.0
    
    return {'mean': mean_val, 'std': std_val}

def load_existing_csv(csv_path):
    """Load existing CSV results if they exist"""
    existing = defaultdict(dict)
    if not csv_path.exists():
        return existing
    
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row['dataset']
            if 'method' in row:
                method = row['method']
                existing[dataset][method] = row
            else:
                # Hyperparameter format
                tau = row['tau_thres']
                beta = row['beta']
                key = f'tau{tau}_beta{beta}'
                existing[dataset][key] = row
    
    return existing

def merge_results(existing, new_results):
    """Merge new results with existing CSV data"""
    merged = defaultdict(dict)
    
    # Copy existing results
    for dataset in existing:
        for key in existing[dataset]:
            merged[dataset][key] = existing[dataset][key]
    
    # Add/update with new results
    for dataset in new_results:
        for key in new_results[dataset]:
            merged[dataset][key] = new_results[dataset][key]
    
    return merged

def aggregate_results(results):
    """Aggregate multiple runs into mean/std statistics"""
    aggregated = defaultdict(dict)
    
    for dataset in results:
        for method in results[dataset]:
            runs = results[dataset][method]
            
            # Extract metrics from all runs
            clean_accs = [run['clean_acc'] for run in runs if not math.isnan(run['clean_acc'])]
            clean_ttc_accs = [run['clean_ttc_acc'] for run in runs if not math.isnan(run['clean_ttc_acc'])]
            adv_accs = [run['adv_acc'] for run in runs if not math.isnan(run['adv_acc'])]
            adv_ttc_accs = [run['adv_ttc_acc'] for run in runs if not math.isnan(run['adv_ttc_acc'])]
            
            aggregated[dataset][method] = {
                'clean_acc': compute_statistics(clean_accs),
                'clean_ttc_acc': compute_statistics(clean_ttc_accs),
                'adv_acc': compute_statistics(adv_accs),
                'adv_ttc_acc': compute_statistics(adv_ttc_accs)
            }
    
    return aggregated


def find_used_checkpoint_for_method(exp_dir, dataset, method):
    """Inspect run directories under exp_dir to find run_info.txt and determine used checkpoint for a method."""
    used = []
    # runs are named like {method}_{dataset}_run{idx}
    pattern = f"{method}_{dataset}_run"
    for d in exp_dir.iterdir():
        if not d.is_dir():
            continue
        # look for run dirs inside exp_dir that start with method_dataset_run
        for run_dir in d.iterdir():
            name = run_dir.name
            if name.startswith(pattern):
                info_path = run_dir / 'run_info.txt'
                if info_path.exists():
                    try:
                        txt = info_path.read_text(encoding='utf-8')
                        for line in txt.splitlines():
                            if line.startswith('victim_resume='):
                                val = line.split('=', 1)[1].strip()
                                used.append(val)
                    except Exception:
                        continue
    if not used:
        return ''
    # return the most common value
    from collections import Counter
    c = Counter(used)
    return c.most_common(1)[0][0]

# Process each experiment
experiments = [
    ('table1_pgd_eps1', 'standard'),
    ('table3_ttc_on_aft', 'standard'),
    ('table4_cw_eps1', 'standard'),
    ('table5_pgd_eps4_aft', 'standard'),
    ('table6_hyperparam', 'hyperparam')
]

for exp_name, exp_type in experiments:
    print(f"Processing {exp_name}...")
    
    exp_dir = results_dir / exp_name
    results = collect_experiment_results(exp_dir, exp_type)
    
    if not results:
        print(f"No results found for {exp_name}")
        continue

    # If processing Table 5, remove any per-method TTC variants that include '+' (e.g. 'TeCoA+TTC')
    # This ensures these per-method TTC variants are not considered, logged, or exported for Table 5.
    if exp_name == 'table5_pgd_eps4_aft':
        for ds in list(results.keys()):
            for m in list(results[ds].keys()):
                if '+' in m:
                    del results[ds][m]
    
    # Aggregate results
    aggregated = aggregate_results(results)
    
    # Write CSV with incremental update support
    csv_path = results_dir / f"{exp_name}_results.csv"
    
    # Load existing results and merge
    existing = load_existing_csv(csv_path)
    
    # Convert aggregated to same format as existing for merging
    aggregated_formatted = defaultdict(dict)
    for dataset in aggregated:
        for method in aggregated[dataset]:
            stats = aggregated[dataset][method]
            aggregated_formatted[dataset][method] = {
                'clean_acc_mean': stats['clean_acc']['mean'],
                'clean_acc_std': stats['clean_acc']['std'],
                'adv_acc_mean': stats['adv_acc']['mean'],
                'adv_acc_std': stats['adv_acc']['std'],
                'clean_ttc_acc_mean': stats['clean_ttc_acc']['mean'],
                'clean_ttc_acc_std': stats['clean_ttc_acc']['std'],
                'adv_ttc_acc_mean': stats['adv_ttc_acc']['mean'],
                'adv_ttc_acc_std': stats['adv_ttc_acc']['std']
            }
    
    # Merge with existing
    merged = merge_results(existing, aggregated_formatted)
    
    if exp_type == 'hyperparam':
        # Special format for hyperparameter study
        with csv_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['dataset', 'tau_thres', 'beta', 'clean_acc_mean', 'clean_acc_std', 'adv_acc_mean', 'adv_acc_std', 'clean_ttc_acc_mean', 'clean_ttc_acc_std', 'adv_ttc_acc_mean', 'adv_ttc_acc_std'])
            
            for dataset in sorted(merged.keys()):
                for method_key in sorted(merged[dataset].keys()):
                    # Parse tau and beta from method_key
                    tau_val = method_key.split('_')[0].replace('tau', '')
                    beta_val = method_key.split('_')[1].replace('beta', '')
                    
                    data = merged[dataset][method_key]
                    writer.writerow([
                        dataset, tau_val, beta_val,
                        data.get('clean_acc_mean', ''), data.get('clean_acc_std', ''),
                        data.get('adv_acc_mean', ''), data.get('adv_acc_std', ''),
                        data.get('clean_ttc_acc_mean', ''), data.get('clean_ttc_acc_std', ''),
                        data.get('adv_ttc_acc_mean', ''), data.get('adv_ttc_acc_std', '')
                    ])
    else:
        # Standard format for Tables 1-5
        with csv_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['dataset', 'method', 'used_checkpoint', 'clean_acc_mean', 'clean_acc_std', 'adv_acc_mean', 'adv_acc_std', 'clean_ttc_acc_mean', 'clean_ttc_acc_std', 'adv_ttc_acc_mean', 'adv_ttc_acc_std'])

            for dataset in sorted(merged.keys()):
                # For Table 5 we explicitly remove any per-method TTC variants from the output
                methods = sorted(merged[dataset].keys())
                if exp_name == 'table5_pgd_eps4_aft':
                    methods = [m for m in methods if '+' not in m]

                for method in methods:
                    data = merged[dataset][method]
                    # attempt to find used checkpoint for this method/dataset
                    used_ckpt = find_used_checkpoint_for_method(exp_dir, dataset, method)
                    writer.writerow([
                        dataset, method, used_ckpt,
                        data.get('clean_acc_mean', ''), data.get('clean_acc_std', ''),
                        data.get('adv_acc_mean', ''), data.get('adv_acc_std', ''),
                        data.get('clean_ttc_acc_mean', ''), data.get('clean_ttc_acc_std', ''),
                        data.get('adv_ttc_acc_mean', ''), data.get('adv_ttc_acc_std', '')
                    ])
    
    print(f"Wrote {csv_path}")

print("All results parsed and saved!")