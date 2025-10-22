"""
Export CSV files for paper tables (Table 1, 3, 4, 5, 6).
Table 2 (PGD eps=4/255) has been removed per user request and is not exported.

Usage:
    python scripts/export_tables_csv.py --results_dir <results_dir>
"""

import argparse
import csv
from pathlib import Path
import math
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', required=True, help='directory containing CSV results')
args = parser.parse_args()

results_dir = Path(args.results_dir)


def fmt_csv(val):
    """Format value for CSV (simple number or empty)"""
    try:
        v = float(val)
        if math.isnan(v):
            return ''
        return f'{v:.2f}'
    except:
        return ''


def fmt_csv_with_std(mean, std):
    """Format mean ± std for CSV"""
    try:
        m = float(mean)
        s = float(std)
        if math.isnan(m) or math.isnan(s):
            return ''
        if s == 0:
            return f'{m:.2f}'
        return f'{m:.2f}±{s:.2f}'
    except:
        return ''


def read_csv_results(csv_path):
    """Read CSV results into nested dict[dataset][method] = row"""
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found")
        return {}

    results = {}
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row.get('dataset')
            if not dataset:
                continue

            # Ensure dataset key exists
            if dataset not in results:
                results[dataset] = {}

            # Hyperparameter CSVs do NOT have 'method' column
            if 'tau_thres' in row and 'beta' in row:
                tau = row.get('tau_thres', '')
                beta = row.get('beta', '')
                key = f'tau{tau}_beta{beta}'
                results[dataset][key] = row
            else:
                # Standard format should have 'method'
                method = row.get('method')
                if not method:
                    # skip malformed rows
                    continue
                results[dataset][method] = row

    return results


def finetune_label(method_key, out_format='csv'):
    """Map internal method keys to human-readable labels with epsilon when applicable.
    out_format: 'csv' -> 'TeCoA (ε=1/255)'; 'latex' -> 'TeCoA$^{1}'
    """
    if not method_key:
        return ''
    mk = method_key.lower()
    # Detect epsilon suffix
    if mk.endswith('_eps1') or mk.endswith('_eps1.0'):
        eps = '1'
    elif mk.endswith('_eps4') or mk.endswith('_eps4.0'):
        eps = '4'
    else:
        eps = None

    if 'tecoa' in mk:
        base = 'TeCoA'
    elif 'pmg' in mk:
        base = 'PMG-AFT'
    elif 'fare' in mk:
        base = 'FARE'
    else:
        # generic fallback: return the raw key
        return method_key

    if eps:
        if out_format == 'latex':
            return f"{base}$^{{{eps}}}$"
        # CSV: show full epsilon as fraction per user request
        if eps == '1':
            return f"{base} (ε=1/255)"
        if eps == '4':
            return f"{base} (ε=4/255)"
    return base


# Table 1: PGD eps=1/255
print("Exporting Table 1 CSV...")
table1_results = read_csv_results(results_dir / 'table1_pgd_eps1_results.csv')

table1_data = []
table1_headers = ['Dataset', 'Metric', 'CLIP', 'TeCoA', 'PMG-AFT', 'FARE', 'TTC', 'Δ_TTC_CLIP']

for dataset in sorted(table1_results.keys()):
    data = table1_results[dataset]

    clip = data.get('baseline', {})
    fare = data.get('fare', {})
    tecoa = data.get('tecoa', {})
    pmg = data.get('pmg_aft', {})

    # Calculate deltas
    clip_rob = fmt_csv(clip.get('adv_acc_mean', ''))
    ttc_rob = fmt_csv(clip.get('adv_ttc_acc_mean', ''))
    clip_acc = fmt_csv(clip.get('clean_acc_mean', ''))
    ttc_acc = fmt_csv(clip.get('clean_ttc_acc_mean', ''))

    delta_rob = ''
    delta_acc = ''
    try:
        if clip_rob and ttc_rob:
            delta_rob = f'{float(ttc_rob) - float(clip_rob):.2f}'
        if clip_acc and ttc_acc:
            delta_acc = f'{float(ttc_acc) - float(clip_acc):.2f}'
    except:
        pass

    # Robustness row
    table1_data.append([
        dataset, 'Robustness',
        clip_rob,
        fmt_csv_with_std(tecoa.get('adv_acc_mean', ''), tecoa.get('adv_acc_std', '')),
        fmt_csv_with_std(pmg.get('adv_acc_mean', ''), pmg.get('adv_acc_std', '')),
        fmt_csv_with_std(fare.get('adv_acc_mean', ''), fare.get('adv_acc_std', '')),
        ttc_rob,
        delta_rob
    ])

    # Accuracy row
    table1_data.append([
        dataset, 'Accuracy',
        clip_acc,
        fmt_csv_with_std(tecoa.get('clean_acc_mean', ''), tecoa.get('clean_acc_std', '')),
        fmt_csv_with_std(pmg.get('clean_acc_mean', ''), pmg.get('clean_acc_std', '')),
        fmt_csv_with_std(fare.get('clean_acc_mean', ''), fare.get('clean_acc_std', '')),
        ttc_acc,
        delta_acc
    ])

# Write Table 1 CSV
df1 = pd.DataFrame(table1_data, columns=table1_headers)
csv_path1 = results_dir / 'Table1_PGD_eps1.csv'
df1.to_csv(csv_path1, index=False)
print(f"Exported: {csv_path1}")


# Table 3: TTC on AFT models
print("Exporting Table 3 CSV...")
# parser produces 'table3_ttc_on_aft_results.csv'
table3_results = read_csv_results(results_dir / 'table3_ttc_on_aft_results.csv')

table3_data = []
table3_headers = ['Dataset', 'Metric', 'AFT_Model', 'Original', 'TTC', 'Δ_TTC_Original']

for dataset in sorted(table3_results.keys()):
    data = table3_results[dataset]

    for aft_model in ['fare', 'tecoa', 'pmg_aft']:
        if aft_model not in data:
            continue

        method_data = data[aft_model]

        orig_rob = fmt_csv(method_data.get('adv_acc_mean', ''))
        ttc_rob = fmt_csv(method_data.get('adv_ttc_acc_mean', ''))
        orig_acc = fmt_csv(method_data.get('clean_acc_mean', ''))
        ttc_acc = fmt_csv(method_data.get('clean_ttc_acc_mean', ''))

        delta_rob = ''
        delta_acc = ''
        try:
            if orig_rob and ttc_rob:
                delta_rob = f'{float(ttc_rob) - float(orig_rob):.2f}'
            if orig_acc and ttc_acc:
                delta_acc = f'{float(ttc_acc) - float(orig_acc):.2f}'
        except:
            pass

        # Robustness row
        table3_data.append([
            dataset, 'Robustness', aft_model.upper(),
            orig_rob, ttc_rob, delta_rob
        ])

        # Accuracy row
        table3_data.append([
            dataset, 'Accuracy', aft_model.upper(),
            orig_acc, ttc_acc, delta_acc
        ])

# Write Table 3 CSV
df3 = pd.DataFrame(table3_data, columns=table3_headers)
csv_path3 = results_dir / 'Table3_TTC_AFT.csv'
df3.to_csv(csv_path3, index=False)
print(f"Exported: {csv_path3}")


# Table 4: CW attack
print("Exporting Table 4 CSV...")
# parser produces 'table4_cw_eps1_results.csv'
table4_results = read_csv_results(results_dir / 'table4_cw_eps1_results.csv')

table4_data = []
table4_headers = ['Dataset', 'Metric', 'CLIP', 'TeCoA', 'FARE', 'PMG-AFT', 'TTC', 'Δ_TTC_CLIP']

for dataset in sorted(table4_results.keys()):
    data = table4_results[dataset]

    clip = data.get('baseline', {})
    fare = data.get('fare', {})
    tecoa = data.get('tecoa', {})
    pmg = data.get('pmg_aft', {})

    # Calculate deltas
    clip_rob = fmt_csv(clip.get('adv_acc_mean', ''))
    ttc_rob = fmt_csv(clip.get('adv_ttc_acc_mean', ''))
    clip_acc = fmt_csv(clip.get('clean_acc_mean', ''))
    ttc_acc = fmt_csv(clip.get('clean_ttc_acc_mean', ''))

    delta_rob = ''
    delta_acc = ''
    try:
        if clip_rob and ttc_rob:
            delta_rob = f'{float(ttc_rob) - float(clip_rob):.2f}'
        if clip_acc and ttc_acc:
            delta_acc = f'{float(ttc_acc) - float(clip_acc):.2f}'
    except:
        pass

    # Robustness row
    table4_data.append([
        dataset, 'Robustness',
        clip_rob,
        fmt_csv_with_std(tecoa.get('adv_acc_mean', ''), tecoa.get('adv_acc_std', '')),
        fmt_csv_with_std(fare.get('adv_acc_mean', ''), fare.get('adv_acc_std', '')),
        fmt_csv_with_std(pmg.get('adv_acc_mean', ''), pmg.get('adv_acc_std', '')),
        ttc_rob,
        delta_rob
    ])

    # Accuracy row
    table4_data.append([
        dataset, 'Accuracy',
        clip_acc,
        fmt_csv_with_std(tecoa.get('clean_acc_mean', ''), tecoa.get('clean_acc_std', '')),
        fmt_csv_with_std(fare.get('clean_acc_mean', ''), fare.get('clean_acc_std', '')),
        fmt_csv_with_std(pmg.get('clean_acc_mean', ''), pmg.get('clean_acc_std', '')),
        ttc_acc,
        delta_acc
    ])

# Write Table 4 CSV
df4 = pd.DataFrame(table4_data, columns=table4_headers)
csv_path4 = results_dir / 'Table4_CW_attack.csv'
df4.to_csv(csv_path4, index=False)
print(f"Exported: {csv_path4}")


# Table 5: PGD eps=4/255 with AFT superscripts
print("Exporting Table 5 CSV...")
table5_results = read_csv_results(results_dir / 'table5_pgd_eps4_aft_results.csv')

table5_data = []
# Remove per-method TTC variants for Table 5 per user request. Keep only the base methods
# and show the TTC (from CLIP) and the delta against CLIP like other tables.
table5_headers = ['Dataset', 'Metric', 'CLIP', 'TeCoA', 'FARE', 'PMG-AFT', 'TTC', 'Δ_TTC_CLIP']

for dataset in sorted(table5_results.keys()):
    data = table5_results[dataset]

    clip = data.get('baseline', {})
    fare = data.get('fare', {})
    tecoa = data.get('tecoa', {})
    pmg = data.get('pmg_aft', {})

    # Calculate deltas between CLIP and TTC (CLIP's TTC columns hold the 'TTC (ours)' values)
    clip_rob = fmt_csv(clip.get('adv_acc_mean', ''))
    ttc_rob = fmt_csv(clip.get('adv_ttc_acc_mean', ''))
    clip_acc = fmt_csv(clip.get('clean_acc_mean', ''))
    ttc_acc = fmt_csv(clip.get('clean_ttc_acc_mean', ''))

    delta_rob = ''
    delta_acc = ''
    try:
        if clip_rob and ttc_rob:
            delta_rob = f'{float(ttc_rob) - float(clip_rob):.2f}'
        if clip_acc and ttc_acc:
            delta_acc = f'{float(ttc_acc) - float(clip_acc):.2f}'
    except:
        pass

    # Robustness row: do not include per-method TTC variant columns
    table5_data.append([
        dataset, 'Robustness',
        clip_rob,
        fmt_csv_with_std(tecoa.get('adv_acc_mean', ''), tecoa.get('adv_acc_std', '')),
        fmt_csv_with_std(fare.get('adv_acc_mean', ''), fare.get('adv_acc_std', '')),
        fmt_csv_with_std(pmg.get('adv_acc_mean', ''), pmg.get('adv_acc_std', '')),
        ttc_rob,
        delta_rob
    ])

    # Accuracy row
    table5_data.append([
        dataset, 'Accuracy',
        clip_acc,
        fmt_csv_with_std(tecoa.get('clean_acc_mean', ''), tecoa.get('clean_acc_std', '')),
        fmt_csv_with_std(fare.get('clean_acc_mean', ''), fare.get('clean_acc_std', '')),
        fmt_csv_with_std(pmg.get('clean_acc_mean', ''), pmg.get('clean_acc_std', '')),
        ttc_acc,
        delta_acc
    ])

# Write Table 5 CSV
df5 = pd.DataFrame(table5_data, columns=table5_headers)
csv_path5 = results_dir / 'Table5_PGD_eps4_AFT.csv'
df5.to_csv(csv_path5, index=False)
print(f"Exported: {csv_path5}")


# Table 6: Hyperparameter study
print("Exporting Table 6 CSV...")
# parser produces 'table6_hyperparam_results.csv'
table6_results = read_csv_results(results_dir / 'table6_hyperparam_results.csv')

table6_data = []
table6_headers = ['Dataset', 'Metric', 'tau_thres', 'beta', 'Clean_Accuracy', 'Robust_Accuracy']

for dataset in sorted(table6_results.keys()):
    data = table6_results[dataset]

    for key, method_data in data.items():
        if key.startswith('tau'):
            # Extract tau and beta from key like 'tau0.1_beta1.0'
            parts = key.split('_')
            tau = parts[0].replace('tau', '')
            beta = parts[1].replace('beta', '')

            clean_acc = fmt_csv_with_std(method_data.get('clean_ttc_acc_mean', ''), method_data.get('clean_ttc_acc_std', ''))
            robust_acc = fmt_csv_with_std(method_data.get('adv_ttc_acc_mean', ''), method_data.get('adv_ttc_acc_std', ''))

            table6_data.append([
                dataset, 'Results', tau, beta, clean_acc, robust_acc
            ])

# Write Table 6 CSV
df6 = pd.DataFrame(table6_data, columns=table6_headers)
csv_path6 = results_dir / 'Table6_Hyperparams.csv'
df6.to_csv(csv_path6, index=False)
print(f"Exported: {csv_path6}")

print("\n=== CSV Export Summary ===")
print(f"Exported tables to {results_dir}:")
print(f"  - Table1_PGD_eps1.csv")
print(f"  - Table3_TTC_AFT.csv")
print(f"  - Table4_CW_attack.csv")
print(f"  - Table5_PGD_eps4_AFT.csv")
print(f"  - Table6_Hyperparams.csv")
print("\nEach CSV contains readable format with:")
print("  - Dataset names")
print("  - Robustness/Accuracy metrics")
print("  - All method results")
print("  - Delta calculations (TTC - baseline)")
print("  - Mean±Std format where applicable")