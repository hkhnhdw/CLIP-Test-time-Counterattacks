"""Parse TTC_results logs from ALL run folders and produce a comprehensive CSV summary.
Usage:
    python scripts/parse_logs.py /path/to/TTC_results
If no argument given, defaults to 'TTC_results'.
Output: summary_parsed.csv in the specified folder, with columns for each method (CLIP, FARE, TeCoA, PMG_AFT, TTC).
"""
import re
from pathlib import Path
import sys
import csv
from collections import defaultdict

if len(sys.argv) > 1:
    results_dir = Path(sys.argv[1])
else:
    results_dir = Path('TTC_results')

if not results_dir.exists():
    print(f'Results directory {results_dir} does not exist')
    sys.exit(1)

# Find all run folders (e.g., CLIP_baseline_pgd_..., FARE.pth_pgd_..., etc.)
run_folders = [d for d in results_dir.iterdir() if d.is_dir()]
if not run_folders:
    print('No run folders under', results_dir)
    sys.exit(1)

print(f'Found {len(run_folders)} run folders')

# Pattern to extract dataset results from logs
# Allow an optional leading timestamp + ' - INFO - ' prefix (produced by logging) before the dataset name
pattern = re.compile(r"(?m)^(?:[0-9\-:\., ]+ - INFO -\s*)?(.+?):\n\t- clean acc\.\s*([0-9.+-]+|nan|NaN) \(ttc:\s*([0-9.+-]+|nan|NaN)\)\n\t- robust acc\.\s*([0-9.+-]+|nan|NaN) \(ttc:\s*([0-9.+-]+|nan|NaN)\)")

def tofloat(x):
    try:
        return float(x)
    except:
        return float('nan')

# Collect results: {dataset: {method: {clean, adv, ttc_clean, ttc_adv}}}
all_results = defaultdict(lambda: defaultdict(dict))

for folder in run_folders:
    # Infer method name from folder name
    folder_name = folder.name
    if 'CLIP_baseline' in folder_name:
        method = 'CLIP'
    elif 'FARE' in folder_name:
        method = 'FARE'
    elif 'TeCoA' in folder_name:
        method = 'TeCoA'
    elif 'PMG_AFT' in folder_name or 'PMG-AFT' in folder_name:
        method = 'PMG_AFT'
    else:
        # fallback: try to extract checkpoint name
        method = folder_name.split('_pgd_')[0] if '_pgd_' in folder_name else folder_name
    
    print(f'Processing {folder_name} as method: {method}')
    
    # Find log files recursively in this folder
    logs = list(folder.rglob('*.log'))
    if not logs:
        print(f'  No log files in {folder_name}')
        continue
    
    # Parse the first/most recent log
    logfile = sorted(logs, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    print(f'  Parsing log: {logfile.relative_to(results_dir)}')
    text = logfile.read_text(encoding='utf-8', errors='ignore')
    
    for m in pattern.findall(text):
        dataset, clean, ttc_clean, adv, ttc_adv = m
        # dataset may include a logging prefix like '2025-10-03 03:57:23,919 - INFO - minidata'
        dataset = dataset.strip()
        # Remove any remaining leading timestamp/level prefix if present
        dataset = re.sub(r'^\s*[0-9\-:\., ]+\s*-\s*INFO\s*-\s*', '', dataset)
        # Normalize to the last token (e.g. 'minidata') so different runs map to the same dataset key
        m_ds = re.search(r'([A-Za-z0-9_\-]+)$', dataset)
        if m_ds:
            dataset = m_ds.group(1)
        all_results[dataset][method] = {
            'clean': tofloat(clean),
            'adv': tofloat(adv),
            'ttc_clean': tofloat(ttc_clean),
            'ttc_adv': tofloat(ttc_adv),
        }

# Build CSV with columns: dataset, CLIP_clean, CLIP_adv, FARE_clean, FARE_adv, ..., TTC_clean, TTC_adv
datasets = sorted(all_results.keys())
methods = ['CLIP', 'FARE', 'TeCoA', 'PMG_AFT']
fieldnames = ['dataset']
for m in methods:
    fieldnames.extend([f'{m}_clean', f'{m}_adv'])
fieldnames.extend(['TTC_clean', 'TTC_adv'])

rows = []
for ds in datasets:
    row = {'dataset': ds}
    for m in methods:
        if m in all_results[ds]:
            row[f'{m}_clean'] = all_results[ds][m]['clean']
            row[f'{m}_adv'] = all_results[ds][m]['adv']
        else:
            row[f'{m}_clean'] = float('nan')
            row[f'{m}_adv'] = float('nan')
    # TTC columns: use ttc_clean/ttc_adv from CLIP baseline if available, else first method with TTC
    ttc_clean = float('nan')
    ttc_adv = float('nan')
    if 'CLIP' in all_results[ds]:
        ttc_clean = all_results[ds]['CLIP']['ttc_clean']
        ttc_adv = all_results[ds]['CLIP']['ttc_adv']
    row['TTC_clean'] = ttc_clean
    row['TTC_adv'] = ttc_adv
    rows.append(row)

out = results_dir / 'summary_parsed.csv'
with out.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f'\nWrote {out}')
print(f'Parsed {len(datasets)} datasets across {len(methods)} methods')
for row in rows:
    print(row)
