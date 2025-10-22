"""Generate a comprehensive LaTeX table from summary_parsed.csv with all methods.
This script reads the CSV (with columns for CLIP, FARE, TeCoA, PMG_AFT, TTC) and produces
 a LaTeX table with Robustness and Accuracy rows per dataset, plus an AVG row and Δ = TTC - CLIP.
Also exports a readable CSV file for easy editing.
Columns without data are filled with '-'.
"""
import csv
from pathlib import Path
import sys
import math
import pandas as pd


def find_csv(path_arg=None):
    if path_arg:
        csvf = Path(path_arg)
    else:
        csvf = Path('TTC_results')
        candidates = list(csvf.glob('**/summary_parsed.csv'))
        if not candidates:
            return None
        csvf = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return csvf


csvf = find_csv(sys.argv[1] if len(sys.argv) > 1 else None)
if csvf is None or not csvf.exists():
    print('No summary_parsed.csv found')
    sys.exit(1)

outname_tex = csvf.parent / 'table.tex'
outname_csv = csvf.parent / 'table_readable.csv'

rows = []
with csvf.open('r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)


def tofloat(v):
    try:
        x = float(v)
        if math.isnan(x):
            return None
        return x
    except:
        return None

def fmt(v):
    if v is None:
        return '-'
    return f'{v:.2f}'

def fmt_csv(v):
    """Format for CSV - return float or empty string"""
    if v is None:
        return ''
    return f'{v:.2f}'

# accumulate averages
accums = {}
count = 0
for r in rows:
    count += 1
    for col in r:
        if col == 'dataset':
            continue
        val = tofloat(r[col])
        accums.setdefault(col, 0.0)
        if val is not None:
            accums[col] += val

def avg(col):
    s = accums.get(col, 0.0)
    if count == 0:
        return None
    return s / count

# Prepare data for CSV export
csv_data = []
csv_headers = ['Dataset', 'Metric', 'CLIP', 'TeCoA', 'FARE', 'PMG_AFT', 'TTC', 'Delta_TTC_CLIP']

# For each dataset, add Rob. and Acc. rows
for r in rows:
    name = r['dataset']
    clip_acc = tofloat(r.get('CLIP_clean'))
    clip_rob = tofloat(r.get('CLIP_adv'))
    tecoa_acc = tofloat(r.get('TeCoA_clean'))
    tecoa_rob = tofloat(r.get('TeCoA_adv'))
    fare_acc = tofloat(r.get('FARE_clean'))
    fare_rob = tofloat(r.get('FARE_adv'))
    pmg_acc = tofloat(r.get('PMG_AFT_clean'))
    pmg_rob = tofloat(r.get('PMG_AFT_adv'))
    ttc_acc = tofloat(r.get('TTC_clean'))
    ttc_rob = tofloat(r.get('TTC_adv'))

    # Δ = TTC - CLIP
    delta_acc = None if (ttc_acc is None or clip_acc is None) else (ttc_acc - clip_acc)
    delta_rob = None if (ttc_rob is None or clip_rob is None) else (ttc_rob - clip_rob)

    # Robustness row
    csv_data.append([
        name, 'Robustness', 
        fmt_csv(clip_rob), fmt_csv(tecoa_rob), fmt_csv(fare_rob), fmt_csv(pmg_rob), 
        fmt_csv(ttc_rob), fmt_csv(delta_rob)
    ])
    
    # Accuracy row  
    csv_data.append([
        name, 'Accuracy',
        fmt_csv(clip_acc), fmt_csv(tecoa_acc), fmt_csv(fare_acc), fmt_csv(pmg_acc),
        fmt_csv(ttc_acc), fmt_csv(delta_acc)
    ])

# Add average row
avg_clip_acc = avg('CLIP_clean')
avg_tecoa_acc = avg('TeCoA_clean') 
avg_fare_acc = avg('FARE_clean')
avg_pmg_acc = avg('PMG_AFT_clean')
avg_ttc_acc = avg('TTC_clean')
avg_delta_acc = None if (avg_ttc_acc is None or avg_clip_acc is None) else (avg_ttc_acc - avg_clip_acc)

csv_data.append([
    'AVG', 'Accuracy',
    fmt_csv(avg_clip_acc), fmt_csv(avg_tecoa_acc), fmt_csv(avg_fare_acc), fmt_csv(avg_pmg_acc),
    fmt_csv(avg_ttc_acc), fmt_csv(avg_delta_acc)
])

# Write readable CSV
with outname_csv.open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(csv_headers)
    writer.writerows(csv_data)

print(f'Wrote readable CSV: {outname_csv}')

# Write LaTeX table
with outname_tex.open('w', encoding='utf-8') as f:
    f.write('% LaTeX table generated from ' + str(csvf) + '\n')
    f.write('\\begin{tabular}{lccccccccccr}\\n')
    f.write('\\hline\\n')
    f.write('Dataset & CLIP & TeCoA & FARE & RN & TTE & PMG-AFT & TTC (ours) & Δ \\\\ \n')
    f.write('\\hline\\n')

    # For each dataset, write Rob. then Acc.
    for r in rows:
        name = r['dataset']
        clip_acc = tofloat(r.get('CLIP_clean'))
        clip_rob = tofloat(r.get('CLIP_adv'))
        tecoa_acc = tofloat(r.get('TeCoA_clean'))
        tecoa_rob = tofloat(r.get('TeCoA_adv'))
        fare_acc = tofloat(r.get('FARE_clean'))
        fare_rob = tofloat(r.get('FARE_adv'))
        pmg_acc = tofloat(r.get('PMG_AFT_clean'))
        pmg_rob = tofloat(r.get('PMG_AFT_adv'))
        ttc_acc = tofloat(r.get('TTC_clean'))
        ttc_rob = tofloat(r.get('TTC_adv'))

        # Δ = TTC - CLIP
        delta_acc = None if (ttc_acc is None or clip_acc is None) else (ttc_acc - clip_acc)
        delta_rob = None if (ttc_rob is None or clip_rob is None) else (ttc_rob - clip_rob)

        # Robustness row
        f.write(f"{name} (Rob.) & {fmt(clip_rob)} & {fmt(tecoa_rob)} & {fmt(fare_rob)} & - & {fmt(pmg_rob)} & - & {fmt(ttc_rob)} & {fmt(delta_rob)} \\\\ \n")
        # Accuracy row
        f.write(f"{name} (Acc.) & {fmt(clip_acc)} & {fmt(tecoa_acc)} & {fmt(fare_acc)} & - & {fmt(pmg_acc)} & - & {fmt(ttc_acc)} & {fmt(delta_acc)} \\\\ \hline\n")

    # Average row across datasets
    f.write('AVG & ')
    f.write(f"{fmt(avg('CLIP_clean'))} & {fmt(avg('TeCoA_clean'))} & {fmt(avg('FARE_clean'))} & - & {fmt(avg('PMG_AFT_clean'))} & - & {fmt(avg('TTC_clean'))} & {fmt(None if (avg('TTC_clean') is None or avg('CLIP_clean') is None) else (avg('TTC_clean')-avg('CLIP_clean')))} \\\\ \n")
    f.write('\\hline\\n')

print(f'Wrote LaTeX: {outname_tex}')
print(f'Generated table with {len(rows)} datasets')
print('Note: RN and other columns are placeholders (-) - fill manually if you have those results.')

# Also create a pandas-friendly version for easier analysis
df = pd.DataFrame(csv_data, columns=csv_headers)
excel_file = csvf.parent / 'table_readable.xlsx'
df.to_excel(excel_file, index=False)
print(f'Wrote Excel file: {excel_file}')

# Print preview of the readable table
print('\nPreview of readable CSV:')
print(df.to_string(index=False))
