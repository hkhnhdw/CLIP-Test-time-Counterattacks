#!/usr/bin/env python3
import sys
import json
from collections import defaultdict
from statistics import mean

def print_table(results_summary):
    # Lấy tất cả các metric có trong mọi dataset
    all_metrics = set()
    for metrics in results_summary.values():
        all_metrics.update(metrics.keys())

    all_metrics = sorted(all_metrics)

    # Tiêu đề
    header = ["Dataset"] + [m.upper() for m in all_metrics] + ["#Parts"]
    col_widths = [max(len(h), 10) for h in header]

    # In tiêu đề
    line = " | ".join(h.ljust(w) for h, w in zip(header, col_widths))
    print(line)
    print("-" * len(line))

    # In từng dòng dataset
    for dataset, metrics in results_summary.items():
        row = [dataset.ljust(col_widths[0])]
        for i, m in enumerate(all_metrics, start=1):
            val = metrics.get(m, None)
            if val is None:
                row.append("N/A".rjust(col_widths[i]))
            else:
                row.append(f"{val:.2f}".rjust(col_widths[i]))
        # Số part
        row.append(str(metrics.get("#parts", 0)).rjust(col_widths[-1]))
        print(" | ".join(row))


def main():
    results = defaultdict(list)

    for line in sys.stdin:
        line = line.strip()
        if not line or "\t" not in line:
            continue

        dataset, metrics_str = line.split("\t", 1)
        if "ERROR" in metrics_str:
            continue

        try:
            metrics = json.loads(metrics_str)
            results[dataset].append(metrics)
        except json.JSONDecodeError:
            continue

    # Tổng hợp kết quả
    summary = {}
    for dataset, metrics_list in results.items():
        keys = set()
        for m in metrics_list:
            keys.update(m.keys())

        summary[dataset] = {}
        for k in keys:
            values = [m[k] for m in metrics_list if k in m]
            if values:
                summary[dataset][k] = mean(values)
        summary[dataset]["#parts"] = len(metrics_list)

    # In bảng
    if summary:
        print_table(summary)
    else:
        print("⚠️  No valid results found.")


if __name__ == "__main__":
    main()
