"""
step4_evaluate.py
Reads all math_*.jsonl files in --resultsdir and prints a summary table.
Usage: python step4_evaluate.py [--resultsdir results/] [--save-csv]
"""
import json
import os
import argparse
from collections import Counter, defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--resultsdir", default="results", help="Directory with math_*.jsonl files")
parser.add_argument("--save-csv",   action="store_true", help="Also save summary as CSV")
args = parser.parse_args()

RESULTSDIR = args.resultsdir
jsonl_files = sorted([
    os.path.join(RESULTSDIR, f)
    for f in os.listdir(RESULTSDIR)
    if f.startswith("math_") and f.endswith(".jsonl")
])

if not jsonl_files:
    print(f"No math_*.jsonl files found in '{RESULTSDIR}'. Run step3 first.")
    exit(1)

print(f"Found {len(jsonl_files)} result file(s) in {RESULTSDIR}\n")

diff_labels = {0: "Basic", 1: "Elementary", 2: "Intermediate", 3: "Advanced", 4: "Expert"}

# ── aggregate ────────────────────────────────────────────────────────────
summary_rows = []

for fpath in jsonl_files:
    with open(fpath, encoding="utf-8") as f:
        data = [json.loads(l) for l in f if l.strip()]
    if not data:
        continue

    total  = len(data)
    v      = Counter(r["verification"] for r in data)
    d      = Counter(r["difficulty"]   for r in data)
    model  = data[0].get("model_used", "?")
    source = data[0].get("source", "?")

    pass_pct    = v["pass"]    / total * 100
    fail_pct    = v["fail"]    / total * 100
    unknown_pct = v["unknown"] / total * 100

    print(f"{'='*55}")
    print(f"  {model}  ×  {source}")
    print(f"{'='*55}")
    print(f"  Total records : {total}")
    print(f"  PASS          : {v['pass']:>4}  ({pass_pct:.1f}%)")
    print(f"  FAIL          : {v['fail']:>4}  ({fail_pct:.1f}%)")
    print(f"  UNKNOWN       : {v['unknown']:>4}  ({unknown_pct:.1f}%)")
    print(f"  Difficulty breakdown:")
    for lvl in sorted(d):
        print(f"    Level {lvl} ({diff_labels.get(lvl,'?'):<12}): {d[lvl]}")
    print()

    summary_rows.append({
        "model":    model,
        "dataset":  source,
        "total":    total,
        "pass":     v["pass"],
        "fail":     v["fail"],
        "unknown":  v["unknown"],
        "pass_pct": round(pass_pct, 1),
    })

# ── summary table ─────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  FULL SUMMARY TABLE")
print(f"{'='*70}")
print(f"{'Model':<20} {'Dataset':<25} {'Total':>6} {'Pass':>6} {'Pass%':>7}")
print(f"{'-'*70}")
for row in summary_rows:
    print(f"{row['model']:<20} {row['dataset']:<25} {row['total']:>6} {row['pass']:>6} {row['pass_pct']:>6.1f}%")
print(f"{'='*70}")

if args.save_csv:
    import csv
    csv_path = os.path.join(RESULTSDIR, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary CSV saved → {csv_path}")