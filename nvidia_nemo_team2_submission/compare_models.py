# compare_models.py
# Reads all model outputs and produces comparison report
import json
import os
import pandas as pd
from collections import Counter

MODELS = [
    "glm-5-fp8",
    "minimax-m2.5",
    "kimi-k2.5",
    "deepseek-v3.2",
    "gpt-oss",
]

TOTAL_INPUT = 20  # how many problems were in input.jsonl

results = []

for model in MODELS:
    # look for parquet file
    parquet_files = [
        f for f in os.listdir(".")
        if f.startswith(f"math_{model}") and f.endswith(".parquet")
    ]
    if not parquet_files:
        print(f"WARNING: No parquet file found for {model}, skipping")
        continue

    df = pd.read_parquet(parquet_files[0])
    total      = len(df)
    verif      = Counter(df["verification"])
    has_think  = df["response"].str.contains("<think>").sum()
    avg_len    = int(df["response"].str.len().mean())
    pass_rate  = verif["pass"] / total * 100 if total else 0
    gen_rate   = total / TOTAL_INPUT * 100

    results.append({
        "model":      model,
        "generated":  f"{total}/{TOTAL_INPUT} ({gen_rate:.0f}%)",
        "pass":       f"{verif['pass']} ({pass_rate:.1f}%)",
        "fail":       f"{verif['fail']} ({verif['fail']/total*100:.1f}%)",
        "unknown":    f"{verif['unknown']} ({verif['unknown']/total*100:.1f}%)",
        "has_think":  f"{has_think}/{total}",
        "avg_length": avg_len,
        "pass_count": verif["pass"],   # for sorting
    })

if not results:
    print("No results found. Run step2 and step3 first.")
    exit()

# sort by pass rate
results.sort(key=lambda x: x["pass_count"], reverse=True)
best_model = results[0]["model"]

# print report
print("=" * 65)
print("          MATH PIPELINE — MODEL COMPARISON REPORT")
print("=" * 65)
print(f"{'Model':<18} {'Generated':<14} {'PASS':<12} {'FAIL':<12} {'Avg Len'}")
print("-" * 65)
for r in results:
    print(f"{r['model']:<18} {r['generated']:<14} {r['pass']:<12} {r['fail']:<12} {r['avg_length']}")

print("=" * 65)
print(f"BEST MODEL : {best_model} ({results[0]['pass']} correct)")
print("=" * 65)

# also save as CSV for documentation
df_report = pd.DataFrame(results).drop(columns=["pass_count"])
df_report.to_csv("model_comparison_report.csv", index=False)
print(f"\nReport saved to: model_comparison_report.csv")

# print sample from best model
best_files = [
    f for f in os.listdir(".")
    if f.startswith(f"math_{best_model}") and f.endswith(".parquet")
]
if best_files:
    df_best = pd.read_parquet(best_files[0])
    print(f"\n--- Sample from best model ({best_model}) ---")
    row = df_best.iloc[0]
    print(f"Problem  : {row['prompt'][:100]}...")
    print(f"Answer   : {row['response'].split(chr(10))[-1][:80]}")
    print(f"Expected : {row['ground_truth']}")
    print(f"Verify   : {row['verification']}")