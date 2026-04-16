"""
run_all.py  —  Math Pipeline Orchestrator (works on Windows + Linux)
Runs 4 datasets × 5 models, resume-safe, organized output.

Usage:
  python run_all.py                        # run everything
  python run_all.py --limit 10             # quick test with 10 problems
  python run_all.py --dataset GAIR/LIMO    # single dataset only
  python run_all.py --model glm-5-fp8      # single model only
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────

DATASETS = [
    "GAIR/LIMO",
    "simplescaling/s1K",
    "twinkle-ai/tw-math-reasoning-2k",
    "TIGER-Lab/TheoremQA",
]

MODELS = [
    "glm-5-fp8",
    "minimax-m2.5",
    "kimi-k2.5",
    "deepseek-v3.2",
    "gpt-oss-120b",
]

LIMIT = 200

# Output goes here — change this if you want a different location
# On cluster with full /projects/data, use: os.path.expanduser("~/math_pipeline_results")
OUTBASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_output")

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--limit",   type=int, default=LIMIT)
parser.add_argument("--dataset", default=None, help="Run only this dataset")
parser.add_argument("--model",   default=None, help="Run only this model")
args = parser.parse_args()

if args.dataset:
    DATASETS = [args.dataset]
if args.model:
    MODELS = [args.model]
LIMIT = args.limit

# ── SETUP ─────────────────────────────────────────────────────────────────
LOGS_DIR    = os.path.join(OUTBASE, "logs")
RESULTS_DIR = os.path.join(OUTBASE, "results")
INPUTS_DIR  = os.path.join(OUTBASE, "inputs")

os.makedirs(LOGS_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(INPUTS_DIR,  exist_ok=True)

CHECKPOINT_FILE = os.path.join(LOGS_DIR, "completed_pairs.txt")

def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE) as f:
        return set(l.strip() for l in f if l.strip())

def mark_done(pair):
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(pair + "\n")

def run_cmd(cmd, log_file):
    """Run a command, tee output to log file and stdout."""
    print(f"  CMD: {' '.join(cmd)}")
    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace"
        )
        for line in proc.stdout:
            print(line, end="")
            lf.write(line)
        proc.wait()
    return proc.returncode

# ── MAIN ──────────────────────────────────────────────────────────────────
completed = load_checkpoint()
total_pairs = len(DATASETS) * len(MODELS)
done_count  = 0
fail_count  = 0

print("=" * 60)
print(f"  MATH PIPELINE  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Datasets : {len(DATASETS)}")
print(f"  Models   : {len(MODELS)}")
print(f"  Limit    : {LIMIT} problems per dataset")
print(f"  Output   : {OUTBASE}")
print("=" * 60)

for DATASET in DATASETS:
    SOURCE = DATASET.split("/")[-1]   # e.g. LIMO, s1K, TheoremQA

    print(f"\n{'━'*60}")
    print(f"  DATASET: {DATASET}")
    print(f"{'━'*60}")

    INPUT_FILE = os.path.join(INPUTS_DIR, f"input_{SOURCE}.jsonl")

    # ── STEP 1: download dataset ──────────────────────────────────────────
    if os.path.exists(INPUT_FILE):
        print(f"  [STEP1] Already downloaded → {INPUT_FILE}")
    else:
        print(f"  [STEP1] Downloading {DATASET}...")
        rc = run_cmd(
            [sys.executable, "step1_prepare_data.py",
             "--dataset", DATASET,
             "--limit",   str(LIMIT),
             "--outfile", INPUT_FILE],
            os.path.join(LOGS_DIR, f"{SOURCE}_step1.log")
        )
        if rc != 0 or not os.path.exists(INPUT_FILE):
            print(f"  ERROR: step1 failed for {DATASET}")
            fail_count += 1
            continue

    # ── STEP 2+3: per model ───────────────────────────────────────────────
    for MODEL in MODELS:
        done_count += 1
        PAIR = f"{SOURCE}__{MODEL}"

        if PAIR in completed:
            print(f"\n  [SKIP] {MODEL} × {SOURCE} — already done")
            continue

        print(f"\n  ── {MODEL}  ({done_count}/{total_pairs})  {datetime.now().strftime('%H:%M:%S')} ──")

        OUTPUT_FILE = os.path.join(OUTBASE, f"output_{MODEL}_{SOURCE}.jsonl")

        # step 2
        print(f"  [STEP2] Generating with {MODEL}...")
        rc = run_cmd(
            [sys.executable, "step2_generate.py",
             "--model",   MODEL,
             "--infile",  INPUT_FILE,
             "--outfile", OUTPUT_FILE],
            os.path.join(LOGS_DIR, f"{SOURCE}_{MODEL}_step2.log")
        )

        if not os.path.exists(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0:
            print(f"  ERROR: step2 produced no output for {MODEL} × {SOURCE}")
            fail_count += 1
            continue

        # step 3
        print(f"  [STEP3] Postprocessing...")
        rc = run_cmd(
            [sys.executable, "step3_postprocess.py",
             "--model",  MODEL,
             "--source", SOURCE,
             "--infile", OUTPUT_FILE,
             "--outdir", RESULTS_DIR],
            os.path.join(LOGS_DIR, f"{SOURCE}_{MODEL}_step3.log")
        )

        mark_done(PAIR)
        print(f"  ✓ Completed: {MODEL} × {SOURCE}")

# ── STEP 4: final evaluation ──────────────────────────────────────────────
print(f"\n{'━'*60}")
print(f"  FINAL EVALUATION")
print(f"{'━'*60}")
run_cmd(
    [sys.executable, "step4_evaluate.py",
     "--resultsdir", RESULTS_DIR,
     "--save-csv"],
    os.path.join(LOGS_DIR, "step4_evaluate.log")
)

print(f"\n{'='*60}")
print(f"  ALL DONE  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Failed pairs : {fail_count}")
print(f"  Results in   : {RESULTS_DIR}")
print(f"  Summary CSV  : {os.path.join(RESULTS_DIR, 'summary.csv')}")
print(f"  Logs in      : {LOGS_DIR}")
print(f"{'='*60}")