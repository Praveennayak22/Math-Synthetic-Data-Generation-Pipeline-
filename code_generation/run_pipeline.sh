#!/usr/bin/env bash
# =============================================================================
#  run_pipeline.sh — Team 2 Code Pipeline Orchestrator
#
#  USAGE — Generate variants from seeds (main cluster mode):
#    export BATCH_ID="batch1"
#    nohup bash run_pipeline.sh --source variants input/seeds_batch1.jsonl 3 > output/batch1_pipeline.log 2>&1 &
#
#    export BATCH_ID="batch2"
#    nohup bash run_pipeline.sh --source variants input/seeds_batch2.jsonl 3 > output/batch2_pipeline.log 2>&1 &
#
#  OTHER MODES:
#    bash run_pipeline.sh --source excel       input/Code-Math_websites.xlsx 3
#    bash run_pipeline.sh --source huggingface open-r1/codeforces-cots 3
#    bash run_pipeline.sh input/seeds_prepared.jsonl 3   (classic mode)
#
#  OUTPUT STRUCTURE (new):
#    output/{dataset_name}/{run_name}/sandbox/
#    output/{dataset_name}/{run_name}/dataset/
#
#  Example for deepmind/code_contests batch1:
#    output/deepmind_code_contests/batch1_variants_batch1/sandbox/
#    output/deepmind_code_contests/batch1_variants_batch1/dataset/
#
#  MERGING AFTER ALL BATCHES COMPLETE:
#    cat output/deepmind_code_contests/*/dataset/team2_code_sft.jsonl > output/final/team2_code_sft_final.jsonl
#    cat output/*/*/dataset/team2_code_sft.jsonl > output/final/team2_code_sft_all_datasets.jsonl
#    wc -l output/final/team2_code_sft_final.jsonl
# =============================================================================

set -e

# ── Load .env if present ──────────────────────────────────────────────────────
[ -f .env ] && source .env

# ── Batch identity — stamped on every output entry for traceability ───────────
BATCH_ID="${BATCH_ID:-batch1}"

# ── Parse arguments ───────────────────────────────────────────────────────────
SOURCE_MODE="classic"
INPUT_ARG=""
NUM_TRIALS=3
HF_LIMIT=500

if [ "$1" = "--source" ]; then
    SOURCE_MODE="$2"
    INPUT_ARG="$3"
    NUM_TRIALS="${4:-3}"
    HF_LIMIT="${5:-500}"
else
    INPUT_ARG="$1"
    NUM_TRIALS="${2:-3}"
fi

# ── Validate ──────────────────────────────────────────────────────────────────
if [ -z "$INPUT_ARG" ]; then
    echo "Usage:"
    echo "  export BATCH_ID=batch1"
    echo "  bash run_pipeline.sh --source variants input/seeds_batch1.jsonl 3"
    echo ""
    echo "  bash run_pipeline.sh --source excel       input/Code-Math_websites.xlsx 3"
    echo "  bash run_pipeline.sh --source huggingface open-r1/codeforces-cots 3"
    echo "  bash run_pipeline.sh input/seeds_prepared.jsonl 3"
    exit 1
fi

# ── Derive intermediate file paths (batch-specific to prevent race conditions) ─
RAW_FILE="input/raw_${BATCH_ID}.jsonl"
SEEDS_FILE="input/seeds_${BATCH_ID}.jsonl"
VARIANTS_FILE="input/variants_${BATCH_ID}.jsonl"

# ── Derive dataset folder name and run name ───────────────────────────────────
# FIX: Outputs are now nested under a dataset-named parent folder so that
# runs from different datasets don't all dump into the same output/ directory.
#
# Structure:
#   output/{DATASET_FOLDER}/{RUN_NAME}/sandbox/
#   output/{DATASET_FOLDER}/{RUN_NAME}/dataset/
#
# DATASET_FOLDER — derived from the input source:
#   huggingface mode : "deepmind_code_contests", "codeparrot_apps", etc.
#   excel mode       : basename of the xlsx file
#   variants/classic : basename of the seeds file (e.g. "seeds_batch1")
#
# RUN_NAME — the specific batch run identifier within that dataset folder.

if [ "$SOURCE_MODE" = "excel" ]; then
    DATASET_FOLDER=$(basename "$INPUT_ARG" .xlsx | sed 's/_websites//')
    RUN_NAME="${BATCH_ID}"

elif [ "$SOURCE_MODE" = "huggingface" ]; then
    # e.g. "deepmind/code_contests" → "deepmind_code_contests"
    DATASET_FOLDER=$(echo "$INPUT_ARG" | tr '/' '_' | tr '-' '_')
    RUN_NAME="${BATCH_ID}"

elif [ "$SOURCE_MODE" = "variants" ]; then
    SEEDS_FILE="$INPUT_ARG"
    # Use parent folder name if seeds file is inside a dataset folder,
    # otherwise fall back to the seeds filename
    SEEDS_PARENT=$(basename "$(dirname "$INPUT_ARG")")
    if [ "$SEEDS_PARENT" = "input" ] || [ "$SEEDS_PARENT" = "." ]; then
        DATASET_FOLDER=$(basename "$INPUT_ARG" .jsonl | sed 's/seeds_//' | sed 's/variants_//')
    else
        DATASET_FOLDER="$SEEDS_PARENT"
    fi
    RUN_NAME="${BATCH_ID}_$(basename "$INPUT_ARG" .jsonl)"

else
    # classic mode
    SEEDS_FILE="$INPUT_ARG"
    if [ ! -f "$SEEDS_FILE" ]; then
        echo "Error: File not found: $SEEDS_FILE"
        exit 1
    fi
    DATASET_FOLDER=$(basename "$SEEDS_FILE" _prepared.jsonl | sed 's/seeds_//')
    RUN_NAME="${BATCH_ID}"
fi

# ── Build full output paths ───────────────────────────────────────────────────
SANDBOX_DIR="output/${DATASET_FOLDER}/${RUN_NAME}/sandbox"
DATASET_DIR="output/${DATASET_FOLDER}/${RUN_NAME}/dataset"

# ── Print banner ──────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Team 2 Code Pipeline"
echo "============================================================"
echo "  Mode           : $SOURCE_MODE"
echo "  Input          : $INPUT_ARG"
echo "  Dataset folder : $DATASET_FOLDER"
echo "  Run name       : $RUN_NAME"
echo "  Batch ID       : $BATCH_ID"
echo "  Trials         : $NUM_TRIALS"
echo "  Seeds file     : $SEEDS_FILE"
echo "  Sandbox        : $SANDBOX_DIR"
echo "  Dataset        : $DATASET_DIR"
echo "  Gen model      : deepseek-v3.2  (soketlab-node049:30000)"
echo "  Brain model    : openai/gpt-oss-120b  (tensorstudio)"
echo "============================================================"
echo ""

# ── Clean previous sandbox for this run ──────────────────────────────────────
if [ -d "$SANDBOX_DIR" ]; then
    echo "[ Pre-run ] Cleaning sandbox: $SANDBOX_DIR"
    rm -rf "$SANDBOX_DIR"
fi
mkdir -p "$SANDBOX_DIR" "$DATASET_DIR" input output/final

# =============================================================================
#  STAGE 0A — SCRAPER (Excel mode only)
# =============================================================================
if [ "$SOURCE_MODE" = "excel" ]; then
    echo ""
    echo "[ Stage 0A ] Scraper — reading Excel and scraping URLs..."
    python3 pipeline/scrape_questions.py \
        --excel "$INPUT_ARG" \
        --out   "$RAW_FILE"

    if [ $? -ne 0 ] || [ ! -f "$RAW_FILE" ]; then
        echo "Error: scrape_questions.py failed. Exiting."
        exit 1
    fi
    echo "[ Stage 0A ] Done — $RAW_FILE"
fi

# =============================================================================
#  STAGE 0B — HUGGINGFACE FETCHER (HuggingFace mode only)
# =============================================================================
if [ "$SOURCE_MODE" = "huggingface" ]; then
    echo ""
    echo "[ Stage 0B ] HuggingFace Fetcher — loading $INPUT_ARG..."
    python3 pipeline/fetch_huggingface.py \
        --dataset "$INPUT_ARG" \
        --out     "$RAW_FILE"  \
        --limit   "$HF_LIMIT"

    if [ $? -ne 0 ] || [ ! -f "$RAW_FILE" ]; then
        echo "Error: fetch_huggingface.py failed. Exiting."
        exit 1
    fi
    echo "[ Stage 0B ] Done — $RAW_FILE"
fi

# =============================================================================
#  STAGE 0C — SYNTHESIZER (Excel + HuggingFace modes)
# =============================================================================
if [ "$SOURCE_MODE" = "excel" ] || [ "$SOURCE_MODE" = "huggingface" ]; then
    echo ""
    echo "[ Stage 0C ] Synthesizer — classifying domains, rating difficulty..."
    python3 pipeline/synthesize_seeds.py \
        --input "$RAW_FILE" \
        --out   "$SEEDS_FILE"

    if [ $? -ne 0 ] || [ ! -f "$SEEDS_FILE" ]; then
        echo "Error: synthesize_seeds.py failed. Exiting."
        exit 1
    fi
    echo "[ Stage 0C ] Done — $SEEDS_FILE"
fi

# =============================================================================
#  STAGE 0D — VARIANT GENERATOR
# =============================================================================
if [ "$SOURCE_MODE" = "variants" ] || [ "$SOURCE_MODE" = "huggingface" ] || [ "$SOURCE_MODE" = "excel" ]; then
    echo ""
    echo "[ Stage 0D ] Variant Generator — generating diverse questions from seeds..."
    python3 pipeline/generate_variants.py \
        --input        "$SEEDS_FILE"    \
        --out          "$VARIANTS_FILE" \
        --num_variants 10

    if [ $? -ne 0 ] || [ ! -f "$VARIANTS_FILE" ]; then
        echo "Error: generate_variants.py failed. Exiting."
        exit 1
    fi
    echo "[ Stage 0D ] Done — $VARIANTS_FILE"
    SEEDS_FILE="$VARIANTS_FILE"
fi

# =============================================================================
#  STAGE 1 — THE BRAIN
# =============================================================================
echo ""
echo "[ Step 1/4 ] Brain — calling GPT-OSS (TensorStudio)..."
python3 pipeline/completion_tensorstudio.py \
    --input_file "$SEEDS_FILE" \
    --num_trials "$NUM_TRIALS"

if [ $? -ne 0 ]; then
    echo "Error: completion_tensorstudio.py failed. Exiting."
    exit 1
fi
echo "[ Step 1/4 ] Done"

# =============================================================================
#  STAGE 2 — THE SORTER
# =============================================================================
echo ""
echo "[ Step 2/4 ] Sorter — extracting sandboxes..."
SEEDS_BASE=$(basename "$SEEDS_FILE" .jsonl)
SEEDS_DIR=$(dirname "$SEEDS_FILE")

for TRIAL in $(seq 0 $((NUM_TRIALS - 1))); do
    RESULTS_FILE="${SEEDS_DIR}/${SEEDS_BASE}_results${TRIAL}.jsonl"
    if [ ! -f "$RESULTS_FILE" ]; then
        echo "  [Warning] Results file not found: $RESULTS_FILE — skipping trial $TRIAL"
        continue
    fi
    echo "  Sorting trial $TRIAL..."
    python3 pipeline/extract_tir_sandbox.py \
        --input_file "$RESULTS_FILE" \
        --base_out   "$SANDBOX_DIR"
done
echo "[ Step 2/4 ] Done"

# =============================================================================
#  STAGE 3 — THE INSPECTOR
# =============================================================================
echo ""
echo "[ Step 3/4 ] Inspector — running pytest on all sandboxes..."
bash pipeline/execute_sandbox.sh "$SANDBOX_DIR"
echo "[ Step 3/4 ] Done"

# =============================================================================
#  STAGE 4 — THE SHIPPING BOX
# =============================================================================
echo ""
echo "[ Step 4/4 ] Shipping Box — packing dataset (batch: $BATCH_ID)..."
python3 pipeline/pack_dataset.py \
    --sandbox_dir "$SANDBOX_DIR" \
    --out_dir     "$DATASET_DIR" \
    --version     "$BATCH_ID"
echo "[ Step 4/4 ] Done"

# =============================================================================
#  CLEANUP — Remove sandbox to save disk space
# =============================================================================
echo ""
echo "[ Cleanup ] Removing sandbox to free disk space..."
rm -rf "$SANDBOX_DIR"
echo "[ Cleanup ] Done."

# =============================================================================
#  FINAL SUMMARY
# =============================================================================
TOTAL_SFT=$(wc -l < "$DATASET_DIR/team2_code_sft.jsonl" 2>/dev/null || echo "0")

echo ""
echo "============================================================"
echo "  Pipeline Complete!  [Batch: $BATCH_ID]"
echo "============================================================"
echo "  Dataset folder : output/${DATASET_FOLDER}/"
echo "  Run folder     : output/${DATASET_FOLDER}/${RUN_NAME}/"
echo "  SFT entries    : $TOTAL_SFT"
echo "  SFT     → $DATASET_DIR/team2_code_sft.jsonl"
echo "  RL      → $DATASET_DIR/team2_code_rl.jsonl"
echo "  Rej     → $DATASET_DIR/team2_code_rejected.jsonl"
echo "  Log     → $DATASET_DIR/run_log.jsonl"
echo "============================================================"
echo ""
echo "To merge all runs for this dataset:"
echo "  cat output/${DATASET_FOLDER}/*/dataset/team2_code_sft.jsonl > output/final/${DATASET_FOLDER}_sft.jsonl"
echo ""
echo "To merge ALL datasets:"
echo "  cat output/*/*/dataset/team2_code_sft.jsonl > output/final/team2_code_sft_all.jsonl"
echo "============================================================"