"""
convert_opencodereasoning.py — Schema Converter for nvidia/OpenCodeReasoning

Converts nvidia/OpenCodeReasoning (split_0 + split_1) into Team 2's SFT schema.
Does NOT touch the existing pipeline — outputs are written to a separate directory.

Split notes:
  - split_0 (567,850 rows): `input` field contains the full question.
  - split_1 (167,405 rows): `input` is "-"; question must be fetched from
    TACO (BAAI/TACO) or APPS (codeparrot/apps) by `dataset`, `split`, `index`.

Output schema (matches pack_dataset.py / master tracking spreadsheet):
  Instruction, Question, <Think>, Answer, Difficulty level,
  Number of tokens in Instruction, Number of tokens in Question,
  Number of tokens in Answer, Task, Domain, Language, Source,
  Made by, Reasoning/SFT, Model

Usage:
  python pipeline/convert_opencodereasoning.py --out_dir data/ocr_converted
  python pipeline/convert_opencodereasoning.py --out_dir data/ocr_converted --splits split_0
  python pipeline/convert_opencodereasoning.py --out_dir data/ocr_converted --splits split_1
"""

import argparse
import json
import re
import sys
from pathlib import Path

# ── Optional parquet support ──────────────────────────────────────────────────
try:
    import pandas as pd
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

# ── HuggingFace datasets (required) ──────────────────────────────────────────
try:
    from datasets import load_dataset
except ImportError:
    print("[ERROR] 'datasets' package not found. Install it with:")
    print("  pip install datasets")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_NAME  = "deepseek-r1"
INSTRUCTION = (
    "Solve the given coding problem. Provide a step-by-step reasoning trace "
    "inside <think> tags, followed by the complete Python solution."
)


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def approx_tokens(text: str) -> int:
    """Word-count × 1.3 approximation, same as pack_dataset.py."""
    return int(len(text.split()) * 1.3)


def extract_think(output: str) -> str:
    """
    Pull the <think>...</think> block out of R1's raw output.
    Returns the block WITH the tags (matching pack_dataset.py format).
    If no tags found, wraps the whole output in <think> tags as fallback.
    """
    match = re.search(r"(<think>.*?</think>)", output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: treat the entire output as the think trace
    stripped = output.strip()
    return f"<think>\n{stripped}\n</think>" if stripped else ""


def build_entry(question: str, think_trace: str, solution: str,
                difficulty, source_name: str) -> dict:
    """Assemble one record in Team 2 SFT schema."""
    return {
        "Instruction":                     INSTRUCTION,
        "Question":                        question,
        "<Think>":                         think_trace,
        "Answer":                          solution,
        "Difficulty level":                difficulty if difficulty is not None else -1,
        "Number of tokens in Instruction": approx_tokens(INSTRUCTION),
        "Number of tokens in Question":    approx_tokens(question),
        "Number of tokens in Answer":      approx_tokens(solution),
        "Task":                            "Code Generation",
        "Domain":                          "Code",
        "Language":                        "English",
        "Source":                          source_name,
        "Made by":                         "Team 2",
        "Reasoning/SFT":                   "Reasoning",
        "Model":                           MODEL_NAME,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SPLIT-1 QUESTION LOADER
# ═══════════════════════════════════════════════════════════════════════════════
def load_lookup_datasets() -> dict:
    """
    Load TACO and APPS into memory as fast lookup dicts.
    Structure: {dataset_name: {split_name: {index: question_str}}}
    """
    print("[Split-1] Loading BAAI/TACO lookup dataset …")
    taco_raw = load_dataset("BAAI/TACO", trust_remote_code=True)
    taco = {}
    for split_name, split_data in taco_raw.items():
        taco[split_name] = {i: row["question"] for i, row in enumerate(split_data)}

    print("[Split-1] Loading codeparrot/apps lookup dataset …")
    apps_raw = load_dataset("codeparrot/apps", trust_remote_code=True)
    apps = {}
    for split_name, split_data in apps_raw.items():
        apps[split_name] = {i: row["question"] for i, row in enumerate(split_data)}

    return {"taco": taco, "apps": apps}


def get_question_split1(row: dict, lookups: dict) -> str | None:
    """Resolve the question for a split_1 row via external lookup."""
    dataset_name = row.get("dataset", "").lower()
    split_name   = row.get("split", "")
    try:
        idx = int(row.get("index", -1))
    except (TypeError, ValueError):
        return None

    if dataset_name not in lookups:
        return None
    split_lookup = lookups[dataset_name].get(split_name)
    if split_lookup is None:
        return None
    return split_lookup.get(idx)


# ═══════════════════════════════════════════════════════════════════════════════
#  PROCESS SPLITS
# ═══════════════════════════════════════════════════════════════════════════════
def process_split_0(out_f, stats: dict) -> list:
    """Load and convert split_0. Returns list of records."""
    print("\n[Split-0] Loading nvidia/OpenCodeReasoning split_0 …")
    ds = load_dataset("nvidia/OpenCodeReasoning", "split_0")
    data = ds["split_0"]
    records = []

    for i, row in enumerate(data):
        if i % 50_000 == 0:
            print(f"  [Split-0] Processing row {i:,} / {len(data):,} …")

        question = (row.get("input") or "").strip()
        solution = (row.get("solution") or "").strip()
        output   = (row.get("output") or "").strip()

        if not question or not solution:
            stats["skipped_missing"] += 1
            continue

        think_trace = extract_think(output)
        entry = build_entry(
            question=question,
            think_trace=think_trace,
            solution=solution,
            difficulty=row.get("difficulty"),
            source_name="nvidia/OpenCodeReasoning (split_0)",
        )
        out_f.write(json.dumps(entry) + "\n")
        records.append(entry)
        stats["converted"] += 1

    print(f"  [Split-0] Done — {len(records):,} records written.")
    return records


def process_split_1(out_f, stats: dict) -> list:
    """Load and convert split_1 (requires TACO/APPS lookup). Returns list of records."""
    print("\n[Split-1] Loading nvidia/OpenCodeReasoning split_1 …")
    ds = load_dataset("nvidia/OpenCodeReasoning", "split_1")
    data = ds["split_1"]

    lookups = load_lookup_datasets()
    records = []

    for i, row in enumerate(data):
        if i % 20_000 == 0:
            print(f"  [Split-1] Processing row {i:,} / {len(data):,} …")

        solution = (row.get("solution") or "").strip()
        output   = (row.get("output") or "").strip()

        question = get_question_split1(row, lookups)
        if not question:
            stats["skipped_missing"] += 1
            continue
        question = question.strip()

        if not solution:
            stats["skipped_missing"] += 1
            continue

        think_trace = extract_think(output)
        entry = build_entry(
            question=question,
            think_trace=think_trace,
            solution=solution,
            difficulty=row.get("difficulty"),
            source_name="nvidia/OpenCodeReasoning (split_1)",
        )
        out_f.write(json.dumps(entry) + "\n")
        records.append(entry)
        stats["converted"] += 1

    print(f"  [Split-1] Done — {len(records):,} records written.")
    return records


# ═══════════════════════════════════════════════════════════════════════════════
#  PARQUET
# ═══════════════════════════════════════════════════════════════════════════════
def write_parquet(records: list, path: Path):
    if not PARQUET_AVAILABLE or not records:
        return
    try:
        df = pd.DataFrame(records)
        df.to_parquet(str(path), index=False, engine="pyarrow")
        print(f"  Parquet written → {path}")
    except Exception as e:
        print(f"  [Warning] Parquet write failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  ARGS
# ═══════════════════════════════════════════════════════════════════════════════
def get_args():
    parser = argparse.ArgumentParser(
        description="Convert nvidia/OpenCodeReasoning into Team 2 SFT schema."
    )
    parser.add_argument(
        "--out_dir", type=str, required=True,
        help="Directory where output files will be written."
    )
    parser.add_argument(
        "--splits", nargs="+", choices=["split_0", "split_1"],
        default=["split_0", "split_1"],
        help="Which splits to process (default: both)."
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to existing output file instead of overwriting (use when resuming split_1)."
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    args    = get_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl  = out_dir / "team2_code_ocr.jsonl"
    out_parquet = out_dir / "team2_code_ocr.parquet"

    stats = {"converted": 0, "skipped_missing": 0}
    all_records = []

    print(f"\n{'='*60}")
    print(f"  OpenCodeReasoning → Team 2 Schema Converter")
    print(f"  Splits : {args.splits}")
    print(f"  Output : {out_jsonl}")
    print(f"{'='*60}")

    open_mode = "a" if args.append else "w"
    if args.append and out_jsonl.exists():
        print(f"  [Append mode] Appending to existing file: {out_jsonl}")
    with open(out_jsonl, open_mode, encoding="utf-8") as out_f:
        if "split_0" in args.splits:
            records = process_split_0(out_f, stats)
            all_records.extend(records)

        if "split_1" in args.splits:
            records = process_split_1(out_f, stats)
            all_records.extend(records)

    # ── Parquet ───────────────────────────────────────────────────────────────
    if PARQUET_AVAILABLE:
        write_parquet(all_records, out_parquet)
    else:
        print("\n  [Info] Parquet skipped — install pyarrow + pandas to enable.")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = stats["converted"] + stats["skipped_missing"]
    print(f"\n{'='*60}")
    print(f"  Conversion Complete")
    print(f"{'='*60}")
    print(f"  Total rows processed : {total:,}")
    print(f"  Converted            : {stats['converted']:,}")
    print(f"  Skipped (no Q or A)  : {stats['skipped_missing']:,}")
    print(f"  JSONL output         : {out_jsonl}")
    if PARQUET_AVAILABLE:
        print(f"  Parquet output       : {out_parquet}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
