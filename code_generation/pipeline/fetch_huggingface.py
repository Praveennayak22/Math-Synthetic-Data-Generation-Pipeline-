"""
fetch_huggingface.py — Stage 0B: HuggingFace Fetcher (Mode 2)

Loads ANY HuggingFace dataset using streaming mode (no full download).
Prints all columns + sample values, then writes raw_questions.jsonl.

Team 2 assigned datasets:
    deepmind/code_contests   → text_field = "description"
    codeparrot/apps          → text_field = "question"
These are auto-resolved without any GLM call.

For any other dataset, GLM auto-detects the correct field.

Bug fixes (Gemini review):
  - List fields are joined instead of val[0] to prevent data loss
  - ds.skip().take() used for fast batch offsets instead of slow manual loop
  - IndexError protection added to the config retry block

FIX (auth): call_glm now attaches Authorization: Bearer header using
  TENSORSTUDIO_API_KEY env var. Previously it used the OpenAI client which
  reads the key via a different path, causing 401 errors when calling
  the TensorStudio endpoint directly.

Usage:
    # Batch 1 — seeds 1 to 500:
    python fetch_huggingface.py --dataset deepmind/code_contests --limit 500 --offset 0 --out input/raw_batch1.jsonl

    # Batch 2 — seeds 501 to 1000:
    python fetch_huggingface.py --dataset deepmind/code_contests --limit 500 --offset 500 --out input/raw_batch2.jsonl

    # Specify field manually:
    python fetch_huggingface.py --dataset some/dataset --text_field problem --out input/raw_questions.jsonl
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

try:
    from datasets import load_dataset, get_dataset_config_names
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("[Error] HuggingFace datasets not installed. Run: pip install datasets")


# ── Known text fields for all Team 2 datasets ────────────────────────────────
KNOWN_TEXT_FIELDS = {
    # Original
    "deepmind/code_contests":             "description",
    "codeparrot/apps":                    "question",
    "open-r1/codeforces-cots":            "description",
    # New
    "LingoIITGN/PythonSaga":              "prompt",           # function sig + docstring
    "bennny674/leveled_code_bench":       "response_message", # JSON string, parsed in serialize_row
    "livecodebench/code_generation_lite": "question_content", # needs version_tag config
    # Mirror of livecodebench that works with modern datasets library (no loading script)
    "bzantium/livecodebench":             "question_content",
}

# ── Special loading configs ───────────────────────────────────────────────────
# livecodebench needs a version_tag config or it raises ValueError on load.
# "release_v5" is the latest version with ~880 problems.
DATASET_CONFIGS = {
    "livecodebench/code_generation_lite": {"name": "release_v5"},
    "bzantium/livecodebench":             {"name": "release_v5"},
}

# ── Datasets that require trust_remote_code=True to load ─────────────────────
# NOTE: trust_remote_code is no longer supported in recent datasets versions.
# These datasets use old-style .py loading scripts which are now blocked.
# Solution: load them directly from HuggingFace's auto-converted Parquet files
# using the "parquet" path — HF auto-converts every dataset to parquet at
# refs/convert/parquet even when the original has a loading script.
# Format: dataset_name -> (parquet_repo, split_prefix)
PARQUET_FALLBACK_DATASETS = {
    "codeparrot/apps":                    "codeparrot/apps",
    "livecodebench/code_generation_lite": "livecodebench/code_generation_lite",
}

# ── Datasets that need special text extraction in serialize_row ───────────────
# leveled_code_bench : one column, whole row is a JSON string — extract ["question"]
# PythonSaga         : prompt = function_sig + """docstring""" — extract docstring
SPECIAL_PARSE_DATASETS = {
    "bennny674/leveled_code_bench",
    "LingoIITGN/PythonSaga",
}

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_URL  = os.environ.get("LLM_URL",   "https://api.tensorstudio.ai/sglang/v1/chat/completions")
MODEL_NAME = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")


# ═══════════════════════════════════════════════════════════════════════════════
#  GLM CLIENT
#  FIX: Replaced OpenAI client with direct requests call so the
#  TENSORSTUDIO_API_KEY env var is picked up the same way as every
#  other script in the pipeline (Bearer token in Authorization header).
#  The old OpenAI client read the key via OPENAI_API_KEY, which is a
#  different env var — causing silent 401 failures on the cluster.
# ═══════════════════════════════════════════════════════════════════════════════
def get_client():
    """No-op — kept for API compatibility with other pipeline scripts."""
    return None


def call_glm(client, prompt: str) -> str:
    """
    Call TensorStudio for field detection.
    FIX: Uses requests directly with Authorization header instead of
    OpenAI client, matching the auth pattern used by all other pipeline scripts.
    """
    try:
        import requests as _requests

        # ── FIX: Read API key from env and attach as Bearer token ─────────────
        api_key = os.environ.get("TENSORSTUDIO_API_KEY", "")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        # ─────────────────────────────────────────────────────────────────────

        payload = {
            "model":       MODEL_NAME,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens":  50,
        }
        resp = _requests.post(MODEL_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data    = resp.json()
        content = (data["choices"][0]["message"].get("content") or "").strip()
        return content
    except Exception as e:
        print(f"    [API Error] {e}")
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
#  COLUMN INSPECTOR
# ═══════════════════════════════════════════════════════════════════════════════
def inspect_and_print(columns: list, first_row: dict) -> dict:
    print(f"\n{'─'*65}")
    print(f"  Dataset columns ({len(columns)} total) — sample from first row:")
    print(f"{'─'*65}")

    samples = {}
    for col in columns:
        val = first_row.get(col)
        if val is None:
            display = "(None)"
        elif isinstance(val, str):
            display = val[:120].replace("\n", " ")
        elif isinstance(val, list):
            display = f"[list len={len(val)}]  e.g.: {str(val[0])[:80]}" if val else "[empty list]"
        elif isinstance(val, dict):
            display = f"[dict keys={list(val.keys())}]"
        else:
            display = str(val)[:120]

        print(f"  {col:<35} {display}")
        samples[col] = val

    print(f"{'─'*65}\n")
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
#  FIELD DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════
def is_model_output(text: str) -> bool:
    if not isinstance(text, str):
        return False
    s = text.strip()
    return (
        s.startswith("<think>") or
        s.startswith("Okay,") or
        s.startswith("Okay let") or
        s.startswith("Let me") or
        "I need to solve" in s[:80] or
        "let's figure out" in s[:80].lower()
    )


def detect_text_field(columns: list, samples: dict, client, dataset_name: str = "") -> str:
    # 1. Known datasets — instant, no API call
    if dataset_name in KNOWN_TEXT_FIELDS:
        field = KNOWN_TEXT_FIELDS[dataset_name]
        if field in columns:
            print(f"[Field detect] Known dataset — field: '{field}'")
            return field

    # 2. GLM detection
    col_summary = ""
    for col in columns:
        val = samples.get(col, "")
        if isinstance(val, str):
            preview = val[:150].replace("\n", " ")
        elif isinstance(val, list):
            preview = f"[list] first: {str(val[0])[:100]}" if val else "[empty]"
        else:
            preview = str(val)[:100]
        col_summary += f"  '{col}': {preview}\n"

    prompt = (
        f"This is a HuggingFace competitive programming dataset.\n"
        f"Columns with sample values:\n\n{col_summary}\n"
        f"Which column contains the raw problem statement — the question text "
        f"a programmer reads to understand what to solve?\n"
        f"Do NOT pick columns with model-generated reasoning, solutions, or system prompts.\n"
        f"Reply with ONLY the exact column name, nothing else."
    )

    detected = call_glm(client, prompt).strip().strip('"').strip("'")
    if detected in columns:
        print(f"[Field detect] GLM identified field: '{detected}'")
        return detected
    if detected:
        print(f"[Field detect] GLM returned '{detected}' — not valid, falling back.")

    # 3. Fallback — longest string that is not a model output
    best, best_len = columns[0], 0
    for col in columns:
        val = samples.get(col, "")
        if isinstance(val, str) and len(val) > best_len and not is_model_output(val):
            best_len = len(val)
            best = col

    print(f"[Field detect] Fallback — using field: '{best}'")
    return best


# ═══════════════════════════════════════════════════════════════════════════════
#  ROW SERIALIZER
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_pythonsaga_text(prompt: str) -> str:
    """
    PythonSaga prompt format:
        from typing import List
        def function_name(args) -> ReturnType:
            \"\"\"Problem description here.
            Example: ...
            \"\"\"
    We extract everything inside the triple-quoted docstring as the question text.
    If no docstring found, fall back to full prompt.
    """
    match = re.search(r'"""(.*?)"""', prompt, re.DOTALL)
    if match:
        text = match.group(1).strip()
        if len(text) > 30:
            return text
    # Fallback — return full prompt
    return prompt.strip()


def _extract_leveled_code_bench_text(response_message: str) -> tuple[str, dict]:
    """
    leveled_code_bench has one column 'response_message' which is a JSON string:
        {
          "question": "...",
          "test": {"input": "...", "output": "..."}
        }
    Returns (question_text, extra_dict).
    """
    try:
        parsed = json.loads(response_message)
        question = parsed.get("question", "").strip()
        test     = parsed.get("test", {})
        extra    = {"test_input": str(test.get("input", "")),
                    "test_output": str(test.get("output", ""))}
        return question, extra
    except (json.JSONDecodeError, Exception):
        # If JSON parse fails, use raw string
        return response_message.strip(), {}


def serialize_row(row: dict, text_field: str, dataset_name: str, language: str) -> dict | None:
    """
    Serialize one dataset row into raw_questions.jsonl format.
    Handles special parsing for:
      - bennny674/leveled_code_bench  (JSON string column)
      - LingoIITGN/PythonSaga         (extract docstring from function prompt)
    """
    extra_override = {}

    # ── Special dataset parsing ───────────────────────────────────────────────
    if dataset_name in SPECIAL_PARSE_DATASETS:
        raw_val = row.get(text_field, "")
        if not isinstance(raw_val, str):
            raw_val = str(raw_val)

        if dataset_name == "bennny674/leveled_code_bench":
            text, extra_override = _extract_leveled_code_bench_text(raw_val)

        elif dataset_name == "LingoIITGN/PythonSaga":
            text = _extract_pythonsaga_text(raw_val)
            # Also preserve canonical_solution and test as extra
            extra_override = {
                "canonical_solution": row.get("canonical_solution", ""),
                "test_code":          row.get("test", ""),
                "entry_point":        row.get("entry_point", ""),
            }

    else:
        # ── Standard extraction ───────────────────────────────────────────────
        val = row.get(text_field, "")
        if isinstance(val, list):
            text = "\n".join(map(str, val)) if val else ""
        elif isinstance(val, dict):
            text = str(next(iter(val.values()), ""))
        else:
            text = str(val) if val else ""

    text = text.strip()
    if len(text) < 30:
        return None

    # ── Build extra dict from all remaining columns ───────────────────────────
    extra = {}
    for col, v in row.items():
        if col == text_field:
            continue
        extra[col] = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v

    # Merge any special-case extra fields (these take priority)
    extra.update(extra_override)

    return {
        "question_text":  text,
        "options":        [],
        "correct_answer": None,
        "answer_source":  "sandbox_verified",
        "source_name":    dataset_name.replace("/", "_").replace("-", "_"),
        "source_url":     f"https://huggingface.co/datasets/{dataset_name}",
        "language":       language,
        "scraped_at":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "extra":          extra,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ARGS
# ═══════════════════════════════════════════════════════════════════════════════
def get_args():
    parser = argparse.ArgumentParser(
        description="Fetch questions from any HuggingFace dataset into raw_questions.jsonl."
    )
    parser.add_argument("--dataset",    type=str, required=True)
    parser.add_argument("--out",        type=str, default="input/raw_questions.jsonl")
    parser.add_argument("--limit",      type=int, default=500)
    parser.add_argument("--offset",     type=int, default=0,
                        help="Number of rows to skip from the start. Default 0.")
    parser.add_argument("--split",      type=str, default="train")
    parser.add_argument("--text_field", type=str, default="")
    parser.add_argument("--language",   type=str, default="English")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    args = get_args()

    if not HF_AVAILABLE:
        print("[Error] Install HuggingFace datasets: pip install datasets")
        return

    client = get_client()

    print(f"\n[HuggingFace] Loading (streaming): {args.dataset}  split={args.split}")
    print(f"  Offset : {args.offset}  (skipping first {args.offset} rows)")
    print(f"  Limit  : {args.limit}  (fetching rows {args.offset + 1} to {args.offset + args.limit})")

    try:
        # ── Use DATASET_CONFIGS if this dataset needs special load params ─────
        load_kwargs = DATASET_CONFIGS.get(args.dataset, {})
        if load_kwargs:
            print(f"  Config  : {load_kwargs}  (required for this dataset)")

        # ── Parquet fallback for datasets with blocked loading scripts ─────────
        # codeparrot/apps and livecodebench have old .py loading scripts that
        # datasets v4.0+ refuses to run. The official fix is to pass
        # revision="refs/convert/parquet" which loads the auto-converted
        # Parquet version that HuggingFace maintains for every dataset.
        if args.dataset in PARQUET_FALLBACK_DATASETS:
            print(f"  [Info] Loading via Parquet fallback (revision=refs/convert/parquet)")
            ds = load_dataset(
                args.dataset,
                split=args.split,
                streaming=True,
                revision="refs/convert/parquet",
                **load_kwargs,
            )
        else:
            ds = load_dataset(args.dataset, split=args.split, streaming=True, **load_kwargs)
            ds = load_dataset(args.dataset, split=args.split, streaming=True, **load_kwargs)

    except Exception as e:
        print(f"[Error] {e}")
        try:
            configs = get_dataset_config_names(args.dataset)
            if not configs:
                print("[Error] No configs found or dataset is gated/private.")
                return
            print(f"  Available configs: {configs}")
            print(f"  Retrying with: {configs[0]}")
            ds = load_dataset(args.dataset, configs[0], split=args.split, streaming=True)
        except Exception as e2:
            print(f"[Error] Could not load: {e2}")
            return

    try:
        first_row = next(iter(ds))
    except StopIteration:
        print("[Error] Dataset is completely empty.")
        return

    columns = list(first_row.keys())
    samples = inspect_and_print(columns, first_row)

    if args.text_field and args.text_field in columns:
        text_field = args.text_field
        print(f"[Field] Using --text_field: '{text_field}'")
    elif args.text_field:
        print(f"[Warning] --text_field '{args.text_field}' not in columns: {columns}")
        text_field = detect_text_field(columns, samples, client, args.dataset)
    else:
        text_field = detect_text_field(columns, samples, client, args.dataset)

    print(f"\n[Field] question_text ← '{text_field}'")
    print(f"[Field] all other columns preserved in 'extra'\n")

    print(f"  Fetching rows {args.offset + 1} to {args.offset + args.limit} natively...")
    batch_ds = ds.skip(args.offset).take(args.limit)
    all_rows = list(batch_ds)

    if not all_rows:
        print("[Error] No rows collected. Check your --offset and --limit values.")
        return

    print(f"  Collected {len(all_rows)} rows.\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with open(args.out, "w", encoding="utf-8") as out_f:
        for row in all_rows:
            entry = serialize_row(row, text_field, args.dataset, args.language)
            if entry is None:
                skipped += 1
                continue

            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

            if written % 50 == 0:
                print(f"  Written: {written}", end="\r")

    print(f"\n{'='*55}")
    print(f"  HuggingFace Fetch Complete")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Offset   : {args.offset}  (rows {args.offset + 1} to {args.offset + args.limit})")
    print(f"  Field    : '{text_field}'")
    print(f"  Written  : {written}")
    print(f"  Skipped  : {skipped}  (too short / empty)")
    print(f"  Output   : {args.out}")
    print(f"{'='*55}")
    print(f"\nNext step:")
    print(f"  python pipeline/synthesize_seeds.py --input {args.out} --out input/seeds_prepared.jsonl")


if __name__ == "__main__":
    main()
    os._exit(0)