"""
step1_prepare_data.py
Loads ANY of the 6 HuggingFace math datasets and writes input.jsonl
Called by run_all.sh with:  python step1_prepare_data.py --dataset GAIR/LIMO --limit 200
"""
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, help="HuggingFace dataset path")
parser.add_argument("--limit",   type=int, default=200)
parser.add_argument("--outfile", default="input.jsonl")
args = parser.parse_args()

DATASET = args.dataset
LIMIT   = args.limit
OUTFILE = args.outfile

# ── per-dataset field mappings ─────────────────────────────────────────────
# Each entry: (question_key, answer_key, config, split)
# config=None means no config argument
DATASET_CONFIG = {
    "GAIR/LIMO": {
        "question": "question",
        "answer":   "answer",
        "config":   None,
        "split":    "train",
    },
    "simplescaling/s1K": {
        "question": "problem",
        "answer":   "solution",     # long solution; we store as-is
        "config":   None,
        "split":    "train",
    },
    "twinkle-ai/tw-math-reasoning-2k": {
        "question": "problem",
        "answer":   "solution",
        "config":   None,
        "split":    "train",
    },
    "TIGER-Lab/TheoremQA": {
        "question": "Question",
        "answer":   "Answer",
        "config":   None,
        "split":    "test",         # TheoremQA only has test split
    },
    "WNJXYK/MATH-Reasoning-Paths": {
        "question": "problem",
        "answer":   "answer",
        "config":   None,
        "split":    "train",
    },
    "netop/TeleMath": {
        "question": "question",
        "answer":   "answer",
        "config":   None,
        "split":    "train",
    },
}

# fallback keys if exact name not found
FALLBACK_QUESTION_KEYS = [
    "question", "Question", "problem", "Problem",
    "instruction", "prompt", "input", "query", "text", "content"
]
FALLBACK_ANSWER_KEYS = [
    "answer", "Answer", "solution", "Solution",
    "output", "response", "target", "label",
    "gold", "answerKey", "correct_answer"
]

CONFIGS_TO_TRY = [None, "main", "all", "default"]
SPLITS_TO_TRY  = ["train", "test", "validation"]

from datasets import load_dataset

def load_ds(name, cfg_hint, split_hint):
    """Try the hinted config/split first, then fall back to brute-force."""
    attempts = [(cfg_hint, split_hint)]
    for cfg in CONFIGS_TO_TRY:
        for spl in SPLITS_TO_TRY:
            if (cfg, spl) not in attempts:
                attempts.append((cfg, spl))
    for cfg, spl in attempts:
        try:
            ds = load_dataset(name, cfg, split=spl) if cfg else load_dataset(name, split=spl)
            print(f"  Loaded: config={cfg}, split={spl}")
            print(f"  Columns: {ds.column_names}")
            return ds
        except Exception:
            continue
    raise RuntimeError(f"Could not load {name} with any config/split")

def find_val(item, preferred_key, fallback_keys):
    if preferred_key and preferred_key in item and item[preferred_key]:
        return str(item[preferred_key])
    for k in fallback_keys:
        if k in item and item[k]:
            return str(item[k])
    return None

# ── main ──────────────────────────────────────────────────────────────────
print(f"Loading {DATASET} (limit={LIMIT})...")

info      = DATASET_CONFIG.get(DATASET, {})
q_key     = info.get("question")
a_key     = info.get("answer")
cfg_hint  = info.get("config")
spl_hint  = info.get("split", "train")

dataset = load_ds(DATASET, cfg_hint, spl_hint)

source_tag = DATASET.split("/")[-1]   # e.g. "LIMO", "s1K", "TheoremQA"

count = 0
skipped = 0
with open(OUTFILE, "w", encoding="utf-8") as f:
    for i, item in enumerate(dataset):
        if LIMIT and count >= LIMIT:
            break
        problem = find_val(item, q_key, FALLBACK_QUESTION_KEYS)
        if not problem:
            skipped += 1
            continue
        answer  = find_val(item, a_key, FALLBACK_ANSWER_KEYS)
        row = {
            "index":     count,
            "problem":   problem,
            "subdomain": source_tag,
            "dataset":   DATASET,
        }
        if answer:
            row["expected_answer"] = answer
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        count += 1

print(f"Saved {count} problems → {OUTFILE}")
if skipped:
    print(f"Skipped {skipped} rows (no question field found)")

# preview
with open(OUTFILE) as f:
    d = json.loads(f.readline())
print(f"\nPreview:")
print(f"  problem : {d['problem'][:100]}...")
print(f"  answer  : {str(d.get('expected_answer','N/A'))[:80]}")
print(f"  subdomain: {d['subdomain']}")