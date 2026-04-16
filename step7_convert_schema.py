"""
step7_convert_schema.py
Converts HuggingFace datasets to the standard math pipeline schema.
NO generation needed — just reads and reformats.

Prof's instruction:
  - AM-DeepSeek-R1-Distilled-1.4M  → just convert to schema
  - SynthLabsAI/Big-Math-RL-Verified → just convert to schema
  - math-ai/AutoMathText             → just convert to schema
  - zwhe99/DeepMath-103K             → just convert, pick best of 3 solutions

Usage:
  python step7_convert_schema.py --dataset a-m-team/AM-DeepSeek-R1-Distilled-1.4M --limit 2000
  python step7_convert_schema.py --dataset SynthLabsAI/Big-Math-RL-Verified --limit 2000
  python step7_convert_schema.py --dataset math-ai/AutoMathText --limit 2000
  python step7_convert_schema.py --dataset zwhe99/DeepMath-103K --limit 2000
"""

import json
import argparse
import os
import re
import pandas as pd
from datetime import datetime

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, help="HuggingFace dataset path")
parser.add_argument("--limit",   type=int, default=2000)
parser.add_argument("--outdir",  default="/projects/data/datasets/code_post_training_data/math_pipeline_team2/NeMo-Math-Pipeline/data/converted")
parser.add_argument("--config",  default=None)
parser.add_argument("--split",   default=None)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

DATASET_NAME = args.dataset
SOURCE_TAG   = DATASET_NAME.split("/")[-1]

# ── FIXED SCHEMA CONSTANTS ────────────────────────────────────────────────
SYSTEM_PROMPT = "You are an AI assistant that helps people find information. Use clear, accurate language and provide concise, correct answers."
INSTRUCTION   = "Solve the following math question. Show your reasoning in <think>...</think> and then provide the final answer clearly."

# ── AM-DeepSeek raw JSONL path (bypasses broken HF schema) ───────────────
AM_DEEPSEEK_RAW = (
    "/home/iitgn_pt_data/.cache/huggingface/hub/"
    "datasets--a-m-team--AM-DeepSeek-R1-Distilled-1.4M/snapshots/"
    "53531c06634904118a2dcd83961918c4d69d1cdf/am_0.9M_sample_1k.jsonl"
)

# ── DATASET CONFIGS ───────────────────────────────────────────────────────
DATASET_CONFIG = {
    "a-m-team/AM-DeepSeek-R1-Distilled-1.4M": {
        "question_key":   "problem",
        "answer_key":     "answer",
        "thinking_key":   "r1_solution",
        "domain_key":     None,
        "config":         "am_0.9M_sample_1k",
        "split":          "train",
        "model":          "DeepSeek-R1",
        "reasoning_type": "reasoning",
    },
    "SynthLabsAI/Big-Math-RL-Verified": {
        "question_key":   "problem",
        "answer_key":     "answer",
        "thinking_key":   None,
        "domain_key":     "source",
        "config":         None,
        "split":          "train",
        "model":          "Llama-3.1-8B",
        "reasoning_type": "reasoning",
    },
    "math-ai/AutoMathText": {
        "question_key":   "text",
        "answer_key":     None,
        "thinking_key":   None,
        "domain_key":     None,
        "config":         None,
        "split":          "web-0.50-to-1.00",
        "model":          "Qwen-72B",
        "reasoning_type": "SFT",
        "min_score":      0.5,
    },
    "zwhe99/DeepMath-103K": {
        "question_key":   "problem",
        "answer_key":     "answer",
        "thinking_key":   None,
        "domain_key":     "source",
        "config":         None,
        "split":          "train",
        "model":          "kimi-k2.5",
        "reasoning_type": "reasoning",
    },
}

FALLBACK_QUESTION_KEYS = ["problem", "question", "Question", "text", "input", "prompt"]
FALLBACK_ANSWER_KEYS   = ["answer", "Answer", "solution", "output", "target", "label"]

# ── HELPERS ───────────────────────────────────────────────────────────────
def map_difficulty(text):
    if not text: return "intermediate"
    words = len(text.split())
    if words < 30:    return "basic"
    elif words < 60:  return "elementary"
    elif words < 100: return "intermediate"
    elif words < 150: return "advanced"
    else:             return "expert"

def map_domain(text, domain_hint=None):
    if domain_hint and domain_hint not in ["train", "test", None, "None"]:
        return domain_hint
    t = (text or "").lower()
    if any(w in t for w in ['integral', 'derivative', 'limit', 'calculus']):     return "calculus"
    if any(w in t for w in ['triangle', 'circle', 'angle', 'area', 'geometry']): return "geometry"
    if any(w in t for w in ['prime', 'divisib', 'modulo', 'number theory']):      return "number_theory"
    if any(w in t for w in ['probability', 'random', 'expected', 'chance']):      return "probability"
    if any(w in t for w in ['combination', 'permutation', 'counting']):           return "combinatorics"
    if any(w in t for w in ['sequence', 'series', 'arithmetic', 'geometric']):    return "sequences_and_series"
    if any(w in t for w in ['sin', 'cos', 'tan', 'trigon']):                      return "trigonometry"
    return "algebra"

def word_count(text):
    return len(str(text).split()) if text else 0

def pick_best_solution(item):
    candidates = []
    for key in ["solution_1", "solution_2", "solution_3", "solutions", "r1_solution"]:
        if key in item and item[key]:
            val = item[key]
            candidates.extend([str(v) for v in val] if isinstance(val, list) else [str(val)])
    return max(candidates, key=len) if candidates else ""

# ── AM-DEEPSEEK SPECIAL READER (bypasses broken HF schema) ───────────────
def convert_am_deepseek(limit, outdir):
    print(f"\n  Using direct JSONL reader for AM-DeepSeek (bypasses HF schema bug)")
    print(f"  Source: {AM_DEEPSEEK_RAW}\n")

    records = []
    skipped = 0

    with open(AM_DEEPSEEK_RAW, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and len(records) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            messages = item.get("messages", [])
            user_msg = next((m for m in messages if m.get("role") == "user"), None)
            asst_msg = next((m for m in messages if m.get("role") == "assistant"), None)

            if not user_msg:
                skipped += 1
                continue

            question = str(user_msg.get("content", "")).strip()
            if not question or len(question) < 10:
                skipped += 1
                continue

            # extract from assistant info block
            thinking = ""
            answer   = ""
            if asst_msg:
                info       = asst_msg.get("info", {}) or {}
                think_raw  = info.get("think_content")  or asst_msg.get("content", "")
                answer_raw = info.get("answer_content") or info.get("reference_answer", "")
                thinking   = "" if str(think_raw)  == "None" else str(think_raw).strip()
                answer     = "" if str(answer_raw) == "None" else str(answer_raw).strip()

                # fallback: parse <think> from content if think_content is empty
                if not thinking and asst_msg.get("content"):
                    content     = asst_msg["content"]
                    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                    if think_match:
                        thinking = think_match.group(1).strip()

            domain     = map_domain(question)
            difficulty = map_difficulty(question)

            record = {
                "SFT/Reasoning":                   "Reasoning",
                "System prompt":                    SYSTEM_PROMPT,
                "Instruction":                      INSTRUCTION,
                "Question":                         question,
                "<Think>":                          thinking,
                "Answer":                           answer,
                "Difficulty level":                 difficulty,
                "Number of words in instruction":   word_count(INSTRUCTION),
                "Number of words in system prompt": word_count(SYSTEM_PROMPT),
                "Number of words in question":      word_count(question),
                "Number of words in answer":        word_count(answer),
                "Task":                             "QA",
                "Domain":                           domain,
                "Language":                         "English",
                "Source":                           "AM-DeepSeek-R1-Distilled-1.4M",
                "Made by":                          "math_pipeline",
                "Reasoning/SFT":                    "reasoning",
                "Model":                            "DeepSeek-R1",
                "sample_id":                        f"AM-DeepSeek:train:{i}",
                "verified":                         False,
            }
            records.append(record)

            if len(records) % 100 == 0:
                print(f"  Converted {len(records)} records...")

    count        = len(records)
    jsonl_file   = os.path.join(outdir, f"AM-DeepSeek-R1-Distilled-1.4M_schema_{count}.jsonl")
    parquet_file = os.path.join(outdir, f"AM-DeepSeek-R1-Distilled-1.4M_schema_{count}.parquet")

    with open(jsonl_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pd.DataFrame(records).to_parquet(parquet_file, index=False)

    print(f"\n{'='*60}")
    print(f"  DONE — {datetime.now().strftime('%H:%M:%S')}")
    print(f"  Converted : {count}")
    print(f"  Skipped   : {skipped}")
    print(f"  JSONL     : {jsonl_file}")
    print(f"  Parquet   : {parquet_file}")
    print(f"{'='*60}\n")

# ── LOAD DATASET (for non-AM-DeepSeek datasets) ───────────────────────────
def load_dataset_safe(name, cfg, spl, limit):
    from datasets import load_dataset
    info = DATASET_CONFIG.get(name, {})
    cfg  = args.config or cfg or info.get("config")
    spl  = args.split  or spl or info.get("split", "train")

    splits_to_try  = [spl, "train", "test", "validation"]
    configs_to_try = [cfg, None, "main", "default"] if cfg else [None, "main", "default"]

    for c in configs_to_try:
        for s in splits_to_try:
            try:
                ds = load_dataset(name, c, split=s, streaming=False, features=None) if c \
                     else load_dataset(name, split=s, streaming=False, features=None)
                print(f"  Loaded: config={c}, split={s}, size={len(ds)}")
                return ds, s
            except Exception:
                continue

    print("  Trying streaming mode...")
    try:
        ds = load_dataset(name, cfg, split=spl, streaming=True, features=None) if cfg \
             else load_dataset(name, split=spl, streaming=True, features=None)
        print(f"  Loaded in streaming mode")
        return ds, spl
    except Exception as e:
        raise RuntimeError(f"Could not load {name}: {e}")

# ── MAIN ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  STEP 7 — Schema Conversion (No Generation)")
print(f"  Dataset : {DATASET_NAME}")
print(f"  Limit   : {args.limit}")
print(f"  Output  : {args.outdir}")
print(f"{'='*60}\n")

# Route AM-DeepSeek to special direct-reader (bypasses broken HF schema)
if DATASET_NAME == "a-m-team/AM-DeepSeek-R1-Distilled-1.4M":
    convert_am_deepseek(args.limit, args.outdir)
    exit(0)

# ── Standard path for all other datasets ─────────────────────────────────
info       = DATASET_CONFIG.get(DATASET_NAME, {})
q_key      = info.get("question_key")
a_key      = info.get("answer_key")
t_key      = info.get("thinking_key")
d_key      = info.get("domain_key")
model_name = info.get("model", "unknown")
r_type     = info.get("reasoning_type", "reasoning")
min_score  = info.get("min_score", 0.0)

print(f"Loading {DATASET_NAME}...")
ds, split_used = load_dataset_safe(DATASET_NAME, info.get("config"), info.get("split", "train"), args.limit)

records = []
skipped = 0
count   = 0

for i, item in enumerate(ds):
    if args.limit and count >= args.limit:
        break

    if "lm_q1q2_score" in item and item["lm_q1q2_score"] < min_score:
        skipped += 1
        continue

    question = None
    if q_key and q_key in item and item[q_key]:
        question = str(item[q_key]).strip()
    else:
        for k in FALLBACK_QUESTION_KEYS:
            if k in item and item[k]:
                question = str(item[k]).strip()
                break

    if not question or len(question) < 10:
        skipped += 1
        continue

    answer = ""
    if a_key and a_key in item and item[a_key]:
        answer = str(item[a_key]).strip()
    else:
        for k in FALLBACK_ANSWER_KEYS:
            if k in item and item[k]:
                answer = str(item[k]).strip()
                break

    thinking = ""
    if DATASET_NAME == "zwhe99/DeepMath-103K":
        thinking = pick_best_solution(item)
    elif t_key and t_key in item and item[t_key]:
        val = item[t_key]
        if isinstance(val, dict):   thinking = str(list(val.values())[0]).strip()
        elif isinstance(val, list): thinking = str(val[0]).strip() if val else ""
        else:                       thinking = str(val).strip()

    domain_hint = str(item[d_key]) if d_key and d_key in item else None
    domain      = map_domain(question, domain_hint)
    difficulty  = map_difficulty(question)

    record = {
        "SFT/Reasoning":                   r_type.capitalize(),
        "System prompt":                    SYSTEM_PROMPT,
        "Instruction":                      INSTRUCTION,
        "Question":                         question,
        "<Think>":                          thinking,
        "Answer":                           answer,
        "Difficulty level":                 difficulty,
        "Number of words in instruction":   word_count(INSTRUCTION),
        "Number of words in system prompt": word_count(SYSTEM_PROMPT),
        "Number of words in question":      word_count(question),
        "Number of words in answer":        word_count(answer),
        "Task":                             "QA",
        "Domain":                           domain,
        "Language":                         "English",
        "Source":                           SOURCE_TAG,
        "Made by":                          "math_pipeline",
        "Reasoning/SFT":                    r_type,
        "Model":                            model_name,
        "sample_id":                        f"{SOURCE_TAG}:{split_used}:{i}",
        "verified":                         False,
    }
    records.append(record)
    count += 1
    if count % 100 == 0:
        print(f"  Converted {count} records...")

print(f"\n  Total converted : {count}")
print(f"  Skipped         : {skipped}")

jsonl_file   = os.path.join(args.outdir, f"{SOURCE_TAG}_schema_{count}.jsonl")
parquet_file = os.path.join(args.outdir, f"{SOURCE_TAG}_schema_{count}.parquet")

with open(jsonl_file, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
pd.DataFrame(records).to_parquet(parquet_file, index=False)

print(f"\n{'='*60}")
print(f"  DONE — {datetime.now().strftime('%H:%M:%S')}")
print(f"  JSONL   : {jsonl_file}")
print(f"  Parquet : {parquet_file}")
print(f"  Records : {count}")
print(f"{'='*60}\n")