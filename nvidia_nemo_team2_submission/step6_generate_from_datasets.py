"""
step6_generate_from_datasets.py
Uses questions from HuggingFace datasets as INSPIRATION only.
Kimi-k2.5 generates completely NEW, DIVERSE questions — not copies.
Progressive difficulty 1→5 per seed question.
Duplicate detection ensures no repeated questions.

Usage:
  python step6_generate_from_datasets.py --dataset netop/TeleMath --outfile telemath_generated.jsonl
  python step6_generate_from_datasets.py --dataset KbsdJames/Omni-MATH --outfile omnimath_generated.jsonl
  python step6_generate_from_datasets.py --dataset WNJXYK/MATH-Reasoning-Paths --outfile mathpaths_generated.jsonl
  python step6_generate_from_datasets.py --dataset allenai/ai2_arc --outfile arc_generated.jsonl

Output format (same as step5):
{
  "source": "kimi-k2.5-generated-from-TeleMath",
  "question": "...",
  "model_response": "<think>...</think>\nSolution...",
  "model_answer": "X",
  "ground_truth": "X",
  "topic": "algebra",
  "difficulty": 3,
  "difficulty_label": "Intermediate",
  "seed_question": "original question used as inspiration",
  "seed_dataset": "netop/TeleMath",
  "augmented": true,
  "generated_by": "kimi-k2.5",
  "team": "math_pipeline_team2"
}
"""

import json
import requests
import time
import os
import re
import argparse
import hashlib
from datetime import datetime

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",    required=True, help="HuggingFace dataset path e.g. netop/TeleMath")
parser.add_argument("--outfile",    default=None,  help="Output file path (auto-named if not set)")
parser.add_argument("--limit",      type=int, default=200, help="Max seed questions to use from dataset")
parser.add_argument("--config",     default=None,  help="Dataset config if needed e.g. ARC-Challenge")
parser.add_argument("--split",      default="train", help="Dataset split (train/test)")
args = parser.parse_args()

# auto-name output file from dataset name
DATASET_NAME = args.dataset
SOURCE_TAG   = DATASET_NAME.split("/")[-1]   # e.g. TeleMath, Omni-MATH
OUTFILE      = args.outfile or f"generated_from_{SOURCE_TAG}.jsonl"

# ── KIMI CONFIG ───────────────────────────────────────────────────────────
KIMI_URL = "http://soketlab-node054:30000/v1/chat/completions"
HEADERS  = {"Content-Type": "application/json"}

# ── DIFFICULTY ────────────────────────────────────────────────────────────
DIFFICULTY_LABELS = {
    1: "Basic",
    2: "Elementary",
    3: "Intermediate",
    4: "Advanced",
    5: "Expert",
}

DIFFICULTY_DESCRIPTIONS = {
    1: "simple single-step problem suitable for high school beginners, uses basic formulas directly",
    2: "two or three step problem requiring some algebraic manipulation, suitable for grade 10-11",
    3: "multi-step problem requiring deeper reasoning and multiple concepts, suitable for grade 12 or JEE Mains level",
    4: "challenging competition-style problem requiring creative insight, suitable for JEE Advanced level",
    5: "olympiad-level problem requiring deep mathematical insight, elegant proof or non-obvious technique",
}

# ── DATASET FIELD MAPPINGS ────────────────────────────────────────────────
DATASET_CONFIG = {
    "netop/TeleMath": {
        "question": "question",
        "config":   None,
        "split":    "train",
    },
    "WNJXYK/MATH-Reasoning-Paths": {
        "question": "problem",
        "config":   None,
        "split":    "train",
    },
    "KbsdJames/Omni-MATH": {
        "question": "problem",
        "config":   None,
        "split":    "test",
    },
    "allenai/ai2_arc": {
        "question": "question",
        "config":   "ARC-Challenge",
        "split":    "test",
    },
    "zwhe99/DeepMath-103K": {
        "question": "problem",
        "config":   None,
        "split":    "train",
    },
}

FALLBACK_QUESTION_KEYS = [
    "question", "Question", "problem", "Problem",
    "instruction", "prompt", "input", "query", "text"
]

# ── LOAD DATASET ──────────────────────────────────────────────────────────
def load_seed_questions(dataset_name, limit):
    from datasets import load_dataset

    info      = DATASET_CONFIG.get(dataset_name, {})
    cfg       = args.config or info.get("config")
    spl       = args.split  or info.get("split", "train")
    q_key     = info.get("question")

    # try configured split first, then fallbacks
    splits_to_try = [spl, "train", "test", "validation"]
    configs_to_try = [cfg, None, "main", "default"] if cfg else [None, "main", "default"]

    ds = None
    for c in configs_to_try:
        for s in splits_to_try:
            try:
                ds = load_dataset(dataset_name, c, split=s) if c else load_dataset(dataset_name, split=s)
                print(f"  Loaded: config={c}, split={s}, columns={ds.column_names}")
                break
            except Exception:
                continue
        if ds:
            break

    if not ds:
        raise RuntimeError(f"Could not load {dataset_name}")

    questions = []
    for item in ds:
        if len(questions) >= limit:
            break
        # find question field
        text = None
        if q_key and q_key in item and item[q_key]:
            text = str(item[q_key])
        else:
            for k in FALLBACK_QUESTION_KEYS:
                if k in item and item[k]:
                    text = str(item[k])
                    break
        if text and len(text.strip()) > 20:
            questions.append(text.strip())

    print(f"  Loaded {len(questions)} seed questions from {dataset_name}")
    return questions

# ── PROMPT ────────────────────────────────────────────────────────────────
def make_prompt(seed_question, difficulty):
    desc  = DIFFICULTY_DESCRIPTIONS[difficulty]
    label = DIFFICULTY_LABELS[difficulty]
    return f"""You are an expert math problem creator.

Below is a SEED question for inspiration only. Do NOT copy it or solve it.
Use it only to understand the TOPIC and CONCEPT, then create a completely NEW and ORIGINAL question.

SEED QUESTION (for topic reference only):
{seed_question[:300]}

Your task:
- Create a brand new math problem on the SAME TOPIC but completely different from the seed
- Difficulty: Level {difficulty} ({label}) — {desc}
- The problem must be 100% original — different numbers, different scenario, different structure
- Must have a clean numeric or algebraic final answer
- Do NOT copy the seed question in any way

Respond in EXACTLY this format and nothing else:

QUESTION:
[Write your completely new original problem here]

SOLUTION:
[Write the complete step-by-step solution here]

FINAL ANSWER:
[Write only the final answer here, just the number or expression]"""

# ── API CALL ──────────────────────────────────────────────────────────────
def call_kimi(seed_question, difficulty):
    payload = {
        "model": "kimi-k2.5",
        "messages": [
            {"role": "system", "content": "You are an expert math problem creator. Create original problems with full solutions."},
            {"role": "user",   "content": make_prompt(seed_question, difficulty)},
        ],
        "temperature": 0.9,
        "max_tokens":  16000,
    }
    try:
        resp = requests.post(KIMI_URL, headers=HEADERS, json=payload, timeout=900)
        if resp.status_code == 200:
            msg       = resp.json()["choices"][0]["message"]
            content   = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            return reasoning, content
        else:
            print(f"  API error {resp.status_code}: {resp.text[:200]}")
            return None, None
    except requests.exceptions.Timeout:
        print("  Timeout")
        return None, None
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
        return None, None

# ── PARSE RESPONSE ────────────────────────────────────────────────────────
def parse_response(content):
    question = solution = answer = ""

    q_match = re.search(r'QUESTION:\s*(.*?)(?=SOLUTION:|$)', content, re.DOTALL | re.IGNORECASE)
    if q_match:
        question = q_match.group(1).strip()

    s_match = re.search(r'SOLUTION:\s*(.*?)(?=FINAL ANSWER:|$)', content, re.DOTALL | re.IGNORECASE)
    if s_match:
        solution = s_match.group(1).strip()

    a_match = re.search(r'FINAL ANSWER:\s*(.*?)$', content, re.DOTALL | re.IGNORECASE)
    if a_match:
        answer = a_match.group(1).strip().split('\n')[0].strip()

    return question, solution, answer

# ── VALIDATION ────────────────────────────────────────────────────────────
def is_valid(question, solution, answer):
    if not question or len(question) < 20:
        return False, "question too short"
    if not solution or len(solution) < 50:
        return False, "solution too short"
    if not answer:
        return False, "no answer extracted"
    return True, "ok"

# ── DUPLICATE DETECTION ───────────────────────────────────────────────────
def question_hash(text):
    """Hash first 100 chars of question for duplicate detection."""
    normalized = re.sub(r'\s+', ' ', text.strip().lower())[:100]
    return hashlib.md5(normalized.encode()).hexdigest()

# ── TOPIC EXTRACTOR ───────────────────────────────────────────────────────
def guess_topic(question_text):
    """Guess math topic from question text for metadata."""
    t = question_text.lower()
    if any(w in t for w in ['integral', 'derivative', 'limit', 'differentiat', 'calculus']):
        return 'calculus'
    if any(w in t for w in ['triangle', 'circle', 'angle', 'area', 'perimeter', 'geometry', 'polygon']):
        return 'geometry'
    if any(w in t for w in ['prime', 'divisib', 'modulo', 'remainder', 'factor', 'gcd', 'lcm']):
        return 'number theory'
    if any(w in t for w in ['probability', 'random', 'likely', 'chance', 'expected']):
        return 'probability'
    if any(w in t for w in ['combination', 'permutation', 'arrange', 'choose', 'select', 'ways']):
        return 'combinatorics'
    if any(w in t for w in ['sequence', 'series', 'arithmetic', 'geometric', 'sum of']):
        return 'sequences and series'
    if any(w in t for w in ['sin', 'cos', 'tan', 'trigon', 'angle']):
        return 'trigonometry'
    return 'algebra'

# ── RESUME — load already generated questions ─────────────────────────────
existing_count  = 0
existing_hashes = set()
existing_seeds  = set()   # track which (seed_idx, difficulty) already done

if os.path.exists(OUTFILE):
    with open(OUTFILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                existing_count += 1
                existing_hashes.add(question_hash(obj.get("question", "")))
                key = (obj.get("seed_index", -1), obj.get("difficulty", 0))
                existing_seeds.add(key)
            except:
                pass
    print(f"Resuming — {existing_count} already generated, {len(existing_hashes)} unique questions tracked")
else:
    print("Starting fresh.")

# ── LOAD SEED QUESTIONS ───────────────────────────────────────────────────
print(f"\nLoading seed questions from {DATASET_NAME}...")
seed_questions = load_seed_questions(DATASET_NAME, args.limit)

if not seed_questions:
    print("ERROR: No seed questions loaded. Check dataset name and config.")
    exit(1)

# ── MAIN ──────────────────────────────────────────────────────────────────
total_target = len(seed_questions) * len(DIFFICULTY_LABELS)
print(f"\n{'='*60}")
print(f"  STEP 6 — Dataset-Inspired Math Question Generation")
print(f"  Model       : kimi-k2.5 @ {KIMI_URL}")
print(f"  Seed dataset: {DATASET_NAME}")
print(f"  Seed questions: {len(seed_questions)}")
print(f"  Difficulties: 5 levels per seed")
print(f"  Target      : {total_target} questions")
print(f"  Output      : {OUTFILE}")
print(f"{'='*60}\n")

ok = fail = skip = dup = 0

with open(OUTFILE, "a", encoding="utf-8") as out:
    for seed_idx, seed_q in enumerate(seed_questions):
        print(f"\n  ── Seed [{seed_idx+1}/{len(seed_questions)}]: {seed_q[:60]}...")

        for difficulty, label in DIFFICULTY_LABELS.items():
            key = (seed_idx, difficulty)

            # skip if already done
            if key in existing_seeds:
                print(f"    [SKIP] Level {difficulty} — already done")
                skip += 1
                continue

            print(f"    [Level {difficulty} — {label}] generating...", end=" ", flush=True)
            reasoning, content = call_kimi(seed_q, difficulty)

            if not content:
                print("FAILED (no response)")
                fail += 1
                continue

            question, solution, answer = parse_response(content)
            valid, reason = is_valid(question, solution, answer)

            if not valid:
                print(f"FAILED ({reason})")
                fail += 1
                continue

            # duplicate check
            qhash = question_hash(question)
            if qhash in existing_hashes:
                print("SKIPPED (duplicate)")
                dup += 1
                continue
            existing_hashes.add(qhash)

            # build record
            model_response = f"<think>{reasoning}</think>\n{solution}" if reasoning else solution
            topic          = guess_topic(seed_q + " " + question)

            record = {
                "source":           f"kimi-k2.5-generated-from-{SOURCE_TAG}",
                "question":         question,
                "model_response":   model_response,
                "model_answer":     answer,
                "ground_truth":     answer,
                "topic":            topic,
                "difficulty":       difficulty,
                "difficulty_label": label,
                "seed_question":    seed_q[:200],
                "seed_dataset":     DATASET_NAME,
                "seed_index":       seed_idx,
                "augmented":        True,
                "generated_by":     "kimi-k2.5",
                "team":             "math_pipeline_team2",
                "timestamp":        datetime.utcnow().isoformat(),
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            ok += 1
            existing_seeds.add(key)
            print(f"OK — Q: {question[:50]}... A: {answer}")
            time.sleep(0.5)

print(f"\n{'='*60}")
print(f"  DONE — {datetime.now().strftime('%H:%M:%S')}")
print(f"  Generated  : {ok}")
print(f"  Failed     : {fail}")
print(f"  Skipped    : {skip}")
print(f"  Duplicates : {dup}")
print(f"  Total in file: {ok + existing_count}")
print(f"  Saved → {OUTFILE}")
print(f"{'='*60}")