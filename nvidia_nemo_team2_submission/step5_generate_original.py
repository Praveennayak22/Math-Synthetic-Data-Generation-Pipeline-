"""
step5_generate_original.py
Generates original math questions + solutions using kimi-k2.5.
No dataset needed — kimi creates questions from scratch.

Usage:
  python step5_generate_original.py --per-combo 10 --outfile generated_math.jsonl
  python step5_generate_original.py --per-combo 50 --outfile generated_math.jsonl  # larger run
  python step5_generate_original.py --topic algebra --difficulty 3 --per-combo 5   # single combo test

Output format matches teammate's jeebench format:
{
  "source": "kimi-k2.5-generated",
  "question": "...",
  "model_response": "<think>...</think>\nFinal answer: X",
  "model_answer": "X",
  "ground_truth": "X",
  "topic": "algebra",
  "difficulty": 3,
  "difficulty_label": "Advanced",
  "augmented": false,
  "team": "math_pipeline_team2",
  "generated_by": "kimi-k2.5"
}
"""

import json
import requests
import time
import os
import re
import argparse
from datetime import datetime

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--per-combo",  type=int, default=10,
                    help="Questions to generate per topic×difficulty combo")
parser.add_argument("--outfile",    default="generated_math.jsonl",
                    help="Output file path")
parser.add_argument("--topic",      default=None,
                    help="Single topic only (optional)")
parser.add_argument("--difficulty", type=int, default=None,
                    help="Single difficulty only 1-5 (optional)")
args = parser.parse_args()

# ── KIMI CONFIG ───────────────────────────────────────────────────────────
KIMI_URL = "http://soketlab-node054:30000/v1/chat/completions"
HEADERS  = {"Content-Type": "application/json"}

# ── TOPICS & DIFFICULTY ───────────────────────────────────────────────────
TOPICS = [
    "algebra",
    "geometry",
    "number theory",
    "combinatorics",
    "calculus",
    "trigonometry",
    "probability",
    "sequences and series",
]

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

if args.topic:
    TOPICS = [args.topic]
if args.difficulty:
    DIFFICULTY_LABELS = {args.difficulty: DIFFICULTY_LABELS[args.difficulty]}

# ── PROMPT ────────────────────────────────────────────────────────────────
def make_prompt(topic, difficulty):
    desc = DIFFICULTY_DESCRIPTIONS[difficulty]
    label = DIFFICULTY_LABELS[difficulty]
    return f"""You are an expert math problem creator. Create a completely original {topic} problem.

Requirements:
- Difficulty: Level {difficulty} ({label}) — {desc}
- The problem must be completely original and NOT copied from any textbook or online source
- The problem must have a clean numeric or algebraic final answer
- Do NOT use problems from JEE, AMC, AIME, or any known competition directly — create inspired-by but original problems

Respond in EXACTLY this format and nothing else:

QUESTION:
[Write the complete problem statement here]

SOLUTION:
[Write the complete step-by-step solution here]

FINAL ANSWER:
[Write only the final answer here, just the number or expression]"""

# ── API CALL ──────────────────────────────────────────────────────────────
def call_kimi(topic, difficulty):
    payload = {
        "model": "kimi-k2.5",
        "messages": [
            {"role": "system", "content": "You are a math expert. Solve the problem step by step. Write the final answer clearly."},
            {"role": "user",   "content": make_prompt(topic, difficulty)},
        ],
        "temperature": 0.9,   # higher temp for more diverse generation
        "max_tokens":  4000,
    }
    try:
        resp = requests.post(KIMI_URL, headers=HEADERS, json=payload, timeout=300)
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
    """Extract question, solution, and final answer from kimi's response."""
    question = ""
    solution = ""
    answer   = ""

    # extract QUESTION block
    q_match = re.search(r'QUESTION:\s*(.*?)(?=SOLUTION:|$)', content, re.DOTALL | re.IGNORECASE)
    if q_match:
        question = q_match.group(1).strip()

    # extract SOLUTION block
    s_match = re.search(r'SOLUTION:\s*(.*?)(?=FINAL ANSWER:|$)', content, re.DOTALL | re.IGNORECASE)
    if s_match:
        solution = s_match.group(1).strip()

    # extract FINAL ANSWER block
    a_match = re.search(r'FINAL ANSWER:\s*(.*?)$', content, re.DOTALL | re.IGNORECASE)
    if a_match:
        answer = a_match.group(1).strip().split('\n')[0].strip()

    return question, solution, answer

# ── VALIDATION ────────────────────────────────────────────────────────────
def is_valid(question, solution, answer):
    """Basic validation — ensure all parts are non-empty and reasonable."""
    if not question or len(question) < 20:
        return False, "question too short"
    if not solution or len(solution) < 50:
        return False, "solution too short"
    if not answer:
        return False, "no answer extracted"
    return True, "ok"

# ── RESUME ────────────────────────────────────────────────────────────────
existing_count = 0
existing_combos = {}   # (topic, difficulty) -> count already generated

if os.path.exists(args.outfile):
    with open(args.outfile, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                existing_count += 1
                key = (obj.get("topic", ""), obj.get("difficulty", 0))
                existing_combos[key] = existing_combos.get(key, 0) + 1
            except:
                pass
    print(f"Resuming — {existing_count} already generated")
else:
    print("Starting fresh.")

# ── MAIN ──────────────────────────────────────────────────────────────────
total_combos = len(TOPICS) * len(DIFFICULTY_LABELS)
print(f"\n{'='*55}")
print(f"  STEP 5 — Original Math Question Generation")
print(f"  Model     : kimi-k2.5 @ {KIMI_URL}")
print(f"  Topics    : {len(TOPICS)}")
print(f"  Difficulties: {len(DIFFICULTY_LABELS)}")
print(f"  Per combo : {args.per_combo}")
print(f"  Target    : {total_combos * args.per_combo} questions")
print(f"  Output    : {args.outfile}")
print(f"{'='*55}\n")

ok = fail = skip = 0

with open(args.outfile, "a", encoding="utf-8") as out:
    for topic in TOPICS:
        for difficulty, label in DIFFICULTY_LABELS.items():
            key = (topic, difficulty)
            already = existing_combos.get(key, 0)
            needed  = args.per_combo - already

            if needed <= 0:
                print(f"  [SKIP] {topic} × Level {difficulty} — already have {already}")
                skip += already
                continue

            print(f"\n  ── {topic.upper()} × Level {difficulty} ({label}) — generating {needed} ──")

            for i in range(needed):
                print(f"    [{i+1}/{needed}] generating...", end=" ", flush=True)
                reasoning, content = call_kimi(topic, difficulty)

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

                # build output record in teammate's format
                model_response = f"<think>{reasoning}</think>\n{solution}" if reasoning else solution

                record = {
                    "source":           "kimi-k2.5-generated",
                    "question":         question,
                    "model_response":   model_response,
                    "model_answer":     answer,
                    "ground_truth":     answer,
                    "topic":            topic,
                    "difficulty":       difficulty,
                    "difficulty_label": label,
                    "augmented":        False,
                    "generated_by":     "kimi-k2.5",
                    "team":             "math_pipeline_team2",
                    "timestamp":        datetime.utcnow().isoformat(),
                }

                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                ok += 1
                print(f"OK — Q: {question[:50]}... A: {answer}")
                time.sleep(0.5)   # avoid hammering the API

print(f"\n{'='*55}")
print(f"  DONE — {datetime.now().strftime('%H:%M:%S')}")
print(f"  Generated : {ok}")
print(f"  Failed    : {fail}")
print(f"  Skipped   : {skip}")
print(f"  Total in file: {ok + existing_count}")
print(f"  Saved → {args.outfile}")
print(f"{'='*55}")