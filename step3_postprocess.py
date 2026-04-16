"""
step3_postprocess.py — UNIVERSAL FINAL VERSION
Handles all 5 model output formats automatically.
Usage: python step3_postprocess.py --model glm-5-fp8 --source LIMO --outdir results/
"""
import json
import re
import argparse
import os
import pandas as pd
from collections import Counter

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",   required=True, help="Model name, e.g. glm-5-fp8")
parser.add_argument("--source",  required=True, help="Dataset short name, e.g. LIMO")
parser.add_argument("--infile",  default=None,  help="Defaults to output_<model>.jsonl")
parser.add_argument("--outdir",  default=".",   help="Where to write output files")
args = parser.parse_args()

MODEL_NAME = args.model
SOURCE     = args.source
INFILE     = args.infile or f"output_{MODEL_NAME}.jsonl"
OUTDIR     = args.outdir
DOMAIN     = "math"

os.makedirs(OUTDIR, exist_ok=True)

# ── answer extraction ─────────────────────────────────────────────────────

def extract_reasoning_and_answer(generation):
    """
    Split on </think> / </thinking> / </thought>.
    Returns (reasoning_str, answer_str).
    """
    for tag in [r'</think>', r'</thinking>', r'</thought>']:
        m = re.search(tag, generation, re.IGNORECASE)
        if m:
            reasoning = generation[:m.start()]
            reasoning = re.sub(r'<think[^>]*>', '', reasoning, flags=re.IGNORECASE).strip()
            after_tag = generation[m.end():].strip()
            return reasoning, get_best_answer(after_tag, reasoning)
    # no think tag — treat whole thing as reasoning
    return generation.strip(), get_best_answer("", generation)


def get_best_answer(after_tag, reasoning):
    """Try extraction strategies in order of reliability."""
    full_text = (after_tag or "") + " " + (reasoning or "")

    # 1. \boxed{} in after_tag (deepseek / kimi)
    if after_tag:
        box = re.search(r'\\boxed\{([^}]+)\}', after_tag)
        if box:
            return box.group(1).replace(',', '').strip()

    # 2. plain number right after tag — strip minimax <...>wrapper</...> first
    if after_tag:
        cleaned = re.sub(r'<\.\.\.>.*?</\.\.\.>', '', after_tag, flags=re.DOTALL).strip()
        if cleaned:
            nums = re.findall(r'\$?\s*(-?\d+(?:[.,]\d+)?(?:/\d+)?)', cleaned)
            if nums:
                return nums[0].replace(',', '')

    # 3. \boxed{} in reasoning
    boxes = re.findall(r'\\boxed\{([^}]+)\}', reasoning)
    if boxes:
        return boxes[-1].replace(',', '').strip()

    # 4. "Final answer: X" / "The answer is X"
    fa = re.search(
        r'(?:[Ff]inal\s+[Aa]nswer|[Tt]he\s+answer\s+is)\s*[:\-=]?\s*\$?\s*(-?\d+(?:\.\d+)?(?:/\d+)?)',
        full_text
    )
    if fa:
        return fa.group(1).replace(',', '')

    # 5. bold number **X**
    bolds = re.findall(r'\*\*\$?(-?[\d,]+\.?\d*)\*\*', full_text)
    if bolds:
        return bolds[-1].replace(',', '')

    # 6. last number after last = sign
    equals = re.findall(r'=\s*\$?(-?[\d,]+\.?\d*)\s*(?:\.|$|\n)', reasoning)
    if equals:
        return equals[-1].replace(',', '')

    # 7. absolute last number
    all_nums = re.findall(r'-?\b\d+(?:\.\d+)?\b', full_text)
    return all_nums[-1] if all_nums else ""


# ── helpers ───────────────────────────────────────────────────────────────

def extract_expected(text):
    """GSM8K-style: extract number after ####; else last number."""
    m = re.search(r'####\s*([\d,\.]+)', str(text))
    if m:
        return m.group(1).replace(',', '').strip()
    nums = re.findall(r'-?\b\d+(?:\.\d+)?\b', str(text))
    return nums[-1] if nums else str(text).strip()


def verify(predicted, expected_raw):
    if not predicted or not expected_raw:
        return "unknown"
    expected = extract_expected(expected_raw)
    try:
        p = float(str(predicted).replace(',', ''))
        e = float(str(expected).replace(',', ''))
        return "pass" if abs(p - e) < 1e-4 else "fail"
    except Exception:
        return "pass" if str(predicted).strip() == str(expected).strip() else "unknown"


def get_difficulty(problem):
    t     = problem.lower()
    words = len(t.split())
    nums  = len(re.findall(r'\d+', t))
    ops   = sum(t.count(w) for w in ['percent','ratio','fraction','rate','total','average','per','each'])
    score = 0
    if words > 80:   score += 2
    elif words > 50: score += 1
    if nums > 6:     score += 1
    if ops > 2:      score += 1
    return min(score, 4)


# ── main ──────────────────────────────────────────────────────────────────
if not os.path.exists(INFILE):
    print(f"ERROR: {INFILE} not found. Did step2 run for model '{MODEL_NAME}'?")
    exit(1)

items = []
with open(INFILE, encoding="utf-8") as f:
    for lineno, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"  Skipping bad JSON at line {lineno}: {line[:50]}")

print(f"Read {len(items)} records from {INFILE}")

results = []
for item in items:
    problem    = item.get("problem", "")
    generation = item.get("generation", "")
    if not generation:
        continue                          # step2 marked this as failed — skip

    reasoning, answer = extract_reasoning_and_answer(generation)
    expected          = item.get("expected_answer", "")
    status            = verify(answer, expected)

    results.append({
        "prompt":       problem,
        "response":     f"<think>{reasoning}</think>\n{answer}",
        "ground_truth": extract_expected(expected) if expected else None,
        "difficulty":   get_difficulty(problem),
        "domain":       DOMAIN,
        "subdomain":    item.get("subdomain", SOURCE),
        "verification": status,
        "source":       SOURCE,
        "model_used":   MODEL_NAME,
        "dataset":      item.get("dataset", ""),
    })

count        = len(results)
jsonl_file   = os.path.join(OUTDIR, f"{DOMAIN}_{MODEL_NAME}_{SOURCE}_{count}.jsonl")
parquet_file = os.path.join(OUTDIR, f"{DOMAIN}_{MODEL_NAME}_{SOURCE}_{count}.parquet")

with open(jsonl_file, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

pd.DataFrame(results).to_parquet(parquet_file, index=False)

v = Counter(r["verification"] for r in results)
print(f"Saved   : {count} records")
print(f"JSONL   : {jsonl_file}")
print(f"Parquet : {parquet_file}")
print(f"PASS    : {v['pass']} ({v['pass']/count*100:.1f}%)" if count else "PASS: 0")
print(f"FAIL    : {v['fail']} ({v['fail']/count*100:.1f}%)" if count else "FAIL: 0")
print(f"UNKNOWN : {v['unknown']}")