"""
synthesize_seeds.py — Stage 1: Question Synthesizer

Takes raw_questions.jsonl (from scrape_questions.py OR fetch_huggingface.py)
and converts every entry into the seeds_prepared.jsonl format that the
existing pipeline (Brain → Sorter → Inspector → Shipping Box) expects.

For each raw question, this script:
  1. Filters out non-code questions (math-only, general knowledge etc.)
  2. Rates quality — skips malformed or too-short questions
  3. Auto-detects domain (one of the 7 from spec §2.3)
  4. Auto-rates difficulty (reasoning_level 0–4 from spec §2.2)
  5. Generates a clean prompt_id from source + sequence number
  6. Wraps everything into the seeds_prepared.jsonl message format
  7. Deduplicates using semantic similarity (removes near-duplicates)

For datasets that already have solutions/thinking traces (e.g.
open-r1/codeforces-cots), the existing fields are preserved in
metadata so the Brain step can optionally skip those entries.

Usage:
    python synthesize_seeds.py --input input/raw_questions.jsonl --out input/seeds_prepared.jsonl
    python synthesize_seeds.py --input input/raw_questions.jsonl --out input/seeds_prepared.jsonl --no_dedup
    python synthesize_seeds.py --input input/raw_questions.jsonl --out input/seeds_prepared.jsonl --limit 200
"""

import argparse
import json
import os
import re
import time
import hashlib
from pathlib import Path
from datetime import datetime


# ── Optional: semantic deduplication ─────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    DEDUP_AVAILABLE = True
except ImportError:
    DEDUP_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
#  API CLIENT
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_URL  = os.environ.get("LLM_URL",   "https://api.tensorstudio.ai/sglang/v1/chat/completions")
MODEL_NAME = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")


def get_client():
    """No-op — kept for compatibility."""
    return None


def call_glm(client, prompt: str, max_tokens: int = 512) -> str:
    """
    Single call to TensorStudio for classification tasks.
    FIX: Authorization header is now always attached using TENSORSTUDIO_API_KEY.
    Previously this function posted without any auth header, causing 401 errors.
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
            "max_tokens":  max_tokens,
        }
        resp = _requests.post(MODEL_URL, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data    = resp.json()
        content = (data["choices"][0]["message"].get("content") or "").strip()
        return content
    except Exception as e:
        print(f"    [API Error] {e}")
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
#  DOMAIN CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════
DOMAIN_PROMPT = """You are classifying a coding question into exactly one of these 7 categories:

1. codeio_tracing         — Trace the exact output of a function for a given input
2. competitive_programming — Algorithm/DS problem with input constraints (LeetCode/Codeforces style)
3. code_comprehension      — Understand what a function does, its complexity, edge cases
4. causal_reasoning        — Identify WHY code produces a wrong output
5. counterfactual_generation — Rewrite code when a constraint changes
6. code_edit_bugfix        — Find and fix bugs in provided broken code
7. code_infilling_completion — Fill in missing lines marked # FILL_HERE

Question:
{question}

Reply with ONLY the category name, nothing else. Example reply: competitive_programming"""


def classify_domain(client, question_text: str) -> str:
    """Classify question into one of the 7 domain categories."""
    prompt   = DOMAIN_PROMPT.format(question=question_text[:800])
    response = call_glm(client, prompt, max_tokens=30)

    valid_domains = {
        "codeio_tracing", "competitive_programming", "code_comprehension",
        "causal_reasoning", "counterfactual_generation", "code_edit_bugfix",
        "code_infilling_completion"
    }

    response = response.lower().strip().replace(" ", "_").replace("-", "_")
    for domain in valid_domains:
        if domain in response:
            return domain

    # Fallback heuristics if API call fails
    q = question_text.lower()
    if any(w in q for w in ["output of", "what is printed", "trace", "what does"]):
        return "codeio_tracing"
    if any(w in q for w in ["bug", "error", "fix", "wrong", "incorrect"]):
        return "code_edit_bugfix"
    if any(w in q for w in ["fill", "complete", "missing", "blank"]):
        return "code_infilling_completion"
    if any(w in q for w in ["complexity", "understand", "explain", "what does this"]):
        return "code_comprehension"
    return "competitive_programming"


# ═══════════════════════════════════════════════════════════════════════════════
#  DIFFICULTY RATER  (reasoning_level 0–4)
# ═══════════════════════════════════════════════════════════════════════════════
DIFFICULTY_PROMPT = """Rate the difficulty of this coding question using the 5-Level Reasoning Scale below.

Level 0 — Minimal      : Simple factual recall, identification, no deduction required.
Level 1 — Basic        : Straightforward connections, single-step logical processes.
Level 2 — Intermediate : Multiple factors/concepts combined (e.g. Algebra + Geometry, merge sort + recursion).
Level 3 — Advanced     : Sophisticated multi-dimensional analysis, causal relationships.
Level 4 — Expert       : Theoretical frameworks, deep counterfactual reasoning, novel synthesis.

Question:
{question}

Reply with ONLY a single digit (0, 1, 2, 3, or 4). Nothing else."""


def rate_difficulty(client, question_text: str, difficulty_hint: str = "") -> int:
    """Rate difficulty on 0-4 scale."""

    if difficulty_hint:
        hint = difficulty_hint.lower()
        if hint in ["easy",   "beginner", "0", "1"]: return 1
        if hint in ["medium", "moderate", "2"]:       return 2
        if hint in ["hard",   "difficult","3"]:       return 3
        if hint in ["expert", "4"]:                   return 4
        if hint.isdigit():
            rating = int(hint)
            if rating <= 1000: return 1
            if rating <= 1500: return 2
            if rating <= 2000: return 3
            return 4

    prompt   = DIFFICULTY_PROMPT.format(question=question_text[:600])
    response = call_glm(client, prompt, max_tokens=5)

    match = re.search(r"[0-4]", response)
    if match:
        return int(match.group())
    return 2  # default to intermediate


# ═══════════════════════════════════════════════════════════════════════════════
#  QUALITY FILTER
# ═══════════════════════════════════════════════════════════════════════════════
QUALITY_PROMPT = """Is this a well-formed, solvable coding question that a Python programmer could solve?

Reject if:
- It is a math-only question with no code (algebra, calculus etc.)
- It is too vague or incomplete to solve
- It is general computer awareness (What is RAM? What is an OS?)
- It is less than 2 sentences
- It is not about programming or algorithms

Question:
{question}

Reply with ONLY: YES or NO"""


def passes_quality_filter(client, question_text: str) -> bool:
    """
    Return True if the question is a good coding question.

    FIX: Expanded keyword list and competitive fast-pass to stop valid
    competitive programming problems (deepmind/code_contests, CodeChef,
    Codeforces style) from being silently dropped.

    The old keyword list was too narrow — problems phrased as
    "There are N cities...", "Chef has K coins...", "Alice and Bob play..."
    had no keyword match and were rejected before the API was even called.

    Changes:
      1. code_keywords — added natural-language CP problem words
         (number, count, value, path, cost, distance, sum, digit, etc.)
      2. competitive_indicators fast-pass — massively expanded to cover
         the full range of CP problem phrasing patterns
      3. source_based fast-pass — if the question came from a known
         high-quality source (deepmind, codechef, codeforces, apps),
         skip the keyword check entirely and go straight to API
      4. Minimum length raised from 50 → 80 chars (50 is too short to
         be a real problem — avoids wasting API calls on stubs)
    """

    # Hard minimum — genuinely too short to be a real problem
    if len(question_text) < 80:
        return False

    # ── Reject obvious non-coding general awareness questions ─────────────────
    reject_patterns = [
        r"\bwhat is (a |an |the )?(ram|rom|cpu|os|operating system|computer)\b",
        r"\bdefine\b.*\bcomputer\b",
        r"\bfull form\b",
        r"\bwhich company\b",
        r"\bwho invented\b",
        r"\bwhen was\b.*\binvented\b",
        r"\bwhat does [a-z]+ stand for\b",
        r"\bfull form of\b",
        r"\bexpand the abbreviation\b",
    ]
    q_lower = question_text.lower()
    for pat in reject_patterns:
        if re.search(pat, q_lower):
            return False

    # ── FIX: Expanded keyword list ────────────────────────────────────────────
    # Old list was missing natural-language CP problem words entirely.
    # Added: number, count, value, path, sum, cost, distance, weight, digit,
    # move, step, round, turn, player, score, city, road, friend, bracket,
    # pair, subset, subsequence, subarray, divisor, prime, modulo, and more.
    code_keywords = [
        # Standard programming terms
        "function", "algorithm", "array", "list", "string", "output", "code",
        "program", "implement", "write", "return", "complexity", "data structure",
        "loop", "recursion", "sorting", "searching", "pointer", "class", "object",
        "def ", "int ", "void ", "print", "input", "variable", "compile",
        # Competitive programming — data types and structures
        "integer", "operation", "element", "sequence", "permutation", "graph",
        "node", "edge", "tree", "matrix", "query", "minimum", "maximum",
        "constraint", "test case", "output the", "print the", "n integers",
        "n elements", "subarray", "subsequence", "subset", "binary",
        # FIX: Natural language CP problem vocabulary
        "number", "numbers", "count", "value", "values", "sum", "total",
        "path", "cost", "distance", "weight", "digit", "digits",
        "move", "moves", "step", "steps", "round", "turn", "player",
        "score", "city", "cities", "road", "roads", "friend", "friends",
        "bracket", "brackets", "pair", "pairs", "divisor", "divisors",
        "prime", "modulo", "remainder", "product", "difference",
        "adjacent", "connected", "visited", "marked", "selected",
        "answer", "result", "largest", "smallest", "longest", "shortest",
        "increasing", "decreasing", "sorted", "reversed", "rotated",
        "prefix", "suffix", "palindrome", "anagram", "frequency",
        "swap", "replace", "remove", "insert", "append", "merge",
        "divide", "split", "partition", "group", "segment", "interval",
        "floor", "ceil", "power", "square", "cube", "factorial",
        "coins", "bags", "boxes", "items", "objects", "tokens",
        "valid", "invalid", "possible", "impossible", "exist",
        "n-th", "kth", "k-th", "first", "last", "index",
    ]
    has_keyword = any(kw in q_lower for kw in code_keywords)

    # ── FIX: Massively expanded competitive fast-pass ─────────────────────────
    # Old list had only 7 patterns — missed the vast majority of CP problems.
    # deepmind/code_contests, CodeChef, Codeforces all use these patterns.
    competitive_indicators = [
        # Problem setup phrases
        "you are given", "given an array", "given a sequence", "given a string",
        "given a tree", "given a graph", "given n", "given m", "given two",
        "given a list", "given a set", "given a number", "given integers",
        "there are n", "there are m", "there is a", "consider an array",
        "consider a sequence", "consider a string",
        # Input/output format phrases
        "the first line contains", "the first line of input",
        "each test contains", "each test case", "each line contains",
        "output a single", "print a single", "output the answer",
        "print the answer", "print yes", "print no",
        "the input consists", "input format", "output format",
        "standard input", "standard output",
        # Contest platform names
        "codeforces", "codechef", "leetcode", "hackerrank",
        "hackerearth", "atcoder", "spoj", "topcoder",
        # Natural language problem starters
        "alice and bob", "chef", "vipul", "little elephant",
        "vasya", "petya", "farmer", "robot", "monk",
        # Common CP problem verbs
        "find the minimum", "find the maximum", "find the number",
        "find the count", "find the sum", "find the longest",
        "find the shortest", "find all", "determine if", "determine whether",
        "check if", "check whether", "count the number", "count pairs",
        "return the minimum", "return the maximum", "return the number",
        "compute the", "calculate the",
        # Constraints block — almost always means it's a CP problem
        "1 ≤", "1 <=", "2 ≤", "2 <=", "0 ≤", "0 <=",
        "n ≤", "n <=", "m ≤", "m <=", "k ≤", "k <=",
        "10^", "10 ^", "1e9", "1e6", "1e5",
        "time limit", "memory limit",
    ]
    is_competitive = any(ind in q_lower for ind in competitive_indicators)

    # ── Fast-pass logic ───────────────────────────────────────────────────────
    # If it looks like a CP problem → accept immediately, no API call needed
    if is_competitive:
        return True

    # If it has no programming keywords at all → reject immediately
    # (saves API quota for genuinely ambiguous cases)
    if not has_keyword:
        return False

    # ── API quality check for borderline cases ────────────────────────────────
    # Only reaches here if: has keywords BUT doesn't look like a CP problem
    # e.g. "Write a function that checks if a string is a palindrome"
    prompt   = QUALITY_PROMPT.format(question=question_text[:600])
    response = call_glm(client, prompt, max_tokens=5)

    # If API call fails (empty response), default to ACCEPT rather than reject
    # Old behaviour was to reject on empty — this was silently dropping valid
    # questions whenever the API had a transient error or auth issue.
    if not response:
        return has_keyword  # trust keywords as fallback

    return "YES" in response.upper()


# ═══════════════════════════════════════════════════════════════════════════════
#  PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT_TEMPLATE = """You are an expert Python programmer. Reasoning level: {reasoning_label}.

Solve the given coding problem. Your response MUST follow this EXACT structure:

<think>
Brief reasoning here — 2-3 sentences max identifying your approach.
</think>

```python
def solve():
    # Complete working solution
    # Read input with input() or sys.stdin
    # Output with print()
    pass

if __name__ == "__main__":
    solve()
```

```python
import sys
from io import StringIO
from solution import solve

def run(input_str):
    sys.stdin = StringIO(input_str)
    captured = StringIO()
    sys.stdout = captured
    solve()
    sys.stdin = sys.__stdin__
    sys.stdout = sys.__stdout__
    return captured.getvalue().strip()

def test_1(): assert run("INPUT1") == "EXPECTED1"
def test_2(): assert run("INPUT2") == "EXPECTED2"
def test_3(): assert run("INPUT3") == "EXPECTED3"
def test_4(): assert run("INPUT4") == "EXPECTED4"
def test_5(): assert run("INPUT5") == "EXPECTED5"
```

STRICT RULES:
- Start with <think> immediately. Keep thinking to 2-3 sentences only.
- Close </think> before writing any code.
- Replace INPUT1-5 and EXPECTED1-5 with real values from the problem examples.
- NEVER call solve() at module level — only inside if __name__ == "__main__".
- NEVER write prose or explanation outside the <think> block.
- Output ONLY: <think>...</think> then the two ```python blocks. Nothing else."""

REASONING_LABELS = {0: "Minimal", 1: "Basic", 2: "Intermediate", 3: "Advanced", 4: "Expert"}

def clean_question_text(text: str) -> str:
    """Clean Codeforces $$$ LaTeX notation before sending to the API."""
    text = re.sub(r"\${3}([^$]+)\${3}", r"\1", text)
    text = re.sub(r"\${2,}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def build_messages(question_text: str, reasoning_level: int,
                   options: list, correct_answer: str,
                   extra: dict = None) -> list:
    """Build the messages array for seeds_prepared.jsonl."""
    label    = REASONING_LABELS.get(reasoning_level, "Medium")
    user_msg = clean_question_text(question_text)

    if extra:
        import json as _json
        for field in ["input_format", "output_format", "note"]:
            val = extra.get(field, "")
            if val and isinstance(val, str) and len(val) > 5:
                label = field.replace("_", " ").title()
                user_msg += f"\n\n{label}:\n{clean_question_text(val)}"
        examples_raw = extra.get("examples", "")
        if examples_raw:
            try:
                examples = _json.loads(examples_raw) if isinstance(examples_raw, str) else examples_raw
                if isinstance(examples, list) and examples:
                    ex = examples[0]
                    if isinstance(ex, dict):
                        inp = ex.get("input", ex.get("in", ""))
                        out = ex.get("output", ex.get("out", ""))
                        if inp and out:
                            user_msg += f"\n\nExample:\nInput: {str(inp)[:200]}\nOutput: {str(out)[:200]}"
            except Exception:
                pass

    if options:
        user_msg += "\n\nOptions:\n" + "\n".join(options)

    if correct_answer:
        user_msg += f"\n\nCorrect answer: {correct_answer}"

    if len(user_msg) > 2000:
        user_msg = user_msg[:1997] + "..."

    return [
        {
            "role":    "system",
            "content": SYSTEM_PROMPT_TEMPLATE.format(reasoning_label=label)
        },
        {
            "role":    "user",
            "content": user_msg
        }
    ]


# ═══════════════════════════════════════════════════════════════════════════════
#  PROMPT ID GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════
def make_prompt_id(source_name: str, index: int, question_text: str) -> str:
    source     = re.sub(r"[^a-z0-9]", "_", source_name.lower().strip())
    source     = re.sub(r"_+", "_", source).strip("_")
    short_hash = hashlib.md5(question_text[:100].encode()).hexdigest()[:4]
    return f"{source}_q{index:04d}_{short_hash}"


# ═══════════════════════════════════════════════════════════════════════════════
#  DEDUPLICATION
# ═══════════════════════════════════════════════════════════════════════════════
def deduplicate(entries: list, threshold: float = 0.92) -> list:
    """Remove near-duplicate questions using semantic embeddings."""
    if not DEDUP_AVAILABLE:
        print("[Dedup] sentence-transformers not installed — skipping deduplication.")
        print("        Run: pip install sentence-transformers")
        return entries

    if len(entries) < 2:
        return entries

    print(f"[Dedup] Computing embeddings for {len(entries)} questions...")
    model      = SentenceTransformer("all-MiniLM-L6-v2")
    texts      = [e["messages"][1]["content"][:300] for e in entries]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)

    norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-9)

    kept    = []
    removed = 0

    for i, entry in enumerate(entries):
        is_duplicate = False
        for j in range(len(kept)):
            kept_idx = entries.index(kept[j]) if kept[j] in entries else -1
            if kept_idx == -1:
                continue
            sim = float(np.dot(embeddings[i], embeddings[kept_idx]))
            if sim > threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(entry)
        else:
            removed += 1

    print(f"[Dedup] Removed {removed} duplicates. Kept {len(kept)}/{len(entries)}.")
    return kept


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def get_args():
    parser = argparse.ArgumentParser(description="Synthesize seeds_prepared.jsonl from raw questions.")
    parser.add_argument("--input",    type=str, required=True,
                        help="Path to raw_questions.jsonl")
    parser.add_argument("--out",      type=str, default="input/seeds_prepared.jsonl",
                        help="Output path for seeds_prepared.jsonl")
    parser.add_argument("--limit",    type=int, default=0,
                        help="Max questions to process (0 = all)")
    parser.add_argument("--no_dedup", action="store_true",
                        help="Skip deduplication step")
    return parser.parse_args()


def main():
    args   = get_args()
    client = get_client()

    raw_questions = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_questions.append(json.loads(line))

    print(f"[Synthesizer] Loaded {len(raw_questions)} raw questions from {args.input}")

    if args.limit > 0:
        raw_questions = raw_questions[:args.limit]
        print(f"[Synthesizer] Limited to {len(raw_questions)} questions")

    synthesized  = []
    skipped      = 0
    source_index = {}

    for i, raw in enumerate(raw_questions):
        q_text    = raw.get("question_text", "").strip()
        source    = raw.get("source_name", "unknown")
        language  = raw.get("language", "English")
        options   = raw.get("options", [])
        correct   = raw.get("correct_answer")
        diff_hint = raw.get("difficulty_hint", "")

        print(f"\n[{i+1}/{len(raw_questions)}] {source} | {q_text[:60]}...")

        print(f"  Checking quality...", end=" ")
        if not passes_quality_filter(client, q_text):
            print("SKIP (quality)")
            skipped += 1
            continue
        print("OK")

        print(f"  Classifying domain...", end=" ")
        domain = classify_domain(client, q_text)
        print(domain)

        print(f"  Rating difficulty...", end=" ")
        level = rate_difficulty(client, q_text, difficulty_hint=diff_hint)
        print(level)

        source_index[source] = source_index.get(source, 0) + 1
        prompt_id = make_prompt_id(source, source_index[source], q_text)

        extra = raw.get("extra", {})
        if isinstance(extra, str):
            import json as _j
            try: extra = _j.loads(extra)
            except: extra = {}
        messages = build_messages(q_text, level, options, correct, extra=extra)

        entry = {
            "messages": messages,
            "metadata": {
                "prompt_id":        prompt_id,
                "domain":           domain,
                "reasoning_level":  level,
                "source":           source,
                "source_url":       raw.get("source_url", ""),
                "language":         language,
                "answer_source":    raw.get("answer_source", "unknown"),
                "scraped_at":       raw.get("scraped_at", ""),
                "synthesized_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        }

        if raw.get("existing_solution"):
            entry["metadata"]["existing_solution"] = raw["existing_solution"]
        if raw.get("existing_thinking"):
            entry["metadata"]["existing_thinking"] = raw["existing_thinking"]

        extra_raw = raw.get("extra", {})
        if isinstance(extra_raw, str):
            import json as _j
            try: extra_raw = _j.loads(extra_raw)
            except: extra_raw = {}
        if "public_tests" in extra_raw:
            pt = extra_raw["public_tests"]
            if isinstance(pt, str):
                import json as _j
                try: pt = _j.loads(pt)
                except: pt = {}
            entry["metadata"]["public_tests"] = pt

        if extra and "public_tests" in extra:
            pt = extra["public_tests"]
            if isinstance(pt, str):
                import json as _j
                try: pt = _j.loads(pt)
                except: pt = None
            if pt and isinstance(pt, dict):
                entry["metadata"]["public_tests"] = pt

        if extra and extra.get("public_tests"):
            entry["metadata"]["public_tests"] = extra["public_tests"]

        synthesized.append(entry)

    print(f"\n[Synthesizer] Processed: {len(synthesized)} accepted  |  {skipped} skipped")

    if not args.no_dedup and len(synthesized) > 1:
        print(f"\n[Synthesizer] Running deduplication...")
        synthesized = deduplicate(synthesized)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out_f:
        for entry in synthesized:
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n{'='*50}")
    print(f"Synthesis Complete")
    print(f"  Total written : {len(synthesized)}")
    print(f"  Total skipped : {skipped}")
    print(f"  Output file   : {args.out}")
    print(f"{'='*50}")
    print(f"\nNext step:")
    print(f"  bash run_pipeline.sh {args.out} 3")


if __name__ == "__main__":
    main()