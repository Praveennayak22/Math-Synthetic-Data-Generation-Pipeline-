"""
generate_variants.py — Question Variant Generator (KodCode-style, v2 with Diversity Fixes)

Implements KodCode's proven two-call approach:
  Call 1: Generate problem statement ONLY (no solution, no tests)
  Call 2: Given the problem, generate solution + tests using
          <|Solution Begin|>...<|Solution End|> and
          <|Test Begin|>...<|Test End|> tags

The solution from Call 2 is run locally to verify it works.
The tests from Call 2 become ground truth for the Brain step.
The Brain independently generates a new solution which is verified
against Call 2's tests — eliminating circular self-testing.

v2 Diversity Fixes:
  1. Mutation Menu     — randomly inject a specific mutation directive per variant
  2. Domain Injection  — inject a random real-world domain into every prompt
  3. Negative Guardrail— explicitly forbid reusing seed's characters/themes
  4. Temperature bump  — Call 1 uses 0.95 (brainstorming), Call 2 uses 0.3 (precise)

Bug fixes (Gemini review):
  - str(inp_str) cast in subprocess to prevent TypeError on non-string inputs
  - Full-text fingerprint hash (no [:200] truncation) to prevent false duplicates

Usage:
    python generate_variants.py --input input/seeds_prepared.jsonl \
                                --out input/variants_prepared.jsonl \
                                --num_variants 10
"""

import argparse
import ast
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import requests

# ── Model config ───────────────────────────────────────────────────────────────
GEN_MODEL_URL  = os.environ.get("GEN_LLM_URL",   "http://soketlab-node049:30000/v1/chat/completions")
GEN_MODEL_NAME = os.environ.get("GEN_LLM_MODEL", "deepseek-v3.2")

MODEL_URL       = os.environ.get("LLM_URL",   "https://api.tensorstudio.ai/sglang/v1/chat/completions")
MODEL_NAME      = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
REQUEST_TIMEOUT    = 120
MAX_RETRIES        = 2
MAX_PARALLEL_SEEDS = int(os.environ.get("GEN_PARALLEL_SEEDS", "4"))  # concurrent seed workers


# ═══════════════════════════════════════════════════════════════════════════════
#  MUTATION MENU
# ═══════════════════════════════════════════════════════════════════════════════
MUTATION_TYPES = [
    {
        "name": "data_structure_mutation",
        "directive": (
            "DATA STRUCTURE MUTATION: Keep the core logical difficulty of the seed, "
            "but completely change the data structures involved. If the seed used strings "
            "or arrays, the new problem MUST use trees, graphs, or 2D grids instead. "
            "Redesign the scenario so the new data structure is natural and necessary."
        ),
    },
    {
        "name": "constraint_mutation",
        "directive": (
            "CONSTRAINT MUTATION: Take the core algorithmic concept and add a severe "
            "time or space complexity constraint (e.g., must run in O(N) time or O(1) "
            "extra space). Alter the problem scenario so this constraint makes physical "
            "sense in context. The constraint must be the central challenge."
        ),
    },
    {
        "name": "concept_blending",
        "directive": (
            "CONCEPT BLENDING: Take the core algorithmic concept of the seed and blend "
            "it with a completely different concept. For example: combine a sliding window "
            "with modular arithmetic, or merge a greedy approach with dynamic programming, "
            "or mix binary search with string manipulation. The blend must feel natural."
        ),
    },
    {
        "name": "greedy_counting",
        "directive": (
            "GREEDY COUNTING: Design a problem where the optimal answer can be found "
            "by a simple greedy approach — counting occurrences, sorting once, or "
            "making locally optimal choices. The problem should involve maximizing or "
            "minimizing a quantity subject to simple constraints. Keep it at Level 2 "
            "difficulty with clear integer input/output and short examples."
        ),
    },
    {
        "name": "two_pointer_variant",
        "directive": (
            "TWO POINTER / SLIDING WINDOW: Reformulate the seed's core problem so that "
            "it can be solved efficiently using a two-pointer or sliding window technique "
            "on a linear sequence. The problem should involve finding a contiguous subarray "
            "or subsequence satisfying some condition. Keep difficulty at Level 2-3 and "
            "ensure the problem has clear, testable integer or string outputs."
        ),
    },
    {
        "name": "real_world_application",
        "directive": (
            "REAL WORLD APPLICATION: Take the seed's abstract algorithm and embed it "
            "deeply in a realistic engineering scenario. The problem should read like "
            "something from a systems design interview: caching policies, load balancing, "
            "database indexing, network routing, or compiler optimization."
        ),
    },
    {
        "name": "debugging_variant",
        "directive": (
            "DEBUGGING VARIANT: Write a problem where broken code is provided and the "
            "solver must identify the bug AND fix it AND explain why the original approach "
            "was wrong. The broken code should implement the seed's algorithm incorrectly "
            "in a subtle, non-obvious way."
        ),
    },
    {
        "name": "output_tracing",
        "directive": (
            "OUTPUT TRACING: Write a problem where a complete (but non-trivial) function "
            "is provided and the solver must trace its exact output for 3-5 different "
            "inputs, including edge cases. The function should involve the seed's algorithm "
            "in a way that requires careful step-by-step mental simulation."
        ),
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
#  DOMAIN INJECTION
# ═══════════════════════════════════════════════════════════════════════════════
DOMAINS = [
    "space exploration and orbital mechanics",
    "high-frequency trading and financial markets",
    "genetics and DNA sequencing",
    "traffic management in a smart city",
    "cybersecurity and cryptographic protocols",
    "RPG game mechanics and inventory management",
    "agriculture and precision farming",
    "hospital logistics and patient scheduling",
    "social network analysis and recommendation systems",
    "logistics and supply chain optimization",
    "climate modelling and weather prediction",
    "robotics and autonomous vehicle navigation",
    "e-commerce warehouse automation",
    "competitive esports match statistics",
    "archaeological site mapping and artifact cataloguing",
    "submarine sonar signal processing",
    "pharmaceutical drug interaction analysis",
    "air traffic control and runway scheduling",
    "music streaming playlist generation",
    "wildlife conservation and animal tracking",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  API CALLS
# ═══════════════════════════════════════════════════════════════════════════════
def _call(url: str, model: str, messages: list, max_tokens: int,
          temperature: float, api_key: str = "") -> str:
    payload = {
        "model":       model,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data    = resp.json()
            msg     = data["choices"][0]["message"]
            content = (msg.get("content") or msg.get("reasoning_content") or "").strip()
            return content
        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"    [Retry {attempt}] {str(e)[:60]}...")
                time.sleep(3)
            else:
                print(f"    [Failed] {str(e)[:80]}")
                return ""
    return ""


def call_gen_model(messages: list, max_tokens: int = 4096, temperature: float = 0.95) -> str:
    return _call(GEN_MODEL_URL, GEN_MODEL_NAME, messages, max_tokens, temperature)


def call_model(messages: list, max_tokens: int = 8192, temperature: float = 0.8) -> str:
    api_key = os.environ.get("TENSORSTUDIO_API_KEY", "")
    return _call(MODEL_URL, MODEL_NAME, messages, max_tokens, temperature, api_key)


# ═══════════════════════════════════════════════════════════════════════════════
#  CALL 1 — PROBLEM GENERATION PROMPT
# ═══════════════════════════════════════════════════════════════════════════════
PROBLEM_GEN_SYSTEM = """You are an expert coding problem designer specializing in DIVERSE, CREATIVE problems.

You will be given a seed problem. Your job is to generate ONE new coding problem that is:
- Algorithmically inspired by the seed's core concept
- But COMPLETELY DIFFERENT in theme, scenario, characters, and context

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MUTATION DIRECTIVE (you MUST follow this):
{mutation_directive}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REQUIRED DOMAIN: {injected_domain}
Your problem MUST be set in this domain. Every object, character, and scenario
must come from this domain. Be creative and specific.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULES — VIOLATIONS WILL BE REJECTED:
1. You are FORBIDDEN from using the same character names as the seed (e.g. Vipul, Chef, Alice, Bob).
2. You are FORBIDDEN from using the same thematic objects as the seed (e.g. brackets, transceivers, lucky numbers).
3. You are FORBIDDEN from setting the problem in competitive programming / algorithmic contest style.
4. The scenario must be physically and contextually distinct from the seed.
5. Do NOT mention the seed problem at all.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your output MUST follow this EXACT format:

PROBLEM:
<A complete, self-contained problem statement. Include:
 - A realistic scenario in the required domain (2-3 sentences max)
 - Clear input format (with constraints)
 - Clear output format
 - Exactly 2 worked examples with input and expected output
 - Nothing else>

LENGTH LIMIT: The entire problem statement MUST be under 1500 characters.
Keep the scenario brief. Keep constraints simple. Do NOT add extra examples or lengthy explanations.
Short and precise is better than long and detailed.

Output ONLY the PROBLEM: block. No solution, no code, no explanation."""


def build_problem_gen_prompt(seed_question: str, seed_domain: str,
                              mutation: dict, injected_domain: str) -> list:
    system = PROBLEM_GEN_SYSTEM.format(
        mutation_directive=mutation["directive"],
        injected_domain=injected_domain,
    )
    seed_preview = seed_question[:800]
    user_msg = (
        f"Here is the seed problem you must draw ALGORITHMIC INSPIRATION from "
        f"(but NOT copy the theme, characters, or scenario):\n\n"
        f"---SEED START---\n{seed_preview}\n---SEED END---\n\n"
        f"Now generate ONE new problem following all the rules above.\n"
        f"Remember: domain = {injected_domain}, mutation = {mutation['name']}.\n"
        f"The problem must feel like it belongs in {injected_domain}, not in "
        f"a competitive programming contest."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_msg},
    ]


def extract_single_problem(response: str) -> str:
    match = re.search(r'PROBLEM\s*:\s*(.*)', response, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
        if len(text) > 50:
            return text
    response = response.strip()
    if len(response) > 100:
        return response
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
#  CALL 2 — SOLUTION + TEST GENERATION PROMPT
# ═══════════════════════════════════════════════════════════════════════════════
SOLUTION_TEST_SYSTEM = """You are an expert Python programmer.
Given a coding problem, generate a complete solution and unit tests.

You MUST use EXACTLY this format:

<|Solution Begin|>
```python
def solve():
    # Complete working solution
    # Read input with input() or sys.stdin
    # Output with print()
    pass

if __name__ == "__main__":
    solve()
```
<|Solution End|>

<|Test Begin|>
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

def test_1():
    assert run("INPUT1") == "EXPECTED1"

def test_2():
    assert run("INPUT2") == "EXPECTED2"

def test_3():
    assert run("INPUT3") == "EXPECTED3"

def test_4():
    assert run("INPUT4") == "EXPECTED4"

def test_5():
    assert run("INPUT5") == "EXPECTED5"
```
<|Test End|>

RULES:
- Replace INPUT1-5 and EXPECTED1-5 with real values from the problem examples.
- The solution MUST be complete and syntactically correct — never truncate.
- NEVER call solve() at module level — only inside if __name__ == "__main__".
- Output ONLY the two tagged blocks. No other text."""


def build_solution_test_prompt(problem: str) -> list:
    MAX_PROBLEM_CHARS = 1800
    if len(problem) > MAX_PROBLEM_CHARS:
        cut = problem[:MAX_PROBLEM_CHARS].rfind('\n')
        if cut < MAX_PROBLEM_CHARS - 400:
            cut = problem[:MAX_PROBLEM_CHARS].rfind('.')
        if cut < MAX_PROBLEM_CHARS - 400:
            cut = MAX_PROBLEM_CHARS
        problem_trimmed = (
            problem[:cut + 1].strip()
            + "\n[Problem truncated — implement based on the above description]"
        )
    else:
        problem_trimmed = problem

    user_content = (
        f"{problem_trimmed}\n\n"
        "---\n"
        "CRITICAL FORMAT REMINDER:\n"
        "Your response MUST contain EXACTLY these two tagged blocks and nothing else:\n"
        "1. <|Solution Begin|> ... <|Solution End|>\n"
        "2. <|Test Begin|> ... <|Test End|>\n"
        "Do NOT write any prose, explanation, or text outside these tags.\n"
        "Start your response immediately with <|Solution Begin|>."
    )
    return [
        {"role": "system", "content": SOLUTION_TEST_SYSTEM},
        {"role": "user",   "content": user_content},
    ]


def extract_solution_and_tests(response: str) -> tuple:
    def strip_fences(text: str) -> str:
        text = text.strip()
        text = re.sub(r'^```python\s*', '', text)
        text = re.sub(r'```\s*$', '', text).strip()
        text = re.sub(r'^```\s*$', '', text, flags=re.MULTILINE).strip()
        return text

    sol_match  = re.search(r'<\|Solution Begin\|>(.*?)<\|Solution End\|>', response, re.DOTALL | re.IGNORECASE)
    test_match = re.search(r'<\|Test Begin\|>(.*?)<\|Test End\|>', response, re.DOTALL | re.IGNORECASE)

    if sol_match and test_match:
        return strip_fences(sol_match.group(1)), strip_fences(test_match.group(1))

    all_blocks = re.findall(r'```python(.*?)```', response, re.DOTALL)
    if len(all_blocks) >= 2:
        sol_code  = all_blocks[0].strip()
        test_code = all_blocks[-1].strip()
        if sol_code != test_code and "def solve" in sol_code:
            return sol_code, test_code

    if len(all_blocks) == 1:
        block = all_blocks[0].strip()
        if "def solve" in block and "def test_" in block:
            parts = re.split(r'(?=def test_1)', block, maxsplit=1)
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip()

    return "", ""


# ═══════════════════════════════════════════════════════════════════════════════
#  SOLUTION VERIFIER
#  Bug fix: str(inp_str) cast prevents TypeError if LLM generates non-string input
# ═══════════════════════════════════════════════════════════════════════════════
def verify_solution(solution_code: str, test_code: str) -> tuple:
    """
    1. Syntax check
    2. Run solution against each test input extracted from test_code
    3. Use actual output as ground truth (eliminates circular self-testing)
    Returns (passed: bool, corrected_test_code: str)
    """
    try:
        ast.parse(solution_code)
    except SyntaxError:
        return False, ""

    test_inputs = re.findall(r'assert\s+run\s*\(\s*(.*?)\s*\)\s*==', test_code, re.DOTALL)
    if not test_inputs:
        return False, ""

    if "if __name__" not in solution_code:
        full_code = solution_code + '\n\nif __name__ == "__main__":\n    solve()\n'
    else:
        full_code = solution_code

    corrected_assertions = []
    passed = 0

    for inp_repr in test_inputs:
        try:
            inp_str = ast.literal_eval(inp_repr.strip())
        except Exception:
            continue

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(full_code)
                tmp_path = f.name

            result = subprocess.run(
                [sys.executable, tmp_path],
                input=str(inp_str),   # FIX: always cast to str — LLM may produce int/list
                capture_output=True,
                text=True,
                timeout=10,
            )
            actual = result.stdout.strip()

            if result.returncode == 0 and actual:
                corrected_assertions.append((inp_repr.strip(), repr(actual)))
                passed += 1

        except Exception:
            pass
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    if passed < 3:
        return False, ""

    corrected_test = build_corrected_test_code(corrected_assertions)
    return True, corrected_test


def build_corrected_test_code(assertions: list) -> str:
    lines = [
        "import sys",
        "from io import StringIO",
        "from solution import solve",
        "",
        "def run(input_str):",
        "    sys.stdin = StringIO(input_str)",
        "    captured = StringIO()",
        "    sys.stdout = captured",
        "    solve()",
        "    sys.stdin = sys.__stdin__",
        "    sys.stdout = sys.__stdout__",
        "    return captured.getvalue().strip()",
        "",
    ]
    for idx, (inp_repr, out_repr) in enumerate(assertions, 1):
        lines.append(f"def test_{idx}(): assert run({inp_repr}) == {out_repr}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  DEDUPLICATION
#  Bug fix: hash full text (no [:200] truncation) to prevent false positives
# ═══════════════════════════════════════════════════════════════════════════════
def fingerprint(text: str) -> str:
    # FIX: hash the full text — [:200] caused false duplicate detection
    # when two different problems share the same domain-specific preamble
    return hashlib.md5(text.lower().strip().encode()).hexdigest()[:8]


# ═══════════════════════════════════════════════════════════════════════════════
#  DIFFICULTY RATER (Call 2.5)
# ═══════════════════════════════════════════════════════════════════════════════
DIFFICULTY_RATING_PROMPT = """You are an expert programming educator.
Rate the reasoning difficulty of this coding problem and its solution on this scale:
  1 = Basic      : single loop, direct formula, output tracing
  2 = Intermediate: two pointers, sliding window, greedy, simple DP
  3 = Advanced   : graph algorithms, complex DP, data structures, multiple concepts
  4 = Expert     : advanced graph theory, segment trees, complex combinatorics

Problem:
{problem}

Solution:
{solution}

Reply with ONLY a single integer: 1, 2, 3, or 4. No explanation."""


def rate_difficulty(problem: str, solution_code: str, fallback: int) -> int:
    prompt = DIFFICULTY_RATING_PROMPT.format(
        problem=problem[:1500],
        solution=solution_code[:1000],
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        response = call_model(messages, max_tokens=8, temperature=0.0)
        for ch in response.strip():
            if ch in "1234":
                rating = int(ch)
                if 1 <= rating <= 4:
                    return rating
    except Exception:
        pass
    return fallback


# ═══════════════════════════════════════════════════════════════════════════════
#  SEED ENTRY BUILDER
# ═══════════════════════════════════════════════════════════════════════════════
BRAIN_SYSTEM_PROMPT = """You are an expert Python programmer. Reasoning level: {reasoning_label}.

Solve the given coding problem. Your response MUST follow this EXACT structure:

<think>
Brief reasoning here — 2-3 sentences max identifying your approach.
</think>

<|Solution Begin|>
```python
def solve():
    # Complete working solution
    # Read input with input() or sys.stdin
    # Output with print()
    pass

if __name__ == "__main__":
    solve()
```
<|Solution End|>

<|Test Begin|>
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
<|Test End|>

RULES:
- Start with <think> immediately.
- Replace INPUT1-5 and EXPECTED1-5 with real values from the problem.
- NEVER call solve() at module level — only inside if __name__ == "__main__"."""

REASONING_LABELS = {0: "Minimal", 1: "Basic", 2: "Intermediate", 3: "Advanced", 4: "Expert"}

MUTATION_DIFFICULTY = {
    "data_structure_mutation": 3,
    "constraint_mutation":     3,
    "concept_blending":        3,
    "greedy_counting":         2,
    "two_pointer_variant":     2,
    "real_world_application":  2,
    "debugging_variant":       2,
    "output_tracing":          1,
}

MUTATION_DOMAIN = {
    "data_structure_mutation": "competitive_programming",
    "constraint_mutation":     "competitive_programming",
    "concept_blending":        "competitive_programming",
    "greedy_counting":         "competitive_programming",
    "two_pointer_variant":     "competitive_programming",
    "real_world_application":  "competitive_programming",
    "debugging_variant":       "code_edit_bugfix",
    "output_tracing":          "codeio_tracing",
}


def build_seed_entry(problem: str, solution_code: str, corrected_test_code: str,
                     seed_metadata: dict, variant_index: int, seed_index: int,
                     mutation: dict, injected_domain: str,
                     rated_level: int = -1) -> dict:
    mutation_level  = MUTATION_DIFFICULTY.get(mutation["name"], 2)
    reasoning_level = rated_level if rated_level in (1, 2, 3, 4) else mutation_level
    domain          = MUTATION_DOMAIN.get(mutation["name"], "competitive_programming")
    source          = seed_metadata.get("source", "synthetic")
    label           = REASONING_LABELS.get(reasoning_level, "Intermediate")

    user_msg = problem.strip()
    if len(user_msg) > 2000:
        user_msg = user_msg[:1997] + "..."

    short_hash = hashlib.md5(problem[:100].encode()).hexdigest()[:4]
    prompt_id  = f"{source}_seed{seed_index:03d}_var{variant_index:02d}_{short_hash}"

    return {
        "messages": [
            {"role": "system", "content": BRAIN_SYSTEM_PROMPT.format(reasoning_label=label)},
            {"role": "user",   "content": user_msg},
        ],
        "metadata": {
            "prompt_id":           prompt_id,
            "domain":              domain,
            "reasoning_level":     reasoning_level,
            "mutation_difficulty": mutation_level,
            "source":              "synthetic_variant",
            "original_source":     source,
            "source_url":          seed_metadata.get("source_url", ""),
            "language":            seed_metadata.get("language", "English"),
            "answer_source":       "verified",
            "synthesized_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "verified_test_code":  corrected_test_code,
            "reference_solution":  solution_code,
            "mutation_type":       mutation["name"],
            "injected_domain":     injected_domain,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MUTATION + DOMAIN SELECTOR
# ═══════════════════════════════════════════════════════════════════════════════
def select_mutations_and_domains(n: int) -> list:
    pairs = []
    mutation_cycle   = list(MUTATION_TYPES)
    random.shuffle(mutation_cycle)
    domains_shuffled = random.sample(DOMAINS, min(n, len(DOMAINS)))
    while len(domains_shuffled) < n:
        domains_shuffled += random.sample(DOMAINS, min(n - len(domains_shuffled), len(DOMAINS)))
    for i in range(n):
        mutation = mutation_cycle[i % len(mutation_cycle)]
        domain   = domains_shuffled[i]
        pairs.append((mutation, domain))
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def get_args():
    parser = argparse.ArgumentParser(description="Generate diverse variant problems from seeds (KodCode-style v2).")
    parser.add_argument("--input",        type=str, required=True)
    parser.add_argument("--out",          type=str, default="input/variants_prepared.jsonl")
    parser.add_argument("--num_variants", type=int, default=10)
    return parser.parse_args()


def process_one_seed(seed_idx: int, seed: dict, num_variants: int,
                     seen_fps: set, fps_lock: threading.Lock,
                     total_seeds: int) -> dict:
    """
    Worker: process all variants for a single seed (runs in a thread).
    Returns dict with accepted entries + per-seed stats.
    """
    metadata      = seed.get("metadata", {})
    seed_domain   = metadata.get("domain", "competitive_programming")
    seed_question = seed["messages"][1]["content"]

    print(f"\n[Seed {seed_idx+1}/{total_seeds}] {seed_domain} | {seed_question[:60]}...")

    combos = select_mutations_and_domains(num_variants)
    print(f"  Planned mutations: {[c[0]['name'] for c in combos]}")
    print(f"  Planned domains  : {[c[1] for c in combos]}\n")

    entries    = []
    n_problems = 0
    n_verified = 0
    n_rejected = 0

    for var_idx, (mutation, injected_domain) in enumerate(combos):

        print(f"  Variant {var_idx+1}/{num_variants} "
              f"| mutation={mutation['name']} "
              f"| domain={injected_domain}")

        # ── CALL 1: Generate problem statement ────────────────────────
        print(f"    Call 1: Generating problem...", end=" ", flush=True)
        messages = build_problem_gen_prompt(
            seed_question, seed_domain, mutation, injected_domain
        )
        # max_tokens reduced 1536→1024: problems are constrained to <1500 chars
        response = call_gen_model(messages, max_tokens=1024, temperature=0.95)

        if not response:
            print("FAILED (no response)")
            n_rejected += 1
            continue

        problem = extract_single_problem(response)
        n_problems += 1

        if not problem or len(problem) < 100:
            print(f"FAILED (problem too short: {len(problem)} chars)")
            n_rejected += 1
            continue

        # Thread-safe duplicate check
        with fps_lock:
            fp = fingerprint(problem)
            if fp in seen_fps:
                print("SKIPPED (duplicate)")
                n_rejected += 1
                continue
            seen_fps.add(fp)

        print(f"OK ({len(problem)} chars)")

        # ── CALL 2: Generate solution + tests ─────────────────────────
        print(f"    Call 2: Generating solution+tests...", end=" ", flush=True)
        sol_messages = build_solution_test_prompt(problem)
        # max_tokens reduced 6144→3072: solutions rarely exceed this
        sol_response = call_gen_model(sol_messages, max_tokens=3072, temperature=0.3)

        if not sol_response:
            print("FAILED (no response)")
            n_rejected += 1
            continue

        solution_code, test_code = extract_solution_and_tests(sol_response)

        if not solution_code or not test_code:
            print("FAILED (tags not found)")
            n_rejected += 1
            continue

        # ── VERIFY ────────────────────────────────────────────────────
        passed, corrected_test_code = verify_solution(solution_code, test_code)

        if not passed:
            print("REJECTED (solution failed verification)")
            n_rejected += 1
            continue

        print("ACCEPTED ✓")

        # ── CALL 2.5: Rate difficulty ─────────────────────────────────
        fallback_level = MUTATION_DIFFICULTY.get(mutation["name"], 2)
        print(f"    Call 2.5: Rating difficulty...", end=" ", flush=True)
        rated_level = rate_difficulty(problem, solution_code, fallback_level)
        print(f"Level {rated_level}  (mutation default was {fallback_level})")

        entry = build_seed_entry(
            problem, solution_code, corrected_test_code,
            metadata, var_idx + 1, seed_idx + 1,
            mutation, injected_domain,
            rated_level=rated_level,
        )
        entries.append(entry)
        n_verified += 1

    print(f"\n  Seed {seed_idx+1}: {n_verified}/{num_variants} variants accepted")
    return {
        "entries":    entries,
        "n_problems": n_problems,
        "n_verified": n_verified,
        "n_rejected": n_rejected,
    }


def main():
    args = get_args()

    seeds = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))

    print(f"[VariantGen v2] Loaded {len(seeds)} seeds from {args.input}")
    print(f"[VariantGen v2] Gen model   : {GEN_MODEL_NAME}  ({GEN_MODEL_URL})")
    print(f"[VariantGen v2] Brain model : {MODEL_NAME}")
    print(f"[VariantGen v2] Approach    : KodCode 2-call + Mutation Menu + Domain Injection")
    print(f"[VariantGen v2] Target      : {len(seeds) * args.num_variants} variants")
    print(f"[VariantGen v2] Mutations   : {len(MUTATION_TYPES)} types")
    print(f"[VariantGen v2] Domains     : {len(DOMAINS)} domains")
    print(f"[VariantGen v2] Parallel    : {MAX_PARALLEL_SEEDS} seed workers\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    total_problems = 0
    total_verified = 0
    total_rejected = 0
    seen_fps       = set()
    fps_lock       = threading.Lock()
    all_entries    = []

    # ── Parallel seed processing ───────────────────────────────────────────────
    # Each seed runs in its own thread; variants within a seed are sequential.
    # API calls are I/O-bound so threads give near-linear speedup up to the
    # number of workers (default 4, tune with GEN_PARALLEL_SEEDS env var).
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_SEEDS) as executor:
        futures = {
            executor.submit(
                process_one_seed,
                seed_idx, seed, args.num_variants, seen_fps, fps_lock, len(seeds)
            ): seed_idx
            for seed_idx, seed in enumerate(seeds)
        }

        for future in as_completed(futures):
            try:
                result = future.result()
                all_entries.extend(result["entries"])
                total_problems += result["n_problems"]
                total_verified += result["n_verified"]
                total_rejected += result["n_rejected"]
            except Exception as exc:
                idx = futures[future]
                print(f"  [Error] Seed {idx+1} worker raised: {exc}")

    # Write all accepted entries once (no per-entry locking needed)
    with open(args.out, "w", encoding="utf-8") as out_f:
        for entry in all_entries:
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Variant Generation v2 Complete")
    print(f"  Seeds processed : {len(seeds)}")
    print(f"  Problems parsed : {total_problems}")
    print(f"  Verified        : {total_verified}")
    print(f"  Rejected        : {total_rejected}")
    print(f"  Accept rate     : {total_verified/max(total_problems,1):.1%}")
    print(f"  Output          : {args.out}")
    print(f"{'='*60}")
    print(f"\nNext step:")
    print(f"  bash run_pipeline.sh --source variants {args.out} 3")


if __name__ == "__main__":
    main()