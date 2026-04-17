"""
extract_tir_sandbox.py — The Sorter (Step 3 of Team 2 Code Pipeline)

Reads a *_results{i}.jsonl, extracts the <think> trace, solution,
and tests from each assistant response, and writes per-problem trial folders.

Folder layout:
    <base_out>/<prompt_id>/trial_<N>/
        solution.py
        test_solution.py     ← sys.path trick so pytest finds solution.py
        run_test.sh
        data.json

Usage:
    python extract_tir_sandbox.py \
        --input_file input/variants_test5_run_results0.jsonl \
        --base_out   output/sandbox
"""

import argparse
import json
import os
import re


# ── Extraction helpers ────────────────────────────────────────────────────────

def extract_think(text):
    """
    Extract <think>…</think> trace.
    Falls back to everything before the first code block.
    """
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: grab preamble before first solution block
    split = re.split(r'<\|Solution Begin\|>|```python', text, maxsplit=1, flags=re.IGNORECASE)
    return split[0].strip() if len(split) > 1 else "No reasoning found."


def extract_blocks(text):
    """
    Extract solution and test code blocks.
    Uses KodCode <|Solution Begin|>/<|Test Begin|> tags (primary).
    Falls back to first/last ```python fences if tags not found.
    """
    # Try KodCode tags first
    sol_match = re.search(
        r'<\|Solution Begin\|>(.*?)<\|Solution End\|>',
        text, re.DOTALL | re.IGNORECASE
    )
    test_match = re.search(
        r'<\|Test Begin\|>(.*?)<\|Test End\|>',
        text, re.DOTALL | re.IGNORECASE
    )

    if sol_match and test_match:
        # Strip ```python fences from within tags
        sol_raw  = sol_match.group(1).strip()
        solution = re.sub(r'^```python\s*', '', sol_raw)
        solution = re.sub(r'```\s*$', '', solution).strip()

        test_raw = test_match.group(1).strip()
        tests    = re.sub(r'^```python\s*', '', test_raw)
        tests    = re.sub(r'```\s*$', '', tests).strip()
        return solution, tests

    # Fallback to ```python fences
    all_blocks = re.findall(r'```python(.*?)```', text, re.DOTALL | re.IGNORECASE)
    solution   = all_blocks[0].strip() if all_blocks else "pass"
    tests      = all_blocks[-1].strip() if len(all_blocks) > 1 else ""
    return solution, tests


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract code and think traces into sandbox folders.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to *_results{N}.jsonl")
    parser.add_argument("--base_out",   type=str, default="demo/test_run/sandbox",
                        help="Output sandbox base directory")
    args = parser.parse_args()

    trial_num = "0"
    m = re.search(r'results(\d+)', args.input_file)
    if m:
        trial_num = m.group(1)

    print(f"Extracting trial_{trial_num} from: {args.input_file}")
    count = 0

    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            metadata  = data.get("metadata", {})
            prompt_id = metadata.get("prompt_id", "UNKNOWN")

            think        = extract_think(data["messages"][-1]["content"])
            sol, tests   = extract_blocks(data["messages"][-1]["content"])

            folder = os.path.join(args.base_out, prompt_id, f"trial_{trial_num}")
            os.makedirs(folder, exist_ok=True)

            # solution.py
            with open(os.path.join(folder, "solution.py"), "w") as f_out:
                f_out.write(sol)

            # test_solution.py — prefer public_tests from dataset over GLM-generated tests
            public_tests = metadata.get("public_tests")
            if public_tests and isinstance(public_tests, dict):
                inputs  = public_tests.get("input",  [])
                outputs = public_tests.get("output", [])
                # Build verified tests from dataset's own test cases
                test_lines = [
                    "import sys, os",
                    "sys.path.insert(0, os.path.dirname(__file__))",
                    "from solution import solve",
                    "from io import StringIO",
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
                # Use up to 5 test cases from public_tests
                pairs = list(zip(inputs, outputs))[:5]
                for idx, (inp, out) in enumerate(pairs, 1):
                    inp_repr = repr(inp.strip())
                    out_repr = repr(out.strip())
                    test_lines.append(f"def test_{idx}(): assert run({inp_repr}) == {out_repr}")
                # Pad to 5 tests if fewer public tests available
                for idx in range(len(pairs) + 1, 6):
                    if pairs:
                        inp_repr = repr(pairs[0][0].strip())
                        out_repr = repr(pairs[0][1].strip())
                        test_lines.append(f"def test_{idx}(): assert run({inp_repr}) == {out_repr}")
                tests_content = "\n".join(test_lines)
            else:
                # Fall back to GLM-generated tests
                tests_content = (
                    "import sys, os\n"
                    "sys.path.insert(0, os.path.dirname(__file__))\n"
                    "from solution import *\n\n"
                    + tests
                )

            with open(os.path.join(folder, "test_solution.py"), "w") as f_out:
                f_out.write(tests_content)

            # FIX: Robustly extract the ACTUAL problem text (the user's message)
            actual_prompt = ""
            for msg in data.get("messages", []):
                if msg.get("role") == "user":
                    actual_prompt = msg.get("content", "")
                    break
            if not actual_prompt and len(data.get("messages", [])) > 1:
                actual_prompt = data["messages"][1]["content"]

            # data.json — reasoning_level and domain promoted to top-level
            # so pack_dataset.py can read them without digging into metadata
            with open(os.path.join(folder, "data.json"), "w") as f_out:
                json.dump({
                    "prompt":          actual_prompt,  # <--- FIXED
                    "think_trace":     think,
                    "solution_code":   sol,
                    "test_code":       tests,
                    "metadata":        metadata,
                    "reasoning_level": metadata.get("reasoning_level", -1),
                    "domain":          metadata.get("domain", "unknown"),
                    "messages":        data.get("messages", []) # Preserving this for the packer script
                }, f_out, indent=2)

            # run_test.sh
            sh_path = os.path.join(folder, "run_test.sh")
            with open(sh_path, "w") as f_out:
                f_out.write(
                    '#!/bin/bash\n'
                    '# Always run from the trial folder so imports resolve correctly\n'
                    'cd "$(dirname "$0")"\n'
                    'timeout --kill-after=5s 30 pytest test_solution.py -v --tb=short '
                    '> test_log.log 2>&1\n'
                    'if [ $? -eq 0 ]; then\n'
                    '    echo "Pass" > test_result.txt\n'
                    'else\n'
                    '    echo "Fail" > test_result.txt\n'
                    'fi\n'
                )
            os.chmod(sh_path, 0o755)
            count += 1

    print(f"Done. Created {count} sandboxes → {args.base_out}")

if __name__ == "__main__":
    main()