"""
pack_dataset.py — The Shipping Box (Step 4 of Team 2 Code Pipeline)

Walks the sandbox directory, reads test_result.txt from every trial folder,
and packs all verified (Pass) samples into SFT and RL training datasets.

Key features:
  - Exact SFT schema matching the master tracking spreadsheet.
  - Fix: Extracts actual generated problem from messages[1].
  - Fix: Restores <think> tags around the reasoning trace.
  - Removes internal trackers so 'Model' is the last column.
"""

import argparse
import json
import os
import re
import ast
from datetime import datetime
from pathlib import Path

# ── Hardcoded model names — always accurate regardless of env var state ────────
GEN_MODEL_NAME   = "deepseek-v3.2"
BRAIN_MODEL_NAME = "openai/gpt-oss-120b"

# ── Parquet support (optional) ────────────────────────────────────────────────
try:
    import pandas as pd
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
#  CODE SAFETY SCANNER
# ═══════════════════════════════════════════════════════════════════════════════
SAFETY_BANNED = [
    r"\bos\.system\s*\(",
    r"\bsubprocess\s*\.",
    r"\bos\.popen\s*\(",
    r"\bos\.execv\s*\(",
    r"\bos\.spawn",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bcompile\s*\(",
    r"__import__\s*\(",
    r"\bsocket\s*\.",
    r"\burllib\s*\.",
    r"\brequests\s*\.",
    r"\bopen\s*\(.*['\"]w['\"]",
    r"\bshutil\.rmtree\s*\(",
    r"\bos\.remove\s*\(",
    r"\bos\.unlink\s*\(",
    r"shell\s*=\s*True",
    r"\bimport\s+ctypes\b",
    r"\bimport\s+pickle\b",
]

SAFETY_WARNINGS = [
    r"\bimport\s+sys\b",
    r"\bimport\s+os\b",
    r"\bgetattr\s*\(",
    r"\bsetattr\s*\(",
]

def check_code_safety(code: str) -> dict:
    violations = [p for p in SAFETY_BANNED   if re.search(p, code, re.MULTILINE)]
    warnings   = [p for p in SAFETY_WARNINGS if re.search(p, code, re.MULTILINE)]

    syntax_valid = True
    try:
        ast.parse(code)
    except SyntaxError:
        syntax_valid = False

    return {
        "passed":       len(violations) == 0,
        "violations":   violations,
        "warnings":     warnings,
        "syntax_valid": syntax_valid,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PYTEST LOG PARSER
# ═══════════════════════════════════════════════════════════════════════════════
def parse_pytest_log(log_text: str) -> dict:
    tests_passed   = 0
    tests_failed   = 0
    execution_time = None
    failed_test    = None

    passed_match = re.search(r"(\d+) passed", log_text)
    failed_match = re.search(r"(\d+) failed", log_text)
    error_match  = re.search(r"(\d+) error",  log_text)

    if passed_match:
        tests_passed = int(passed_match.group(1))
    if failed_match:
        tests_failed = int(failed_match.group(1))
    elif error_match:
        tests_failed = int(error_match.group(1))

    time_match = re.search(r"in\s+([\d.]+)s", log_text)
    if time_match:
        execution_time = float(time_match.group(1))

    failed_test_match = re.search(r"FAILED\s+test_solution\.py::(test_\w+)", log_text)
    if failed_test_match:
        failed_test = failed_test_match.group(1)

    return {
        "tests_passed":        tests_passed,
        "tests_failed":        tests_failed,
        "execution_time_secs": execution_time,
        "failed_test":         failed_test,
        "full_traceback":      log_text.strip(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PARQUET WRITER
# ═══════════════════════════════════════════════════════════════════════════════
def write_parquet(records: list, path: Path):
    if not PARQUET_AVAILABLE or not records:
        return
    try:
        df = pd.DataFrame(records)
        df.to_parquet(str(path), index=False, engine="pyarrow")
    except Exception as e:
        print(f"[Warning] Parquet write failed for {path}: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  ARGS
# ═══════════════════════════════════════════════════════════════════════════════
def get_args():
    parser = argparse.ArgumentParser(description="Pack verified sandbox results into training datasets.")
    parser.add_argument("--sandbox_dir", type=str, required=True)
    parser.add_argument("--out_dir",     type=str, required=True)
    parser.add_argument("--version",     type=str, default="batch1",
                        help="Batch ID stamped on every entry (e.g. batch1, batch2)")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    args       = get_args()
    out_dir    = Path(args.out_dir)
    version    = args.version
    batch_date = datetime.now().strftime("%Y-%m-%d")

    out_dir.mkdir(parents=True, exist_ok=True)

    sft_output      = out_dir / "team2_code_sft.jsonl"
    rl_output       = out_dir / "team2_code_rl.jsonl"
    rejected_output = out_dir / "team2_code_rejected.jsonl"

    sft_records = []
    rl_records  = []
    rej_records = []

    success_count        = 0
    rejected_count       = 0
    safety_blocked_count = 0

    seen_sft_prompts = set()

    with open(sft_output,      "w", encoding="utf-8") as sft_f, \
         open(rl_output,       "w", encoding="utf-8") as rl_f,  \
         open(rejected_output, "w", encoding="utf-8") as rej_f:

        for root, dirs, files in os.walk(args.sandbox_dir):
            if "test_result.txt" not in files or "data.json" not in files:
                continue

            with open(os.path.join(root, "test_result.txt")) as tr:
                result = tr.read().strip()

            try:
                with open(os.path.join(root, "data.json"), encoding="utf-8") as dj:
                    data = json.load(dj)
            except Exception as e:
                print(f"  [Warning] Skipping corrupted data.json in {root}: {e}")
                continue

            full_log = ""
            log_path = os.path.join(root, "test_log.log")
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as lf:
                    full_log = lf.read()

            parsed        = parse_pytest_log(full_log)
            metadata      = data.get("metadata", {})
            reasoning_lvl = data.get("reasoning_level", metadata.get("reasoning_level", -1))
            prompt_id     = metadata.get("prompt_id", os.path.basename(os.path.dirname(root)))
            trial         = os.path.basename(root)

            solution_code = data.get("solution_code", "")
            safety        = check_code_safety(solution_code)

            if not safety["passed"]:
                safety_blocked_count += 1

            # ── Pack into SFT + RL datasets ───────────────────────────────────
            if result == "Pass" and safety["passed"]:

                # FIX 1: Extract the ACTUAL problem from messages[1] (User role)
                # If messages array exists, grab the user prompt. Otherwise fallback to raw prompt.
                messages_array = data.get("messages", [])
                if len(messages_array) >= 2:
                    real_question = messages_array[1].get("content", "")
                else:
                    real_question = data.get("prompt", "")

                # FIX 2: Restore the <think> tags around the trace
                raw_think   = data.get("think_trace", "").strip()
                think_trace = f"<think>\n{raw_think}\n</think>" if raw_think else ""
                
                instruction    = "Solve the given coding problem. Provide a step-by-step reasoning trace inside <think> tags, followed by the complete Python solution."
                
                # Approximate tokens (word_count * 1.3)
                instr_tokens = int(len(instruction.split()) * 1.3)
                q_tokens     = int(len(real_question.split()) * 1.3)
                ans_tokens   = int(len(solution_code.split()) * 1.3)

                # ── 1. SFT ENTRY (Deduplicated, Tagged as "SFT") ───────────────
                if prompt_id not in seen_sft_prompts:
                    sft_entry = {
                        "Instruction":                       instruction,
                        "Question":                          real_question,
                        "<Think>":                           think_trace,
                        "Answer":                            solution_code,
                        "Difficulty level":                  reasoning_lvl,
                        "Number of tokens in Instruction":   instr_tokens,
                        "Number of tokens in Question":      q_tokens,
                        "Number of tokens in Answer":        ans_tokens,
                        "Task":                              "Code Generation",
                        "Domain":                            "Code",
                        "Language":                          "English",
                        "Source":                            "synthetic_data",
                        "Made by":                           "Team 2",
                        "Reasoning/SFT":                     "Reasoning",
                        "Model":                             f"{GEN_MODEL_NAME} + {BRAIN_MODEL_NAME}"
                    }
                    sft_f.write(json.dumps(sft_entry) + "\n")
                    sft_records.append(sft_entry)
                    seen_sft_prompts.add(prompt_id)

                # ── 2. RL ENTRY (All Trials, Tagged as "Reasoning") ────────────
                rl_entry = {
                    "Instruction":                       instruction,
                    "Question":                          real_question,
                    "<Think>":                           think_trace,
                    "Answer":                            solution_code,
                    "Difficulty level":                  reasoning_lvl,
                    "Number of tokens in Instruction":   instr_tokens,
                    "Number of tokens in Question":      q_tokens,
                    "Number of tokens in Answer":        ans_tokens,
                    "Task":                              "Code Generation",
                    "Domain":                            "Code",
                    "Language":                          "English",
                    "Source":                            "synthetic_data",
                    "Made by":                           "Team 2",
                    "Reasoning/SFT":                     "Reasoning",
                    "Model":                             f"{GEN_MODEL_NAME} + {BRAIN_MODEL_NAME}"
                }
                rl_f.write(json.dumps(rl_entry) + "\n")
                rl_records.append(rl_entry)

                success_count += 1

            else:
                rej_entry = {
                    "prompt_id":         prompt_id,
                    "trial":             trial,
                    "rejection_reason":  f"pytest {result}" if result != "Pass" else "safety_violation",
                    "tests_passed":      parsed["tests_passed"],
                    "tests_failed":      parsed["tests_failed"],
                }
                rej_f.write(json.dumps(rej_entry) + "\n")
                rej_records.append(rej_entry)
                rejected_count += 1

    # ── Parquet output ────────────────────────────────────────────────────────
    if PARQUET_AVAILABLE:
        write_parquet(sft_records, out_dir / "team2_code_sft.parquet")
        write_parquet(rl_records,  out_dir / "team2_code_rl.parquet")
        write_parquet(rej_records, out_dir / "team2_code_rejected.parquet")
        parquet_note = "written"
    else:
        parquet_note = "skipped (pip install pyarrow pandas to enable)"

    # ── Summary ───────────────────────────────────────────────────────────────
    total     = success_count + rejected_count
    pass_rate = success_count / total if total > 0 else 0.0

    print(f"\n{'='*56}")
    print(f"  Packing Complete  [Batch: {version}  |  {batch_date}]")
    print(f"{'='*56}")
    print(f"  Batch ID        : {version}")
    print(f"  Accepted        : {success_count} / {total}  ({pass_rate:.1%})")
    print(f"  SFT unique      : {len(seen_sft_prompts)}  (deduplicated)")
    print(f"  RL entries      : {len(rl_records)}  (all passing trials)")
    print(f"  Rejected        : {rejected_count}")
    print(f"  Safety blocked  : {safety_blocked_count}")
    print(f"  Parquet         : {parquet_note}")
    print(f"{'='*56}\n")

if __name__ == "__main__":
    main()