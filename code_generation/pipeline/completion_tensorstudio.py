"""
completion_tensorstudio.py — The Brain (Step 2 of Team 2 Code Pipeline)

Cluster version: DeepSeek-V3.2 endpoint.
- DeepSeek puts all output in content (not reasoning_content)
- DeepSeek follows structured output instructions reliably
- Single streaming call, no restructuring needed
"""

import argparse
import json
import os
import re
import sys
import time

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

MODEL_URL             = os.environ.get("LLM_URL", "https://api.tensorstudio.ai/sglang/v1/chat/completions")
MODEL_NAME            = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
MODEL_CONTEXT_LIMIT   = 65536   # DeepSeek-V3 context window
CONTEXT_BUFFER        = 200
MAX_COMPLETION_TOKENS = 8192
REQUEST_TIMEOUT       = 180
MAX_RETRIES           = 3
MAX_PARALLEL_TASKS    = int(os.environ.get("COMPLETION_PARALLEL", "4"))  # concurrent Brain calls


def get_args():
    parser = argparse.ArgumentParser(description="DeepSeek Cluster API Call Manager.")
    parser.add_argument("--input_file",  type=str, required=True,
                        help="Input file (must end with prepared.jsonl)")
    parser.add_argument("--num_trials",  type=int, default=1,
                        help="Number of independent Pass@k trials")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    return parser.parse_args()


def estimate_input_tokens(messages: list) -> int:
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return (total_chars // 4) + (len(messages) * 10)


def safe_max_tokens(messages: list) -> int:
    input_estimate = estimate_input_tokens(messages)
    available      = MODEL_CONTEXT_LIMIT - input_estimate - CONTEXT_BUFFER
    return max(512, min(available, MAX_COMPLETION_TOKENS))


def call_api(messages: list, temperature: float, max_tokens: int) -> str:
    """
    Call model endpoint. Returns assembled reply string.
    Supports optional API key via TENSORSTUDIO_API_KEY env var (for GPT-OSS).
    Handles both content-only and dual-channel (content + reasoning_content) responses.
    """
    payload = {
        "model":       MODEL_NAME,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("TENSORSTUDIO_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = requests.post(MODEL_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    msg       = data["choices"][0]["message"]
    content   = (msg.get("content")           or "").strip()
    reasoning = (msg.get("reasoning_content") or "").strip()

    # If both channels have data (e.g. GPT-OSS), prepend reasoning as <think> trace
    if reasoning and content:
        return f"<think>\n{reasoning}\n</think>\n{content}"
    return content or reasoning


def call_with_retry(messages: list, temperature: float, max_tokens: int) -> str | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            reply = call_api(messages, temperature, max_tokens)

            if not reply:
                if attempt < MAX_RETRIES:
                    print(f"\n  [Retry {attempt}/{MAX_RETRIES}] Empty response, retrying...")
                    time.sleep(2 ** attempt)
                    continue
                return None

            return reply

        except Exception as e:
            err_str = str(e)
            if "400" in err_str and "token" in err_str.lower():
                print(f"\n  [Skip] Token limit exceeded.")
                return None
            if attempt < MAX_RETRIES:
                wait = 2 ** attempt
                print(f"\n  [Retry {attempt}/{MAX_RETRIES}] {err_str[:80]}... waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"\n  [Failed] {err_str[:120]}")
                return None

    return None


def main():
    args = get_args()

    # if not args.input_file.endswith("prepared.jsonl"):
    #     print("Error: Input file must end with 'prepared.jsonl'")
    #     sys.exit(1)

    with open(args.input_file, 'r', encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(tasks)} problems from {args.input_file}")
    print(f"Model    : {MODEL_NAME}")
    print(f"Endpoint : {MODEL_URL}")
    print(f"Trials   : {args.num_trials} | Temp: {args.temperature} | Parallel: {MAX_PARALLEL_TASKS} | Timeout: {REQUEST_TIMEOUT}s")

    def _solve_task(task):
        """Call Brain for a single task; returns (task, reply)."""
        messages   = task["messages"]
        max_tokens = safe_max_tokens(messages)
        reply      = call_with_retry(messages, args.temperature, max_tokens)
        return task, reply

    for trial in range(args.num_trials):
        print(f"\n── Trial {trial} ──────────────────────────────────────")
        output_file = args.input_file.replace(".jsonl", f"_results{trial}.jsonl")
        saved   = 0
        skipped = 0
        results = []

        # ── Parallel Brain calls ──────────────────────────────────────────────
        # Each task (problem) gets an independent API call in its own thread.
        # Tune worker count via COMPLETION_PARALLEL env var (default 4).
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_TASKS) as executor:
            futures = {executor.submit(_solve_task, task): task for task in tasks}
            pbar = tqdm(total=len(tasks), desc=f"Trial {trial}")
            for future in as_completed(futures):
                try:
                    task, reply = future.result()
                    pid = task.get("metadata", {}).get("prompt_id", "?")
                    if reply is None:
                        skipped += 1
                        tqdm.write(f"  Skipped: {pid}")
                    else:
                        results.append({
                            "messages": task["messages"] + [{"role": "assistant", "content": reply}],
                            "metadata": task.get("metadata", {}),
                        })
                        saved += 1
                except Exception as exc:
                    skipped += 1
                    tqdm.write(f"  [Error] {exc}")
                pbar.update(1)
            pbar.close()

        with open(output_file, 'w', encoding="utf-8") as out_f:
            for r in results:
                out_f.write(json.dumps(r) + "\n")

        print(f"Saved {saved}/{len(tasks)} → {output_file}  (skipped: {skipped})")


if __name__ == "__main__":
    main()