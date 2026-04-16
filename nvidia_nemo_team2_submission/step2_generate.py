"""
step2_generate.py
Calls model API for every problem in input.jsonl.
Usage: python step2_generate.py --model glm-5-fp8 [--infile input.jsonl] [--outfile output_glm-5-fp8.jsonl]

gpt-oss-120b uses an external URL (api.tensorstudio.ai).
If the cluster cannot reach it, pass --skip-external to skip it cleanly
instead of hanging/failing every single request.
"""
import json
import requests
import time
import os
import argparse

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",          required=True)
parser.add_argument("--infile",         default="input.jsonl")
parser.add_argument("--outfile",        default=None)       # auto-set below
parser.add_argument("--skip-external",  action="store_true",
                    help="Skip gpt-oss-120b (external URL blocked on this cluster)")
args = parser.parse_args()

MODEL_NAME = args.model

# ── model registry ────────────────────────────────────────────────────────
MODELS = {
    "glm-5-fp8":     {"url": "http://soketlab-node064:30000/v1/chat/completions", "key": None,    "external": False},
    "minimax-m2.5":  {"url": "http://soketlab-node043:30000/v1/chat/completions", "key": None,    "external": False},
    "kimi-k2.5":     {"url": "http://soketlab-node019:30000/v1/chat/completions", "key": None,    "external": False},
    "deepseek-v3.2": {"url": "http://soketlab-node017:30000/v1/chat/completions", "key": None,    "external": False},
    "gpt-oss-120b":  {"url": "http://172.17.99.11:30000/v1/chat/completions",
                      "key": "AVTXOTWZab9v8WExZMNcGXdCFPCmon4LQPMWP6iS32w2",                                                           "external": False},
}

SYSTEM_PROMPTS = {
    "glm-5-fp8":     "Reasoning: Low",
    "minimax-m2.5":  "Reasoning: Low",
    "kimi-k2.5":     "You are a math expert. Solve the problem step by step. Write the final answer clearly.",
    "deepseek-v3.2": "You are a math expert. Solve the problem step by step. Write the final answer clearly.",
    "gpt-oss-120b":  "Reasoning: Low",
}

if MODEL_NAME not in MODELS:
    print(f"ERROR: Unknown model '{MODEL_NAME}'. Known: {list(MODELS.keys())}")
    exit(1)

cfg        = MODELS[MODEL_NAME]
API_URL    = cfg["url"]
API_KEY    = cfg["key"]
IS_EXTERNAL = cfg["external"]

# ── graceful skip for blocked external model ──────────────────────────────
if IS_EXTERNAL and args.skip_external:
    print(f"SKIPPING {MODEL_NAME} — --skip-external flag set (cluster cannot reach {API_URL})")
    print("Ask Rajvee for the internal API URL or VPN access, then re-run without --skip-external")
    exit(0)

INFILE  = args.infile
OUTFILE = args.outfile or f"output_{MODEL_NAME}.jsonl"

HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["Authorization"] = f"Bearer {API_KEY}"

SYSTEM_PROMPT = SYSTEM_PROMPTS[MODEL_NAME]

# ── API call ──────────────────────────────────────────────────────────────
def call_api(problem):
    model_id = "openai/gpt-oss-120b" if MODEL_NAME == "gpt-oss-120b" else MODEL_NAME
    payload  = {
        "model":       model_id,
        "messages":    [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Problem: {problem}"},
        ],
        "temperature": 0.1,
        "max_tokens":  16000,
    }
    if MODEL_NAME == "gpt-oss-120b":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=600)
        if resp.status_code == 200:
            msg       = resp.json()["choices"][0]["message"]
            content   = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            full      = f"<think>{reasoning}</think>\n{content}" if reasoning else content
            return full if full.strip() else None
        else:
            print(f"  API error {resp.status_code}: {resp.text[:300]}")
            return None
    except requests.exceptions.Timeout:
        print("  Timeout — skipping")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"  ConnectionError: {e}")
        if IS_EXTERNAL:
            print(f"  Hint: cluster may not reach {API_URL}. Re-run with --skip-external to skip this model.")
        return None
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
        return None

# ── main ──────────────────────────────────────────────────────────────────
with open(INFILE, encoding="utf-8") as f:
    problems = [json.loads(l) for l in f if l.strip()]

print(f"Model   : {MODEL_NAME}")
print(f"URL     : {API_URL}")
print(f"Problems: {len(problems)}")
print(f"Output  : {OUTFILE}")

# resume support — read already-done indices
already_done = set()
if os.path.exists(OUTFILE):
    with open(OUTFILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    already_done.add(json.loads(line)["index"])
                except Exception:
                    pass
    print(f"Resuming — {len(already_done)} already done")
else:
    print("Starting fresh.")

with open(OUTFILE, "a", encoding="utf-8") as out:
    ok = fail = skip = 0
    for item in problems:
        idx = item["index"]
        if idx in already_done:
            skip += 1
            continue

        problem    = item["problem"]
        print(f"[{idx+1}/{len(problems)}] {problem[:60]}...")
        generation = call_api(problem)

        if generation:
            item["generation"] = generation
            ok  += 1
            print(f"  OK ({len(generation)} chars)")
        else:
            item["generation"] = ""
            fail += 1
            print("  FAILED")

        item["model_used"] = MODEL_NAME
        out.write(json.dumps(item, ensure_ascii=False) + "\n")
        out.flush()
        time.sleep(0.3)

print(f"\nDone — OK={ok}  FAILED={fail}  SKIPPED={skip}")
print(f"Saved → {OUTFILE}")