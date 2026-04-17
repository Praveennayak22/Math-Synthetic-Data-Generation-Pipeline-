# Code Synthetic Data Pipeline 💻

A comprehensive data processing pipeline for generating and refining high-quality synthetic code training datasets. This pipeline integrates multiple code problem sources (Hugging Face, Excel scraping, local datasets), generates problem variants, creates AI-generated solutions with reasoning traces, executes and validates code, and packages everything into SFT (Supervised Fine-Tuning) and RL (Reinforcement Learning) training datasets.

## Overview

The pipeline transforms diverse code problem datasets into a unified, high-quality training corpus through 7 sequential processing stages:

```
Stage 0B: HuggingFace Fetcher (HF mode) — Load dataset from Hub
          ↓
Stage 0C: Synthesizer — Domain classification & difficulty rating
          ↓
Stage 1:  Question Variant Generator — Create diverse problem variants
          ↓
Stage 2:  Brain (Completion) — AI model generates solutions + reasoning
          ↓
Stage 3:  Execute Sandbox — Run & test generated code
          ↓
Stage 4:  Pack Dataset — Serialize to SFT/RL training format
```

## Features

- **Multi-source dataset ingestion**: 
  - Hugging Face Hub datasets
  - Local JSONL seed files
  
- **Problem variant generation**: Creates diverse code problems from seeds with:
  - Mutation-based difficulty manipulation
  - Domain injection for real-world context
  - Negative guardrails to prevent seed reuse
  - Temperature-tuned generation (brainstorming → precision)

- **Code reasoning annotation**: 
  - AI-generated step-by-step solutions with `<think>` reasoning tokens
  - Configurable model backends (DeepSeek, OpenAI, custom)
  - Parallel batch processing

- **Code execution & verification**: 
  - Local sandbox execution with timeout protection
  - Test case validation
  - Safety scanning (blocks dangerous operations: `os.system`, `exec`, `socket`, etc.)
  - Rejection sampling for correctness

- **Semantic deduplication**: Remove near-duplicate problems using sentence embeddings

- **Complexity scoring**: Auto-assign 0-4 reasoning levels based on problem structure

- **Batch processing**: SLURM-ready orchestration with traceability
  - Per-batch isolation prevents race conditions
  - Dataset-organized output structure
  - Mergeable results across batches

- **Distributed processing**: Cluster-optimized with configurable parallelism

## Input Data Formats

### Mode 1: Hugging Face Datasets
Any HuggingFace dataset with `problem` and optionally `solution` fields:
```python
{
    "problem": str,           # Code problem statement
    "solution": str,          # (optional) Existing solution
    "reasoning": str,         # (optional) Existing reasoning
}
```

### Mode 2: Excel Files
Excel spreadsheet with columns: `title`, `description`, `url`, etc.
```
Title          | Description            | URL
Fibonacci      | Compute nth Fibonacci  | https://...
Binary Search  | Find element in array  | https://...
```

### Mode 3: Local JSONL (Classic)
Pre-prepared seeds file format:
```jsonl
{"prompt_id": "seed_001", "problem": "...", "reasoning_level": 2, "domain": "algorithms"}
{"prompt_id": "seed_002", "problem": "...", "reasoning_level": 3, "domain": "data-structures"}
```

## Output Data Formats

### SFT (Supervised Fine-Tuning) Format
```jsonl
{
    "messages": [
        {"role": "user", "content": "{problem_statement}"},
        {"role": "assistant", "content": "<think>..reasoning..</think>\n{solution_code}"}
    ],
    "answer": "code",
    "verified": true,
    "metadata": {
        "source_dataset": "deepmind_code_contests",
        "variant_id": "var_001_trial_1",
        "domain": "algorithms",
        "reasoning_level": 3,
        "model": "deepseek-v3.2",
        "brain_model": "openai/gpt-oss-120b"
    }
}
```

### RL (Reinforcement Learning) Format
Similar to SFT but includes reward/feedback signals in metadata for RL training.

## Installation

### Prerequisites
- **Python**: 3.9+
- **Compute**: 
  - For local runs: GPU recommended (for model inference)
  - For cluster runs: Access to internal node via VPN/tunnel
- **Storage**: ~50GB for cached datasets + outputs
- **Dependencies**: See [requirements.txt](requirements.txt)

### Pre-Flight Checklist

Before starting, ensure you have:
- [ ] Python 3.9+ installed: `python --version`
- [ ] GPU available (optional): `nvidia-smi`
- [ ] ~50GB free disk space: `df -h`
- [ ] HuggingFace account with API token
- [ ] Access to model inference endpoints (TensorStudio/cluster)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd code_generation

# Verify installation prerequisites
python --version              # Should be 3.9+
which pip                     # Should point to correct Python

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Verify venv activation
which python                  # Should be /path/to/venv/bin/python

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import datasets, torch, tqdm; print('✓ Core deps OK')"

# Configure environment variables
cp .env.example .env
# Edit .env with your credentials (see next section):
```

### Environment Configuration

Edit `.env` file with the following variables:

```bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. HuggingFace Dataset Access
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export HF_TOKEN="hf_xxxxxxxxxxxxxxxx"
# Get token: https://huggingface.co/settings/tokens
# Create "Read" token for dataset access
# Set here or run: huggingface-cli login

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Brain Model — Solution Generator (TensorStudio/Internal Cluster)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export TENSORSTUDIO_API_KEY="your_api_key_here"
# Usage: Generates complete solutions with reasoning (Stage 2)
# Source: TensorStudio API or internal cluster endpoint
# Contact: [admin@cluster.org]

export LLM_URL="https://api.tensorstudio.ai/sglang/v1/chat/completions"
# Or for internal cluster:
# export LLM_URL="http://soketlab-node049:30000/v1/chat/completions"

export LLM_MODEL="openai/gpt-oss-120b"
# Options: openai/gpt-oss-120b, other LLM names supported by endpoint

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Generator Model — Variant Creator (Internal Cluster Node)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export GEN_LLM_URL="http://soketlab-node049:30000/v1/chat/completions"
# For direct cluster access (no tunnel needed)
# Generates diverse problem variants from seeds (Stage 1)

export GEN_LLM_MODEL="deepseek-v3.2"
# Model for variant generation with high creativity

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Parallelism Configuration — Tune based on cluster capacity
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export GEN_PARALLEL_SEEDS=8
# Concurrent variant generations per batch
# Recommendation: Match CPU cores (8-16 for typical cluster node)

export COMPLETION_PARALLEL=8
# Concurrent Brain model API calls
# Recommendation: Match API rate limit / queue size (4-8 typical)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Optional: Caching & Deduplication
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export HF_CACHE_DIR="./data/hf_cache"
# Where to cache downloaded datasets locally

export DEDUP_THRESHOLD=0.85
# Semantic similarity threshold for deduplication (0-1)
# Higher = stricter (0.95 = remove only near-identical)
# Lower = looser (0.7 = remove more variations)

export DEDUP_BATCH_SIZE=128
# Embedding batch size for semantic dedup

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Safety & Validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export SAFETY_SCAN_ENABLED=true
# Enable code safety scanning before execution

export MIN_SOLUTION_LENGTH=50
# Reject solutions shorter than N characters
```

### Verifying Environment Setup

```bash
# Test environment variables are loaded
source .env
echo $GEN_LLM_URL              # Should print cluster URL
echo $TENSORSTUDIO_API_KEY    # Should print key (careful: do not commit!)

# Test Python environment
python -c "import dotenv; print('✓ dotenv available')"

# Test cluster connectivity (if using internal cluster)
curl -X GET http://soketlab-node049:30000/health

# Test HuggingFace token
huggingface-cli whoami

# List cached datasets
du -sh data/hf_cache/         # Show cache size
```

## Configuration

### Step-Specific Settings

Each stage can be customized (add to `.env` or modify in scripts):

- **Stage 0B (HuggingFace)**:
  - `HF_LIMIT=NULL` — Max samples per dataset
  - `HF_CACHE_DIR` — Local cache location

- **Stage 0C (Synthesizer)**:
  - `DEDUP_THRESHOLD=0.85` — Semantic similarity threshold
  - `DEDUP_BATCH_SIZE=128` — Embedding batch size

- **Stage 1 (Variants)**:
  - `NUM_VARIANTS=10` — Variants per seed (default: 3)
  - `MUTATION_DEPTH=3` — Difficulty scaling depth
  - `GEN_TEMPERATURE=0.95` — Problem generation creativity

- **Stage 2 (Brain)**:
  - `COMPLETION_TIMEOUT=60` — Max seconds per completion
  - `COMPLETION_MAX_TOKENS=2048` — Max output length

- **Stage 3 (Execute)**:
  - `SANDBOX_TIMEOUT=10` — Max code execution time
  - `SANDBOX_MEMORY_MB=512` — Max memory per execution

- **Stage 4 (Pack)**:
  - `SAFETY_SCAN_ENABLED=true` — Enable code safety checks
  - `MIN_SOLUTION_LENGTH=50` — Reject short solutions

## Usage

### Verification: Setup Test

Before running real data, verify everything works:

```bash
# Create minimal test input
mkdir -p input output logs
echo '{"prompt_id": "test_001", "problem": "Write a function to add two numbers", "reasoning_level": 1, "domain": "algorithms"}' > input/test_seed.jsonl

# Run just the Synthesizer (Stage 0C) — fastest check
python pipeline/synthesize_seeds.py \
    --input input/test_seed.jsonl \
    --out input/test_seeds.jsonl \
    --limit 1

# Check output exists and has correct format
head input/test_seeds.jsonl | python -m json.tool

# If this succeeds ✓, your environment is correctly configured
```

### Quick Start

#### Mode 1: HuggingFace Dataset

```bash
export BATCH_ID="batch1"
bash run_pipeline.sh --source huggingface deepmind/code_contests 3 500
```

This will:
1. Fetch 500 samples from `deepmind/code_contests` HuggingFace dataset
2. Synthesize and auto-classify them
3. Generate 3 variants per seed
4. Run Brain completion, sandbox execution, and packing
5. Output to: `output/deepmind_code_contests/batch1_{datetime}/`

#### Mode 2: Local Seeds File

```bash
export BATCH_ID="batch1"
bash run_pipeline.sh --source variants input/seeds_batch1.jsonl 3
```

This will:
1. Load pre-prepared seeds from `input/seeds_batch1.jsonl`
2. Generate 3 variants per seed
3. Run full pipeline
4. Output to: `output/{dataset_name}/batch1_variants_batch1/`


### Parallel Batch Processing

Run multiple batches concurrently on cluster:

```bash
# Terminal 1: Batch 1
export BATCH_ID="batch1"
nohup bash run_pipeline.sh --source variants input/seeds_batch1.jsonl 3 > logs/batch1.log 2>&1 &

# Terminal 2: Batch 2
export BATCH_ID="batch2"
nohup bash run_pipeline.sh --source variants input/seeds_batch2.jsonl 3 > logs/batch2.log 2>&1 &

# Terminal 3: Batch 3
export BATCH_ID="batch3"
nohup bash run_pipeline.sh --source variants input/seeds_batch3.jsonl 3 > logs/batch3.log 2>&1 &
```

Each batch runs in isolation:
- Separate input files: `seeds_batch{N}.jsonl`
- Separate outputs: `output/{dataset}/batch{N}_{run_name}/`
- No race conditions or output mixing

### Monitoring Progress

```bash

# Check dataset output as it's being built
wc -l output/*/*/dataset/team2_code_sft.jsonl

# View detailed logs
tail -f logs/batch1.log
```

### Post-Processing: Merge Batches

After all batches complete, combine outputs:

```bash
# Merge all batches for a single dataset
cat output/deepmind_code_contests/*/dataset/team2_code_sft.jsonl \
    > output/final/team2_code_sft_deepmind.jsonl

# Merge all datasets
cat output/*/*/dataset/team2_code_sft.jsonl \
    > output/final/team2_code_sft_all.jsonl

# Count final samples
wc -l output/final/team2_code_sft_all.jsonl

# Verify format (peek at first entry)
head -1 output/final/team2_code_sft_all.jsonl | python -m json.tool
```

### Testing & Validation

#### Quick Sanity Check

```bash
# After running pipeline, validate output format
python -c "
import json
with open('output/final/team2_code_sft_all.jsonl') as f:
    first = json.loads(f.readline())
    assert 'messages' in first
    assert first['verified'] in [True, False]
    assert 'metadata' in first
print('✓ Output format valid')
"
```

#### Inspect Dataset Statistics

```bash
# Count samples by dataset
for dataset in output/*/dataset/team2_code_sft.jsonl; do
    count=$(wc -l < "$dataset")
    name=$(echo "$dataset" | cut -d'/' -f2)
    echo "$name: $count samples"
done

# Check pass/fail ratio
python -c "
import json
with open('output/final/team2_code_sft_all.jsonl') as f:
    verified = sum(1 for line in f if json.loads(line)['verified'])
    total = sum(1 for line in open('output/final/team2_code_sft_all.jsonl'))
    print(f'Pass rate: {verified}/{total} = {100*verified/total:.1f}%')
"
```

#### Validate Sample Content

```bash
# Show sample problem and solution
python -c "
import json
with open('output/final/team2_code_sft_all.jsonl') as f:
    sample = json.loads(f.readline())
    print('Problem:', sample['messages'][0]['content'][:200], '...')
    print('Solution:', sample['messages'][1]['content'][:300], '...')
    print('Verified:', sample['verified'])
    print('Domain:', sample['metadata']['domain'])
    print('Reasoning Level:', sample['metadata']['reasoning_level'])
"
```

## Individual Stage Usage

Run specific stages independently for debugging:

### Stage 0C: Synthesize Seeds Only

```bash
python pipeline/synthesize_seeds.py \
    --input input/raw_questions.jsonl \
    --out input/seeds_prepared.jsonl \
    --limit 100
```

**Options:**
- `--no_dedup` — Skip semantic deduplication
- `--limit N` — Process only first N samples

### Stage 1: Generate Variants Only

```bash
python pipeline/generate_variants.py \
    --input input/seeds_prepared.jsonl \
    --out input/variants_prepared.jsonl \
    --num_variants 5
```

**Options:**
- `--num_variants N` — Variants per seed (default: 3)

### Stage 2: Complete with Brain Model Only

```bash
python pipeline/completion_tensorstudio.py \
    --input input/variants_prepared.jsonl \
    --out output/completions.jsonl
```

### Stage 3: Execute & Test Only

```bash
bash pipeline/execute_sandbox.sh output/completions.jsonl output/sandbox 10
```

**Parameters:**
- `NUM_TRIALS=10` — Trials per problem (default: 3)

### Stage 4: Pack Dataset Only

```bash
python pipeline/pack_dataset.py \
    --sandbox output/sandbox \
    --out output/dataset/team2_code_sft.jsonl
```

**Options:**
- `--include-failed` — Include failed completions (for analysis)

## Cluster Execution (SLURM)

### Prerequisites
- SLURM job scheduler installed
- Access to cluster nodes (via VPN/SSH tunnel if needed)
- Model endpoints running on cluster nodes

### Running on Cluster with nohup

```bash
# Setup for cluster execution
mkdir -p logs output/final

# Create batch
export BATCH_ID="prod_batch1"

# Run in background with output logging
nohup bash run_pipeline.sh \
    --source huggingface \
    deepmind/code_contests 3 500 \
    > logs/batch1_full.log 2>&1 &

# Capture PID for tracking
echo $! > logs/batch1.pid

# Monitor progress in real-time
tail -f logs/batch1_full.log

# Check status later
kill -0 $(cat logs/batch1.pid) && echo "Still running" || echo "Completed"
```

### Creating Custom SLURM Scripts

If your cluster uses SLURM job submission:

```bash
# Create batch_runner.slurm
cat > batch_runner.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=code_pipeline
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%j.log

# Load environment
source .env

# Run pipeline
export BATCH_ID="slurm_batch_${SLURM_JOB_ID}"
bash run_pipeline.sh --source variants input/seeds.jsonl 3

# Merge results
cat output/*/*/dataset/team2_code_sft.jsonl > output/final/team2_code_sft_${SLURM_JOB_ID}.jsonl
EOF

# Submit to SLURM
sbatch batch_runner.slurm
```

## Pipeline Architecture

### Data Flow

```
Raw Input (JSONL/Hugging Face dataset)
    ↓
[Synthesizer] → Seeds (classified, rated, deduplicated)
    ↓
[Variant Generator] → Variants (10 per seed by default)
    ↓
[Brain Model] → Completions (solutions + thinking)
    ↓
[Sandbox Executor] → Results (pass/fail + test outputs)
    ↓
[Pack Dataset] → SFT/RL Training Data
```

### Code Safety

All generated code is safety-scanned before execution. Blocked patterns include:
- System calls: `os.system()`, `subprocess.*`, `os.popen()`
- Dynamic execution: `eval()`, `exec()`, `compile()`
- Dangerous imports: `socket`, `urllib`, `requests`
- File writes: `open(..., 'w')`

See [pack_dataset.py](pipeline/pack_dataset.py) for full safety rules.

### Deduplication

Semantic deduplication removes near-duplicates:
- Uses `sentence-transformers` to embed problems
- Cosine similarity threshold: 0.85 (configurable)
- Keeps highest-quality version (longest, most detailed)

Requires: `pip install sentence-transformers`

## Output Structure

```
output/
├── {dataset_name}/
│   ├── {run_name}/
│   │   ├── sandbox/
│   │   │   ├── trial_0/
│   │   │   │   ├── generated_problem.txt
│   │   │   │   ├── generated_solution.py
│   │   │   │   ├── test_cases.txt
│   │   │   │   ├── test_result.txt          (Pass/Fail)
│   │   │   │   └── reasoning_trace.txt
│   │   │   ├── trial_1/
│   │   │   └── ...
│   │   └── dataset/
│   │       ├── team2_code_sft.jsonl          (SFT training data)
│   │       └── team2_code_rl.jsonl           (RL training data, optional)
│   └── {another_run_name}/
└── final/
    ├── team2_code_sft_all.jsonl               (merged all datasets)
    └── team2_code_sft_{dataset}.jsonl         (per-dataset merge)
```

## Troubleshooting

### Common Issues

**Issue**: "Error: HF_TOKEN not set"
```bash
# Solution: Set Hugging Face token
export HF_TOKEN="hf_xxxxxxxxxxxx"
```

**Issue**: "Error: LLM_URL unreachable"
```bash
# Solution: Check cluster connection
curl -X GET http://soketlab-node049:30000/health
# Or set up SSH tunnel if needed
ssh -L 30000:soketlab-node049:30000 cluster.server.com
```

**Issue**: "Error: CUDA out of memory"
```bash
# Solution: Reduce parallelism
export GEN_PARALLEL_SEEDS=4
export COMPLETION_PARALLEL=4
export SANDBOX_PARALLEL=8
```

**Issue**: "Test execution timeout"
```bash
# Solution: Increase sandbox timeout
export SANDBOX_TIMEOUT=20  # seconds
```

**Issue**: "Deduplication is slow"
```bash
# Solution: Skip deduplication
python pipeline/synthesize_seeds.py \
    --input input/raw.jsonl \
    --out input/seeds.jsonl \
    --no_dedup
```

## Performance Tuning

### For Cluster Execution

- **GEN_PARALLEL_SEEDS**: Match number of CPU cores (typically 8-16)
- **COMPLETION_PARALLEL**: Match API rate limit (typically 4-8)


### Batch Size

- **Smaller batches** (1-2 seeds) → Faster feedback, easier debugging
- **Larger batches** (1000+) → Better resource utilization
- **Recommended**: 10-100 seeds per batch for dev, 500+ for prod

### Hardware Requirements

- **CPU**: 8+ cores (for parallel batch execution)
- **GPU**: Optional (for faster inference if using local models)
- **RAM**: 16GB minimum (64GB recommended for large dedup)
- **Disk**: 100GB+ (for cached models + outputs)

## Dependencies

See [requirements.txt](requirements.txt) for all packages:

```
openai                   # Brain model API client
tqdm                     # Progress bars
pytest pytest-timeout    # Testing framework
pandas pyarrow          # Data manipulation
datasets                # HuggingFace datasets library
sentence-transformers   # Semantic deduplication
numpy openpyxl         # Numerical & Excel support
```

## Contributing & Development

### Running Tests

```bash
pytest -v --timeout=10
```

### Local Testing (without cluster)

1. Use mock models or local inference
2. Set `SANDBOX_TIMEOUT=5` for faster iteration
3. Use `--limit 10` to process small samples

### Extending the Pipeline

To add a new source mode (e.g., GitHub API):

1. Create `pipeline/fetch_github.py` with similar interface
2. Add stage to `run_pipeline.sh` after Stage 0B
3. Ensure output format matches `raw_questions.jsonl`
4. Update this README with usage example

## Publication & Citation

This pipeline was developed as part of the Post-Training Synthetic Data generation efforts. Key papers:

- **KodCode** — Variant generation approach
- **Evol-Instruct** — Difficulty scaling methodology
- **Math-Verify** — Verification techniques applied to code validation

## License

[Specify license: MIT, Apache-2.0, etc.]

## Support

For issues, questions, or feature requests:
- Check [troubleshooting section](#troubleshooting)
- Review stage-specific logs in `logs/`
- Inspect individual sandbox outputs for code execution errors

---

**Last Updated**: April 2026
**Version**: 1.0
**Maintainers**: Team 2 - Code Generation Pipeline
