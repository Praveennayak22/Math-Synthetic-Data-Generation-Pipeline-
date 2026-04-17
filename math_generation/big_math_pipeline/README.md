# Math Synthetic Data Pipeline 🧮

A comprehensive data processing pipeline for generating and refining high-quality synthetic math training datasets. This pipeline integrates multiple open-source math problem datasets, performs difficulty scaling, format conversion, model-based reasoning annotation, and verification.

## Overview

The pipeline transforms diverse math problem datasets into a unified, high-quality training corpus through 7 sequential processing steps:

```
Step 1: Load seed data (NuminaMath-CoT, FineMath, S1K, etc.)
        ↓
Step 2: Evol-Instruct difficulty scaling (Kimi-K2.5 mutations)
        ↓
Step 3: MCQ → open-ended conversion (unify problem formats)
        ↓
Step 4: Kimi-K2.5 teacher reasoning traces (generate <think> tokens)
        ↓
Step 5: Math-Verify validation + rejection sampling (verify correctness)
        ↓
Step 6: Reasoning-level tagging (assign 0-4 complexity scores)
        ↓
Step 7: Serialize to Parquet (final output dataset)
```

## Features

- **Multi-source dataset ingestion**: Supports 10+ math datasets from Hugging Face Hub
- **Difficulty scaling**: Generate harder/easier variants using Evol-Instruct technique
- **Format standardization**: Convert multiple-choice questions to open-ended format
- **Reasoning annotation**: Generate step-by-step reasoning traces via teacher models
- **Answer verification**: Validate math solutions using symbolic computation + rejection sampling
- **Complexity scoring**: Assign 0-4 reasoning level based on problem structure and solution depth
- **Deterministic output**: All outputs include timestamps to prevent data loss on reruns
- **Distributed processing**: SLURM job submission scripts for cluster execution

## Supported Datasets

- **NuminaMath-CoT**: AI-MO/NuminaMath-CoT
- **FineMath**: HuggingFaceTB/finemath
- **S1K**: simplescaling/s1K
- **LIMO**: GAIR/LIMO
- **OmniMath**: KbsdJames/omni-math
- **TwMath**: twinkle-ai/tw-math-reasoning-2k
- **TeleMath**: netop/TeleMath
- **TheoremQA**: TIGER-Lab/theorem_qa
- **AIME**: Contest math problems
- **DeepMath**: Large-scale mathematical reasoning dataset

## Installation

### Pre-Flight Checklist

Before starting, verify you have:

- [ ] **Python 3.9+** — `python --version` should return 3.9 or higher
- [ ] **CUDA 12.0+** (GPU recommended) — `nvidia-smi` should work
- [ ] **100GB+ free disk space** — `df -h` or disk usage in Settings
- [ ] **HuggingFace account** with API token from https://huggingface.co/settings/tokens
- [ ] **Model endpoint access** — cluster or TensorStudio API key
- [ ] **Git installed** — `git --version` should work

### Prerequisites

- **Python**: 3.9+
- **GPU**: CUDA 12.0+ (recommended for inference)
- **Storage**: ~100GB disk space (for cached datasets)
- **Network**: Access to HuggingFace Hub + model endpoint

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/math-pipeline.git
cd math-pipeline

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Verify Python in venv
python --version           # Should be 3.9+

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, datasets, yaml; print('✓ Core deps OK')"

# Configure environment
cp .env.example .env
# Edit .env with credentials (see next section)
```

## Environment Configuration

### .env Setup

Create `.env` file with the following variables:

```bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. HuggingFace Dataset Access
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export HF_TOKEN="hf_xxxxxxxxxxxxxxxx"
# Get token: https://huggingface.co/settings/tokens
# Create a "Read" token for dataset access

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Model API Access
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export MODEL_API_KEY="your_api_key_here"
# Used for teacher model inference
# Obtain from: TensorStudio, OpenAI, or your cluster admin

export MODEL_ENDPOINT="http://soketlab-node054:30000/v1/chat/completions"
# Override endpoint from config.yaml if needed
# For local cluster: http://soketlab-nodeXXX:30000/v1/chat/completions
# For TensorStudio: https://api.tensorstudio.ai/...

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Cache & Storage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export HF_CACHE_DIR="./data/hf_cache"
# Where to store downloaded datasets locally
```

### Verify Environment

```bash
# Load .env and test
source .env
echo "HF_TOKEN: ${HF_TOKEN:0:10}..."    # Show first 10 chars
echo "MODEL_ENDPOINT: $MODEL_ENDPOINT"  # Show endpoint

# Test HuggingFace token
huggingface-cli whoami                  # Should show username

# Test cluster endpoint (if using internal node)
curl -X GET http://soketlab-node054:30000/health
```

## Configuration

Edit `config/config.yaml` to customize pipeline behavior:

```yaml
huggingface:
  token_env: HF_TOKEN              # Environment variable for HF token
  model_api_key_env: MODEL_API_KEY # Environment variable for API key
  cache_dir: data/hf_cache         # Local cache for downloaded datasets

datasets:
  # Enable/disable datasets by setting enabled: true/false
  theoremqa:
    enabled: true                  # Currently enabled dataset
    repo_id: TIGER-Lab/TheoremQA  # HuggingFace dataset ID
    split: test                    # Dataset split (train/test/validation)
    max_samples: null              # null = use entire; or specify count
    math_only: true                # Filter for math problems only
  
  # Other available datasets (set enabled: true to use):
  numina_math:
    enabled: false
    repo_id: AI-MO/NuminaMath-CoT
    max_samples: 200
  
  limo:
    enabled: false
    repo_id: GAIR/LIMO
    max_samples: 817
```

### Model & Pipeline Configuration

```yaml
evol_instruct:
  model: kimi-k2.5                 # Model for difficulty mutations
  num_mutations: 1                 # 1 = fast; 3 = full mutations
  temperature: 0.8                 # Creativity level (0.0-1.0)
  max_new_tokens: 2048             # Max output length

teacher_model:
  model: kimi-k2.5                 # Reasoning trace generator
  endpoint: http://soketlab-node054:30000/v1/chat/completions
  temperature: 0.6                 # Determinism (lower = more stable)
  max_new_tokens: 1500             # Max trace length
  parallel_workers: 2              # Concurrent API requests
  timeout_seconds: 120             # Request timeout

verification:
  timeout_seconds: 30              # Max verification time
  numeric_precision: 1e-6           # Comparison tolerance

reasoning_scale:
  levels:
    0: "Minimal"     # Simple factual recall
    1: "Basic"       # Straightforward logic
    2: "Intermediate" # Multiple concepts
    3: "Advanced"    # Sophisticated analysis
    4: "Expert"      # Theoretical frameworks
```

### Step-Specific Configuration

Each pipeline step is configurable:

- **Step 2** (Evol-Instruct): `evol_instruct.mutation_strategies`, `temperature`, `num_mutations`
- **Step 3** (Format Converter): `format_converter.remove_choices`, `keep_original`
- **Step 4** (Teacher Model): `teacher_model.temperature`, `max_new_tokens`, `enable_thinking`
- **Step 5** (Verification): `verification.timeout_seconds`, `numeric_precision`
- **Step 6** (Reasoning Tagger): `reasoning_scale.levels`

## Usage

### Verify Setup (First Time Only)

Before running the full pipeline, test your environment:

```bash
# Test with minimal data - no API calls
python -m src.pipeline --steps 1 --dry-run

# Expected output (should complete in <30 seconds):
# ✓ Loaded seed data from config
# ✓ Applied mutations
# (No external API calls made)
```

### Run Full Pipeline

```bash
# Process all enabled datasets through all 7 steps
python -m src.pipeline

# With custom sample count
python -m src.pipeline --seed-count 100

# Show all available options
python -m src.pipeline --help
```

**Output**: Timestamped parquet files in `data/` with format:
```
{domain}_{model}_{source}_{count}_{timestamp}.parquet

Example:
math_kimi-k2.5_theoremqa_250_2026-04-17_143022.parquet
```

### Run Specific Steps

Skip expensive API calls by running specific steps:

```bash
# Run only steps 1-2 (load data + difficulty scaling)
python -m src.pipeline --steps 1,2

# Run only step 4 (teacher reasoning generation)
python -m src.pipeline --steps 4

# Dry-run: test everything without API calls
python -m src.pipeline --dry-run

# Available step numbers:
# 1 = Load seed data (NuminaMath, FineMath, etc.)
# 2 = Evol-Instruct difficulty mutations
# 3 = MCQ → open-ended conversion
# 4 = Teacher model reasoning traces (<think> tags)
# 5 = Math-Verify validation + rejection sampling
# 6 = Reasoning-level tagging (0-4 complexity)
# 7 = Serialize to Parquet files
```

### Monitor Pipeline Execution

```bash
# Watch pipeline output in real-time
tail -f logs/pipeline.log

# Count generated parquet files
watch -n 5 'ls -1 data/*.parquet | wc -l'

# Check total output size
du -sh data/

# Inspect specific output file
python -c "
import pandas as pd
df = pd.read_parquet('data/math_kimi-k2.5_theoremqa_250_*.parquet')
print(f'Samples: {len(df)}')
print(f'Verified: {df[\"verified\"].sum()}/{len(df)}')
print(f'Pass rate: {100*df[\"verified\"].sum()/len(df):.1f}%')
print(df.head(2))
"
```

## Dataset Output Format

Each row in the output parquet contains:

```python
{
    "SFT/Reasoning": str,                    # Type: "Reasoning" (with <think> traces)
    "System prompt": str,                    # System context for the problem
    "Instruction": str,                      # Instruction text
    "Question": str,                         # The math problem statement
    "<Think>": str,                          # Step-by-step reasoning trace
    "Answer": str,                           # Model-generated answer
    "Difficulty level": str,                 # "minimal", "basic", "intermediate", "advanced", "expert"
    "Number of words in instruction (preferably no. of tokens)": int,  # Token count
    "Number of words in system prompt": int,      # Word count
    "Number of words in question": int,           # Word count
    "Number of words in answer": int,             # Word count
    "Task": str,                             # Task type (e.g., "QA")
    "Domain": str,                           # Problem domain (e.g., "math", "algebra")
    "Language": str,                         # Language of problem (e.g., "English")
    "Source": str,                           # Dataset source (e.g., "LIMO", "TheoremQA")
    "Made by": str,                          # Always "math_pipeline"
    "Reasoning/SFT": str,                    # Always "reasoning"
    "Model": str,                            # Model used for generation (e.g., "kimi-k2.5")
    "sample_id": str,                        # Unique sample identifier
    "verified": bool                         # Whether answer passed Math-Verify validation
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `SFT/Reasoning` | str | Always "Reasoning" (indicates reasoning traces included) |
| `System prompt` | str | System-level context for the model |
| `Instruction` | str | Task instruction/context |
| `Question` | str | The actual math problem to solve |
| `<Think>` | str | Step-by-step reasoning/chain-of-thought (from teacher model) |
| `Answer` | str | Final answer generated by model |
| `Difficulty level` | str | Reasoning level: "minimal", "basic", "intermediate", "advanced", "expert" |
| `Number of words in instruction (preferably no. of tokens)` | int | Token count for instruction |
| `Number of words in system prompt` | int | Word count for system prompt |
| `Number of words in question` | int | Word count for problem statement |
| `Number of words in answer` | int | Word count for answer |
| `Task` | str | Task classification (always "QA" for question-answering) |
| `Domain` | str | Subject domain (math, algebra, geometry, calculus, etc.) |
| `Language` | str | Language code (always "English") |
| `Source` | str | Original dataset source (LIMO, TheoremQA, AIME, etc.) |
| `Made by` | str | Generator pipeline (always "math_pipeline") |
| `Reasoning/SFT` | str | Dataset type (always "reasoning" for these outputs) |
| `Model` | str | Teacher/reasoning model used (e.g., "kimi-k2.5") |
| `sample_id` | str | Unique identifier for this sample |
| `verified` | bool | True if answer passed Math-Verify validation, False/None otherwise |

## Reasoning Level Scale

The pipeline assigns complexity scores (0-4) internally and converts them to string labels in the final output:

| Internal | Output Label | Description |
|----------|--------------|-------------|
| 0 | "minimal" | Simple factual recall, no deduction required |
| 1 | "basic" | Straightforward connections, single-step logic |
| 2 | "intermediate" | Multiple factors/concepts combined (e.g., Algebra + Geometry) |
| 3 | "advanced" | Sophisticated multi-dimensional analysis, causal relationships |
| 4 | "expert" | Theoretical frameworks, deep counterfactual reasoning, novel synthesis |

**Note**: In the final parquet output, the `"Difficulty level"` field contains the lowercase string names (e.g., "minimal", "basic") not the numeric values.

## Cluster Submission

### Running on SLURM

```bash
# Submit a single dataset processing job
sbatch run_aime_parallel_8.slurm

# Submit all dataset jobs
bash submit_jobs.sh

# Monitor job status
squeue -u $USER

# View job logs
tail -f slurm_output_*.log
```

Each `.slurm` file configures:
- GPU allocation (typically 1-2 GPUs per job)
- Job runtime (12-24 hours typical)
- Memory allocation (40-80GB)
- Batch size and parallelization
- Output logging

### Running Locally (nohup)

For non-SLURM cluster environments:

```bash
# Setup
mkdir -p logs output

# Run in background with logging
nohup python -m src.pipeline > logs/pipeline_$(date +%s).log 2>&1 &

# Monitor progress
tail -f logs/pipeline_*.log

# Check if still running
ps aux | grep "src.pipeline"

# View complete output after job finishes
cat logs/pipeline_*.log
```

## Testing & Validation

### Quick Format Check

```bash
# After pipeline completes, validate output format
python -c "
import pandas as pd

df = pd.read_parquet('data/math_kimi-k2.5_theoremqa_*.parquet')

# Check required columns exist
required_cols = ['Question', 'Answer', '<Think>', 'Difficulty level', 
                 'verified', 'Model', 'Source', 'sample_id']
for col in required_cols:
    assert col in df.columns, f'Missing column: {col}'

# Check field types
assert df['verified'].dtype == bool or df['verified'].dtype == object, 'verified should be bool/nullable'
assert df['Difficulty level'].dtype == object, 'Difficulty level should be str'

print('✓ Output format valid')
print(f'✓ {len(df)} total samples')
print(f'✓ {df[\"verified\"].sum()} verified samples ({100*df[\"verified\"].sum()/len(df):.1f}%)')
print(f'✓ Difficulty levels:')
print(df['Difficulty level'].value_counts())
"
```

### Inspect Sample Data

```bash
# View first sample with all fields
python -c "
import pandas as pd
df = pd.read_parquet('data/math_kimi-k2.5_theoremqa_*.parquet')
sample = df.iloc[0]

print('=' * 70)
print('SAMPLE OUTPUT RECORD')
print('=' * 70)
print()
print('=== Question ===')
print(sample['Question'][:300] + '...' if len(sample['Question']) > 300 else sample['Question'])
print()
print('=== Reasoning (<Think> - first 400 chars) ===')
print(sample['<Think>'][:400] + '...' if len(sample['<Think>']) > 400 else sample['<Think>'])
print()
print('=== Answer ===')
print(sample['Answer'])
print()
print('=== Metadata ===')
print(f'Domain: {sample[\"Domain\"]}')
print(f'Source: {sample[\"Source\"]}')
print(f'Model: {sample[\"Model\"]}')
print(f'Difficulty level: {sample[\"Difficulty level\"]}')
print(f'Task: {sample[\"Task\"]}')
print(f'Language: {sample[\"Language\"]}')
print(f'Verified: {sample[\"verified\"]}')
print()
print('=== Word Counts ===')
print(f'System prompt tokens: {sample[\"Number of words in system prompt\"]}')
print(f'Instruction tokens: {sample[\"Number of words in instruction (preferably no. of tokens)\"]}')
print(f'Question words: {sample[\"Number of words in question\"]}')
print(f'Answer words: {sample[\"Number of words in answer\"]}')
"
```

### Merge Parquets

After running pipeline multiple times:

```bash
# Merge all outputs for a dataset
python merge_parquets_by_dataset.py

# Or manually combine:
python -c "
import pandas as pd
import glob

files = glob.glob('data/math_kimi-k2.5_*.parquet')
dfs = [pd.read_parquet(f) for f in files]
combined = pd.concat(dfs, ignore_index=True)
combined.to_parquet('data/final/math_combined.parquet')

print(f'Combined {len(files)} files into {len(combined)} total samples')
print(f'Pass rate: {100*combined[\"verified\"].sum()/len(combined):.1f}%')
"
```

## Project Structure

```
├── config/
│   └── config.yaml                 # Main configuration
├── data/
│   ├── checkpoints/                # Intermediate step outputs
│   ├── hf_cache/                   # Hugging Face dataset cache
│   └── [output parquets]           # Final timestamped outputs
├── src/
│   ├── pipeline.py                 # Main orchestrator
│   ├── data_ingestion/             # Dataset loaders
│   ├── evol_instruct/              # Difficulty scaling
│   ├── format_converter/           # MCQ conversion
│   ├── teacher_model/              # Reasoning generation
│   ├── verification/               # Answer validation
│   ├── reasoning_tagger/           # Complexity scoring
│   └── __init__.py
├── requirements.txt                # Python dependencies
├── merge_parquets_by_dataset.py   # Combine parallel outputs
├── [utilities scripts]             # Analysis and testing
└── [SLURM job files]               # Cluster submission scripts
```

## Dependencies

Key dependencies:
- **transformers**, **torch**: Model inference
- **datasets**, **huggingface_hub**: Dataset loading
- **pyarrow**: Parquet file I/O
- **sympy**: Mathematical computation
- **math-verify**: Answer verification engine
- **tenacity**: Retry logic for API calls

See `requirements.txt` for complete list.

## Advanced Usage

### Inspect Dataset Samples

```bash
# View sample problems and solutions
python view_output_samples.py

# Analyze calculator dataset
python inspect_calculator_dataset.py

# Check GLM tool outputs
python inspect_glm.py
```

### Generate Reports

```bash
# Generate pipeline quality report
python generate_final_report.py

# Generate LIMO benchmark analysis
python generate_limo_report.py

# Detailed team documentation
python generate_team_documentation.py
```

### Debugging

Enable verbose logging:

```bash
# Set in pipeline.py or config
LOG_LEVEL=DEBUG python -m src.pipeline
```

Check step outputs:

```bash
python test_math_calculator_integration.py
python test_mutation_quality.py
python test_numina_live.py
```

## Performance Considerations

- **Batch size**: Adjust in config for GPU memory constraints
- **Parallelization**: SLURM scripts use 8 parallel workers by default
- **API rate limits**: Tenacity handles retries; adjust backoff in config
- **Verification timeout**: Set per-step timeout to skip slow problems

## Troubleshooting

### Dataset Download Fails
- Verify `HF_TOKEN` is set and has proper permissions
- Check `data/hf_cache/` disk space
- Try `huggingface-cli login`

### Teacher Model Inference Errors
- Verify `MODEL_API_KEY` is valid
- Check API rate limits and quotas
- Review inference batch size settings

### Verification Fails
- Ensure `sympy` and `math-verify` dependencies installed
- Check answer format matches expected output
- Review verification timeout settings

### Memory Issues
- Reduce `batch_size` in config
- Process datasets one at a time (disable in config)
- Run fewer steps with `--steps` flag

## Contributing

To add a new dataset source:

1. Create a new loader in `src/data_ingestion/` (inherit from `BaseLoader`)
2. Register in `src/data_ingestion/loader.py`
3. Add config section to `config/config.yaml`
4. Test with `python -m src.pipeline --steps 1 --dry-run`

## Citation

If you use this pipeline in research, please cite the underlying datasets:

```bibtex
@misc{numina_math,
  title={NuminaMath: A Math Problem-Solving Dataset},
  publisher={Hugging Face Hub},
  year={2024}
}

@misc{limo,
  title={LIMO: Large-scale Math Problem Corpus},
  author={GAIR},
  year={2024}
}

# Add citations for other datasets as needed
```

## License

[Specify license here]

## Support

For issues, questions, or feedback:
- Open an issue on GitHub
- Check existing notebooks and analysis scripts for examples
- Review test files for usage patterns

---

**Last Updated**: April 2026  
**Python Version**: 3.9+  
**Status**: Active Development
