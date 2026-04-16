# Team 2 — NVIDIA NeMo Skills Math Pipeline Submission

## Final Evaluation Compliance 

### Requirement 1: Documented GitHub repository with executable README
Provided in this folder:
- Installation and dependency setup instructions
- Executable run commands (full + quick + stepwise)
- Output locations and demonstration artifacts
- Complete source scripts used for Team 2 pipeline work

### Requirement 2: Completed tasks +  completion date
Provided in:
- `TASKS_AND_TIMELINE.md` (completed items, pending items, target dates)

## Quick Installation Commands (Copy-Paste)

Run these commands exactly from this folder:

```bash
cd nvidia_nemo_team2_submission

python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux/macOS
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run once:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

## Project Summary

This project implements a **7-step math data pipeline** built using workflows inspired by NVIDIA NeMo Skills:
1. Prepare dataset from HuggingFace (`step1_prepare_data.py`)
2. Generate answers using selected model APIs (`step2_generate.py`)
3. Postprocess/extract/fix answer format (`step3_postprocess.py`)
4. Evaluate and summarize performance (`step4_evaluate.py`)
5. Generate original synthetic math questions (`step5_generate_original.py`)
6. Generate dataset-inspired synthetic questions (`step6_generate_from_datasets.py`)
7. Convert external datasets to common schema (`step7_convert_schema.py`)

The pipeline supports multi-model comparison and result export to JSONL + Parquet + CSV.

---

## Repository Structure

```text
nvidia_nemo_team2_submission/
├── README.md
├── TASKS_AND_TIMELINE.md
├── requirements.txt
├── run_all.py
├── step1_prepare_data.py
├── step2_generate.py
├── step3_postprocess.py
├── step4_evaluate.py
├── step5_generate_original.py
├── step6_generate_from_datasets.py
├── step7_convert_schema.py
├── compare_models.py
├── input.jsonl
├── output.jsonl
├── math_dataset_final.jsonl
├── summary.csv
├── results/            (evaluation outputs: model × dataset)
├── converted/          (Step 7 converted schema outputs)
├── generated/          (Step 5/6 synthetic generated outputs)
└── logs/               (optional run logs, if included)
```

---

## Installation

### Prerequisites
- Python 3.8+
- Network access to configured model API endpoints
- HuggingFace access/token for dataset loading (if required by dataset)

### Setup

```bash
# from submission folder
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Dependencies used:
- `datasets`
- `pandas`
- `pyarrow`
- `requests`

---

## How to Run (Executable Commands)

### A) Full evaluation pipeline

```bash
python run_all.py
```

### B) Quick test run (recommended for live demo)

```bash
python run_all.py --limit 10
```

### C) Single dataset / single model run

```bash
python run_all.py --dataset GAIR/LIMO
python run_all.py --model kimi-k2.5
```

### D) Stepwise execution

```bash
python step1_prepare_data.py --dataset GAIR/LIMO --limit 200 --outfile input.jsonl
python step2_generate.py --model kimi-k2.5 --infile input.jsonl --outfile output_kimi-k2.5.jsonl
python step3_postprocess.py --model kimi-k2.5 --source LIMO --infile output_kimi-k2.5.jsonl --outdir results/
python step4_evaluate.py --resultsdir results --save-csv
```

### E) Synthetic generation tasks

```bash
python step5_generate_original.py --per-combo 10 --outfile generated_math.jsonl
python step6_generate_from_datasets.py --dataset netop/TeleMath --limit 200 --outfile generated_from_telemath.jsonl
```

### F) Schema conversion task

```bash
python step7_convert_schema.py --dataset zwhe99/DeepMath-103K --limit 2000
```

---

## Output & Demo Artifacts

Generated outputs are available in:
- `results/` (model outputs, verified records, parquet files)
- `converted/` (schema-converted datasets from Step 7)
- `generated/` (synthetic datasets from Step 5 and Step 6)
- `summary.csv` (evaluation summary)

Current local submission includes synced cluster artifacts (Team 2):
- Evaluation files (JSONL + Parquet) in `results/`
- Converted schema files in `converted/`
- Generated datasets (`generated_*.jsonl` + parquet) in `generated/`

During final live demonstration, we can show:
1. `run_all.py --limit 10` execution
2. One result file in `results/`
3. One converted file from `converted/`
4. One generated file from `generated/`
5. `summary.csv` generation
6. Model comparison via `compare_models.py`

---

## Cluster-to-Local Result Sync (Used in Submission)

Because most runs were executed on cluster, outputs were pulled to this submission folder via `scp`.

Reference remote paths:
- `/projects/data/datasets/code_post_training_data/math_pipeline_team2/NeMo-Math-Pipeline/data/results`
- `/projects/data/datasets/code_post_training_data/math_pipeline_team2/NeMo-Math-Pipeline/data/converted`
- `/projects/data/datasets/code_post_training_data/math_pipeline_team2/NeMo-Math-Pipeline/data/generated_*.jsonl`

Example download commands (from local PowerShell):

```powershell
$BASE="C:\Users\manoh\OneDrive\Desktop\PROJECT\NVIDIA-MATH\Skills\pipeline\nvidia_nemo_team2_submission"

scp -r iitgn_pt_data@slurm.dev.soket.ai:/projects/data/datasets/code_post_training_data/math_pipeline_team2/NeMo-Math-Pipeline/data/results/* "$BASE\results\"
scp -r iitgn_pt_data@slurm.dev.soket.ai:/projects/data/datasets/code_post_training_data/math_pipeline_team2/NeMo-Math-Pipeline/data/converted/* "$BASE\converted\"
scp iitgn_pt_data@slurm.dev.soket.ai:/projects/data/datasets/code_post_training_data/math_pipeline_team2/NeMo-Math-Pipeline/data/generated_*.jsonl "$BASE\generated\"
```

---

## Completion Status (Team 2)
See detailed tracker: `TASKS_AND_TIMELINE.md`

Quick status:
- Core pipeline implementation: **Completed**
- Multi-model evaluation: **Completed**
- Synthetic generation steps: **Completed**
- Schema conversion: **Completed**
- Documentation + executable README: **Completed**
- Cluster result sync to local submission: **Completed**

---

## Team

Team: **Team 2**
Project: **Math Synthetic Data Curation / Generation Pipeline**

Last updated: 2026-04-17




