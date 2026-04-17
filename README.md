# Post-Training Synthetic Data Pipeline 🚀

A comprehensive suite of three specialized pipelines for generating high-quality synthetic training datasets for code and mathematics domains. Choose the pipeline that best fits your needs!

---

## 📋 Quick Navigation

### 🧮 **[BigMath Pipeline](math_generation/big_math_pipeline/README.md)**
**Generate synthetic math training datasets with reasoning traces**

- **Purpose**: Create high-quality mathematical reasoning datasets
- **Input**: 10+ math datasets from HuggingFace (NuminaMath, FineMath, LIMO, etc.)
- **Output**: Parquet files with problems, solutions, and complexity scores
- **Key Features**:
  - Evol-Instruct difficulty scaling
  - Teacher model reasoning traces with `<think>` tags
  - Math-Verify answer validation
  - 0-4 complexity scoring
  - Reasoning-level tagging

**[📖 Full BigMath README →](math_generation/big_math_pipeline/README.md)**

---

### 💻 **[Code Generation Pipeline](code_generation/README.md)**
**Generate synthetic code training datasets from multiple sources**

- **Purpose**: Create diverse code problem/solution pairs with verification
- **Input**: HuggingFace datasets, Excel files, local JSONL seeds
- **Output**: SFT/RL training datasets (JSONL format)
- **Key Features**:
  - Multi-source ingestion (HuggingFace, Excel, local)
  - Problem variant generation with mutations
  - AI-generated solutions with reasoning
  - Sandbox execution & code verification
  - Parallel batch processing

**[📖 Full Code Pipeline README →](code_generation/README.md)**

---

### 🔬 **[NVIDIA NEMO Pipeline](math_generation/nvidia_nemo_pipeline/nvidia_nemo_team2_submission/README.md)**
**NVIDIA team's specialized math dataset generation approach**

- **Purpose**: Generate math datasets using NVIDIA's methodology
- **Input**: Contest math problems and datasets
- **Output**: Converted and formatted math training data
- **Key Features**:
  - Schema conversion utilities
  - Model comparison tools
  - Structured evaluation framework
  - Team-specific optimizations

**[📖 Full NEMO README →](math_generation/nvidia_nemo_pipeline/nvidia_nemo_team2_submission/README.md)**



## 🚀 Quick Start

### **Which Pipeline Should I Use?**

**Choose BigMath if you:**
- Need mathematical reasoning datasets
- Want to scale difficulty using Evol-Instruct
- Prefer parquet output format
- Need answer verification with sympy/math-verify
- Work with contest/academic math problems

```bash
cd math_generation/big_math_pipeline
python -m src.pipeline --dry-run
```

**Choose Code Generation if you:**
- Need code problem/solution pairs
- Want to work with multiple data sources
- Need sandboxed code execution verification
- Want SFT/RL training formats
- Need parallel batch processing

```bash
cd code_generation
bash run_pipeline.sh --source huggingface deepmind/code_contests 3 100
```

**Choose NEMO if you:**
- Following NVIDIA's specific methodology
- Working with contest math problems
- Need structured evaluation
- Part of NVIDIA research team

```bash
cd math_generation/nvidia_nemo_pipeline/nvidia_nemo_team2_submission
python run_all.py
```

---

## 📁 Directory Structure

```
Post-Training_Synthetic_Data_Pipeline-Code_and_Mathematics/
├── README.md                          # ← You are here (navigation hub)
├── code_generation/                   # 💻 Code Generation Pipeline
│   ├── README.md                      # Detailed code pipeline docs
│   ├── requirements.txt
│   ├── run_pipeline.sh
│   ├── .env.example
│   └── pipeline/
│       ├── synthesize_seeds.py
│       ├── generate_variants.py
│       ├── completion_tensorstudio.py
│       ├── execute_sandbox.sh
│       ├── pack_dataset.py
│       └── ...
│
├── math_generation/
│   ├── big_math_pipeline/             # 🧮 BigMath Pipeline
│   │   ├── README.md                  # Detailed bigmath docs
│   │   ├── requirements.txt
│   │   ├── config/
│   │   │   └── config.yaml
│   │   └── src/
│   │       ├── pipeline.py
│   │       ├── data_ingestion/
│   │       ├── evol_instruct/
│   │       ├── teacher_model/
│   │       ├── verification/
│   │       └── ...
│   │
│   └── nvidia_nemo_pipeline/          # 🔬 NVIDIA NEMO Pipeline
│       └── nvidia_nemo_team2_submission/
│           ├── README.md              # NEMO pipeline docs
│           ├── requirements.txt
│           ├── run_all.py
│           ├── step1_prepare_data.py
│           ├── step2_generate.py
│           └── ...
│
├── CODE_PIPELINE_COMPLETION_REPORT.md # Analysis docs
└── README_COMPARISON_ANALYSIS.md      # Pipeline comparison
```

---

## 🔧 Common Setup Steps

All pipelines share similar setup requirements:

### 1. **Python Environment**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Verify Python version
python --version          # Should be 3.9+
```

### 2. **Install Dependencies**

Each pipeline has its own `requirements.txt`:

```bash
# For BigMath
cd math_generation/big_math_pipeline
pip install -r requirements.txt

# For Code Generation
cd code_generation
pip install -r requirements.txt

# For NEMO
cd math_generation/nvidia_nemo_pipeline/nvidia_nemo_team2_submission
pip install -r requirements.txt
```

### 3. **Configure Environment**

Each pipeline needs a `.env` file:

```bash
# Copy example template (if available)
cp .env.example .env

# Edit with your credentials:
export HF_TOKEN="hf_xxxxxxxxxxxxxxxx"        # HuggingFace API token
export MODEL_API_KEY="your_api_key_here"     # Model inference API key
export MODEL_ENDPOINT="http://..."           # Model endpoint (cluster/TensorStudio)
export HF_CACHE_DIR="./data/hf_cache"        # Dataset cache location
```

### 4. **Verify Setup**

Before running:

```bash
# Test with dry-run (no API calls)
python -m src.pipeline --dry-run              # BigMath
bash run_pipeline.sh --source variants input/seeds.jsonl 1  # Code Gen
python step1_prepare_data.py                   # NEMO
```

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [BigMath README](math_generation/big_math_pipeline/README.md) | Complete BigMath pipeline documentation |
| [Code Pipeline README](code_generation/README.md) | Complete code generation documentation |
| [NEMO README](math_generation/nvidia_nemo_pipeline/nvidia_nemo_team2_submission/README.md) | NVIDIA NEMO methodology docs |
| [Pipeline Comparison](README_COMPARISON_ANALYSIS.md) | Detailed comparison of all pipelines |
| [Code Pipeline Report](CODE_PIPELINE_COMPLETION_REPORT.md) | Code pipeline analysis & improvements |

---

## 🏗️ Architecture Overview

### **BigMath Pipeline** (7 Steps)
```
Load Seed Data (10+ datasets)
    ↓
Evol-Instruct Difficulty Scaling
    ↓
MCQ → Open-ended Conversion
    ↓
Teacher Model Reasoning Traces
    ↓
Math-Verify Validation
    ↓
Reasoning-level Tagging (0-4)
    ↓
Serialize to Parquet
```

### **Code Generation Pipeline** (4+ Stages)
```
Load Source Data (HF/Excel/JSONL)
    ↓
Synthesize & Classify (Domain/Difficulty)
    ↓
Generate Problem Variants
    ↓
Brain Model Completions (Reasoning)
    ↓
Sandbox Execution & Verification
    ↓
Pack into SFT/RL Format
```

### **NEMO Pipeline** (7 Steps)
```
Prepare Data
    ↓
Generate Variants
    ↓
Post-process Results
    ↓
Evaluate Quality
    ↓
Generate from Datasets
    ↓
Convert Schema
    ↓
Compare Models
```

---

## ⚙️ Configuration

Each pipeline has configuration files:

- **BigMath**: `math_generation/big_math_pipeline/config/config.yaml`
- **Code Gen**: Environment variables in `.env` file
- **NEMO**: `math_generation/nvidia_nemo_pipeline/nvidia_nemo_team2_submission/config/`

Refer to each pipeline's README for detailed configuration options.

---

## 🔗 Integration with Other Projects

These pipelines can be:
- **Combined**: Use multiple pipelines to generate diverse datasets
- **Chained**: Output from one pipeline as input to another
- **Compared**: Evaluate which works best for your use case


---

## 💡 Key Features Across All Pipelines

✅ **Multi-source Dataset Ingestion** — Load from diverse sources  
✅ **Reasoning Annotation** — Generate step-by-step thinking traces  
✅ **Quality Verification** — Validate correctness (math or code)  
✅ **Complexity Scoring** — Assign difficulty levels  
✅ **Parallel Processing** — Distributed execution support  
✅ **Deterministic Output** — Reproducible results with timestamps  
✅ **Cluster Ready** — SLURM/nohup support  

---

## 🐛 Troubleshooting

### Common Issues (all pipelines)

**Error: "HF_TOKEN not set"**
```bash
export HF_TOKEN="hf_xxxxx..."
# Or add to .env file and run: source .env
```

**Error: "Model endpoint unreachable"**
```bash
# Test connectivity
curl -X GET http://your-endpoint:port/health
# Or check SSH tunnel if using internal cluster
```

**Error: "CUDA out of memory"**
- Reduce batch size in config
- Run fewer samples with `--limit` flag
- Process datasets sequentially

For pipeline-specific issues, see respective READMEs:
- [BigMath Troubleshooting](math_generation/big_math_pipeline/README.md#troubleshooting)
- [Code Gen Troubleshooting](code_generation/README.md#troubleshooting)
- [NEMO Troubleshooting](math_generation/nvidia_nemo_pipeline/nvidia_nemo_team2_submission/README.md)

---

## 👥 Contributing

Each pipeline welcomes contributions:

1. Choose a pipeline to enhance
2. Follow its specific README guidelines
3. Test thoroughly before submitting
4. Update documentation accordingly

See individual pipeline READMEs for contribution guidelines.

---

## 📄 License

Each pipeline may have its own license. See respective directories for details.

---

## 📞 Support & Questions

For issues related to:
- **BigMath pipeline**: See [BigMath README](math_generation/big_math_pipeline/README.md#support)
- **Code Generation**: See [Code README](code_generation/README.md#support)
- **NEMO pipeline**: See [NEMO README](math_generation/nvidia_nemo_pipeline/nvidia_nemo_team2_submission/README.md)

---

## 🎯 Quick Decision Tree

```
START HERE ↓

Are you working with CODE problems?
├─ YES → Use Code Generation Pipeline 💻
│  └─ Go to: code_generation/README.md
│
└─ NO → Are you part of NVIDIA team?
   ├─ YES → Use NEMO Pipeline 🔬
   │  └─ Go to: math_generation/nvidia_nemo_pipeline/nvidia_nemo_team2_submission/README.md
   │
   └─ NO → Use BigMath Pipeline 🧮
      └─ Go to: math_generation/big_math_pipeline/README.md
```

---

**Last Updated**: April 17, 2026  
**Version**: 1.0  
**Status**: Production Ready ✅

---

## 🚀 Next Steps

1. **Choose your pipeline** using the decision tree above
2. **Open the pipeline's detailed README** (links above)
3. **Follow the setup instructions** in that README
4. **Run the verification test** to ensure configuration works
5. **Launch your pipeline** with real data

**Let's generate high-quality synthetic training data! 🎉**
