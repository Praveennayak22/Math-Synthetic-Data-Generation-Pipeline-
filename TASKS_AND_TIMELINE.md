# Team 2 — Tasks, Status, and Timeline

## 1) Completed Tasks

| Task | Description | Status | Evidence |
|---|---|---|---|
| Step 1 | Dataset ingestion and normalization | Completed | `step1_prepare_data.py`, input JSONL files |
| Step 2 | Multi-model generation API runs | Completed | `step2_generate.py`, model output JSONL files |
| Step 3 | Postprocess, answer extraction, verification labeling | Completed | `step3_postprocess.py`, `results/` files |
| Step 4 | Evaluation summary and reporting | Completed | `step4_evaluate.py`, `summary.csv` |
| Step 5 | Original synthetic generation | Completed | `step5_generate_original.py`, `generated/` |
| Step 6 | Dataset-inspired generation | Completed | `step6_generate_from_datasets.py`, `generated/` |
| Step 7 | Schema conversion for external datasets | Completed | `step7_convert_schema.py`, `converted/` |
| Documentation | README + submission packaging | Completed | `README.md`, this file |
| Cluster sync | Pulled results from cluster to local submission | Completed | `results/`, `generated/`, `converted/` |

---

## 2) Live Demonstration Plan

1. Install dependencies from `requirements.txt`
2. Run quick evaluation: `python run_all.py --limit 10`
3. Show one file each from `results/`, `generated/`, and `converted/`
4. Show `summary.csv` and model comparison output

---

Last updated: 2026-04-17
