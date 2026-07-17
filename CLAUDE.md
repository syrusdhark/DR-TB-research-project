# CLAUDE.md

Reference for working in this repo. Read this before making changes — several
things here are non-obvious from the code alone.

## What this project is

A research prototype for TB screening and Drug-Resistant TB (DR-TB) risk
assessment, served through a Streamlit app (`app.py`). It runs **two
independent models**, not one:

1. **Stage 1 — TB Detector** (`TBImageClassifier` in `model.py`): chest X-ray
   image → TB vs Normal. Image-only.
2. **Stage 2 — DR-TB Risk Model** (`DRTBRiskModel` in `model.py`): clinical
   data + genomic mutations → DR-TB risk. No image input.

**The two never influence each other.** The X-ray does not affect the DR-TB
risk score, and the clinical/genomic form does not affect TB detection. This
is deliberate, not an oversight — see "Why two stages?" below.

## Quick orientation

```
app.py                  Streamlit UI — the only user-facing entry point
model.py                Both model architectures (TBImageClassifier, DRTBRiskModel, MultiHeadAttention)
model_loader.py          load_tb_classifier(), load_drtb_risk_model(), get_model_info()
preprocessing.py         Image + tabular feature encoding (unchanged by consumer)
predictor.py             predict_tb(), predict_drtb_risk() — independent calls, independent models
report_generator.py      Builds the human-readable report from both stage results
config.py                Feature lists, model path prefixes, threshold lookup

scripts/
  prepare_stage1_data.py  Builds data/tb_image_manifest.csv (each real image used once)
  prepare_stage2_data.py  Builds data/drtb_risk_dataset.csv (per-patient, derived label)
  train_tb_classifier.py  Trains Stage 1
  train_drtb_risk.py      Trains Stage 2
  verify_pipeline.py      End-to-end smoke test, no browser needed

docs/
  PROJECT_OVERVIEW.md     Full architecture writeup, "why two stages?", data sources
  QUICK_START.md          Manual testing checklist
  STREAMLIT_DEPLOYMENT.md, MEDICAL_HISTORY_STRUCTURE.md, MEMORY_OPTIMIZATION_GUIDE.md
  data-sources/           Provenance notes for the WHO/clinical/genomic source data
  archive/                Superseded docs (old retrain plans) — historical only, don't follow them

data/                     clinical_data.csv, genomic_mutations.csv (real per-patient tables, tracked in git)
                          + tb_image_manifest.csv, drtb_risk_dataset.csv (derived, gitignored, regenerate via scripts/)
                          + merged_dataset.csv (DEPRECATED, kept for history — see below, don't use it)
TB_Chest_Radiography_Database/   700 Tuberculosis + 3500 Normal CXR images (gitignored, present on disk)
results/models/           tb_classifier.pth + tb_classifier_metrics.json
                          drtb_risk_model.pth + drtb_risk_model_metrics.json
DR_TB_using_RoMIA.ipynb   STALE — defines the old fused-model architecture inline, do not use as reference
```

## Why two stages? (read this before touching model.py or the data pipeline)

The original design was a single fused model (`MultimodalFusionModel`) that
took CXR + clinical + genomic features together and predicted DR-TB directly.
It was replaced because:

1. **The training data faked the image-resistance link.** `data/merged_dataset.csv`
   paired each of the 4,200 real X-ray images with independently-generated
   synthetic patient records, reusing individual images up to 53 times each
   to force a balanced label. The same image appeared under both DR-TB-positive
   and DR-TB-negative labels across different rows — so the CXR branch had no
   real signal to learn for the resistance decision. This is also consistent
   with domain knowledge: X-ray appearance is not a validated indicator of
   drug-resistance status (resistance is a genomic/phenotypic property).

2. **The label itself was inconsistent.** `label_drtb` in the old merged
   dataset contradicted the fields it should be derived from — e.g. patients
   with both MDR-TB and XDR-TB confirmed were labeled DR-TB-**negative** 100%
   of the time (0/10), and 5+ resistance mutations were labeled negative 100%
   of the time. Backwards from basic TB biology.

**The fix, in order:**
- `TBImageClassifier` trains only on `data/tb_image_manifest.csv`, where each
  real image is used exactly once with its real folder-derived label
  (`scripts/prepare_stage1_data.py`).
- `DRTBRiskModel` trains only on `data/drtb_risk_dataset.csv`, one row per
  real patient (`scripts/prepare_stage2_data.py`), with `label_drtb`
  **derived deterministically**:
  ```
  label_drtb = 1  if  mdr_tb OR xdr_tb OR rifampin_resistance
                      OR isoniazid_resistance OR mutation_count >= 1
               0  otherwise
  ```
  Expect Stage 2 to hit very high validation/test AUROC (~0.98-0.99) — that's
  because the label is a deterministic function of a subset of its own input
  features, not evidence the model found some deeper signal. Its metrics
  file (`results/models/drtb_risk_model_metrics.json`) says this explicitly;
  don't quote its AUROC as a real-world accuracy claim without that caveat.

`retrain_model.py` and `retrain_exact_architecture.py` (the old training
scripts for the fused model) were deleted along with their orphaned
checkpoint (`exact_match_nov2025.pth`) — they imported a model class that no
longer exists. `DR_TB_using_RoMIA.ipynb` still defines the old architecture
inline and was intentionally **not** updated; don't treat it as a reference.

## A real constraint you'll hit again: no pretrained weights in this sandbox

`TBImageClassifier` is a compact CNN trained **from scratch**, not a
pretrained EfficientNet-B4 (the original design's backbone). Both
`download.pytorch.org` (torchvision's weight host) and `huggingface.co` are
blocked by this environment's network egress policy — confirmed via
`curl -sS "$HTTPS_PROXY/__agentproxy/status"`, which showed a 403 policy
denial on `download.pytorch.org:443`. If you're working somewhere with normal
network access, switch `TBImageClassifier` back to a pretrained backbone
(see the class docstring in `model.py` for the exact swap) — it should
outperform the from-scratch CNN on a dataset this size (4,200 images).

Because CPU-only training is also slow, `config.IMG_SIZE` was dropped from
380 (EfficientNet-B4's native size) to 192 to keep an epoch under ~2-3
minutes on 4 CPU cores. Raise it back to 380 if retraining with GPU access.

## Data flow

```
Real source data (tracked in git):
  data/clinical_data.csv, data/genomic_mutations.csv   (4,200 patients)
  TB_Chest_Radiography_Database/{Tuberculosis,Normal}/ (4,200 images, gitignored)
        │
        ▼  scripts/prepare_stage1_data.py, scripts/prepare_stage2_data.py
Derived training tables (gitignored, regenerate anytime):
  data/tb_image_manifest.csv     (img_path, label_tb — 1 row per image)
  data/drtb_risk_dataset.csv     (14 clinical + 12 genomic cols, label_drtb — 1 row per patient)
        │
        ▼  scripts/train_tb_classifier.py, scripts/train_drtb_risk.py
Checkpoints + metrics (tracked in git):
  results/models/tb_classifier.pth (+ _metrics.json)
  results/models/drtb_risk_model.pth (+ _metrics.json)
        │
        ▼  model_loader.py picks up the latest file per prefix automatically
app.py inference (via predictor.py, preprocessing.py, report_generator.py)
```

`config.get_latest_model_path(prefix)` picks the newest `.pth` file starting
with `config.TB_MODEL_PREFIX` (`"tb_classifier"`) or
`config.DRTB_RISK_MODEL_PREFIX` (`"drtb_risk_model"`) — so a retrained
checkpoint just needs the right filename prefix to be picked up automatically,
no code change needed. Each stage's decision threshold is read from its
`*_metrics.json`'s `optimal_threshold` field via
`config.get_threshold_for_model()`, not hardcoded.

## Running things

```bash
# App
./run_app.sh                          # or: streamlit run app.py

# Rebuild derived training tables (fast, no GPU needed)
python3 scripts/prepare_stage1_data.py
python3 scripts/prepare_stage2_data.py

# Retrain (CPU-only takes minutes for Stage 2, ~30-45 min for Stage 1)
python3 scripts/train_drtb_risk.py
python3 scripts/train_tb_classifier.py

# End-to-end smoke test without a browser
python3 scripts/verify_pipeline.py
```

All four scripts resolve paths relative to the repo root via
`Path(__file__).resolve().parent.parent`, so they work from any cwd.

## Conventions worth knowing

- **`data/` and `TB_Chest_Radiography_Database/` are gitignored by default**,
  but `data/clinical_data.csv`, `data/genomic_mutations.csv`, and
  `data/merged_dataset.csv` are force-tracked exceptions (see `.gitignore`).
  Derived files (`tb_image_manifest.csv`, `drtb_risk_dataset.csv`) are *not*
  tracked — regenerate them via the `scripts/prepare_*` scripts rather than
  expecting them to exist after a fresh clone.
- **`results/models/*.pth` and `*.json` are tracked** (gitignore has explicit
  exceptions for them) — checkpoints are meant to be committed.
- **Training scripts save the best checkpoint after every validation
  improvement**, not just at the end — a background training run that gets
  killed mid-flight (this has happened in this sandbox) still leaves a usable
  checkpoint. If you write a new training script, keep this pattern.
- Comorbidity fields (COPD, asthma, pneumonia, COVID-19, and 17 others in
  `config.COMORBIDITY_FEATURES`) are captured in the UI and used for the
  report's risk-factor narrative, but deliberately **not** part of either
  model's input tensor — only the original 14 clinical / 12 genomic features
  defined in `config.CLINICAL_FEATURES` / `config.GENOMIC_FEATURES` are.
- No test suite exists yet. `scripts/verify_pipeline.py` is the closest thing
  — a scripted run through both models plus report generation, with asserts
  on the prediction labels.

## Medical/legal framing

This is a research tool, not a diagnostic device. The app and docs carry
explicit disclaimers (should not replace DST, clinical judgment, or standard
diagnostics) — preserve these if you touch `app.py`'s UI copy.
