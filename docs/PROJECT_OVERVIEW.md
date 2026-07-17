# DR-TB Prediction System - Project Overview

## 🎯 Project Description

A **two-stage deep learning system** for TB screening and drug-resistance risk assessment:
- **Stage 1 — TB Detection**: a CXR-image-only model (chest X-ray → TB vs Normal)
- **Stage 2 — DR-TB Risk**: a clinical + genomic-only model (patient profile → drug-resistance risk)

The system provides both simple predictions and detailed diagnostic reports to assist healthcare professionals in early detection and drug-resistance risk triage.

---

## 🆕 Current Snapshot (2026-07-16)

- **Architecture change**: the previous single fused model (CXR + clinical + genomic → one DR-TB prediction) was replaced with two independent models. See "Why two stages?" below — the fused model's image branch had no valid basis to inform the resistance decision.
- **Deployment state**: Streamlit UI (`app.py`) remains the primary entry point; inference runs locally with Python 3.12, PyTorch, on CPU or GPU.
- **Model health**: Stage 1 (TB detection) reaches test AUROC 0.9999 / accuracy 99.8% — a genuine result, since TB vs Normal from real X-rays is a well-established, learnable task. Stage 2 (DR-TB risk) reaches similarly high validation/test metrics (~0.99 AUROC), but for a different reason — its label is deterministically derived from a subset of its own input fields, so treat that number as a sanity check, not a real-world accuracy claim. See `results/models/tb_classifier_metrics.json` and `results/models/drtb_risk_model_metrics.json` for exact numbers.
- **Known limitation**: Stage 1 was trained without ImageNet-pretrained weights because `download.pytorch.org` and `huggingface.co` were both blocked by the training environment's network policy. A pretrained EfficientNet-B4 backbone (the original design) should outperform this from-scratch CNN and is recommended if retraining somewhere with normal network access — see `model.py`'s `TBImageClassifier` docstring for the exact swap.
- **Stale artifacts**: `DR_TB_using_RoMIA.ipynb` still defines the old fused-model architecture inline and is out of date relative to `model.py`. Treat `scripts/train_tb_classifier.py` and `scripts/train_drtb_risk.py` as the canonical training path, not the notebook.

---

## Why two stages? (the bug this replaced)

The original model fused a CXR image with clinical and genomic features into
one DR-TB prediction. Its training data (`merged_dataset.csv`, since deleted) reused each of
the 4,200 real chest X-ray images up to 53 times, pairing the same image with
different independently-generated synthetic patient records to manufacture a
balanced (50/50) label. That means the same X-ray appeared in training under
both DR-TB-positive and DR-TB-negative labels — so the image branch could not
have learned any real signal for the resistance decision. This is also
consistent with domain knowledge: chest X-ray appearance is not a validated
indicator of drug-resistance status in the first place (resistance is a
genomic/phenotypic property).

Separately, the label itself (`label_drtb` in the old merged dataset)
contradicted the fields it should be derived from — e.g. patients with both
MDR-TB and XDR-TB confirmed were labeled DR-TB-**negative** 100% of the time.
The current Stage 2 dataset (`data/drtb_risk_dataset.csv`, built by
`scripts/prepare_stage2_data.py`) instead derives the label deterministically and
consistently:
```
label_drtb = 1  if  mdr_tb OR xdr_tb OR rifampin_resistance
                    OR isoniazid_resistance OR mutation_count >= 1
             0  otherwise
```

## 🏗️ System Architecture

### Stage 1: TB Image Classifier (`TBImageClassifier`)

```
┌─────────────────┐
│  CXR Image      │ → Compact CNN (5 conv blocks) → TB / Normal
│  (192x192)      │
└─────────────────┘
```
Trained on `data/tb_image_manifest.csv` — every real image in
`TB_Chest_Radiography_Database` used exactly once, labeled from its source
folder (no duplication, no synthetic pairing).

### Stage 2: DR-TB Risk Model (`DRTBRiskModel`)

```
┌─────────────────┐  ┌─────────────────┐
│ Clinical Data   │  │ Genomic Data    │
│ (14 features)   │  │ (12 mutations)  │
└────────┬────────┘  └────────┬────────┘
         │                    │
         └─────────┬──────────┘
                    │
           ┌────────▼────────┐
           │ Multi-Head      │
           │ Attention       │
           │ Fusion (2-way)  │
           └────────┬────────┘
                    │
           ┌────────▼────────┐
           │ Classification  │
           │ Head            │
           └────────┬────────┘
                    │
           ┌────────▼────────┐
           │ DR-TB Risk /    │
           │ Low Risk        │
           └─────────────────┘
```
No image input. Trained on `data/drtb_risk_dataset.csv` — 4,200 unique
patients (one row each) from `clinical_data.csv` + `genomic_mutations.csv`.

### Key Components

1. **Stage 1 — CXR Encoder**: compact CNN trained from scratch (5 conv blocks, ~1M params)
2. **Stage 2 — Clinical Encoder**: multi-layer neural network (14 features)
3. **Stage 2 — Genomic Encoder**: multi-layer neural network (12 mutation types)
4. **Stage 2 — Fusion Layer**: multi-head attention between clinical and genomic embeddings only
5. **Two independent classifiers**: TB vs Normal (Stage 1), DR-TB Risk vs Low Risk (Stage 2)

---

## 📊 Model Performance

See `results/models/tb_classifier_metrics.json` and
`results/models/drtb_risk_model_metrics.json` for current numbers (both
include the optimal decision threshold used at inference). Stage 2's metrics
are expected to be very high because its label is a deterministic function of
a subset of its own inputs — that reflects the label's construction, not a
claim of predictive power beyond it.

---

## 🖥️ Web Interface Features

### Input Modalities

#### 1. **Chest X-Ray Image** (Stage 1 input only)
- Supported formats: PNG, JPG, JPEG
- Automatic preprocessing to 192x192 pixels
- ImageNet normalization

#### 2. **Clinical Data** (Stage 2 input only)
- **Demographics**:
  - Age (0-150 years)
  - Gender (Male/Female)
  - Geographic Region (Africa, Americas, Asia, Europe)

- **Medical History**:
  - Previous TB Treatment
  - HIV Status
  - Diabetes
  - Smoking History
  - **COPD (Chronic Obstructive Pulmonary Disease)**
  - **Asthma**
  - **Pneumonia**
  - **COVID-19**

- **Resistance Status**:
  - MDR-TB Confirmed
  - XDR-TB Confirmed
  - Rifampin Resistance
  - Isoniazid Resistance

#### 3. **Genomic Mutations** (Stage 2 input only)
- **Rifampin Resistance (rpoB)**:
  - S531L, S450L, H526Y, H445Y, D435V

- **Isoniazid Resistance**:
  - katG S315T, katG S315N
  - inhA C15T
  - fabG1 -15C>T

- **Other Resistance**:
  - pncA H57D (Pyrazinamide)
  - embB M306V (Ethambutol)

### Output Features

#### Simple Prediction View (two separate results, not one fused number)
- **TB Detection badge** (Stage 1, image-only): Tuberculosis/Normal, probability, confidence
- **DR-TB Risk badge** (Stage 2, clinical+genomic-only): DR-TB Risk/Low Risk, probability, confidence
- **Risk Level**: High, Medium, or Low, shown per stage

#### Detailed Report
- **Stage 1 & Stage 2 Summaries**: Full interpretation with probability and confidence for each
- **Identified Risk Factors**: All relevant clinical risk factors with severity levels
- **Genomic Mutation Analysis**: Detected mutations with significance
- **DR-TB Risk Modality Contributions**: Which of Clinical/Genomic contributed most to the Stage 2 score (Stage 1's image score is separate and not part of this breakdown)
- **Clinical Recommendations**: Priority-based action items

---

## 📁 Project Structure

```
DR-TB research project/
├── app.py                      # Main Streamlit web application
├── model.py                    # Model architecture definitions
├── model_loader.py             # Model loading utilities
├── preprocessing.py             # Input preprocessing functions
├── predictor.py                 # Prediction logic
├── report_generator.py          # Detailed report generation
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── run_app.sh                   # Startup script
│
├── README.md                    # Application documentation
├── CLAUDE.md                    # Project reference for Claude Code
│
├── scripts/
│   ├── prepare_stage1_data.py   # Builds data/tb_image_manifest.csv (Stage 1)
│   ├── prepare_stage2_data.py   # Builds data/drtb_risk_dataset.csv (Stage 2)
│   ├── train_tb_classifier.py   # Trains Stage 1 (TB image classifier)
│   ├── train_drtb_risk.py       # Trains Stage 2 (DR-TB risk model)
│   └── verify_pipeline.py       # End-to-end smoke test (no browser needed)
│
├── docs/
│   ├── PROJECT_OVERVIEW.md      # This file
│   ├── QUICK_START.md           # Quick start guide
│   ├── STREAMLIT_DEPLOYMENT.md
│   ├── MEDICAL_HISTORY_STRUCTURE.md
│   ├── MEMORY_OPTIMIZATION_GUIDE.md
│   ├── data-sources/             # Data provenance notes
│   └── archive/                  # Superseded docs, kept for history
│
├── data/
│   ├── clinical_data.csv        # Per-patient clinical metadata (4,200 patients)
│   ├── genomic_mutations.csv    # Per-patient genomic mutation data (4,200 patients)
│   ├── tb_image_manifest.csv    # Stage 1 training manifest (1 row per real image)
│   └── drtb_risk_dataset.csv    # Stage 2 training table (1 row per patient)
│
├── results/
│   └── models/
│       ├── tb_classifier.pth               # Stage 1 checkpoint
│       ├── tb_classifier_metrics.json      # Stage 1 evaluation metrics + threshold
│       ├── drtb_risk_model.pth             # Stage 2 checkpoint
│       └── drtb_risk_model_metrics.json    # Stage 2 evaluation metrics + threshold
│
├── TB_Chest_Radiography_Database/
│   ├── Tuberculosis/           # TB CXR images
│   └── Normal/                 # Normal CXR images
│
├── data_sources/               # WHO TB data and other sources
└── DR_TB_using_RoMIA.ipynb     # Training notebook
```

---

## 🔧 Technical Stack

### Core Technologies
- **Python 3.12**
- **PyTorch 2.9.1** - Deep learning framework
- **Torchvision 0.24.1** - Computer vision models
- **Streamlit 1.51.0** - Web interface framework

### Key Libraries
- **scikit-learn** - Machine learning utilities
- **Pillow** - Image processing
- **pandas/numpy** - Data processing
- **matplotlib/seaborn** - Visualization
- **grad-cam** - Model interpretability
- **biopython** - Bioinformatics tools

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, for faster inference)
- Trained model checkpoint files in `results/models/`

### Installation

1. **Create Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Application**:
   ```bash
   streamlit run app.py
   # Or use: ./run_app.sh
   ```

4. **Access Application**:
   - Open browser to: `http://localhost:8501`

---

## 📝 Current Status

### ✅ Completed Features

- [x] Two-stage architecture: TB detection (image-only) + DR-TB risk (clinical/genomic-only)
- [x] Model training pipelines for both stages, with clean data preparation scripts
- [x] Web interface with Streamlit showing both results separately
- [x] Image upload and preprocessing
- [x] Clinical data input forms
- [x] Genomic mutation selection
- [x] Real-time prediction
- [x] Detailed diagnostic reports covering both stages
- [x] Risk factor analysis
- [x] Report export functionality
- [x] Additional medical history fields (COPD, Asthma, Pneumonia, COVID-19)

### ⚠️ Known Issues

1. **Stage 1 has no pretrained backbone**: the training environment used to build the current `tb_classifier.pth` could not reach `download.pytorch.org` or `huggingface.co`, so it's a compact CNN trained from scratch rather than a pretrained EfficientNet-B4. Retrain with a pretrained backbone on a GPU-equipped, network-unrestricted environment for better accuracy (see `model.py`).
2. **Stage 2's near-perfect metrics reflect its label construction**: `label_drtb` is deterministically derived from a subset of its own input features (see "Why two stages?" above), so validation/test AUROC in the high 0.90s is expected, not evidence of learned signal beyond that rule.
3. **`DR_TB_using_RoMIA.ipynb` is stale**: it still defines the old fused-model architecture inline and was not updated as part of this change. Use `scripts/train_tb_classifier.py` / `scripts/train_drtb_risk.py` instead.

### 🔄 Future Enhancements

- [ ] Retrain Stage 1 with a pretrained EfficientNet-B4 backbone on unrestricted network access
- [ ] Update or retire `DR_TB_using_RoMIA.ipynb` to match the current two-stage architecture
- [ ] Add batch prediction capability
- [ ] Implement model versioning
- [ ] Add more visualization options
- [ ] Export reports as PDF
- [ ] Add user authentication
- [ ] Database integration for patient records
- [ ] API endpoint for programmatic access

---

## 🎓 Research Context

### Data Sources

1. **CXR Images**: 
   - TB Chest Radiography Database
   - 700 TB cases, 3500 Normal cases

2. **Clinical Data**:
   - WHO TB burden estimates
   - Indonesian clinical dataset
   - Regional epidemiological data

3. **Genomic Data**:
   - Research-based mutation frequencies
   - Real mutation patterns from published studies
   - PMC9225881, PMC8113720, Nature Scientific Reports

### Model Training

- **Stage 1 (TB detection)**: compact CNN trained from scratch, AdamW, weighted random sampling for the 700:3500 class imbalance, horizontal flip + rotation augmentation
- **Stage 2 (DR-TB risk)**: clinical + genomic encoders fused via 2-way multi-head attention, AdamW, weighted random sampling for class imbalance
- Both stages pick their decision threshold from validation-set F1, not a fixed 0.5 cutoff

---

## 📊 Clinical Features

### Current Model Input (14 features)
1. Age
2. Previous TB Treatment
3. HIV Status
4. Diabetes Status
5. Smoking Status
6. MDR-TB
7. XDR-TB
8. Rifampin Resistance
9. Isoniazid Resistance
10. Gender (encoded)
11. Region: Africa
12. Region: Americas
13. Region: Asia
14. Region: Europe

### Additional Features (for reporting only)
- COPD
- Asthma
- Pneumonia
- COVID-19

*Note: These are captured in the UI and included in risk factor analysis but not sent to the model to maintain compatibility.*

---

## 🧬 Genomic Features (12 mutations)

1. rpoB_S531L (Rifampin resistance - 34% frequency)
2. rpoB_S450L (Rifampin resistance - 20% frequency)
3. rpoB_H526Y (Rifampin resistance - 4.4% frequency)
4. rpoB_H445Y (Rifampin resistance - 1.3% frequency)
5. rpoB_D435V (Rifampin resistance - 1.8% frequency)
6. katG_S315T (Isoniazid resistance - 70% frequency)
7. katG_S315N (Isoniazid resistance - rare)
8. inhA_C15T (Isoniazid resistance - 11.6% frequency)
9. fabG1_C15T (Isoniazid resistance - 6.1% frequency)
10. pncA_H57D (Pyrazinamide resistance)
11. embB_M306V (Ethambutol resistance)
12. mutation_count (Total mutation count)

---

## ⚠️ Important Disclaimers

### Medical Disclaimer

**This is a research tool and should NOT replace:**
- Clinical judgment
- Standard diagnostic procedures
- Drug susceptibility testing (DST)
- Expert medical consultation

### Best Practices

- Always correlate predictions with patient symptoms and history
- Use in conjunction with physical examination findings
- Confirm positive predictions with laboratory tests
- Follow local TB treatment guidelines
- Consult with TB specialists for complex cases

---

## 📈 Model Metrics Summary

See `results/models/tb_classifier_metrics.json` (Stage 1) and
`results/models/drtb_risk_model_metrics.json` (Stage 2) for current test-set
metrics and each stage's chosen decision threshold. Stage 2's numbers are
high by construction (see "Why two stages?" above) — treat them as a sanity
check that the model learned the derivation rule, not as a real-world
accuracy claim.

---

## 🔐 Security & Privacy

- All processing is done locally
- No data is sent to external servers
- Model checkpoints are stored locally
- Patient data is not persisted (session-based only)

---

## 📞 Support & Documentation

- **Application Guide**: See `../README.md`
- **Quick Start**: See `QUICK_START.md`
- **Training Details**: See `DR_TB_using_RoMIA.ipynb`

---

## 🏷️ Version Information

- **Project Version**: 1.0.0
- **Last Updated**: 2025-11-19
- **Python Version**: 3.12
- **PyTorch Version**: 2.9.1
- **Streamlit Version**: 1.51.0

---

## 📄 License

This is a research project. Please ensure compliance with:
- Medical device regulations (if applicable)
- Data privacy laws (HIPAA, GDPR, etc.)
- Institutional review board (IRB) requirements

---

**Project Maintained by**: DR-TB Research Project Team

