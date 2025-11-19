# DR-TB Prediction System - Project Overview

## 🎯 Project Description

A **multimodal deep learning system** for predicting **Drug-Resistant Tuberculosis (DR-TB)** using a combination of:
- **Chest X-Ray (CXR) Images** - Visual analysis using EfficientNet-B4
- **Clinical Metadata** - Patient demographics and medical history
- **Genomic Mutations** - Resistance mutation patterns

The system provides both simple predictions and detailed diagnostic reports to assist healthcare professionals in early detection and diagnosis of drug-resistant tuberculosis.

---

## 🆕 Current Snapshot (2025-11-19)

- **Deployment state**: Streamlit UI (`app.py`) remains the primary entry point; inference runs locally with Python 3.12, PyTorch 2.9.1, and EfficientNet-B4 weights in `results/models/`.
- **Modalities in use**: 380x380 CXR uploads, 14-feature clinical form inputs, and 12 curated genomic mutation toggles feed the multimodal fusion pipeline without schema drift.
- **Model health**: Latest evaluation metrics (AUROC 0.933, Accuracy 87.5%, Recall 93.8%, threshold 0.638) are stored in `results/evaluation_results.json` and visualized via `results/roc_curve.png` and friends.
- **Operational caveats**: Checkpoint-to-architecture mismatches persist for some fusion layers; predictions remain functional but require retraining with aligned weights for peak accuracy.
- **Immediate focus**: Track the retraining effort, batch prediction support, PDF export, and model versioning—currently the highest-impact roadmap items before any clinical pilot.

---

## 🏗️ System Architecture

### Model Architecture: Multimodal Fusion Model

```
┌─────────────────┐
│  CXR Image      │ → EfficientNet-B4 → 1792 features
│  (380x380)      │
└─────────────────┘
         │
         ├─────────────────┐
         │                 │
┌────────▼────────┐  ┌─────▼──────────┐
│ Clinical Data   │  │ Genomic Data    │
│ (14 features)   │  │ (12 mutations)  │
└─────────────────┘  └─────────────────┘
         │                 │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │ Multi-Head      │
         │ Attention       │
         │ Fusion          │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │ Classification  │
         │ Head            │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │ DR-TB / Normal  │
         │ Prediction      │
         └─────────────────┘
```

### Key Components

1. **CXR Encoder**: EfficientNet-B4 (pre-trained on ImageNet)
2. **Clinical Encoder**: Multi-layer neural network (14 features)
3. **Genomic Encoder**: Multi-layer neural network (12 mutation types)
4. **Fusion Layer**: Multi-head attention mechanism
5. **Classifier**: Binary classification (DR-TB vs Normal)

---

## 📊 Model Performance

Based on evaluation results:

- **AUROC**: 0.933 (93.3%)
- **Accuracy**: 87.5%
- **Precision**: 16.1%
- **Recall**: 93.8%
- **F1-Score**: 0.275
- **Optimal Threshold**: 0.638

---

## 🖥️ Web Interface Features

### Input Modalities

#### 1. **Chest X-Ray Image**
- Supported formats: PNG, JPG, JPEG
- Automatic preprocessing to 380x380 pixels
- ImageNet normalization

#### 2. **Clinical Data**
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

#### 3. **Genomic Mutations**
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

#### Simple Prediction View
- **Prediction Badge**: Color-coded (Red for DR-TB, Green for Normal)
- **Probability**: Model confidence (0-100%)
- **Confidence Score**: Distance from threshold
- **Risk Level**: High, Medium, or Low

#### Detailed Report
- **Prediction Summary**: Full interpretation with probability and confidence
- **Identified Risk Factors**: All relevant clinical risk factors with severity levels
- **Genomic Mutation Analysis**: Detected mutations with significance
- **Modality Contributions**: Which input type contributed most (CXR/Clinical/Genomic)
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
├── README_APP.md                # Application documentation
├── QUICK_START.md              # Quick start guide
├── PROJECT_OVERVIEW.md         # This file
│
├── data/
│   ├── merged_dataset.csv      # Combined training dataset
│   ├── clinical_data.csv       # Clinical metadata
│   ├── genomic_mutations.csv   # Genomic mutation data
│   └── cache/                  # Cached data
│
├── results/
│   ├── models/                 # Trained model checkpoints (.pth)
│   ├── evaluation_results.json # Model performance metrics
│   ├── confusion_matrix.png    # Evaluation visualization
│   ├── roc_curve.png          # ROC curve
│   └── precision_recall_curve.png
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

- [x] Multimodal model architecture (CXR + Clinical + Genomic)
- [x] Model training pipeline
- [x] Web interface with Streamlit
- [x] Image upload and preprocessing
- [x] Clinical data input forms
- [x] Genomic mutation selection
- [x] Real-time prediction
- [x] Simple prediction display
- [x] Detailed diagnostic reports
- [x] Risk factor analysis
- [x] Report export functionality
- [x] Model loading with flexible architecture matching
- [x] Additional medical history fields (COPD, Asthma, Pneumonia, COVID-19)

### ⚠️ Known Issues

1. **Architecture Mismatch**: 
   - Saved model checkpoints use a slightly different architecture
   - Model loads with flexible matching but some layers use random initialization
   - **Impact**: Predictions may be less accurate
   - **Solution**: Retrain model with current architecture or use matching checkpoint

2. **Model Compatibility**:
   - Some fusion layers are missing from saved checkpoints
   - App works but accuracy may be affected

### 🔄 Future Enhancements

- [ ] Retrain model with current architecture for full accuracy
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

- **Training Strategy**: Multimodal fusion with attention
- **Loss Function**: Combined Focal Loss + Dice Loss
- **Optimization**: AdamW with learning rate scheduling
- **Augmentation**: MixUp, CutMix, geometric transforms
- **Class Balancing**: SMOTE for synthetic DR-TB samples

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

| Metric | Value |
|--------|-------|
| AUROC | 0.933 |
| Accuracy | 87.5% |
| Precision | 16.1% |
| Recall | 93.8% |
| F1-Score | 0.275 |
| Optimal Threshold | 0.638 |

*Note: Metrics from test set evaluation*

---

## 🔐 Security & Privacy

- All processing is done locally
- No data is sent to external servers
- Model checkpoints are stored locally
- Patient data is not persisted (session-based only)

---

## 📞 Support & Documentation

- **Application Guide**: See `README_APP.md`
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

