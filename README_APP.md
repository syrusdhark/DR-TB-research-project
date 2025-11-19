# DR-TB Prediction Web Interface

A Streamlit web application for predicting Drug-Resistant Tuberculosis (DR-TB) using a multimodal deep learning model that combines chest X-ray images, clinical metadata, and genomic mutation data.

## Features

- **Multimodal Input**: Accepts chest X-ray images, clinical data, and genomic mutations
- **Real-time Prediction**: Fast inference using trained deep learning model
- **Detailed Reports**: Comprehensive diagnostic reports with risk factors and recommendations
- **User-friendly Interface**: Intuitive web interface built with Streamlit
- **Exportable Results**: Download prediction reports as text files

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster inference)
- Trained model checkpoint files in `results/models/`

## Installation

### 1. Create Virtual Environment (Recommended)

On Linux systems with externally-managed Python environments, create a virtual environment:

```bash
python3 -m venv venv
```

### 2. Activate Virtual Environment

**Linux/Mac:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Model Files

Ensure you have trained model checkpoint files (`.pth` files) in the `results/models/` directory. The application will automatically use the latest model if multiple checkpoints are available.

### 5. Run the Application

**Option 1: Using the startup script (Linux/Mac):**
```bash
./run_app.sh
```

**Option 2: Manual activation and run:**
```bash
source venv/bin/activate  # Activate virtual environment first
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

**Note:** Make sure the virtual environment is activated before running the app. You'll see `(venv)` in your terminal prompt when it's active.

## Usage

### Step 1: Upload Chest X-Ray Image

1. Click "Browse files" or drag and drop a chest X-ray image
2. Supported formats: PNG, JPG, JPEG
3. Maximum file size: 10MB
4. Recommended minimum dimensions: 100x100 pixels

### Step 2: Enter Clinical Data

Fill in the clinical information form:

- **Age**: Patient's age (0-150 years)
- **Gender**: Select Male or Female
- **Region**: Select patient's geographic region (Africa, Americas, Asia, Europe)
- **Medical History**: Check relevant boxes:
  - Previous TB Treatment
  - HIV Positive
  - Diabetes
  - Smoking History
- **Resistance Status**: Check if any resistance is confirmed:
  - MDR-TB Confirmed
  - XDR-TB Confirmed
  - Rifampin Resistance
  - Isoniazid Resistance

Click "Confirm Clinical Data" to save your inputs.

### Step 3: Enter Genomic Mutations (Optional)

1. Expand the "Select Detected Mutations" section
2. Check boxes for any detected mutations:
   - **Rifampin Resistance (rpoB)**: S531L, S450L, H526Y, H445Y, D435V
   - **Isoniazid Resistance**: katG S315T, katG S315N, inhA C15T, fabG1 -15C>T
   - **Other Resistance**: pncA H57D (Pyrazinamide), embB M306V (Ethambutol)
3. If genomic data is unavailable, leave all unchecked (defaults to no mutations)

### Step 4: Run Prediction

1. Click the "🔬 Run Prediction" button
2. Wait for processing (usually takes a few seconds)
3. View results in the right panel

### Understanding Results

#### Simple Prediction View

- **Prediction Badge**: Shows "DR-TB" or "Normal" with color coding
- **Probability**: Model's predicted probability (0-100%)
- **Confidence**: Confidence score based on distance from threshold
- **Risk Level**: High, Medium, or Low risk classification

#### Detailed Report

Check "Show Detailed Report" to see:

1. **Prediction Summary**
   - Full prediction with interpretation
   - Probability and confidence scores
   - Risk level assessment

2. **Identified Risk Factors**
   - Clinical risk factors detected
   - Severity levels (Critical, High, Medium, Low)
   - Descriptions of each risk factor

3. **Genomic Mutation Analysis**
   - Total mutations detected
   - Individual mutation descriptions
   - Significance of each mutation
   - Interpretation of mutation patterns

4. **Modality Contributions**
   - Contribution percentage from each input type:
     - CXR Image
     - Clinical Data
     - Genomic Data
   - Primary modality identification

5. **Clinical Recommendations**
   - Priority-based recommendations
   - Specific actions to take
   - Detailed descriptions

### Exporting Results

Click "📥 Download Report (TXT)" to download a text file containing the complete prediction report.

## Model Information

- **Architecture**: Multimodal Fusion Model
  - CXR Encoder: EfficientNet-B4
  - Clinical Encoder: Multi-layer neural network (14 features)
  - Genomic Encoder: Multi-layer neural network (12 features)
  - Fusion: Multi-head attention mechanism

- **Input Specifications**:
  - Image size: 380x380 pixels
  - Clinical features: 14 features
  - Genomic features: 12 mutation types

- **Optimal Threshold**: 0.638 (from validation set)

- **Performance** (from evaluation):
  - AUROC: 0.933
  - Accuracy: 87.5%
  - Precision: 16.1%
  - Recall: 93.8%
  - F1-Score: 0.275

## Important Notes

### ⚠️ Medical Disclaimer

**This is a research tool and should NOT replace:**
- Clinical judgment
- Standard diagnostic procedures
- Drug susceptibility testing (DST)
- Expert medical consultation

### ✅ Best Practices

Always correlate predictions with:
- Patient symptoms and history
- Physical examination findings
- Laboratory test results
- Imaging findings from radiologists

### 🔬 Clinical Workflow

1. Use this tool as a screening/decision support system
2. Confirm all positive predictions with DST
3. Consider negative predictions in context of clinical presentation
4. Follow local TB treatment guidelines
5. Consult with TB specialists for complex cases

## Troubleshooting

### Model Not Loading

**Error**: "Model could not be loaded"

**Solutions**:
1. Verify model files exist in `results/models/` directory
2. Check that model files are valid PyTorch checkpoints (`.pth` files)
3. Ensure model architecture matches the expected format

### CUDA Out of Memory

**Error**: "CUDA out of memory"

**Solutions**:
1. Close other applications using GPU
2. Use CPU mode (modify `model_loader.py` to force CPU)
3. Process images one at a time

### Image Loading Errors

**Error**: "Error loading image"

**Solutions**:
1. Verify image format (PNG, JPG, JPEG only)
2. Check file size (max 10MB)
3. Ensure image is not corrupted
4. Try converting image to RGB format

### Input Validation Errors

**Error**: "Input validation error"

**Solutions**:
1. Check that age is between 0-150
2. Verify gender is "Male" or "Female"
3. Ensure region is one of the valid options
4. Check that all binary inputs are 0 or 1

## File Structure

```
DR-TB research project/
├── app.py                 # Main Streamlit application
├── model.py               # Model architecture definitions
├── model_loader.py        # Model loading utilities
├── preprocessing.py       # Input preprocessing functions
├── predictor.py           # Prediction logic
├── report_generator.py    # Detailed report generation
├── config.py              # Configuration settings
├── requirements.txt      # Python dependencies
├── README_APP.md         # This file
└── results/
    └── models/           # Model checkpoint files (.pth)
```

## Development

### Adding New Features

1. **New Input Types**: Modify `preprocessing.py` to add encoding functions
2. **Report Sections**: Extend `report_generator.py` with new analysis functions
3. **UI Components**: Update `app.py` to add new Streamlit components

### Customizing Threshold

Edit `config.py` to change the optimal threshold:

```python
OPTIMAL_THRESHOLD = 0.638  # Change this value
```

### Model Updates

When updating the model:
1. Place new checkpoint in `results/models/`
2. The app will automatically use the latest model
3. Or specify model path in `model_loader.py`

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review error messages for specific guidance
3. Verify all dependencies are installed correctly
4. Ensure model files are compatible with the codebase

## License

This is a research tool. Please ensure compliance with:
- Medical device regulations (if applicable)
- Data privacy laws (HIPAA, GDPR, etc.)
- Institutional review board (IRB) requirements

## Citation

If you use this tool in research, please cite the original research paper and acknowledge the model architecture and training methodology.

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Maintained by**: DR-TB Research Project Team

