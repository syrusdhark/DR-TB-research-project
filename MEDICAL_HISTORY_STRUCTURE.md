# Medical History Structure & Output Processing

## 📋 Overview

This document explains how medical history data is structured, processed, and presented in the DR-TB Prediction System's output.

---

## 🏗️ Medical History Structure

### Current Implementation

Medical history is organized into **three main categories**:

#### 1. **Core Medical History** (Used in Model Prediction)
These are sent directly to the model for prediction:

- **Previous TB Treatment** (Binary: 0/1)
- **HIV Status** (Binary: 0/1)
- **Diabetes Status** (Binary: 0/1)
- **Smoking History** (Binary: 0/1)

#### 2. **Respiratory Conditions** (For Risk Analysis Only)
These are captured but **not sent to the model** (to maintain compatibility with the 14-feature model):

- **COPD** (Chronic Obstructive Pulmonary Disease)
- **Asthma**
- **Pneumonia**
- **COVID-19**

#### 3. **Resistance Status** (Clinical Confirmation)
These indicate confirmed resistance patterns:

- **MDR-TB Confirmed** (Multi-Drug Resistant TB)
- **XDR-TB Confirmed** (Extensively Drug-Resistant TB)
- **Rifampin Resistance** (Confirmed)
- **Isoniazid Resistance** (Confirmed)

---

## 🔄 Data Flow

```
┌─────────────────────────────────────┐
│   USER INPUT (Web Interface)       │
│                                     │
│   Medical History Checkboxes:       │
│   ☐ Previous TB Treatment           │
│   ☐ HIV Positive                    │
│   ☐ Diabetes                        │
│   ☐ Smoking History                 │
│   ☐ COPD                            │
│   ☐ Asthma                          │
│   ☐ Pneumonia                       │
│   ☐ COVID-19                        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   DATA CAPTURE (app.py)             │
│                                     │
│   - All checkboxes captured          │
│   - Stored in session_state         │
│   - Converted to binary (0/1)       │
└──────────────┬──────────────────────┘
               │
               ├──────────────────────┐
               │                      │
               ▼                      ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│   MODEL INPUT            │  │   REPORT GENERATION      │
│   (preprocessing.py)      │  │   (report_generator.py)  │
│                           │  │                          │
│   Only 14 features sent:  │  │   All conditions used:   │
│   - Core 4 conditions     │  │   - All 8 conditions    │
│   - Demographics          │  │   - Resistance status   │
│   - Resistance status     │  │   - Age-based factors   │
│                           │  │                          │
│   ❌ COPD, Asthma,        │  │   ✅ All included in    │
│      Pneumonia, COVID-19  │  │      risk factor analysis│
│      NOT included         │  │                          │
└───────────┬──────────────┘  └───────────┬──────────────┘
            │                              │
            ▼                              ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│   MODEL PREDICTION        │  │   DETAILED REPORT        │
│   (predictor.py)          │  │   (report_generator.py)   │
│                           │  │                          │
│   Uses 14 features only   │  │   Shows all risk factors │
│   Returns:                │  │   with severity levels:  │
│   - Prediction            │  │   - Critical             │
│   - Probability           │  │   - High                  │
│   - Confidence            │  │   - Medium                │
│   - Risk Level            │  │   - Low                  │
└──────────────────────────┘  └──────────────────────────┘
```

---

## 📊 Output Structure

### 1. **Simple Prediction View**

Medical history is **not directly shown** in the simple view, but it influences:
- **Prediction Result** (DR-TB/Normal)
- **Probability Score** (0-100%)
- **Confidence Level** (0-100%)
- **Risk Level** (High/Medium/Low)

### 2. **Detailed Report - Risk Factors Section**

Medical history appears in the **"Identified Risk Factors"** section with:

#### Structure:
```python
{
    'factor': 'Condition Name',
    'description': 'Medical explanation',
    'severity': 'Critical/High/Medium/Low'
}
```

#### Severity Classification:

**🔴 Critical:**
- MDR-TB Confirmed
- XDR-TB Confirmed

**🟠 High:**
- Previous TB Treatment
- HIV Co-infection
- Rifampin Resistance
- Isoniazid Resistance

**🟡 Medium:**
- Diabetes
- Smoking
- COPD
- Asthma
- Pneumonia
- COVID-19
- Advanced Age (>65 years)

**🟢 Low:**
- (None currently - low risk factors are implicit)

---

## 🔍 Risk Factor Analysis Logic

### Current Implementation (`report_generator.py`)

The system analyzes medical history in this order:

1. **TB-Related History**
   - Previous TB Treatment → High severity
   - MDR-TB → Critical
   - XDR-TB → Critical

2. **Co-morbidities**
   - HIV → High severity
   - Diabetes → Medium severity

3. **Lifestyle Factors**
   - Smoking → Medium severity

4. **Respiratory Conditions** (New)
   - COPD → Medium severity
   - Asthma → Medium severity
   - Pneumonia → Medium severity
   - COVID-19 → Medium severity

5. **Resistance Status**
   - Rifampin Resistance → High severity
   - Isoniazid Resistance → High severity

6. **Demographic Factors**
   - Age > 65 → Medium severity

---

## 📝 Example Output

### Input:
```
Medical History:
☑ Previous TB Treatment
☑ HIV Positive
☑ COPD
☐ Diabetes
☐ Smoking
☐ Asthma
☐ Pneumonia
☐ COVID-19

Resistance Status:
☑ MDR-TB Confirmed
☐ XDR-TB
☑ Rifampin Resistance
☐ Isoniazid Resistance
```

### Output in Detailed Report:

```
IDENTIFIED RISK FACTORS
─────────────────────────────────────────────
• MDR-TB Confirmed (Critical): 
  Multi-drug resistant TB confirmed

• Previous TB Treatment (High): 
  Previous TB treatment increases risk of drug resistance

• HIV Co-infection (High): 
  HIV co-infection is a significant risk factor for TB and drug resistance

• Rifampin Resistance (High): 
  Rifampin resistance detected

• COPD (Medium): 
  COPD (Chronic Obstructive Pulmonary Disease) increases risk of 
  respiratory infections including TB
```

---

## 🎯 How Medical History Affects Output

### 1. **Model Prediction** (Direct Impact)

**Core 4 conditions** directly influence the model:
- Previous TB Treatment → Increases DR-TB probability
- HIV Status → Increases DR-TB probability
- Diabetes → Moderate increase in risk
- Smoking → Moderate increase in risk

**How it works:**
- These are encoded as binary features (0 or 1)
- Sent to the clinical encoder (14 features total)
- Combined with CXR and genomic data
- Affects final prediction probability

### 2. **Risk Factor Analysis** (Indirect Impact)

**All 8 conditions** appear in risk factor analysis:
- Even if not in model input, they're analyzed
- Used to provide clinical context
- Help explain the prediction
- Guide recommendations

### 3. **Clinical Recommendations** (Contextual Impact)

Medical history influences recommendations:

**Example Logic:**
```python
if HIV_status:
    → Add "HIV-TB Co-infection Management" recommendation
    
if COPD or Asthma:
    → Add "Respiratory condition monitoring" note
    
if Previous_TB_Treatment:
    → Add "Monitor for recurrence" recommendation
```

---

## 🔧 Technical Implementation

### Data Structure

```python
clinical_data = {
    # Core model features (14 total)
    'age': 45,
    'previous_tb_treatment': 1,  # Binary
    'hiv_status': 1,              # Binary
    'diabetes_status': 0,         # Binary
    'smoking_status': 0,          # Binary
    'mdr_tb': 1,                  # Binary
    'xdr_tb': 0,                  # Binary
    'rifampin_resistance': 1,     # Binary
    'isoniazid_resistance': 0,    # Binary
    'gender_encoded': 1,          # 0=Female, 1=Male
    'region_Africa': 0,           # One-hot encoding
    'region_Americas': 0,
    'region_Asia': 1,
    'region_Europe': 0,
    
    # Additional for reporting (not in model)
    'copd': 1,                     # Binary
    'asthma': 0,                   # Binary
    'pneumonia': 0,                # Binary
    'covid19': 0                   # Binary
}
```

### Processing Flow

1. **Input Capture** (`app.py`):
   ```python
   copd = st.checkbox("COPD")
   # ... other checkboxes
   ```

2. **Encoding** (`preprocessing.py`):
   ```python
   # Only 14 features sent to model
   features = [age, previous_tb, hiv, diabetes, ...]
   # COPD, Asthma, etc. stored separately for reporting
   ```

3. **Risk Analysis** (`report_generator.py`):
   ```python
   if clinical_data.get('copd', 0):
       risk_factors.append({
           'factor': 'COPD',
           'description': '...',
           'severity': 'Medium'
       })
   ```

4. **Report Generation** (`report_generator.py`):
   ```python
   report = {
       'risk_factors': _identify_risk_factors(...),
       'recommendations': _generate_recommendations(...)
   }
   ```

---

## 📈 Future Enhancements

### Planned Improvements:

1. **Weighted Risk Scoring**
   - Calculate composite risk score from all factors
   - Weight factors by severity
   - Display risk score in output

2. **Temporal History**
   - Track condition onset dates
   - Consider duration of conditions
   - Historical progression analysis

3. **Condition Interactions**
   - Analyze co-morbidity interactions
   - COPD + Smoking = Higher risk
   - HIV + Diabetes = Complex management

4. **Model Integration**
   - Retrain model with all 18 features
   - Include COPD, Asthma, Pneumonia, COVID-19
   - Improve prediction accuracy

5. **Visualization**
   - Risk factor heatmap
   - Timeline of medical history
   - Severity distribution chart

---

## 🎓 Medical Context

### Why This Structure?

1. **Model Compatibility**: 
   - Current model trained on 14 features
   - Adding features requires retraining
   - Flexible approach allows gradual enhancement

2. **Clinical Relevance**:
   - All conditions are medically relevant
   - Even if not in model, they inform clinical judgment
   - Risk factor analysis provides context

3. **User Experience**:
   - Comprehensive data capture
   - Detailed reporting
   - Actionable recommendations

---

## 📋 Summary

### Current State:
- ✅ **8 Medical History Conditions** captured
- ✅ **4 Core Conditions** used in model prediction
- ✅ **4 Additional Conditions** used in risk analysis
- ✅ **4 Resistance Status** indicators
- ✅ **Severity-Based** risk factor classification
- ✅ **Comprehensive Reporting** in detailed output

### Output Includes:
1. **Simple Prediction**: Influenced by core 4 conditions
2. **Risk Factors Section**: All 8 conditions + resistance status
3. **Clinical Recommendations**: Context-aware based on all conditions
4. **Exportable Report**: Complete medical history analysis

---

**Last Updated**: 2024  
**Status**: Active Development

