# DR-TB Model Performance Improvement Plan

## Current Performance Analysis

### Results Summary
- **AUROC**: 0.9305 (Target: 0.98) ⚠️ **-5.0% below target**
- **Accuracy**: 0.8778 (Target: 0.95) ⚠️ **-7.6% below target**
- **Sensitivity (Recall)**: 1.0000 (Target: 0.92) ✅ **Met & exceeded**
- **F1-Score**: 0.2936 (Target: 0.93) ❌ **-68.4% below target**
- **Precision**: 0.1720 (Very low - main problem)

### Key Issues Identified

1. **Critical: Very Low Precision (0.17)**
   - Model predicts DR-TB too frequently (77 false positives vs 16 true positives)
   - Confusion Matrix: 537 TN, 77 FP, 0 FN, 16 TP
   - **Root Cause**: Extreme class imbalance (110 DR-TB vs 4090 Normal = 2.6% positive rate)
   - **Impact**: Low F1-score despite perfect recall

2. **AUROC Below Target**
   - 0.9305 is good but needs improvement to reach 0.98
   - Model discrimination could be better

3. **Accuracy Below Target**
   - Mainly due to high false positive rate

## Immediate Improvements (Already Implemented)

### ✅ 1. Test-Time Augmentation (TTA)
- **Status**: Implemented in Section 10
- **What**: Average predictions from original + horizontally flipped images
- **Expected Impact**: +2-5% F1, reduced false positives

### ✅ 2. Target Recall Constraint Threshold
- **Status**: Implemented in Section 10
- **What**: Find threshold that maintains ≥92% recall while maximizing precision
- **Expected Impact**: Better F1-score by balancing precision/recall

### ✅ 3. Class-Balanced Sampling
- **Status**: Already implemented in Section 7
- **What**: WeightedRandomSampler gives equal weight to both classes
- **Impact**: Model sees more DR-TB examples during training

## Recommended Improvements (Priority Order)

### 🔴 Priority 1: Address Class Imbalance (CRITICAL)

**A. SMOTE or Synthetic Data Generation**
```python
from imblearn.over_sampling import SMOTE

# Apply SMOTE to clinical+genomic features (not images)
# Generate synthetic DR-TB cases
```

**B. Cost-Sensitive Learning**
- Already using Focal Loss (✅)
- Could increase `alpha` parameter (currently 0.25 → try 0.5)
- Could adjust `gamma` parameter (currently 2.0 → try 2.5)

**C. Class Weighting in Loss**
```python
# Add class weights to Focal Loss
pos_weight = class_weights[1]  # ~19x weight for DR-TB
```

**Expected Impact**: +15-25% F1-score, +5-10% precision

### 🟠 Priority 2: Improve Model Architecture

**A. Enhanced Attention Mechanism**
- Current: Basic attention weights
- Improvement: Cross-modal attention (CXR ↔ Clinical ↔ Genomic)
- Add self-attention layers

**B. Ensemble Methods**
- Train 3-5 models with different seeds
- Average predictions
- Use different architectures (EfficientNet-B4, B5, ResNet50)

**Expected Impact**: +3-8% AUROC, +5-10% F1

### 🟡 Priority 3: Advanced Data Augmentation

**A. MixUp Augmentation**
```python
# MixUp combines two images
alpha = 0.2
lambda_param = np.random.beta(alpha, alpha)
```

**B. CutMix Augmentation**
```python
# CutMix replaces part of image with another
```

**C. Stronger Augmentation**
- Increase rotation from 15° to 20°
- Add ElasticTransform
- Stronger ColorJitter

**Expected Impact**: +2-5% generalization, better precision

### 🟢 Priority 4: Training Strategy Improvements

**A. Progressive Image Size Training**
```python
# Start with IMG_SIZE=256, gradually increase to 380
# Train for 5 epochs at each size
```

**B. Learning Rate Scheduling**
- Currently using CosineAnnealingLR ✅
- Add warm restarts
- Use different LR for encoder vs classifier

**C. Longer Training**
- Increase NUM_EPOCHS from 20 to 30-40
- Early stopping patience: 7-10 epochs

**Expected Impact**: +2-4% AUROC, +3-6% F1

### 🔵 Priority 5: Data Quality Improvements

**A. Data Cleaning**
- Remove low-quality CXR images
- Verify DR-TB labels (some might be mislabeled)

**B. More Data**
- Collect more DR-TB cases (currently only 110)
- Use data augmentation more aggressively
- Consider transfer learning from larger TB datasets

**Expected Impact**: +5-10% overall if data quality improves

### 🟣 Priority 6: Hyperparameter Optimization

**A. Grid Search / Random Search**
```python
# Optimize:
- Learning rate: [1e-5, 5e-5, 1e-4, 5e-4]
- Focal Loss alpha: [0.25, 0.5, 0.75]
- Focal Loss gamma: [1.5, 2.0, 2.5]
- Dropout: [0.5, 0.6, 0.7]
- Weight decay: [1e-5, 1e-4, 1e-3]
```

**B. Bayesian Optimization**
- Use Optuna or similar

**Expected Impact**: +2-5% across all metrics

## Quick Wins (Can Implement Today)

### 1. **Adjust Focal Loss Parameters** (5 minutes)
```python
# In Section 8, change:
criterion = FocalLoss(alpha=0.5, gamma=2.5)  # Increased from 0.25, 2.0
```

### 2. **Increase Threshold** (Already done with TTA)
- Re-run Section 10 with TTA enabled
- Should immediately improve F1

### 3. **Add More Aggressive Augmentation** (10 minutes)
```python
# In Section 6, increase:
transforms.RandomRotation(degrees=20)  # From 15
transforms.ColorJitter(brightness=0.3, contrast=0.3)  # From 0.2
```

### 4. **Ensemble 2-3 Models** (30 minutes)
- Train 2 more models with different seeds
- Average predictions

## Expected Timeline

### Week 1: Quick Wins (Immediate)
- ✅ TTA + Target Recall (Already done)
- Adjust Focal Loss parameters
- Stronger augmentation
- **Expected**: F1 → 0.35-0.45, Precision → 0.25-0.35

### Week 2: Class Imbalance (Critical)
- Implement SMOTE for tabular features
- Increase class weights
- **Expected**: F1 → 0.50-0.65, Precision → 0.40-0.55

### Week 3: Architecture Improvements
- Enhanced attention mechanism
- Ensemble 3-5 models
- **Expected**: F1 → 0.65-0.75, AUROC → 0.95-0.96

### Week 4: Advanced Training
- Progressive image size
- Hyperparameter optimization
- Longer training
- **Expected**: F1 → 0.75-0.85, AUROC → 0.96-0.98

## Monitoring Progress

### Key Metrics to Track
1. **F1-Score** (Primary goal - currently 0.29, target 0.93)
2. **Precision** (Critical - currently 0.17, should be >0.80)
3. **AUROC** (Currently 0.93, target 0.98)
4. **Confusion Matrix** (Track false positives)

### Checkpoints
- After each improvement, evaluate on test set
- Compare confusion matrices
- Track validation performance during training

## Notes

- **Sensitivity is already perfect** (1.0) - don't sacrifice this
- **Main issue is precision** - too many false positives
- **Class imbalance is the root cause** - must address this first
- **TTA and threshold optimization** should provide immediate improvement

## Next Steps

1. **Re-run Section 10** with TTA enabled (already done)
2. **Adjust Focal Loss** parameters in Section 8
3. **Implement SMOTE** for tabular features
4. **Train ensemble** of 3 models
5. **Monitor results** and iterate

