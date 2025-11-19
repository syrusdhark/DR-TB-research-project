# Phase 0.1: Retrain Model with Exact Architecture - Implementation Guide

## Critical Issue

The saved model checkpoints don't match the current `model.py` architecture, causing:
- 46 missing keys (critical fusion/classifier layers)
- Random initialization for missing layers
- Real accuracy << reported accuracy

## Solution

Retrain the model from scratch using the **exact** architecture defined in `model.py`.

## Step-by-Step Implementation

### Step 1: Verify Architecture Match

The notebook (`DR_TB_using_RoMIA.ipynb`) currently defines `MultimodalFusionModel` inline. We need to ensure it uses the exact same architecture as `model.py`.

**Action Required:**
1. Open `DR_TB_using_RoMIA.ipynb`
2. Find the cell that defines `MultimodalFusionModel` (around line 1308)
3. Replace the inline definition with an import from `model.py`:

```python
# Replace the entire MultimodalFusionModel class definition with:
from model import MultimodalFusionModel, MultiHeadAttention
```

### Step 2: Run Training from Scratch

1. **Load the merged dataset:**
   ```python
   df = pd.read_csv("data/merged_dataset.csv")
   ```

2. **Create datasets with exact transforms:**
   - Use the same transforms as before
   - Same train/val/test split (70/15/15)
   - Same random seed (42)

3. **Initialize model:**
   ```python
   from model import MultimodalFusionModel
   
   model = MultimodalFusionModel(
       num_clinical_features=14,
       num_genomic_features=12,
       num_classes=1
   ).to(device)
   ```

4. **Train with same hyperparameters:**
   - Learning rate: 1e-4
   - Batch size: 8 (effective: 16 with gradient accumulation)
   - Epochs: 35
   - Loss: Combined Focal + Dice
   - Optimizer: AdamW

5. **Save checkpoint:**
   ```python
   checkpoint = {
       'model_state_dict': model.state_dict(),
       'num_clinical_features': 14,
       'num_genomic_features': 12,
       'validation_auc': best_val_auc,
       'validation_f1': best_val_f1,
       'epoch': best_epoch
   }
   
   torch.save(checkpoint, 'results/models/best_multimodal_dr_tb_2025_v1.pth')
   ```

### Step 3: Verify 100% Weight Loading

After training, test loading:

```python
from model_loader import load_model

model, device = load_model('results/models/best_multimodal_dr_tb_2025_v1.pth')
# Should show: ✅ Model loaded successfully with NO missing keys
```

### Step 4: Re-evaluate

Run evaluation on test set and confirm:
- AUROC matches training (~0.933)
- No architecture mismatch warnings
- All layers loaded correctly

## Expected Outcome

- ✅ Model loads with 0 missing keys
- ✅ 100% architecture match
- ✅ Full reported performance restored
- ✅ No random initialization warnings

## Files to Modify

1. `DR_TB_using_RoMIA.ipynb` - Replace inline model definition with import
2. Training cell - Ensure it saves to `best_multimodal_dr_tb_2025_v1.pth`
3. `model_loader.py` - Update to prioritize new checkpoint (already handles flexible loading)

## Quick Test

After retraining, verify with:

```python
from model_loader import load_model
model, device = load_model()
# Check output - should be clean with no warnings
```

## Next Steps After Phase 0.1

Once model retraining is complete:
1. Proceed to Phase 0.2 (calibration and precision fixes)
2. Then Phase 1 (data integration with TB Portals/CRyPTIC)

