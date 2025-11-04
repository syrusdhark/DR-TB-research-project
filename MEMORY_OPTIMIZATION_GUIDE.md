# Memory Optimization Guide for DR-TB Model Training

## 🚨 Out of Memory (OOM) Error - Solutions

Your GPU has 7.78 GB total memory, and the model is running out. Here are solutions:

## ✅ Solution 1: Use Optimized Settings (Already Applied)

I've already updated the notebook with memory optimizations:
- **Image Size**: Reduced from 456 to 380 (still good quality)
- **Batch Size**: Reduced from 16 to 8
- **Gradient Accumulation**: Enabled (effective batch size = 16)
- **CUDA Cache Clearing**: Enabled

### How to Use:
1. **Restart your kernel** to clear memory
2. **Run all cells from the beginning** (Cell 1 → Cell 2 → ...)
3. The new settings will automatically use less memory

## ✅ Solution 2: Further Reduce Memory (If Still OOM)

If you still get OOM errors, update Cell 2 (Configuration) with these values:

```python
IMG_SIZE = 320  # Further reduce from 380
BATCH_SIZE = 4  # Further reduce from 8
GRADIENT_ACCUMULATION_STEPS = 4  # Increase to maintain effective batch size = 16
NUM_WORKERS = 1  # Reduce to 1
```

## ✅ Solution 3: Use Google Colab (Recommended for Large Models)

### Advantages:
- **Free GPU**: T4 (16GB) or V100 (32GB) - much more memory!
- **No setup needed**: Just upload your notebook
- **Free tier**: 12 hours runtime, can upgrade to Pro for longer sessions

### Steps:
1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload your `DR_TB_using_RoMIA.ipynb` notebook
3. Enable GPU: Runtime → Change runtime type → GPU (T4 or V100)
4. Upload your data:
   - Upload `TB_Chest_Radiography_Database/` folder
   - Upload `data_sources/` folder
5. Update paths in Cell 2 if needed:
   ```python
   DATA_DIR = "/content/TB_Chest_Radiography_Database"
   data_sources_dir = "/content/data_sources"
   ```
6. Run all cells

### For Google Colab, you can use higher settings:
```python
IMG_SIZE = 456  # Can use full size
BATCH_SIZE = 16  # Can use larger batch
GRADIENT_ACCUMULATION_STEPS = 1  # No need for accumulation
```

## ✅ Solution 4: Additional Memory Optimizations

### A. Clear Memory Before Training
Add this cell before training:

```python
# Clear all GPU memory
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
gc.collect()
print("🧹 Memory cleared!")
```

### B. Use CPU for Validation (Optional)
You can move validation to CPU to save GPU memory:

```python
def validate(model, val_loader, criterion, device):
    model.eval()
    # Move model to CPU temporarily
    model_cpu = model.cpu()
    # ... validation code ...
    # Move back to GPU
    model = model_cpu.to(device)
```

### C. Reduce Model Size
If still OOM, you can use EfficientNet-B3 instead of B4:

```python
# In Section 8, change:
self.cxr_encoder = models.efficientnet_b3(pretrained=True)  # Instead of b4
cxr_features = 1536  # Instead of 1792
```

## 📊 Memory Usage Comparison

| Setting | Image Size | Batch Size | GPU Memory | Quality |
|---------|------------|------------|------------|---------|
| Original | 456x456 | 16 | ~8GB | ⭐⭐⭐⭐⭐ |
| Optimized | 380x380 | 8 (accum=2) | ~4-5GB | ⭐⭐⭐⭐ |
| Minimal | 320x320 | 4 (accum=4) | ~2-3GB | ⭐⭐⭐ |
| Google Colab | 456x456 | 16 | ~6GB (16GB GPU) | ⭐⭐⭐⭐⭐ |

## 🎯 Recommended Approach

### For Local Training (8GB GPU):
1. Use the **optimized settings** (already applied)
2. If still OOM, use **minimal settings**
3. Consider using **Google Colab** for better performance

### For Google Colab:
1. Use **full settings** (456x456, batch 16)
2. Much faster training
3. No memory issues

## 🔧 Quick Fix Commands

### Clear CUDA Cache (Run in notebook):
```python
import torch
torch.cuda.empty_cache()
print("Memory cleared!")
```

### Check GPU Memory:
```python
import torch
if torch.cuda.is_available():
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"GPU Memory Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.2f} GB")
```

## 📝 Notes

- **Gradient Accumulation**: Maintains effective batch size while using smaller batches
- **Image Size**: 380x380 is still good quality, minimal quality loss
- **Mixed Precision**: Already enabled (saves ~50% memory)
- **Google Colab**: Best option for large models, free and easy

## 🚀 Next Steps

1. **Restart kernel** and run all cells with new settings
2. If still OOM, use **minimal settings** (320x320, batch 4)
3. Or **upload to Google Colab** for best performance

The notebook is already optimized - just restart and run!

