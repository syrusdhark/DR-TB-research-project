# Implementation Status - November 2025

> **Superseded (2026-07-16)**: this document describes retraining plans for
> the old single fused CXR+clinical+genomic model. That architecture was
> replaced by a two-stage design (`TBImageClassifier` + `DRTBRiskModel` in
> `model.py`) because its training data paired real X-ray images with
> unrelated synthetic patient records — see `PROJECT_OVERVIEW.md`'s "Why two
> stages?" section. The retraining steps below no longer apply; use
> `prepare_stage1_data.py` / `prepare_stage2_data.py` and
> `train_tb_classifier.py` / `train_drtb_risk.py` instead. Left in place for
> historical context.

## ✅ Completed Setup

### 1. Dataset Information Scraped
- ✅ TB Portals: 20,953 TB cases, 4,467 genomes, 23,464 images (16 countries)
- ✅ CRyPTIC: 20,000+ isolates with per-drug MICs
- ✅ BV-BRC: 15,000+ TB genomes with resistance data
- ✅ Belarus Kaggle: 1,049 DR-TB X-rays ready for download

**Access Info**: See `data_sources/DATASET_ACCESS_INFO.md`

### 2. Data Ingestion Scripts Created
- ✅ `data/ingest_tb_portals.py` - TB Portals data processor
- ✅ `data/ingest_cryptic.py` - CRyPTIC dataset processor
- ⚠️  **Note**: These are framework scripts - need actual data files to complete

### 3. Retraining Framework
- ✅ `retrain_model.py` - Framework for retraining
- ✅ `PHASE_0_1_RETRAIN_GUIDE.md` - Step-by-step instructions

---

## 🚀 Phase 0.1: Retrain Model - READY TO EXECUTE

### Current Status
- ✅ Model architecture frozen in `model.py`
- ⚠️  Notebook still defines model inline (needs update)
- ✅ Model loader ready with flexible loading

### Action Required

**Option A: Update Notebook (Recommended)**
1. Open `DR_TB_using_RoMIA.ipynb`
2. Find the cell with `class MultiHeadAttention` and `class MultimodalFusionModel` (around line 1271-1445)
3. Replace entire cell content with:
   ```python
   # Import exact model architecture from model.py
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path.cwd()))
   
   from model import MultimodalFusionModel, MultiHeadAttention
   
   print("✅ Using MultimodalFusionModel from model.py")
   print("   This ensures 100% architecture match!")
   ```
4. Update checkpoint save path to: `best_multimodal_dr_tb_2025_v1.pth`
5. Run training from scratch

**Option B: Use Retrain Script**
- Complete `retrain_model.py` with full training loop
- Run as standalone script

---

## 📋 Next Steps (In Order)

### Immediate (Phase 0.1)
1. **Update notebook** to import from `model.py` ✅ Framework ready
2. **Run training** from scratch with current dataset
3. **Save checkpoint** as `best_multimodal_dr_tb_2025_v1.pth`
4. **Verify** 100% weight loading (no missing keys)

### Short-term (Phase 0.2)
1. Add probability calibration
2. Implement PPV display
3. Add dynamic threshold selection

### Medium-term (Phase 1)
1. **Request TB Portals access** (do this now - takes 1-3 days)
2. **Register for CRyPTIC** (do this now)
3. **Download Belarus Kaggle dataset** (immediate)
4. Run ingestion scripts when data arrives

### Long-term (Phase 2-4)
- Per-drug multi-label prediction
- Model backbone upgrade
- UI enhancements
- Publication preparation

---

## 🔗 Quick Access Links

### Dataset Access
- **TB Portals**: https://datasharing.tbportals.niaid.nih.gov/ → Request Access
- **CRyPTIC**: https://www.crypticproject.org/data → Register
- **Belarus Kaggle**: https://www.kaggle.com/datasets/raddar/drug-resistant-tuberculosis-xrays → Download

### Documentation
- Full Plan: See plan created via mcp_create_plan
- Retrain Guide: `PHASE_0_1_RETRAIN_GUIDE.md`
- Dataset Info: `data_sources/DATASET_ACCESS_INFO.md`
- Scorecard: `data_sources/SCORECARD_2025.md`

---

## ⚡ Ready to Execute

**You can now:**
1. Start Phase 0.1 retraining (update notebook + run training)
2. Request dataset access (TB Portals, CRyPTIC) - do this in parallel
3. Download Belarus dataset immediately

**All frameworks and scripts are ready!**

