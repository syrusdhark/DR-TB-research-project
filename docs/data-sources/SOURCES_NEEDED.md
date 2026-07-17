# Additional Data Sources Needed for Complete DR-TB Research Model

## Current Status
✅ **Available:**
- CXR Images: 700 TB cases, 3500 Normal cases
- Metadata Excel files (basic structure)

⚠️ **Currently Synthetic:**
- Clinical metadata (generated)
- Genomic mutation data (generated)
- DR-TB labels (simulated)

---

## 1. REAL CLINICAL METADATA SOURCES

### A. Public Databases (Require Data Access Agreements)

#### 1. **NIAID TB Portal Program**
- **URL**: https://data.tbportals.niaid.nih.gov/
- **What it provides:**
  - Patient demographics (age, gender, region)
  - Clinical history (previous TB treatment, treatment outcomes)
  - Comorbidities (HIV, diabetes, smoking status)
  - Treatment regimens and outcomes
- **Access**: Requires data-sharing agreement
- **Note**: Already mentioned in your README - 2800 TB images from this source

#### 2. **WHO Global TB Database**
- **URL**: https://www.who.int/teams/global-tuberculosis-programme/data
- **What it provides:**
  - Regional TB statistics
  - Treatment outcome data
  - Drug resistance patterns by region
  - Epidemiological data

#### 3. **NCBI BioProject/BioSample**
- **URL**: https://www.ncbi.nlm.nih.gov/bioproject/
- **What it provides:**
  - Clinical metadata from research studies
  - Patient cohorts with associated data
  - Access via Entrez API (already implemented in your code)
- **How to use**: Search for TB-related projects (BioProject IDs)

#### 4. **PubMed Central (PMC) Research Papers**
- **URL**: https://www.ncbi.nlm.nih.gov/pmc/
- **What it provides:**
  - Supplementary data from published papers
  - Patient cohort information
  - Clinical features from research studies
- **How to use**: Extract tables from supplementary materials

#### 5. **Mendeley Data / Figshare**
- **URL**: https://data.mendeley.com/ | https://figshare.com/
- **What it provides:**
  - Published datasets from research papers
  - Clinical metadata from completed studies
- **Search terms**: "tuberculosis", "drug-resistant TB", "clinical data"

---

## 2. REAL GENOMIC MUTATION DATA SOURCES

### A. TB-Specific Genomic Databases

#### 1. **ReSeqTB Database**
- **URL**: https://platform.reseqtb.org/
- **What it provides:**
  - Comprehensive resistance mutation data
  - rpoB, katG, inhA, pncA, embB mutations
  - Genotype-phenotype correlations
  - API access available
- **Access**: Public, but may require registration

#### 2. **TB Drug Resistance Mutation Database (TBDReaMDB)**
- **URL**: http://www.tbdream.org/
- **What it provides:**
  - Mutation catalog for TB drug resistance
  - Mutation frequency data
  - Drug resistance patterns

#### 3. **NCBI GenBank**
- **URL**: https://www.ncbi.nlm.nih.gov/genbank/
- **What it provides:**
  - Full genomic sequences
  - Annotated mutations
  - Access via Entrez API (already in your code)
- **How to use**: Search for TB strain genomes with resistance markers

#### 4. **CRyPTIC Database (Comprehensive Resistance Prediction for Tuberculosis: an International Consortium)**
- **URL**: https://www.crypticproject.org/
- **What it provides:**
  - Large-scale genomic data with phenotypic DST results
  - Mutation-disease associations
  - Access: Requires data access agreement

#### 5. **PhyResSE (TB Resistance Prediction)**
- **URL**: http://phyresse.org/
- **What it provides:**
  - Mutation-based resistance prediction
  - Genomic data with resistance profiles

#### 6. **Mykrobe Predictor Data**
- **URL**: https://github.com/Mykrobe-tools/mykrobe
- **What it provides:**
  - Resistance mutation databases
  - Open-source mutation catalogs

---

## 3. DRUG SUSCEPTIBILITY TESTING (DST) DATA

### Critical for Real DR-TB Labels

#### 1. **Clinical Laboratory Data**
- **What you need:**
  - Phenotypic DST results (LJ, MGIT, BACTEC)
  - Genotypic DST results (Xpert MTB/RIF, LPA, line probe assays)
  - Minimum inhibitory concentrations (MICs)
  - Resistance profiles for each drug

#### 2. **Sources:**
- Hospital/clinical partnerships
- Research collaborations
- Public health laboratories
- Published research datasets with DST results

#### 3. **Key Variables Needed:**
- Rifampin resistance (RIF-R)
- Isoniazid resistance (INH-R)
- MDR-TB (both RIF-R and INH-R)
- XDR-TB (MDR + additional resistance)
- Pre-XDR-TB (MDR + additional resistance to fluoroquinolones or injectables)

---

## 4. ADDITIONAL IMAGE DATA SOURCES

### A. More CXR Images

#### 1. **NIAID TB Portal** (Already mentioned)
- Additional 2800 TB images (need agreement)

#### 2. **RSNA Pneumonia Detection Challenge**
- **URL**: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data
- **What it provides:**
  - Additional normal CXR images
  - Expert annotations
- **Note**: Already mentioned in your README

#### 3. **MIMIC-CXR Database**
- **URL**: https://physionet.org/content/mimic-cxr/2.0.0/
- **What it provides:**
  - Large CXR database with clinical notes
  - Requires CITI training and data use agreement
- **Access**: Requires registration and training

#### 4. **NIH Chest X-ray Dataset**
- **URL**: https://nihcc.app.box.com/v/ChestXray-NIHCC
- **What it provides:**
  - Additional CXR images
  - Multi-label annotations

#### 5. **CheXpert Dataset**
- **URL**: https://stanfordmlgroup.github.io/competitions/chexpert/
- **What it provides:**
  - Large CXR dataset
  - Clinical labels

---

## 5. EXPERT ANNOTATIONS & SEGMENTATION

### A. For Model Explainability

#### 1. **Lung Segmentation Masks**
- **Sources:**
  - Manual expert annotations
  - Automated segmentation tools (U-Net, etc.)
  - Public datasets with segmentations

#### 2. **Lesion Annotation**
- **What you need:**
  - TB lesion locations (bounding boxes or masks)
  - Severity scores
  - Radiologist annotations

#### 3. **Sources:**
- RSNA Pneumonia Detection (has bounding boxes)
- Research collaborations with radiologists
- Crowdsourcing platforms (if ethical)

---

## 6. VALIDATION & EXTERNAL DATASETS

### A. Independent Test Sets

#### 1. **Geographically Diverse Datasets**
- Different regions (Asia, Africa, Europe, Americas)
- Different hospital systems
- Different imaging equipment

#### 2. **Temporal Validation**
- Historical data for validation
- Prospective data for clinical validation

#### 3. **Sources:**
- Multi-center collaborations
- Public health databases
- Research consortiums

---

## 7. TREATMENT OUTCOME DATA

### A. For Longitudinal Studies

#### 1. **Treatment Response Data**
- **What you need:**
  - Treatment initiation dates
  - Treatment regimens
  - Follow-up imaging
  - Treatment outcomes (cure, failure, relapse)
  - Time to conversion (culture negative)

#### 2. **Sources:**
- Clinical trial databases
- Hospital electronic health records
- Public health surveillance systems
- Research collaborations

---

## 8. ADDITIONAL CLINICAL FEATURES

### A. Laboratory Values

#### 1. **Blood Tests**
- Complete blood count (CBC)
- Inflammatory markers (ESR, CRP)
- Liver function tests
- Kidney function tests

#### 2. **Microbiology**
- Sputum smear results
- Culture results
- Time to positivity

#### 3. **Sources:**
- Hospital partnerships
- Clinical research databases
- Published research datasets

---

## 9. IMPLEMENTATION PRIORITIES

### High Priority (Essential):
1. ✅ **Real DST Data** - Replace synthetic DR-TB labels
2. ✅ **Real Genomic Mutations** - From ReSeqTB or CRyPTIC
3. ✅ **Real Clinical Metadata** - From NIAID TB Portal or hospital partnerships

### Medium Priority (Important):
4. **Additional CXR Images** - For better model generalization
5. **Expert Annotations** - For explainability and validation
6. **Treatment Outcome Data** - For predictive modeling

### Low Priority (Nice to Have):
7. **Longitudinal Data** - For temporal analysis
8. **Laboratory Values** - Additional features
9. **External Validation Sets** - For robust evaluation

---

## 10. HOW TO INTEGRATE THESE SOURCES

### A. Update Your Scraping Functions

#### 1. **ReSeqTB API Integration**
```python
# Add to your scraping utilities
def fetch_reseqtb_mutations(patient_ids):
    """Fetch real mutations from ReSeqTB API"""
    # Implementation needed
    pass
```

#### 2. **NCBI Entrez Enhanced Queries**
```python
# Enhance your existing Entrez queries
def fetch_ncbi_tb_data(search_term):
    """Fetch TB data from NCBI databases"""
    # More specific queries
    pass
```

#### 3. **Hospital Data Integration**
- Data use agreements
- HIPAA compliance
- Data anonymization
- IRB approval

---

## 11. RECOMMENDED ACTION PLAN

### Phase 1: Immediate (1-2 weeks)
1. ✅ Sign up for NIAID TB Portal access
2. ✅ Register for ReSeqTB database
3. ✅ Set up NCBI Entrez API with proper queries
4. ✅ Contact hospital partnerships for clinical data

### Phase 2: Short-term (1-2 months)
1. ✅ Integrate real genomic mutation data
2. ✅ Replace synthetic clinical metadata with real data
3. ✅ Obtain DST results for proper labeling
4. ✅ Add more CXR images from public sources

### Phase 3: Long-term (3-6 months)
1. ✅ Establish hospital partnerships
2. ✅ Obtain expert annotations
3. ✅ Collect treatment outcome data
4. ✅ Build external validation datasets

---

## 12. CONTACT INFORMATION

### Key Organizations:
- **NIAID TB Portal**: tbportal@niaid.nih.gov
- **ReSeqTB**: Check platform.reseqtb.org for contact
- **CRyPTIC**: Contact through website
- **WHO TB Program**: gtbprogram@who.int

### Research Collaboration Opportunities:
- Reach out to TB research groups
- Join TB research consortiums
- Attend TB conferences for networking
- Collaborate with hospitals in TB-endemic regions

---

## 13. DATA ETHICS & COMPLIANCE

### Important Considerations:
1. **Patient Privacy**: Ensure HIPAA/GDPR compliance
2. **Data Sharing Agreements**: Required for most sources
3. **IRB Approval**: Needed for clinical data
4. **Informed Consent**: Required for patient data
5. **Data Anonymization**: Remove all PHI before use

---

## Summary

**Current Model Completeness: ~40%**
- ✅ CXR images: Real data
- ⚠️ Clinical metadata: Synthetic
- ⚠️ Genomic mutations: Synthetic  
- ⚠️ DR-TB labels: Simulated

**With Real Data Sources: ~90%**
- ✅ CXR images: Real data
- ✅ Clinical metadata: Real data
- ✅ Genomic mutations: Real data
- ✅ DR-TB labels: Real DST results

**For Publication-Ready Model: 100%**
- All above +
- Expert annotations
- External validation
- Treatment outcomes
- Multi-center validation

