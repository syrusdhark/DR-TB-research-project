# Scraped Data Summary for DR-TB Research Model

## Data Sources Successfully Scraped

### 1. WHO Global TB Database
- **URL**: https://www.who.int/teams/global-tuberculosis-programme/data
- **Available Data**:
  - CSV files for download (TB burden estimates, case notifications, treatment outcomes)
  - Drug resistance testing data
  - Regional TB statistics
  - Treatment outcome data
- **Access**: Public, requires data sharing agreement for some datasets
- **CSV Download Links**:
  - WHO TB burden estimates: https://extranet.who.int/tme/generateCSV.asp?ds=estimates
  - MDR/RR-TB burden estimates: https://extranet.who.int/tme/generateCSV.asp?ds=mdr_rr_estimates
  - Case notifications: https://extranet.who.int/tme/generateCSV.asp?ds=notifications
  - Treatment outcomes: https://extranet.who.int/tme/generateCSV.asp?ds=outcomes

### 2. Genomic Mutation Data from Research Papers

#### A. Systematic Review (PMC9225881)
**Title**: "Frequency of rpoB, katG, and inhA Gene Polymorphisms Associated with Multidrug-Resistant Mycobacterium tuberculosis Complex Isolates among Ethiopian TB Patients"

**Key Mutations Found**:
- **rpoB mutations** (RIF resistance):
  - S531L: 34.01% frequency
  - S450L: 19.78% frequency
  - H526Y: 4.4% frequency
  - WT8 probe: 15.38% frequency
  - WT7 probe: 4.4% frequency

- **katG mutations** (INH resistance):
  - S315T: 68.6% frequency
  - WT probe: 12.4% frequency

- **inhA mutations** (INH resistance):
  - C15T: 11.57% frequency
  - WT1 probe: 4.13% frequency
  - WT2 probe: 0.83% frequency
  - MUT1 probe: 0.83% frequency

#### B. Iranian Study (PMC8113720)
**Title**: "Detection of genomic mutations in katG and rpoB genes among multidrug-resistant Mycobacterium tuberculosis isolates from Tehran, Iran"

**Key Mutations Found**:
- **katG mutations** (Isoniazid resistance):
  - 315 AGC > ACC (S→T): 70% frequency
  - 315 AGC > ACC (S→T) and 335 ATC > GTC (I→V): 10% frequency

- **rpoB mutations** (Rifampin resistance):
  - 441 ACC > TCC (T→S): 50% frequency
  - 456 CGG > TGG (R→T): 30% frequency

#### C. Large-Scale Genomic Analysis (Nature Scientific Reports)
**Title**: "Large-scale genomic analysis of Mycobacterium tuberculosis reveals extent of target and compensatory mutations linked to multi-drug resistant tuberculosis"

**Key Mutations Found** (from ~32k isolates):
- **katG mutations** (INH resistance):
  - Ser315Thr: 21.9% frequency (n=7165)
  - Plus 31 putative novel mutations identified

- **fabG1 mutations** (INH resistance):
  - -15C>T: 6.1% frequency (n=1989)

- **inhA mutations** (INH resistance):
  - -154G>A: 1.0% frequency (n=332)

- **rpoB mutations** (RIF resistance):
  - Ser450Leu: 15.2% frequency
  - Asp435Val: 1.8% frequency
  - His445Tyr: 1.3% frequency

### 3. Clinical Metadata Sources

#### A. Mendeley Dataset
- **URL**: https://data.mendeley.com/datasets/gn4xjcdvxv
- **Dataset**: Comprehensive Dataset on Suspected Tuberculosis (TBC) Patients in Semarang, Indonesia
- **Contents**:
  - Socio-demographic data (age, gender, region)
  - Clinical data (symptoms, sputum test results)
  - Treatment outcomes
- **Access**: Public, CC BY 4.0 license
- **Download**: Available as Excel files

## Mutation Frequency Summary

### High-Frequency Mutations (Most Common)

#### Rifampin (RIF) Resistance - rpoB gene:
1. **S531L** (Ser531Leu): 34.01% (Ethiopian study)
2. **Ser450Leu**: 15.2% (Large-scale study)
3. **S450L**: 19.78% (Ethiopian study)
4. **441 T→S**: 50% (Iranian study)
5. **H526Y**: 4.4% (Ethiopian study)
6. **Asp435Val**: 1.8% (Large-scale study)
7. **His445Tyr**: 1.3% (Large-scale study)

#### Isoniazid (INH) Resistance - katG gene:
1. **S315T** (Ser315Thr): 
   - 68.6% (Ethiopian study)
   - 70% (Iranian study)
   - 21.9% (Large-scale study, n=7165)
2. **WT probe**: 12.4% (Ethiopian study)

#### Isoniazid (INH) Resistance - inhA gene:
1. **C15T**: 11.57% (Ethiopian study)
2. **fabG1 -15C>T**: 6.1% (Large-scale study, n=1989)
3. **inhA -154G>A**: 1.0% (Large-scale study, n=332)

### Mutation Distribution Patterns

1. **Geographic Variation**: 
   - Ethiopian studies show different mutation frequencies than Iranian studies
   - Regional differences in mutation patterns

2. **Resistance Level**:
   - High-level resistance: katG S315T
   - Low-level resistance: inhA C15T, fabG1 -15C>T

3. **Co-occurrence**:
   - MDR-TB often has both rpoB and katG mutations
   - Compensatory mutations (ahpC, rpoC/rpoA) co-occur with resistance mutations

## Recommended Implementation

### For Real Data Integration:

1. **Use Real Mutation Frequencies**:
   - Replace synthetic probabilities with real frequencies from research
   - Use region-specific frequencies when available

2. **Incorporate Multiple Sources**:
   - Combine data from multiple research papers
   - Use weighted averages based on sample sizes

3. **Clinical Metadata**:
   - Download WHO CSV files for regional statistics
   - Use Mendeley dataset for patient demographics

4. **Update Scraping Functions**:
   - Parse real mutation data from research papers
   - Use NCBI Entrez API for genomic data
   - Query WHO database for regional statistics

## Next Steps

1. ✅ Parse scraped mutation data into structured format
2. ✅ Update `scrape_genomic_mutations()` function with real frequencies
3. ✅ Update `scrape_clinical_metadata()` with real data sources
4. ✅ Integrate WHO CSV download functionality
5. ✅ Add NCBI Entrez API queries for genomic data

