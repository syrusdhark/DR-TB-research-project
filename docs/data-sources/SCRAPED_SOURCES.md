# Scraped Data Sources for DR-TB Research Model

## Summary

Successfully scraped data from multiple public sources using Firecrawl MCP server to enhance the DR-TB research model with real mutation frequencies and clinical metadata sources.

---

## 1. WHO Global TB Database

**URL**: https://www.who.int/teams/global-tuberculosis-programme/data

**Available Data**:
- CSV files for download
- TB burden estimates
- Case notifications
- Treatment outcomes
- Drug resistance testing data
- Regional TB statistics

**CSV Download Links**:
- WHO TB burden estimates: `https://extranet.who.int/tme/generateCSV.asp?ds=estimates`
- MDR/RR-TB burden estimates: `https://extranet.who.int/tme/generateCSV.asp?ds=mdr_rr_estimates`
- Case notifications: `https://extranet.who.int/tme/generateCSV.asp?ds=notifications`
- Treatment outcomes: `https://extranet.who.int/tme/generateCSV.asp?ds=outcomes`
- Drug resistance surveillance: `https://extranet.who.int/tme/generateCSV.asp?ds=dr_surveillance`

**Access**: Public, some datasets may require data sharing agreement

---

## 2. Genomic Mutation Data from Research Papers

### A. Systematic Review (PMC9225881)
**Title**: "Frequency of rpoB, katG, and inhA Gene Polymorphisms Associated with Multidrug-Resistant Mycobacterium tuberculosis Complex Isolates among Ethiopian TB Patients: A Systematic Review"

**Key Findings**:
- **rpoB mutations** (RIF resistance):
  - S531L: **34.01%** frequency (most common)
  - S450L: **19.78%** frequency
  - H526Y: **4.4%** frequency
  - WT8 probe: **15.38%** frequency
  - WT7 probe: **4.4%** frequency

- **katG mutations** (INH resistance):
  - S315T: **68.6%** frequency (most common)
  - WT probe: **12.4%** frequency

- **inhA mutations** (INH resistance):
  - C15T: **11.57%** frequency
  - WT1 probe: **4.13%** frequency
  - WT2 probe: **0.83%** frequency
  - MUT1 probe: **0.83%** frequency

**Sample Size**: 932 culture-positive MTBC isolates from 6 studies

### B. Iranian Study (PMC8113720)
**Title**: "Detection of genomic mutations in katG and rpoB genes among multidrug-resistant Mycobacterium tuberculosis isolates from Tehran, Iran"

**Key Findings**:
- **katG mutations** (Isoniazid resistance):
  - 315 AGC > ACC (S→T): **70%** frequency
  - 315 AGC > ACC (S→T) and 335 ATC > GTC (I→V): **10%** frequency

- **rpoB mutations** (Rifampin resistance):
  - 441 ACC > TCC (T→S): **50%** frequency
  - 456 CGG > TGG (R→T): **30%** frequency

**Sample Size**: 44 M. tuberculosis strains, 10 MDR isolates

### C. Large-Scale Genomic Analysis (Nature Scientific Reports)
**Title**: "Large-scale genomic analysis of Mycobacterium tuberculosis reveals extent of target and compensatory mutations linked to multi-drug resistant tuberculosis"

**Key Findings** (from ~32,669 isolates):
- **katG mutations** (INH resistance):
  - Ser315Thr: **21.9%** frequency (n=7,165 isolates)
  - Plus 31 putative novel mutations identified

- **fabG1 mutations** (INH resistance):
  - -15C>T: **6.1%** frequency (n=1,989 isolates)

- **inhA mutations** (INH resistance):
  - -154G>A: **1.0%** frequency (n=332 isolates)

- **rpoB mutations** (RIF resistance):
  - Ser450Leu: **15.2%** frequency
  - Asp435Val: **1.8%** frequency
  - His445Tyr: **1.3%** frequency

**Sample Size**: 32,669 Mtb isolates with whole genome sequencing

---

## 3. Clinical Metadata Sources

### A. Mendeley Dataset
**URL**: https://data.mendeley.com/datasets/gn4xjcdvxv

**Dataset**: Comprehensive Dataset on Suspected Tuberculosis (TBC) Patients in Semarang, Indonesia

**Contents**:
- Socio-demographic data (age, gender, region)
- Clinical data (symptoms, sputum test results)
- Treatment outcomes
- Patient characteristics

**Access**: Public, CC BY 4.0 license

**Download**: Available as Excel files (17.6 MB)

---

## 4. Real Mutation Frequencies Implemented

### Rifampin (RIF) Resistance - rpoB Gene:
| Mutation | Frequency | Source |
|----------|-----------|--------|
| S531L (Ser531Leu) | 34.01% | Ethiopian study (PMC9225881) |
| S450L (Ser450Leu) | 19.78% / 15.2% | Ethiopian study / Large-scale |
| H526Y (His526Tyr) | 4.4% | Ethiopian study |
| H445Y (His445Tyr) | 1.3% | Large-scale study |
| D435V (Asp435Val) | 1.8% | Large-scale study |

### Isoniazid (INH) Resistance - katG Gene:
| Mutation | Frequency | Source |
|----------|-----------|--------|
| S315T (Ser315Thr) | 68.6% / 70% / 21.9% | Ethiopian / Iranian / Large-scale |
| S315N (Ser315Asn) | Rare | Literature |

### Isoniazid (INH) Resistance - inhA/fabG1 Genes:
| Mutation | Frequency | Source |
|----------|-----------|--------|
| inhA C15T | 11.57% | Ethiopian study |
| fabG1 -15C>T | 6.1% | Large-scale study (n=1,989) |
| inhA -154G>A | 1.0% | Large-scale study (n=332) |

---

## 5. Implementation Status

✅ **Completed**:
- Scraped WHO TB database structure and CSV download links
- Extracted mutation frequencies from 3 research papers
- Identified Mendeley clinical dataset
- Updated `scrape_genomic_mutations()` function with real frequencies
- Created documentation of scraped sources

⚠️ **Pending** (for full integration):
- Download and parse WHO CSV files
- Integrate Mendeley dataset clinical metadata
- Set up NCBI Entrez API queries for genomic data
- Add real-time data fetching capabilities

---

## 6. Next Steps for Full Integration

1. **WHO Data Integration**:
   - Download CSV files from WHO database
   - Parse regional TB statistics
   - Integrate treatment outcome data

2. **NCBI Entrez API**:
   - Set up proper email for Entrez API
   - Query BioProject/BioSample for clinical metadata
   - Fetch genomic sequences from GenBank

3. **Mendeley Dataset**:
   - Download and parse Excel files
   - Extract patient demographics
   - Integrate with existing clinical data

4. **Real-time Scraping**:
   - Add periodic data refresh
   - Cache scraped data
   - Update mutation frequencies from latest research

---

## 7. Data Sources Summary

| Source Type | Status | Data Available |
|-------------|--------|----------------|
| WHO TB Database | ✅ Scraped | CSV download links, structure |
| Research Papers | ✅ Scraped | Mutation frequencies, patterns |
| Mendeley Dataset | ✅ Identified | Clinical metadata dataset |
| NCBI Entrez API | ⚠️ Ready | Requires email setup |
| ReSeqTB Database | ❌ DNS Failed | May require registration |

---

## 8. References

1. Seid, A., Berhane, N., & Nureddin, S. (2022). Frequency of rpoB, katG, and inhA Gene Polymorphisms Associated with Multidrug-Resistant Mycobacterium tuberculosis Complex Isolates among Ethiopian TB Patients: A Systematic Review. *Interdisciplinary Perspectives on Infectious Diseases*, 2022, 1967675. https://doi.org/10.1155/2022/1967675

2. Motavaf, B., et al. (2021). Detection of genomic mutations in katG and rpoB genes among multidrug-resistant Mycobacterium tuberculosis isolates from Tehran, Iran. *New Microbes and New Infections*, 41, 100879. https://doi.org/10.1016/j.nmni.2021.100879

3. Napier, G., et al. (2023). Large-scale genomic analysis of Mycobacterium tuberculosis reveals extent of target and compensatory mutations linked to multi-drug resistant tuberculosis. *Scientific Reports*, 13, 623. https://doi.org/10.1038/s41598-023-27516-4

4. WHO Global Tuberculosis Programme. (2024). Data. https://www.who.int/teams/global-tuberculosis-programme/data

5. Supriyanto, S., & Ilham, A. (2024). Comprehensive Dataset on Suspected Tuberculosis (TBC) Patients in Semarang, Indonesia. Mendeley Data, V2. https://doi.org/10.17632/gn4xjcdvxv.2

