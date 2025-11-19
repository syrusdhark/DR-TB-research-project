"""
Configuration settings for DR-TB Prediction Web Interface
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "results" / "models"
DATA_DIR = BASE_DIR / "data"

# Model configuration
IMG_SIZE = 380
OPTIMAL_THRESHOLD = 0.638  # From evaluation_results.json
NUM_CLINICAL_FEATURES = 14
NUM_GENOMIC_FEATURES = 12

# Clinical feature names (in order)
CLINICAL_FEATURES = [
    'age',
    'previous_tb_treatment',
    'hiv_status',
    'diabetes_status',
    'smoking_status',
    'mdr_tb',
    'xdr_tb',
    'rifampin_resistance',
    'isoniazid_resistance',
    'gender_encoded',
    'region_Africa',
    'region_Americas',
    'region_Asia',
    'region_Europe'
]

# Genomic feature names (in order)
GENOMIC_FEATURES = [
    'rpoB_S531L',
    'rpoB_S450L',
    'rpoB_H526Y',
    'rpoB_H445Y',
    'rpoB_D435V',
    'katG_S315T',
    'katG_S315N',
    'inhA_C15T',
    'fabG1_C15T',
    'pncA_H57D',
    'embB_M306V',
    'mutation_count'
]

BASE_MUTATION_FEATURES = [feature for feature in GENOMIC_FEATURES if feature != 'mutation_count']

# Region options
REGIONS = ['Africa', 'Americas', 'Asia', 'Europe']

# Gender encoding
GENDER_MAPPING = {'Male': 1, 'Female': 0, 'M': 1, 'F': 0}

# Medication history storage key
MEDICATION_HISTORY_FIELD = 'medication_history'

# Comorbidity feature tracking (used for reporting and UI)
COMORBIDITY_FEATURES = [
    'hiv_aids',
    'diabetes_mellitus',
    'silicosis',
    'chronic_kidney_disease',
    'organ_transplantation',
    'tnf_alpha_inhibitors',
    'malnutrition_low_bmi',
    'alcohol_use_disorder',
    'copd',
    'rheumatoid_arthritis',
    'cancer_hematological',
    'cancer_head_neck_lung',
    'gastrectomy_bypass',
    'vitamin_d_deficiency',
    'cystic_fibrosis',
    'sickle_cell_disease',
    'sle',
    'injecting_drug_use',
    'post_covid19_lung_damage',
    'asthma',
    'pneumonia'
]

COMORBIDITY_LABELS = {
    'hiv_aids': 'HIV / AIDS',
    'diabetes_mellitus': 'Diabetes Mellitus',
    'silicosis': 'Silicosis',
    'chronic_kidney_disease': 'Chronic Kidney Disease / ESRD',
    'organ_transplantation': 'Organ Transplantation',
    'tnf_alpha_inhibitors': 'TNF-α Inhibitor Therapy',
    'malnutrition_low_bmi': 'Malnutrition / Low BMI',
    'alcohol_use_disorder': 'Alcohol Use Disorder',
    'copd': 'Chronic Obstructive Pulmonary Disease (COPD)',
    'rheumatoid_arthritis': 'Rheumatoid Arthritis',
    'cancer_hematological': 'Cancer - Hematological',
    'cancer_head_neck_lung': 'Cancer - Head/Neck or Lung',
    'gastrectomy_bypass': 'Gastrectomy / Jejunoileal Bypass',
    'vitamin_d_deficiency': 'Vitamin D Deficiency',
    'cystic_fibrosis': 'Cystic Fibrosis',
    'sickle_cell_disease': 'Sickle Cell Disease',
    'sle': 'Systemic Lupus Erythematosus (SLE)',
    'injecting_drug_use': 'Injecting Drug Use',
    'post_covid19_lung_damage': 'Post-COVID-19 Lung Damage',
    'asthma': 'Asthma',
    'pneumonia': 'Pneumonia'
}

COMORBIDITY_DESCRIPTIONS = {
    'hiv_aids': {
        'description': 'HIV/AIDS with severe CD4 depletion dramatically increases TB reactivation risk.',
        'severity': 'Extremely High',
        'relative_risk': '18–30×'
    },
    'diabetes_mellitus': {
        'description': 'Diabetes (Type 1 or 2) triples the risk of active TB and worsens treatment outcomes.',
        'severity': 'Very High',
        'relative_risk': '2.5–3.5×'
    },
    'silicosis': {
        'description': 'Silicosis causes chronic lung scarring and is one of the strongest non-HIV risk factors for TB.',
        'severity': 'Extremely High',
        'relative_risk': '10–30×'
    },
    'chronic_kidney_disease': {
        'description': 'Chronic kidney disease or dialysis-related uremia suppresses immunity and raises TB risk.',
        'severity': 'Very High',
        'relative_risk': '10–25×'
    },
    'organ_transplantation': {
        'description': 'Solid organ or stem cell transplantation involves lifelong immunosuppression and high TB risk.',
        'severity': 'Very High',
        'relative_risk': '20–74×'
    },
    'tnf_alpha_inhibitors': {
        'description': 'TNF-α inhibitors (e.g., infliximab) impair granuloma maintenance leading to TB reactivation.',
        'severity': 'High',
        'relative_risk': '5–20×'
    },
    'malnutrition_low_bmi': {
        'description': 'Malnutrition (BMI < 18.5) weakens cell-mediated immunity, common in high-burden regions.',
        'severity': 'High',
        'relative_risk': '2–12×'
    },
    'alcohol_use_disorder': {
        'description': 'Chronic alcohol use (>40 g/day) impairs immunity and correlates with poor TB adherence.',
        'severity': 'High',
        'relative_risk': '3–4×'
    },
    'copd': {
        'description': 'COPD with inhaled steroids and structural lung damage increases TB susceptibility.',
        'severity': 'Medium',
        'relative_risk': '2–3×'
    },
    'rheumatoid_arthritis': {
        'description': 'Rheumatoid arthritis and its therapies elevate TB risk via chronic inflammation and steroids.',
        'severity': 'Medium',
        'relative_risk': '2–4×'
    },
    'cancer_hematological': {
        'description': 'Hematological malignancies and chemotherapy cause profound immunosuppression.',
        'severity': 'Medium-High',
        'relative_risk': '2–10×'
    },
    'cancer_head_neck_lung': {
        'description': 'Head, neck, and lung cancers disrupt airway defenses and often require immunosuppressive therapy.',
        'severity': 'Medium-High',
        'relative_risk': '2–10×'
    },
    'gastrectomy_bypass': {
        'description': 'Gastrectomy/jejunoileal bypass reduces gastric acidity, facilitating primary progressive TB.',
        'severity': 'Medium',
        'relative_risk': '2–5×'
    },
    'vitamin_d_deficiency': {
        'description': 'Vitamin D deficiency impairs macrophage activity and is linked to higher TB incidence.',
        'severity': 'Medium',
        'relative_risk': '2–5×'
    },
    'cystic_fibrosis': {
        'description': 'Cystic fibrosis patients have chronic airway damage and frequent antibiotics increasing TB susceptibility.',
        'severity': 'Medium',
        'relative_risk': 'Increased'
    },
    'sickle_cell_disease': {
        'description': 'Sickle cell disease causes functional asplenia and pulmonary infarcts that predispose to infection.',
        'severity': 'Medium',
        'relative_risk': 'Increased'
    },
    'sle': {
        'description': 'Systemic lupus erythematosus plus immunosuppressive therapy elevates TB reactivation risk.',
        'severity': 'Medium',
        'relative_risk': '3–7×'
    },
    'injecting_drug_use': {
        'description': 'Injection drug use overlaps with HIV, malnutrition, and unstable housing, compounding TB risk.',
        'severity': 'Medium',
        'relative_risk': '2–10×'
    },
    'post_covid19_lung_damage': {
        'description': 'Post-COVID-19 lung damage (within 12 months) is linked to 2–6× higher TB incidence.',
        'severity': 'Medium',
        'relative_risk': '2–6×'
    },
    'asthma': {
        'description': 'Asthma may complicate TB diagnosis and treatment, especially with systemic steroids.',
        'severity': 'Medium',
        'relative_risk': 'Increased'
    },
    'pneumonia': {
        'description': 'Recent pneumonia may indicate structural lung damage that increases TB susceptibility.',
        'severity': 'Medium',
        'relative_risk': 'Increased'
    }
}

# Default values for optional inputs
DEFAULT_CLINICAL = {
    'age': 45,
    'previous_tb_treatment': 0,
    'hiv_status': 0,
    'diabetes_status': 0,
    'smoking_status': 0,
    'mdr_tb': 0,
    'xdr_tb': 0,
    'rifampin_resistance': 0,
    'isoniazid_resistance': 0,
    'gender_encoded': 0,  # Female
    'region_Africa': 0,
    'region_Americas': 0,
    'region_Asia': 1,  # Default to Asia
    'region_Europe': 0
}

DEFAULT_GENOMIC = {
    'rpoB_S531L': 0,
    'rpoB_S450L': 0,
    'rpoB_H526Y': 0,
    'rpoB_H445Y': 0,
    'rpoB_D435V': 0,
    'katG_S315T': 0,
    'katG_S315N': 0,
    'inhA_C15T': 0,
    'fabG1_C15T': 0,
    'pncA_H57D': 0,
    'embB_M306V': 0,
    'mutation_count': 0
}

# Image normalization (ImageNet stats)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Multi-label resistance drugs (WHO 2025 catalogue)
TARGET_RESISTANCE_DRUGS = [
    'rifampin',
    'isoniazid',
    'pyrazinamide',
    'ethambutol',
    'moxifloxacin',
    'levofloxacin',
    'amikacin',
    'kanamycin',
    'capreomycin',
    'bedaquiline',
    'linezolid',
    'clofazimine',
    'delamanid',
    'pretomanid',
    'ethionamide',
]

DRUG_DISPLAY_NAMES = {
    'rifampin': 'Rifampin (RIF)',
    'isoniazid': 'Isoniazid (INH)',
    'pyrazinamide': 'Pyrazinamide (PZA)',
    'ethambutol': 'Ethambutol (EMB)',
    'moxifloxacin': 'Moxifloxacin (MXF)',
    'levofloxacin': 'Levofloxacin (LEV)',
    'amikacin': 'Amikacin (AMI)',
    'kanamycin': 'Kanamycin (KAN)',
    'capreomycin': 'Capreomycin (CAP)',
    'bedaquiline': 'Bedaquiline (BDQ)',
    'linezolid': 'Linezolid (LZD)',
    'clofazimine': 'Clofazimine (CFZ)',
    'delamanid': 'Delamanid (DLM)',
    'pretomanid': 'Pretomanid (PTO)',
    'ethionamide': 'Ethionamide (ETH)',
}

# Map CRyPTIC drug codes to target drug names
CRyPTIC_DRUG_CODE_MAPPING = {
    'RIF': 'rifampin',
    'RFB': 'rifampin',  # Rifabutin also informs RIF resistance
    'INH': 'isoniazid',
    'PZA': 'pyrazinamide',
    'EMB': 'ethambutol',
    'MXF': 'moxifloxacin',
    'OFX': 'moxifloxacin',  # use ofloxacin MICs as proxy if MXF missing
    'LEV': 'levofloxacin',
    'AMI': 'amikacin',
    'KAN': 'kanamycin',
    'CAP': 'capreomycin',
    'STM': 'kanamycin',  # streptomycin not target drug but useful for injectables
    'BDQ': 'bedaquiline',
    'LZD': 'linezolid',
    'CFZ': 'clofazimine',
    'DLM': 'delamanid',
    'PTO': 'pretomanid',
    'ETH': 'ethionamide',
    'PAS': None,
    'ETO': 'ethionamide',
}

# WHO 2025 critical concentrations (mg/L) for MGIT/MIC interpretation
DRUG_BREAKPOINTS = {
    'rifampin': 1.0,
    'isoniazid': 0.2,
    'pyrazinamide': 100.0,
    'ethambutol': 2.0,
    'moxifloxacin': 0.5,
    'levofloxacin': 1.0,
    'amikacin': 1.0,
    'kanamycin': 2.5,
    'capreomycin': 2.5,
    'bedaquiline': 0.25,
    'linezolid': 1.0,
    'clofazimine': 0.25,
    'delamanid': 0.016,
    'pretomanid': 0.06,
    'ethionamide': 2.5,
}

FLUOROQUINOLONE_DRUGS = {'moxifloxacin', 'levofloxacin'}
INJECTABLE_DRUGS = {'amikacin', 'kanamycin', 'capreomycin'}

# Mutation descriptions for report generation
MUTATION_DESCRIPTIONS = {
    'rpoB_S531L': 'Rifampin resistance mutation (rpoB S531L) - High frequency (~34%)',
    'rpoB_S450L': 'Rifampin resistance mutation (rpoB S450L) - High frequency (~20%)',
    'rpoB_H526Y': 'Rifampin resistance mutation (rpoB H526Y) - Moderate frequency (~4.4%)',
    'rpoB_H445Y': 'Rifampin resistance mutation (rpoB H445Y) - Low frequency (~1.3%)',
    'rpoB_D435V': 'Rifampin resistance mutation (rpoB D435V) - Low frequency (~1.8%)',
    'katG_S315T': 'Isoniazid resistance mutation (katG S315T) - Very high frequency (~70%)',
    'katG_S315N': 'Isoniazid resistance mutation (katG S315N) - Rare',
    'inhA_C15T': 'Isoniazid resistance mutation (inhA C15T) - Moderate frequency (~11.6%)',
    'fabG1_C15T': 'Isoniazid resistance mutation (fabG1 -15C>T) - Moderate frequency (~6.1%)',
    'pncA_H57D': 'Pyrazinamide resistance mutation (pncA H57D)',
    'embB_M306V': 'Ethambutol resistance mutation (embB M306V)',
}

EXTENDED_MUTATION_FEATURES = [
    *BASE_MUTATION_FEATURES,
    'gyrA_A90V',
    'gyrA_S91P',
    'gyrA_D94G',
    'gyrA_D94A',
    'gyrA_D94N',
    'gyrB_N538D',
    'gyrB_E540V',
    'rrs_A1401G',
    'rrs_G1484T',
    'eis_C14T',
    'eis_G10A',
    'atpE_A63P',
    'pepQ_Q10P',
    'ddn_S33F',
    'fbiC_W700STOP',
]

MUTATION_ALIAS_MAP = {
    ('rpoB', 'S531L'): 'rpoB_S531L',
    ('rpoB', 'S450L'): 'rpoB_S450L',
    ('rpoB', 'H526Y'): 'rpoB_H526Y',
    ('rpoB', 'H445Y'): 'rpoB_H445Y',
    ('rpoB', 'D435V'): 'rpoB_D435V',
    ('katG', 'S315T'): 'katG_S315T',
    ('katG', 'S315N'): 'katG_S315N',
    ('inhA', 'C15T'): 'inhA_C15T',
    ('fabG1', 'C15T'): 'fabG1_C15T',
    ('pncA', 'H57D'): 'pncA_H57D',
    ('embB', 'M306V'): 'embB_M306V',
    ('gyrA', 'A90V'): 'gyrA_A90V',
    ('gyrA', 'S91P'): 'gyrA_S91P',
    ('gyrA', 'D94G'): 'gyrA_D94G',
    ('gyrA', 'D94A'): 'gyrA_D94A',
    ('gyrA', 'D94N'): 'gyrA_D94N',
    ('gyrB', 'N538D'): 'gyrB_N538D',
    ('gyrB', 'E540V'): 'gyrB_E540V',
    ('rrs', 'A1401G'): 'rrs_A1401G',
    ('rrs', 'G1484T'): 'rrs_G1484T',
    ('eis', 'C14T'): 'eis_C14T',
    ('eis', 'G10A'): 'eis_G10A',
    ('atpE', 'A63P'): 'atpE_A63P',
    ('pepQ', 'Q10P'): 'pepQ_Q10P',
    ('ddn', 'S33F'): 'ddn_S33F',
    ('fbiC', 'W700*'): 'fbiC_W700STOP',
    ('fbiC', 'W700STOP'): 'fbiC_W700STOP',
}

# Risk factor descriptions
RISK_FACTOR_DESCRIPTIONS = {
    'previous_tb_treatment': 'Previous TB treatment increases risk of drug resistance',
    'hiv_status': 'HIV co-infection is a significant risk factor for TB and drug resistance',
    'diabetes_status': 'Diabetes increases susceptibility to TB and complications',
    'smoking_status': 'Smoking is associated with increased TB risk and poor outcomes',
    'mdr_tb': 'Multi-drug resistant TB confirmed',
    'xdr_tb': 'Extensively drug-resistant TB confirmed',
    'rifampin_resistance': 'Rifampin resistance detected',
    'isoniazid_resistance': 'Isoniazid resistance detected',
    'copd': COMORBIDITY_DESCRIPTIONS['copd']['description'],
    'asthma': COMORBIDITY_DESCRIPTIONS['asthma']['description'],
    'pneumonia': COMORBIDITY_DESCRIPTIONS['pneumonia']['description'],
    'covid19': COMORBIDITY_DESCRIPTIONS['post_covid19_lung_damage']['description'],
    'silicosis': COMORBIDITY_DESCRIPTIONS['silicosis']['description'],
    'chronic_kidney_disease': COMORBIDITY_DESCRIPTIONS['chronic_kidney_disease']['description'],
    'organ_transplantation': COMORBIDITY_DESCRIPTIONS['organ_transplantation']['description'],
    'tnf_alpha_inhibitors': COMORBIDITY_DESCRIPTIONS['tnf_alpha_inhibitors']['description'],
    'malnutrition_low_bmi': COMORBIDITY_DESCRIPTIONS['malnutrition_low_bmi']['description'],
    'alcohol_use_disorder': COMORBIDITY_DESCRIPTIONS['alcohol_use_disorder']['description'],
    'rheumatoid_arthritis': COMORBIDITY_DESCRIPTIONS['rheumatoid_arthritis']['description'],
    'cancer_hematological': COMORBIDITY_DESCRIPTIONS['cancer_hematological']['description'],
    'cancer_head_neck_lung': COMORBIDITY_DESCRIPTIONS['cancer_head_neck_lung']['description'],
    'gastrectomy_bypass': COMORBIDITY_DESCRIPTIONS['gastrectomy_bypass']['description'],
    'vitamin_d_deficiency': COMORBIDITY_DESCRIPTIONS['vitamin_d_deficiency']['description'],
    'cystic_fibrosis': COMORBIDITY_DESCRIPTIONS['cystic_fibrosis']['description'],
    'sickle_cell_disease': COMORBIDITY_DESCRIPTIONS['sickle_cell_disease']['description'],
    'sle': COMORBIDITY_DESCRIPTIONS['sle']['description'],
    'injecting_drug_use': COMORBIDITY_DESCRIPTIONS['injecting_drug_use']['description'],
}

def get_latest_model_path():
    """Get the path to the latest model checkpoint."""
    if not MODELS_DIR.exists():
        return None
    
    model_files = list(MODELS_DIR.glob("*.pth"))
    if not model_files:
        return None
    
    # Sort by modification time, get latest
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    return latest_model

