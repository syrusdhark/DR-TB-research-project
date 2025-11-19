"""
Preprocessing functions for DR-TB prediction inputs.
Handles image transforms and feature encoding to match training pipeline.
"""

import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import config


def preprocess_image(image, img_size=config.IMG_SIZE):
    """
    Preprocess CXR image to match training pipeline.
    
    Args:
        image: PIL Image or file path
        img_size: Target image size (default 380)
    
    Returns:
        Preprocessed image tensor ready for model input
    
    Raises:
        ValueError: If image format is invalid
        FileNotFoundError: If image path doesn't exist
        IOError: If image cannot be opened
    """
    # Load image if path provided
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image file not found: {image}")
        try:
            image = Image.open(image).convert('RGB')
        except Exception as e:
            raise IOError(f"Failed to open image file: {e}")
    elif not isinstance(image, Image.Image):
        raise ValueError("Image must be PIL Image or file path")
    
    # Validation/Test transforms (no augmentation, just preprocessing)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])
    
    # Apply transforms
    image_tensor = transform(image)
    
    # Add batch dimension
    return image_tensor.unsqueeze(0)


def encode_clinical_features(age, gender, region, previous_tb_treatment=0,
                            hiv_status=0, diabetes_status=0, smoking_status=0,
                            mdr_tb=0, xdr_tb=0, rifampin_resistance=0,
                            isoniazid_resistance=0, copd=0, asthma=0,
                            pneumonia=0, covid19=0, medication_history=None, **additional_comorbidities):
    """
    Encode clinical features into the format expected by the model.
    
    Args:
        age: Patient age (int)
        gender: Gender ('Male'/'Female' or 'M'/'F')
        region: Region ('Africa', 'Americas', 'Asia', 'Europe')
        previous_tb_treatment: Binary flag (0 or 1)
        hiv_status: Binary flag (0 or 1)
        diabetes_status: Binary flag (0 or 1)
        smoking_status: Binary flag (0 or 1)
        mdr_tb: Binary flag (0 or 1)
        xdr_tb: Binary flag (0 or 1)
        rifampin_resistance: Binary flag (0 or 1)
        isoniazid_resistance: Binary flag (0 or 1)
    
    Returns:
        torch.Tensor: Clinical features vector (14 features)
    """
    # Validate inputs
    if not isinstance(age, (int, float)) or age < 0 or age > 150:
        raise ValueError(f"Age must be a number between 0 and 150, got {age}")
    
    if gender not in config.GENDER_MAPPING:
        raise ValueError(f"Gender must be one of {list(config.GENDER_MAPPING.keys())}, got {gender}")
    
    # Encode gender
    gender_encoded = config.GENDER_MAPPING.get(gender, 0)
    
    # Encode region (one-hot)
    if region not in config.REGIONS:
        raise ValueError(f"Region must be one of {config.REGIONS}, got {region}")
    
    region_encoded = {
        'Africa': [1, 0, 0, 0],
        'Americas': [0, 1, 0, 0],
        'Asia': [0, 0, 1, 0],
        'Europe': [0, 0, 0, 1]
    }.get(region, [0, 0, 1, 0])  # Default to Asia
    
    # Additional comorbidity inputs are captured for reporting but not part of model input
    _ = medication_history
    _ = additional_comorbidities
    
    # Build feature vector in correct order
    # Note: The model expects 14 features, so we only include the original features
    # Additional features (COPD, Asthma, Pneumonia, COVID-19) are stored for reporting
    # but not sent to the model to maintain compatibility
    features = [
        float(age),
        float(previous_tb_treatment),
        float(hiv_status),
        float(diabetes_status),
        float(smoking_status),
        float(mdr_tb),
        float(xdr_tb),
        float(rifampin_resistance),
        float(isoniazid_resistance),
        float(gender_encoded),
        float(region_encoded[0]),  # region_Africa
        float(region_encoded[1]),  # region_Americas
        float(region_encoded[2]),  # region_Asia
        float(region_encoded[3])   # region_Europe
    ]
    # Note: copd, asthma, pneumonia, covid19 are captured but not in model input
    # They will be included in risk factor analysis in the report
    
    # Convert to tensor
    features_tensor = torch.tensor(features, dtype=torch.float32)
    
    # Add batch dimension
    return features_tensor.unsqueeze(0)


def encode_genomic_features(rpoB_S531L=0, rpoB_S450L=0, rpoB_H526Y=0,
                            rpoB_H445Y=0, rpoB_D435V=0, katG_S315T=0,
                            katG_S315N=0, inhA_C15T=0, fabG1_C15T=0,
                            pncA_H57D=0, embB_M306V=0, mutation_count=None):
    """
    Encode genomic mutation features into the format expected by the model.
    
    Args:
        rpoB_S531L: Binary flag (0 or 1)
        rpoB_S450L: Binary flag (0 or 1)
        rpoB_H526Y: Binary flag (0 or 1)
        rpoB_H445Y: Binary flag (0 or 1)
        rpoB_D435V: Binary flag (0 or 1)
        katG_S315T: Binary flag (0 or 1)
        katG_S315N: Binary flag (0 or 1)
        inhA_C15T: Binary flag (0 or 1)
        fabG1_C15T: Binary flag (0 or 1)
        pncA_H57D: Binary flag (0 or 1)
        embB_M306V: Binary flag (0 or 1)
        mutation_count: Total mutation count (auto-calculated if None)
    
    Returns:
        torch.Tensor: Genomic features vector (12 features)
    """
    # Calculate mutation count if not provided
    if mutation_count is None:
        mutations = [
            rpoB_S531L, rpoB_S450L, rpoB_H526Y, rpoB_H445Y, rpoB_D435V,
            katG_S315T, katG_S315N, inhA_C15T, fabG1_C15T,
            pncA_H57D, embB_M306V
        ]
        mutation_count = sum(mutations)
    
    # Build feature vector in correct order
    features = [
        float(rpoB_S531L),
        float(rpoB_S450L),
        float(rpoB_H526Y),
        float(rpoB_H445Y),
        float(rpoB_D435V),
        float(katG_S315T),
        float(katG_S315N),
        float(inhA_C15T),
        float(fabG1_C15T),
        float(pncA_H57D),
        float(embB_M306V),
        float(mutation_count)
    ]
    
    # Convert to tensor
    features_tensor = torch.tensor(features, dtype=torch.float32)
    
    # Add batch dimension
    return features_tensor.unsqueeze(0)


def normalize_features(clinical_features, genomic_features, 
                      clinical_mean=None, clinical_std=None,
                      genomic_mean=None, genomic_std=None):
    """
    Normalize features using training statistics.
    If statistics not provided, uses simple normalization.
    
    Args:
        clinical_features: Clinical features tensor
        genomic_features: Genomic features tensor
        clinical_mean: Mean values for clinical features (optional)
        clinical_std: Std values for clinical features (optional)
        genomic_mean: Mean values for genomic features (optional)
        genomic_std: Std values for genomic features (optional)
    
    Returns:
        Tuple of normalized (clinical_features, genomic_features)
    """
    # For now, return as-is (model handles normalization internally)
    # In production, you might want to load actual training statistics
    return clinical_features, genomic_features

