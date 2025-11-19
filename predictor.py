"""
Prediction module for DR-TB diagnosis.
Handles model inference and probability calculations.
"""

import torch
import torch.nn.functional as F
import numpy as np
import config


def predict_drtb(model, cxr_image, clinical_features, genomic_features, 
                 device, threshold=None, return_attention=False):
    """
    Make DR-TB prediction using the multimodal fusion model.
    
    Args:
        model: Loaded MultimodalFusionModel
        cxr_image: Preprocessed CXR image tensor (batch_size=1, 3, H, W)
        clinical_features: Clinical features tensor (batch_size=1, 14)
        genomic_features: Genomic features tensor (batch_size=1, 12)
        device: Device to run inference on
        threshold: Classification threshold (default: config.OPTIMAL_THRESHOLD)
        return_attention: Whether to return attention weights
    
    Returns:
        Dictionary with prediction results:
        - prediction: 'DR-TB' or 'Normal'
        - probability: Raw probability (0-1)
        - confidence: Confidence score (0-100)
        - risk_level: 'High', 'Medium', or 'Low'
        - attention_weights: (optional) Attention weights for interpretability
    """
    if threshold is None:
        threshold = config.OPTIMAL_THRESHOLD
    
    # Move inputs to device
    cxr_image = cxr_image.to(device)
    clinical_features = clinical_features.to(device)
    genomic_features = genomic_features.to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        output, attention_weights = model(cxr_image, clinical_features, genomic_features)
        
        # Apply sigmoid to get probability
        probability = torch.sigmoid(output).cpu().item()
    
    # Binary prediction using threshold
    prediction = 'DR-TB' if probability >= threshold else 'Normal'
    
    # Calculate confidence (distance from threshold, normalized to 0-100)
    if prediction == 'DR-TB':
        # For DR-TB: confidence increases as probability increases above threshold
        confidence = min(100, ((probability - threshold) / (1 - threshold)) * 100 + 50)
    else:
        # For Normal: confidence increases as probability decreases below threshold
        confidence = min(100, ((threshold - probability) / threshold) * 100 + 50)
    
    # Determine risk level
    if probability >= 0.8:
        risk_level = 'High'
    elif probability >= threshold:
        risk_level = 'Medium'
    else:
        risk_level = 'Low'
    
    # Prepare results
    results = {
        'prediction': prediction,
        'probability': probability,
        'confidence': confidence,
        'risk_level': risk_level,
        'threshold_used': threshold
    }
    
    if return_attention:
        # Extract attention weights for interpretability
        attention_weights = attention_weights.cpu().numpy().flatten()
        results['attention_weights'] = {
            'cxr': float(attention_weights[0]),
            'clinical': float(attention_weights[1]),
            'genomic': float(attention_weights[2])
        }
    
    return results


def batch_predict(model, cxr_images, clinical_features_list, genomic_features_list,
                 device, threshold=None):
    """
    Make batch predictions (for multiple samples).
    
    Args:
        model: Loaded MultimodalFusionModel
        cxr_images: Batch of preprocessed CXR images (batch_size, 3, H, W)
        clinical_features_list: List of clinical feature tensors
        genomic_features_list: List of genomic feature tensors
        device: Device to run inference on
        threshold: Classification threshold
    
    Returns:
        List of prediction result dictionaries
    """
    if threshold is None:
        threshold = config.OPTIMAL_THRESHOLD
    
    # Stack features if needed
    if isinstance(clinical_features_list, list):
        clinical_batch = torch.cat(clinical_features_list, dim=0)
    else:
        clinical_batch = clinical_features_list
    
    if isinstance(genomic_features_list, list):
        genomic_batch = torch.cat(genomic_features_list, dim=0)
    else:
        genomic_batch = genomic_features_list
    
    # Move to device
    cxr_images = cxr_images.to(device)
    clinical_batch = clinical_batch.to(device)
    genomic_batch = genomic_batch.to(device)
    
    # Run batch inference
    model.eval()
    with torch.no_grad():
        outputs, _ = model(cxr_images, clinical_batch, genomic_batch)
        probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()
    
    # Process each prediction
    results = []
    for prob in probabilities:
        prediction = 'DR-TB' if prob >= threshold else 'Normal'
        
        if prediction == 'DR-TB':
            confidence = min(100, ((prob - threshold) / (1 - threshold)) * 100 + 50)
        else:
            confidence = min(100, ((threshold - prob) / threshold) * 100 + 50)
        
        if prob >= 0.8:
            risk_level = 'High'
        elif prob >= threshold:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        results.append({
            'prediction': prediction,
            'probability': float(prob),
            'confidence': float(confidence),
            'risk_level': risk_level,
            'threshold_used': threshold
        })
    
    return results

