"""
Prediction module for the two-stage TB / DR-TB pipeline.

predict_tb()        runs Stage 1 (CXR -> TB vs Normal).
predict_drtb_risk()  runs Stage 2 (clinical + genomic -> DR-TB risk).

These are independent calls on independent models -- the DR-TB risk score is
never influenced by the uploaded image, and the TB detection score is never
influenced by clinical/genomic inputs. Keeping them separate is the fix for
the previous fused model, which had no valid basis to let an X-ray inform a
drug-resistance decision (see model.py's module docstring).
"""

import torch


def predict_tb(model, cxr_image, device, threshold=0.5):
    """
    Predict TB vs Normal from a preprocessed CXR image tensor.

    Returns a dict with prediction, probability, confidence, risk_level.
    """
    cxr_image = cxr_image.to(device)

    model.eval()
    with torch.no_grad():
        output = model(cxr_image)
        probability = torch.sigmoid(output).cpu().item()

    prediction = 'Tuberculosis' if probability >= threshold else 'Normal'

    if prediction == 'Tuberculosis':
        confidence = min(100, ((probability - threshold) / max(1e-6, 1 - threshold)) * 100 + 50)
    else:
        confidence = min(100, ((threshold - probability) / max(1e-6, threshold)) * 100 + 50)

    if probability >= 0.8:
        risk_level = 'High'
    elif probability >= threshold:
        risk_level = 'Medium'
    else:
        risk_level = 'Low'

    return {
        'prediction': prediction,
        'probability': probability,
        'confidence': confidence,
        'risk_level': risk_level,
        'threshold_used': threshold,
    }


def predict_drtb_risk(model, clinical_features, genomic_features, device,
                       threshold=0.5, return_modality_weights=True):
    """
    Predict DR-TB risk from clinical + genomic features only (no image).

    Returns a dict with prediction, probability, confidence, risk_level, and
    optionally modality_weights (contribution of clinical vs. genomic).
    """
    clinical_features = clinical_features.to(device)
    genomic_features = genomic_features.to(device)

    model.eval()
    with torch.no_grad():
        output, modality_weights = model(clinical_features, genomic_features)
        probability = torch.sigmoid(output).cpu().item()

    prediction = 'DR-TB Risk' if probability >= threshold else 'Low Risk'

    if prediction == 'DR-TB Risk':
        confidence = min(100, ((probability - threshold) / max(1e-6, 1 - threshold)) * 100 + 50)
    else:
        confidence = min(100, ((threshold - probability) / max(1e-6, threshold)) * 100 + 50)

    if probability >= 0.8:
        risk_level = 'High'
    elif probability >= threshold:
        risk_level = 'Medium'
    else:
        risk_level = 'Low'

    results = {
        'prediction': prediction,
        'probability': probability,
        'confidence': confidence,
        'risk_level': risk_level,
        'threshold_used': threshold,
    }

    if return_modality_weights:
        weights = modality_weights.cpu().numpy().flatten()
        results['modality_weights'] = {
            'clinical': float(weights[0]),
            'genomic': float(weights[1]),
        }

    return results
