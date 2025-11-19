"""
Report generation module for DR-TB predictions.
Creates detailed diagnostic reports with risk factors and recommendations.
"""

import config


def generate_report(prediction_result, clinical_data, genomic_data, attention_weights=None):
    """
    Generate a detailed diagnostic report for DR-TB prediction.
    
    Args:
        prediction_result: Dictionary from predictor.predict_drtb()
        clinical_data: Dictionary with clinical input data
        genomic_data: Dictionary with genomic input data
        attention_weights: Optional attention weights from model
    
    Returns:
        Dictionary with formatted report sections
    """
    report = {
        'summary': _generate_summary(prediction_result),
        'risk_factors': _identify_risk_factors(clinical_data, genomic_data),
        'genomic_analysis': _analyze_genomic_mutations(genomic_data),
        'modality_contributions': _analyze_modality_contributions(attention_weights),
        'recommendations': _generate_recommendations(prediction_result, clinical_data, genomic_data),
        'medication_history': (clinical_data or {}).get('medication_history', '').strip()
    }
    
    return report


def _generate_summary(prediction_result):
    """Generate summary section of report."""
    prediction = prediction_result['prediction']
    probability = prediction_result['probability']
    confidence = prediction_result['confidence']
    risk_level = prediction_result['risk_level']
    
    summary = {
        'prediction': prediction,
        'probability_percent': f"{probability * 100:.2f}%",
        'confidence_percent': f"{confidence:.1f}%",
        'risk_level': risk_level,
        'interpretation': _interpret_prediction(prediction, probability, risk_level)
    }
    
    return summary


def _interpret_prediction(prediction, probability, risk_level):
    """Generate interpretation text for prediction."""
    if prediction == 'DR-TB':
        if risk_level == 'High':
            return (
                "The model predicts Drug-Resistant Tuberculosis (DR-TB) with high confidence. "
                "This patient shows strong indicators of drug-resistant tuberculosis. "
                "Immediate clinical evaluation and drug susceptibility testing (DST) are strongly recommended."
            )
        else:  # Medium
            return (
                "The model predicts Drug-Resistant Tuberculosis (DR-TB) with moderate confidence. "
                "This patient shows indicators of drug-resistant tuberculosis. "
                "Clinical evaluation and drug susceptibility testing (DST) are recommended."
            )
    else:  # Normal
        if risk_level == 'Low':
            return (
                "The model predicts Normal (non-DR-TB) with high confidence. "
                "However, this does not rule out tuberculosis entirely. "
                "Clinical correlation and follow-up are recommended if symptoms persist."
            )
        else:
            return (
                "The model predicts Normal (non-DR-TB). "
                "Clinical correlation is recommended, especially if risk factors are present."
            )


def _identify_risk_factors(clinical_data, genomic_data):
    """Identify and describe risk factors from clinical and genomic data."""
    risk_factors = []
    
    # Clinical risk factors
    if clinical_data.get('previous_tb_treatment', 0):
        risk_factors.append({
            'factor': 'Previous TB Treatment',
            'description': config.RISK_FACTOR_DESCRIPTIONS['previous_tb_treatment'],
            'severity': 'High'
        })
    
    if clinical_data.get('smoking_status', 0):
        risk_factors.append({
            'factor': 'Smoking',
            'description': config.RISK_FACTOR_DESCRIPTIONS['smoking_status'],
            'severity': 'Medium'
        })
    
    if clinical_data.get('mdr_tb', 0):
        risk_factors.append({
            'factor': 'MDR-TB Confirmed',
            'description': config.RISK_FACTOR_DESCRIPTIONS['mdr_tb'],
            'severity': 'Critical'
        })
    
    if clinical_data.get('xdr_tb', 0):
        risk_factors.append({
            'factor': 'XDR-TB Confirmed',
            'description': config.RISK_FACTOR_DESCRIPTIONS['xdr_tb'],
            'severity': 'Critical'
        })
    
    if clinical_data.get('rifampin_resistance', 0):
        risk_factors.append({
            'factor': 'Rifampin Resistance',
            'description': config.RISK_FACTOR_DESCRIPTIONS['rifampin_resistance'],
            'severity': 'High'
        })
    
    if clinical_data.get('isoniazid_resistance', 0):
        risk_factors.append({
            'factor': 'Isoniazid Resistance',
            'description': config.RISK_FACTOR_DESCRIPTIONS['isoniazid_resistance'],
            'severity': 'High'
        })
    
    # Comorbidities and underlying conditions
    comorbidity_flags = {}
    comorbidity_flags.update(clinical_data.get('comorbidities', {}) or {})
    alias_map = {
        'hiv_aids': 'hiv_status',
        'diabetes_mellitus': 'diabetes_status',
        'copd': 'copd',
        'post_covid19_lung_damage': 'covid19',
        'asthma': 'asthma',
        'pneumonia': 'pneumonia'
    }
    for key, alias in alias_map.items():
        if key not in comorbidity_flags and alias in clinical_data:
            comorbidity_flags[key] = clinical_data.get(alias, 0)
    
    for feature in config.COMORBIDITY_FEATURES:
        flag = comorbidity_flags.get(feature, 0)
        if not flag:
            continue
        
        metadata = config.COMORBIDITY_DESCRIPTIONS.get(feature, {})
        factor_label = config.COMORBIDITY_LABELS.get(feature, feature.replace('_', ' ').title())
        risk_factors.append({
            'factor': factor_label,
            'description': metadata.get('description', 'Associated comorbidity increases TB risk.'),
            'severity': metadata.get('severity', 'Medium'),
            'relative_risk': metadata.get('relative_risk', 'N/A')
        })
    
    # Age-based risk (elderly patients)
    age = clinical_data.get('age', 0)
    if age > 65:
        risk_factors.append({
            'factor': 'Advanced Age',
            'description': 'Elderly patients (>65 years) have increased risk of TB complications',
            'severity': 'Medium'
        })
    
    return risk_factors


def _analyze_genomic_mutations(genomic_data):
    """Analyze detected genomic mutations and their significance."""
    mutations_detected = []
    mutation_count = genomic_data.get('mutation_count', 0)
    
    # Check each mutation
    for mutation_name in config.GENOMIC_FEATURES:
        if mutation_name == 'mutation_count':
            continue
        
        if genomic_data.get(mutation_name, 0):
            mutation_desc = config.MUTATION_DESCRIPTIONS.get(
                mutation_name,
                f'{mutation_name} mutation detected'
            )
            
            # Determine significance
            if 'rpoB' in mutation_name:
                significance = 'High - Rifampin resistance'
            elif 'katG' in mutation_name or 'inhA' in mutation_name or 'fabG1' in mutation_name:
                significance = 'High - Isoniazid resistance'
            elif 'pncA' in mutation_name:
                significance = 'Medium - Pyrazinamide resistance'
            elif 'embB' in mutation_name:
                significance = 'Medium - Ethambutol resistance'
            else:
                significance = 'Unknown significance'
            
            mutations_detected.append({
                'mutation': mutation_name,
                'description': mutation_desc,
                'significance': significance
            })
    
    analysis = {
        'total_mutations': mutation_count,
        'mutations_detected': mutations_detected,
        'interpretation': _interpret_mutations(mutation_count, mutations_detected)
    }
    
    return analysis


def _interpret_mutations(mutation_count, mutations_detected):
    """Generate interpretation of mutation analysis."""
    if mutation_count == 0:
        return "No resistance mutations detected in the analyzed genomic regions."
    elif mutation_count == 1:
        return "One resistance mutation detected. Drug susceptibility testing recommended."
    elif mutation_count <= 3:
        return f"{mutation_count} resistance mutations detected. Strong indication of drug resistance."
    else:
        return f"Multiple resistance mutations detected ({mutation_count}). High likelihood of multi-drug resistance."


def _analyze_modality_contributions(attention_weights):
    """Analyze which modalities contributed most to the prediction."""
    if attention_weights is None:
        return {
            'cxr': 'N/A',
            'clinical': 'N/A',
            'genomic': 'N/A',
            'primary_modality': 'N/A'
        }
    
    cxr_weight = attention_weights.get('cxr', 0.33)
    clinical_weight = attention_weights.get('clinical', 0.33)
    genomic_weight = attention_weights.get('genomic', 0.33)
    
    # Find primary modality
    weights = {
        'CXR Image': cxr_weight,
        'Clinical Data': clinical_weight,
        'Genomic Data': genomic_weight
    }
    primary_modality = max(weights, key=weights.get)
    
    return {
        'cxr': f"{cxr_weight * 100:.1f}%",
        'clinical': f"{clinical_weight * 100:.1f}%",
        'genomic': f"{genomic_weight * 100:.1f}%",
        'primary_modality': primary_modality
    }


def _generate_recommendations(prediction_result, clinical_data, genomic_data):
    """Generate clinical recommendations based on prediction and inputs."""
    recommendations = []
    prediction = prediction_result['prediction']
    risk_level = prediction_result['risk_level']
    
    if prediction == 'DR-TB':
        recommendations.append({
            'priority': 'High',
            'action': 'Immediate Drug Susceptibility Testing (DST)',
            'description': 'Perform DST to confirm resistance patterns and guide treatment selection.'
        })
        
        recommendations.append({
            'priority': 'High',
            'action': 'Initiate DR-TB Treatment Protocol',
            'description': 'Begin appropriate drug-resistant TB treatment regimen based on DST results.'
        })
        
        if clinical_data.get('mdr_tb', 0) or clinical_data.get('xdr_tb', 0):
            recommendations.append({
                'priority': 'Critical',
                'action': 'Specialized Care Consultation',
                'description': 'Refer to specialized TB treatment center for MDR/XDR-TB management.'
            })
        
        if clinical_data.get('hiv_status', 0):
            recommendations.append({
                'priority': 'High',
                'action': 'HIV-TB Co-infection Management',
                'description': 'Ensure coordinated care for both HIV and TB, including drug interaction monitoring.'
            })
    else:
        recommendations.append({
            'priority': 'Medium',
            'action': 'Clinical Correlation',
            'description': 'Correlate with patient symptoms, history, and physical examination findings.'
        })
        
        if clinical_data.get('previous_tb_treatment', 0):
            recommendations.append({
                'priority': 'Medium',
                'action': 'Monitor for Recurrence',
                'description': 'Previous TB treatment history warrants close monitoring.'
            })
    
    # General recommendations
    if genomic_data.get('mutation_count', 0) > 0:
        recommendations.append({
            'priority': 'High',
            'action': 'Genomic Confirmation',
            'description': 'Genomic mutations detected. Confirm with phenotypic DST testing.'
        })
    
    recommendations.append({
        'priority': 'Medium',
        'action': 'Follow-up Imaging',
        'description': 'Consider follow-up chest imaging to monitor treatment response.'
    })
    
    return recommendations


def format_report_text(report):
    """Format report dictionary as readable text."""
    lines = []
    
    # Summary
    lines.append("=" * 60)
    lines.append("DR-TB PREDICTION REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    summary = report['summary']
    lines.append("PREDICTION SUMMARY")
    lines.append("-" * 60)
    lines.append(f"Prediction: {summary['prediction']}")
    lines.append(f"Probability: {summary['probability_percent']}")
    lines.append(f"Confidence: {summary['confidence_percent']}%")
    lines.append(f"Risk Level: {summary['risk_level']}")
    lines.append("")
    lines.append(f"Interpretation: {summary['interpretation']}")
    lines.append("")
    
    medication_history = (report.get('medication_history') or "").strip()
    lines.append("MEDICATION HISTORY")
    lines.append("-" * 60)
    if medication_history:
        lines.append(medication_history)
    else:
        lines.append("Not provided.")
    lines.append("")
    
    # Risk Factors
    risk_factors = report['risk_factors']
    lines.append("IDENTIFIED RISK FACTORS")
    lines.append("-" * 60)
    if risk_factors:
        for rf in risk_factors:
            rr = rf.get('relative_risk')
            details = rf['description']
            if rr and rr != 'N/A':
                details = f"{details} (Relative Risk: {rr})"
            lines.append(f"• {rf['factor']} ({rf['severity']}): {details}")
    else:
        lines.append("No significant risk factors identified.")
    lines.append("")
    
    # Genomic Analysis
    genomic = report['genomic_analysis']
    lines.append("GENOMIC MUTATION ANALYSIS")
    lines.append("-" * 60)
    lines.append(f"Total Mutations Detected: {genomic['total_mutations']}")
    if genomic['mutations_detected']:
        for mut in genomic['mutations_detected']:
            lines.append(f"• {mut['mutation']}: {mut['description']}")
            lines.append(f"  Significance: {mut['significance']}")
    lines.append("")
    lines.append(f"Interpretation: {genomic['interpretation']}")
    lines.append("")
    
    # Modality Contributions
    modalities = report['modality_contributions']
    lines.append("MODALITY CONTRIBUTIONS")
    lines.append("-" * 60)
    lines.append(f"CXR Image: {modalities['cxr']}")
    lines.append(f"Clinical Data: {modalities['clinical']}")
    lines.append(f"Genomic Data: {modalities['genomic']}")
    lines.append(f"Primary Modality: {modalities['primary_modality']}")
    lines.append("")
    
    # Recommendations
    recommendations = report['recommendations']
    lines.append("CLINICAL RECOMMENDATIONS")
    lines.append("-" * 60)
    for rec in recommendations:
        lines.append(f"[{rec['priority']} Priority] {rec['action']}")
        lines.append(f"  {rec['description']}")
        lines.append("")
    
    lines.append("=" * 60)
    lines.append("End of Report")
    lines.append("=" * 60)
    
    return "\n".join(lines)

