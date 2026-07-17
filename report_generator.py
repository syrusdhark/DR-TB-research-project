"""
Report generation module for the two-stage TB / DR-TB pipeline.

Builds a report from two independent results:
  - tb_result:    Stage 1 output (CXR -> TB vs Normal)
  - drtb_result:  Stage 2 output (clinical + genomic -> DR-TB risk)

The two are presented as separate findings, deliberately not blended into one
number -- the DR-TB risk score is not informed by the X-ray, and the TB
detection score is not informed by clinical/genomic data.
"""

import config


def generate_report(tb_result, drtb_result, clinical_data, genomic_data, modality_weights=None):
    """
    Generate a detailed diagnostic report.

    Args:
        tb_result: dict from predictor.predict_tb(), or None if no image was provided
        drtb_result: dict from predictor.predict_drtb_risk()
        clinical_data: dict with clinical input data
        genomic_data: dict with genomic input data
        modality_weights: optional dict {'clinical': x, 'genomic': y} from Stage 2

    Returns:
        Dictionary with formatted report sections
    """
    report = {
        'tb_summary': _generate_tb_summary(tb_result),
        'drtb_summary': _generate_drtb_summary(drtb_result),
        'risk_factors': _identify_risk_factors(clinical_data, genomic_data),
        'genomic_analysis': _analyze_genomic_mutations(genomic_data),
        'modality_contributions': _analyze_modality_contributions(modality_weights),
        'recommendations': _generate_recommendations(tb_result, drtb_result, clinical_data, genomic_data),
        'medication_history': (clinical_data or {}).get('medication_history', '').strip()
    }

    return report


def _generate_tb_summary(tb_result):
    """Generate summary section for Stage 1 (TB detection from CXR)."""
    if tb_result is None:
        return {
            'prediction': 'N/A',
            'probability_percent': 'N/A',
            'confidence_percent': 'N/A',
            'risk_level': 'N/A',
            'interpretation': 'No chest X-ray was provided, so TB detection was not run.'
        }

    prediction = tb_result['prediction']
    probability = tb_result['probability']
    confidence = tb_result['confidence']
    risk_level = tb_result['risk_level']

    if prediction == 'Tuberculosis':
        interpretation = (
            "The image-based model detects radiographic features consistent with tuberculosis. "
            "This reflects the X-ray only; it says nothing about drug resistance."
        )
    else:
        interpretation = (
            "The image-based model does not detect radiographic features consistent with tuberculosis. "
            "This does not rule out TB entirely -- clinical correlation is recommended if symptoms persist."
        )

    return {
        'prediction': prediction,
        'probability_percent': f"{probability * 100:.2f}%",
        'confidence_percent': f"{confidence:.1f}%",
        'risk_level': risk_level,
        'interpretation': interpretation,
    }


def _generate_drtb_summary(drtb_result):
    """Generate summary section for Stage 2 (DR-TB risk from clinical + genomic data)."""
    prediction = drtb_result['prediction']
    probability = drtb_result['probability']
    confidence = drtb_result['confidence']
    risk_level = drtb_result['risk_level']

    summary = {
        'prediction': prediction,
        'probability_percent': f"{probability * 100:.2f}%",
        'confidence_percent': f"{confidence:.1f}%",
        'risk_level': risk_level,
        'interpretation': _interpret_drtb_prediction(prediction, risk_level)
    }

    return summary


def _interpret_drtb_prediction(prediction, risk_level):
    """Generate interpretation text for the DR-TB risk prediction."""
    if prediction == 'DR-TB Risk':
        if risk_level == 'High':
            return (
                "The clinical and genomic profile indicates high risk of drug-resistant tuberculosis. "
                "Drug susceptibility testing (DST) is strongly recommended before treatment decisions."
            )
        else:
            return (
                "The clinical and genomic profile indicates moderate risk of drug-resistant tuberculosis. "
                "Drug susceptibility testing (DST) is recommended."
            )
    else:
        return (
            "The clinical and genomic profile does not indicate elevated risk of drug resistance. "
            "This does not replace DST -- correlate with clinical presentation and treatment history."
        )


def _identify_risk_factors(clinical_data, genomic_data):
    """Identify and describe risk factors from clinical and genomic data."""
    risk_factors = []

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

    for mutation_name in config.GENOMIC_FEATURES:
        if mutation_name == 'mutation_count':
            continue

        if genomic_data.get(mutation_name, 0):
            mutation_desc = config.MUTATION_DESCRIPTIONS.get(
                mutation_name,
                f'{mutation_name} mutation detected'
            )

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


def _analyze_modality_contributions(modality_weights):
    """Analyze whether clinical or genomic data contributed more to the DR-TB risk score."""
    if modality_weights is None:
        return {
            'clinical': 'N/A',
            'genomic': 'N/A',
            'primary_modality': 'N/A'
        }

    clinical_weight = modality_weights.get('clinical', 0.5)
    genomic_weight = modality_weights.get('genomic', 0.5)

    weights = {
        'Clinical Data': clinical_weight,
        'Genomic Data': genomic_weight
    }
    primary_modality = max(weights, key=weights.get)

    return {
        'clinical': f"{clinical_weight * 100:.1f}%",
        'genomic': f"{genomic_weight * 100:.1f}%",
        'primary_modality': primary_modality
    }


def _generate_recommendations(tb_result, drtb_result, clinical_data, genomic_data):
    """Generate clinical recommendations based on both predictions and inputs."""
    recommendations = []

    if tb_result is not None and tb_result['prediction'] == 'Tuberculosis':
        recommendations.append({
            'priority': 'High',
            'action': 'Confirm TB Diagnosis',
            'description': 'Follow up radiographic findings with standard TB diagnostic workup (sputum smear, culture, or GeneXpert).'
        })

    if drtb_result['prediction'] == 'DR-TB Risk':
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

    lines.append("=" * 60)
    lines.append("TB / DR-TB PREDICTION REPORT")
    lines.append("=" * 60)
    lines.append("")

    tb_summary = report['tb_summary']
    lines.append("STAGE 1: TB DETECTION (from chest X-ray)")
    lines.append("-" * 60)
    lines.append(f"Prediction: {tb_summary['prediction']}")
    lines.append(f"Probability: {tb_summary['probability_percent']}")
    lines.append(f"Confidence: {tb_summary['confidence_percent']}")
    lines.append(f"Risk Level: {tb_summary['risk_level']}")
    lines.append("")
    lines.append(f"Interpretation: {tb_summary['interpretation']}")
    lines.append("")

    drtb_summary = report['drtb_summary']
    lines.append("STAGE 2: DR-TB RISK (from clinical + genomic data)")
    lines.append("-" * 60)
    lines.append(f"Prediction: {drtb_summary['prediction']}")
    lines.append(f"Probability: {drtb_summary['probability_percent']}")
    lines.append(f"Confidence: {drtb_summary['confidence_percent']}")
    lines.append(f"Risk Level: {drtb_summary['risk_level']}")
    lines.append("")
    lines.append(f"Interpretation: {drtb_summary['interpretation']}")
    lines.append("")

    medication_history = (report.get('medication_history') or "").strip()
    lines.append("MEDICATION HISTORY")
    lines.append("-" * 60)
    lines.append(medication_history if medication_history else "Not provided.")
    lines.append("")

    risk_factors = report['risk_factors']
    lines.append("IDENTIFIED RISK FACTORS")
    lines.append("-" * 60)
    if risk_factors:
        for rf in risk_factors:
            rr = rf.get('relative_risk')
            details = rf['description']
            if rr and rr != 'N/A':
                details = f"{details} (Relative Risk: {rr})"
            lines.append(f"- {rf['factor']} ({rf['severity']}): {details}")
    else:
        lines.append("No significant risk factors identified.")
    lines.append("")

    genomic = report['genomic_analysis']
    lines.append("GENOMIC MUTATION ANALYSIS")
    lines.append("-" * 60)
    lines.append(f"Total Mutations Detected: {genomic['total_mutations']}")
    if genomic['mutations_detected']:
        for mut in genomic['mutations_detected']:
            lines.append(f"- {mut['mutation']}: {mut['description']}")
            lines.append(f"  Significance: {mut['significance']}")
    lines.append("")
    lines.append(f"Interpretation: {genomic['interpretation']}")
    lines.append("")

    modalities = report['modality_contributions']
    lines.append("DR-TB RISK MODALITY CONTRIBUTIONS (clinical vs. genomic)")
    lines.append("-" * 60)
    lines.append(f"Clinical Data: {modalities['clinical']}")
    lines.append(f"Genomic Data: {modalities['genomic']}")
    lines.append(f"Primary Modality: {modalities['primary_modality']}")
    lines.append("")

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
