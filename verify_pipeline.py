"""
End-to-end smoke test for the two-stage pipeline, mirroring what app.py does
on a single "Run Prediction" click, without needing a browser.
"""

from pathlib import Path
from PIL import Image

import config
from model_loader import load_tb_classifier, load_drtb_risk_model, get_model_info
from preprocessing import preprocess_image, encode_clinical_features, encode_genomic_features
from predictor import predict_tb, predict_drtb_risk
from report_generator import generate_report, format_report_text

BASE_DIR = Path(__file__).parent


def main():
    print("=== Loading models ===")
    tb_model, tb_device, tb_threshold = load_tb_classifier()
    drtb_model, drtb_device, drtb_threshold = load_drtb_risk_model()

    tb_info = get_model_info(config.TB_MODEL_PREFIX)
    drtb_info = get_model_info(config.DRTB_RISK_MODEL_PREFIX)
    print("TB model info:", tb_info)
    print("DR-TB risk model info:", drtb_info)

    print("\n=== Stage 1: TB detection on a real TB-labeled image ===")
    tb_image_path = BASE_DIR / "TB_Chest_Radiography_Database" / "Tuberculosis" / "Tuberculosis-2.png"
    image = Image.open(tb_image_path).convert('RGB')
    cxr_tensor = preprocess_image(image)
    tb_result = predict_tb(tb_model, cxr_tensor, tb_device, threshold=tb_threshold)
    print(f"  Image: {tb_image_path.name} (true label: Tuberculosis)")
    print(f"  Result: {tb_result}")

    print("\n=== Stage 1: TB detection on a real Normal-labeled image ===")
    normal_image_path = BASE_DIR / "TB_Chest_Radiography_Database" / "Normal" / "Normal-2.png"
    image2 = Image.open(normal_image_path).convert('RGB')
    cxr_tensor2 = preprocess_image(image2)
    tb_result2 = predict_tb(tb_model, cxr_tensor2, tb_device, threshold=tb_threshold)
    print(f"  Image: {normal_image_path.name} (true label: Normal)")
    print(f"  Result: {tb_result2}")

    print("\n=== Stage 2: DR-TB risk on a high-risk clinical/genomic profile ===")
    clinical_tensor = encode_clinical_features(
        age=55, gender='Male', region='Africa',
        previous_tb_treatment=1, hiv_status=1, diabetes_status=0, smoking_status=1,
        mdr_tb=1, xdr_tb=0, rifampin_resistance=1, isoniazid_resistance=1,
    )
    genomic_tensor = encode_genomic_features(
        rpoB_S531L=1, katG_S315T=1, mutation_count=2
    )
    drtb_result = predict_drtb_risk(drtb_model, clinical_tensor, genomic_tensor, drtb_device, threshold=drtb_threshold)
    print(f"  Result: {drtb_result}")

    print("\n=== Stage 2: DR-TB risk on a low-risk clinical/genomic profile ===")
    clinical_tensor_low = encode_clinical_features(
        age=30, gender='Female', region='Asia',
        previous_tb_treatment=0, hiv_status=0, diabetes_status=0, smoking_status=0,
        mdr_tb=0, xdr_tb=0, rifampin_resistance=0, isoniazid_resistance=0,
    )
    genomic_tensor_low = encode_genomic_features(mutation_count=0)
    drtb_result_low = predict_drtb_risk(drtb_model, clinical_tensor_low, genomic_tensor_low, drtb_device, threshold=drtb_threshold)
    print(f"  Result: {drtb_result_low}")

    print("\n=== Report generation (using the high-risk case) ===")
    clinical_data = {
        'age': 55, 'gender': 'Male', 'region': 'Africa',
        'previous_tb_treatment': 1, 'hiv_status': 1, 'diabetes_status': 0, 'smoking_status': 1,
        'mdr_tb': 1, 'xdr_tb': 0, 'rifampin_resistance': 1, 'isoniazid_resistance': 1,
        'copd': 0, 'asthma': 0, 'pneumonia': 0, 'covid19': 0,
        'medication_history': 'rifampin (past course)', 'comorbidities': {'hiv_aids': 1},
    }
    genomic_data = {
        'rpoB_S531L': 1, 'rpoB_S450L': 0, 'rpoB_H526Y': 0, 'rpoB_H445Y': 0, 'rpoB_D435V': 0,
        'katG_S315T': 1, 'katG_S315N': 0, 'inhA_C15T': 0, 'fabG1_C15T': 0,
        'pncA_H57D': 0, 'embB_M306V': 0, 'mutation_count': 2,
    }
    report = generate_report(tb_result, drtb_result, clinical_data, genomic_data, drtb_result.get('modality_weights'))
    text = format_report_text(report)
    print(text)

    assert tb_result['prediction'] in ('Tuberculosis', 'Normal')
    assert drtb_result['prediction'] in ('DR-TB Risk', 'Low Risk')
    print("\n=== All checks passed ===")


if __name__ == "__main__":
    main()
