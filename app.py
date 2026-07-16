"""
DR-TB Prediction Web Interface
Streamlit application for TB detection and Drug-Resistant TB risk assessment.

This app runs two independent models:
  - Stage 1 (TB detection): chest X-ray only -> TB vs Normal
  - Stage 2 (DR-TB risk):   clinical + genomic data only -> DR-TB risk

They are shown as two separate results. The X-ray never influences the DR-TB
risk score, and clinical/genomic inputs never influence the TB detection
score -- there is no dataset backing this project that pairs a real X-ray
with that same patient's real drug-susceptibility result, so a fused
single-number "DR-TB from X-ray" prediction would not be a valid claim.
"""

import streamlit as st
import torch
from PIL import Image
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import config
from model_loader import load_tb_classifier, load_drtb_risk_model, get_model_info
from preprocessing import preprocess_image, encode_clinical_features, encode_genomic_features
from predictor import predict_tb, predict_drtb_risk
from report_generator import generate_report, format_report_text

st.set_page_config(
    page_title="DR-TB Prediction System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-badge {
        font-size: 2rem;
        font-weight: bold;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .drtb-badge {
        background-color: #ff6b6b;
        color: white;
    }
    .normal-badge {
        background-color: #51cf66;
        color: white;
    }
    .risk-high {
        color: #ff6b6b;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffd43b;
        font-weight: bold;
    }
    .risk-low {
        color: #51cf66;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_cached_models():
    """Load and cache both stage models."""
    try:
        tb_model, tb_device, tb_threshold = load_tb_classifier()
    except Exception as e:
        st.error(f"Failed to load TB image classifier: {e}")
        tb_model, tb_device, tb_threshold = None, None, None

    try:
        drtb_model, drtb_device, drtb_threshold = load_drtb_risk_model()
    except Exception as e:
        st.error(f"Failed to load DR-TB risk model: {e}")
        drtb_model, drtb_device, drtb_threshold = None, None, None

    return {
        'tb_model': tb_model, 'tb_device': tb_device, 'tb_threshold': tb_threshold,
        'drtb_model': drtb_model, 'drtb_device': drtb_device, 'drtb_threshold': drtb_threshold,
    }


def main():
    """Main application function."""
    st.markdown('<h1 class="main-header">🫁 DR-TB Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
    Two independent models:<br>
    <b>TB Detection</b> from Chest X-Ray &nbsp;|&nbsp; <b>DR-TB Risk</b> from Clinical + Genomic Data
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("📊 Model Information")
        tb_info = get_model_info(config.TB_MODEL_PREFIX)
        drtb_info = get_model_info(config.DRTB_RISK_MODEL_PREFIX)

        st.markdown("**Stage 1: TB Detector**")
        if tb_info:
            st.success(f"✅ {tb_info['name']}")
            st.caption(f"{tb_info['size_mb']:.1f} MB")
        else:
            st.warning("⚠️ Not available")

        st.markdown("**Stage 2: DR-TB Risk Model**")
        if drtb_info:
            st.success(f"✅ {drtb_info['name']}")
            st.caption(f"{drtb_info['size_mb']:.1f} MB")
        else:
            st.warning("⚠️ Not available")

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This system runs **two separate models**:
        - **TB Detection**: chest X-ray only
        - **DR-TB Risk**: clinical & genomic data only

        The X-ray does **not** inform the DR-TB risk score, and the
        clinical/genomic data does **not** inform TB detection. Chest X-ray
        appearance is not a validated indicator of drug resistance, so the
        two are kept separate rather than fused into one number.

        **Note:** This is a research tool and should not replace
        clinical judgment or standard diagnostic procedures.
        """)

    models = load_cached_models()
    if models['tb_model'] is None and models['drtb_model'] is None:
        st.error("❌ No models could be loaded. Please check model files in results/models/.")
        st.stop()

    tab1, tab2 = st.tabs(["🔍 Prediction", "📋 Instructions"])

    with tab1:
        prediction_interface(models)

    with tab2:
        show_instructions()


def prediction_interface(models):
    """Main prediction interface (single column scroll)."""
    st.header("📤 Input Data")

    st.subheader("1. Chest X-Ray Image (for TB Detection)")
    uploaded_file = st.file_uploader(
        "Upload CXR Image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a chest X-ray image in PNG, JPG, or JPEG format. Used only for TB detection, not DR-TB risk."
    )
    image = None
    if uploaded_file is not None:
        try:
            if uploaded_file.size > 10 * 1024 * 1024:
                st.error("❌ Image file too large. Maximum size is 10MB.")
                uploaded_file = None
            else:
                image = Image.open(uploaded_file).convert('RGB')
                width, height = image.size
                if width < 100 or height < 100:
                    st.warning("⚠️ Image dimensions are very small. Results may be less accurate.")
                st.image(image, caption="Uploaded CXR Image", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error loading image: {e}")
            st.info("Please ensure the file is a valid image (PNG, JPG, or JPEG)")
            uploaded_file = None

    st.divider()

    st.subheader("2. Clinical Data (for DR-TB Risk)")
    with st.form("clinical_form"):
        age = st.number_input("Age", min_value=0, max_value=150, value=45, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        region = st.selectbox("Region", config.REGIONS)

        st.markdown("**Medical History:**")
        previous_tb = st.checkbox("Previous TB Treatment")
        smoking = st.checkbox("Smoking History")
        asthma = st.checkbox("Asthma")
        pneumonia = st.checkbox("Pneumonia")

        st.markdown("**Resistance Status:**")
        mdr_tb = st.checkbox("MDR-TB Confirmed")
        xdr_tb = st.checkbox("XDR-TB Confirmed")
        rifampin_res = st.checkbox("Rifampin Resistance")
        isoniazid_res = st.checkbox("Isoniazid Resistance")

        medication_history = st.text_area(
            "Medication History",
            placeholder="List current/past medications (e.g., rifampin, steroids, biologics, transplant meds).",
            help="Include immunosuppressants, TB drugs, steroids, biologics, or any therapy that may impact TB risk."
        )

        clinical_submitted = st.form_submit_button("Confirm Clinical Data")

    st.divider()

    st.subheader("3. Comorbidities & Underlying Conditions")
    st.caption("Select all known comorbidities or risk-enhancing conditions for this patient.")
    comorbidity_inputs = {}
    with st.expander("High Priority (strong association with TB progression)", expanded=True):
        comorbidity_inputs['hiv_aids'] = st.checkbox("HIV / AIDS (18–30× risk)")
        comorbidity_inputs['diabetes_mellitus'] = st.checkbox("Diabetes Mellitus (Type 1 or 2)")
        comorbidity_inputs['silicosis'] = st.checkbox("Silicosis")
        comorbidity_inputs['chronic_kidney_disease'] = st.checkbox("Chronic Kidney Disease / ESRD (Dialysis)")
        comorbidity_inputs['organ_transplantation'] = st.checkbox("Organ Transplantation (solid organ / HSCT)")
        comorbidity_inputs['tnf_alpha_inhibitors'] = st.checkbox("TNF-α Inhibitor Therapy (e.g., infliximab, adalimumab)")
        comorbidity_inputs['malnutrition_low_bmi'] = st.checkbox("Malnutrition / BMI < 18.5")
        comorbidity_inputs['alcohol_use_disorder'] = st.checkbox("Alcohol Use Disorder (>40 g/day)")

    with st.expander("Moderate Association / Emerging Risk Factors", expanded=False):
        comorbidity_inputs['copd'] = st.checkbox("Chronic Obstructive Pulmonary Disease (COPD)")
        comorbidity_inputs['rheumatoid_arthritis'] = st.checkbox("Rheumatoid Arthritis")
        comorbidity_inputs['cancer_hematological'] = st.checkbox("Cancer - Hematological")
        comorbidity_inputs['cancer_head_neck_lung'] = st.checkbox("Cancer - Head/Neck or Lung")
        comorbidity_inputs['gastrectomy_bypass'] = st.checkbox("Gastrectomy or Jejunoileal Bypass")
        comorbidity_inputs['vitamin_d_deficiency'] = st.checkbox("Vitamin D Deficiency")
        comorbidity_inputs['cystic_fibrosis'] = st.checkbox("Cystic Fibrosis")
        comorbidity_inputs['sickle_cell_disease'] = st.checkbox("Sickle Cell Disease")
        comorbidity_inputs['sle'] = st.checkbox("Systemic Lupus Erythematosus (SLE)")
        comorbidity_inputs['injecting_drug_use'] = st.checkbox("Injecting Drug Use")
        comorbidity_inputs['post_covid19_lung_damage'] = st.checkbox("Post-COVID-19 Lung Damage (within 12 months)")

    comorbidity_inputs['asthma'] = asthma
    comorbidity_inputs['pneumonia'] = pneumonia

    st.divider()

    st.subheader("4. Genomic Mutations (for DR-TB Risk, Optional)")
    with st.expander("Select Detected Mutations (if sequencing data is available)", expanded=False):
        st.markdown("**Rifampin Resistance (rpoB):**")
        rpoB_S531L = st.checkbox("rpoB S531L")
        rpoB_S450L = st.checkbox("rpoB S450L")
        rpoB_H526Y = st.checkbox("rpoB H526Y")
        rpoB_H445Y = st.checkbox("rpoB H445Y")
        rpoB_D435V = st.checkbox("rpoB D435V")

        st.markdown("**Isoniazid Resistance:**")
        katG_S315T = st.checkbox("katG S315T")
        katG_S315N = st.checkbox("katG S315N")
        inhA_C15T = st.checkbox("inhA C15T")
        fabG1_C15T = st.checkbox("fabG1 -15C>T")

        st.markdown("**Other Resistance:**")
        pncA_H57D = st.checkbox("pncA H57D (Pyrazinamide)")
        embB_M306V = st.checkbox("embB M306V (Ethambutol)")

    st.info("💡 Tip: If genomic sequencing data is unavailable, leave all mutation boxes unchecked (defaults to no mutations).")

    st.divider()

    st.header("📊 Prediction Results")
    if st.button("🔬 Run Prediction", type="primary", use_container_width=True):
        try:
            with st.spinner("Processing prediction..."):
                tb_result = None
                if image is not None and models['tb_model'] is not None:
                    cxr_tensor = preprocess_image(image)
                    tb_result = predict_tb(
                        models['tb_model'], cxr_tensor, models['tb_device'],
                        threshold=models['tb_threshold']
                    )

                hiv_status_flag = 1 if comorbidity_inputs.get('hiv_aids') else 0
                diabetes_flag = 1 if comorbidity_inputs.get('diabetes_mellitus') else 0
                copd_flag = 1 if comorbidity_inputs.get('copd') else 0
                covid19_flag = 1 if comorbidity_inputs.get('post_covid19_lung_damage') else 0

                clinical_tensor = encode_clinical_features(
                    age=age,
                    gender=gender,
                    region=region,
                    previous_tb_treatment=1 if previous_tb else 0,
                    hiv_status=hiv_status_flag,
                    diabetes_status=diabetes_flag,
                    smoking_status=1 if smoking else 0,
                    mdr_tb=1 if mdr_tb else 0,
                    xdr_tb=1 if xdr_tb else 0,
                    rifampin_resistance=1 if rifampin_res else 0,
                    isoniazid_resistance=1 if isoniazid_res else 0,
                    copd=copd_flag,
                    asthma=1 if asthma else 0,
                    pneumonia=1 if pneumonia else 0,
                    covid19=covid19_flag
                )

                mutation_count = sum([
                    rpoB_S531L, rpoB_S450L, rpoB_H526Y, rpoB_H445Y, rpoB_D435V,
                    katG_S315T, katG_S315N, inhA_C15T, fabG1_C15T,
                    pncA_H57D, embB_M306V
                ])

                genomic_tensor = encode_genomic_features(
                    rpoB_S531L=1 if rpoB_S531L else 0,
                    rpoB_S450L=1 if rpoB_S450L else 0,
                    rpoB_H526Y=1 if rpoB_H526Y else 0,
                    rpoB_H445Y=1 if rpoB_H445Y else 0,
                    rpoB_D435V=1 if rpoB_D435V else 0,
                    katG_S315T=1 if katG_S315T else 0,
                    katG_S315N=1 if katG_S315N else 0,
                    inhA_C15T=1 if inhA_C15T else 0,
                    fabG1_C15T=1 if fabG1_C15T else 0,
                    pncA_H57D=1 if pncA_H57D else 0,
                    embB_M306V=1 if embB_M306V else 0,
                    mutation_count=mutation_count
                )

                drtb_result = predict_drtb_risk(
                    models['drtb_model'], clinical_tensor, genomic_tensor,
                    models['drtb_device'], threshold=models['drtb_threshold']
                )

                st.session_state['tb_result'] = tb_result
                st.session_state['drtb_result'] = drtb_result
                st.session_state['clinical_data'] = {
                    'age': age, 'gender': gender, 'region': region,
                    'previous_tb_treatment': 1 if previous_tb else 0,
                    'hiv_status': hiv_status_flag,
                    'diabetes_status': diabetes_flag,
                    'smoking_status': 1 if smoking else 0,
                    'mdr_tb': 1 if mdr_tb else 0,
                    'xdr_tb': 1 if xdr_tb else 0,
                    'rifampin_resistance': 1 if rifampin_res else 0,
                    'isoniazid_resistance': 1 if isoniazid_res else 0,
                    'copd': copd_flag,
                    'asthma': 1 if asthma else 0,
                    'pneumonia': 1 if pneumonia else 0,
                    'covid19': covid19_flag,
                    'medication_history': medication_history.strip(),
                    'comorbidities': comorbidity_inputs
                }
                st.session_state['genomic_data'] = {
                    'rpoB_S531L': 1 if rpoB_S531L else 0,
                    'rpoB_S450L': 1 if rpoB_S450L else 0,
                    'rpoB_H526Y': 1 if rpoB_H526Y else 0,
                    'rpoB_H445Y': 1 if rpoB_H445Y else 0,
                    'rpoB_D435V': 1 if rpoB_D435V else 0,
                    'katG_S315T': 1 if katG_S315T else 0,
                    'katG_S315N': 1 if katG_S315N else 0,
                    'inhA_C15T': 1 if inhA_C15T else 0,
                    'fabG1_C15T': 1 if fabG1_C15T else 0,
                    'pncA_H57D': 1 if pncA_H57D else 0,
                    'embB_M306V': 1 if embB_M306V else 0,
                    'mutation_count': mutation_count
                }
                st.session_state['modality_weights'] = drtb_result.get('modality_weights')

        except ValueError as e:
            st.error(f"❌ Input validation error: {e}")
            st.info("Please check your input values and try again.")
        except FileNotFoundError as e:
            st.error(f"❌ File not found: {e}")
        except RuntimeError as e:
            st.error(f"❌ Model runtime error: {e}")
            st.info("This may be due to insufficient GPU memory. Try using CPU mode.")
        except Exception as e:
            st.error(f"❌ Unexpected error during prediction: {e}")
            if st.checkbox("Show technical details"):
                import traceback
                st.code(traceback.format_exc())

    if 'drtb_result' in st.session_state:
        tb_result = st.session_state.get('tb_result')
        drtb_result = st.session_state['drtb_result']

        col_tb, col_drtb = st.columns(2)

        with col_tb:
            st.markdown("### TB Detection (from X-Ray)")
            if tb_result is None:
                st.info("No X-ray uploaded — TB detection not run.")
            else:
                badge_class = "drtb-badge" if tb_result['prediction'] == "Tuberculosis" else "normal-badge"
                st.markdown(
                    f'<div class="prediction-badge {badge_class}">{tb_result["prediction"]}</div>',
                    unsafe_allow_html=True
                )
                st.metric("Probability", f"{tb_result['probability'] * 100:.2f}%")
                st.metric("Confidence", f"{tb_result['confidence']:.1f}%")

        with col_drtb:
            st.markdown("### DR-TB Risk (from Clinical + Genomic Data)")
            badge_class = "drtb-badge" if drtb_result['prediction'] == "DR-TB Risk" else "normal-badge"
            st.markdown(
                f'<div class="prediction-badge {badge_class}">{drtb_result["prediction"]}</div>',
                unsafe_allow_html=True
            )
            st.metric("Probability", f"{drtb_result['probability'] * 100:.2f}%")
            st.metric("Confidence", f"{drtb_result['confidence']:.1f}%")

        st.markdown("---")
        st.markdown("### Detailed Report")
        show_detailed = st.checkbox("Show Detailed Report", value=False)

        if show_detailed:
            report = generate_report(
                tb_result,
                drtb_result,
                st.session_state.get('clinical_data', {}),
                st.session_state.get('genomic_data', {}),
                st.session_state.get('modality_weights')
            )

            display_detailed_report(report)

            st.markdown("---")
            report_text = format_report_text(report)
            st.download_button(
                "📥 Download Report (TXT)",
                report_text,
                file_name="drtb_prediction_report.txt",
                mime="text/plain"
            )


def display_detailed_report(report):
    """Display detailed report sections."""
    tb_summary = report['tb_summary']
    with st.expander("🫁 Stage 1: TB Detection Summary", expanded=True):
        st.write(f"**Prediction:** {tb_summary['prediction']}")
        st.write(f"**Probability:** {tb_summary['probability_percent']}")
        st.write(f"**Confidence:** {tb_summary['confidence_percent']}")
        st.write(f"**Risk Level:** {tb_summary['risk_level']}")
        st.write(f"**Interpretation:** {tb_summary['interpretation']}")

    drtb_summary = report['drtb_summary']
    with st.expander("🧬 Stage 2: DR-TB Risk Summary", expanded=True):
        st.write(f"**Prediction:** {drtb_summary['prediction']}")
        st.write(f"**Probability:** {drtb_summary['probability_percent']}")
        st.write(f"**Confidence:** {drtb_summary['confidence_percent']}")
        st.write(f"**Risk Level:** {drtb_summary['risk_level']}")
        st.write(f"**Interpretation:** {drtb_summary['interpretation']}")

    medication_history = (report.get('medication_history') or "").strip()
    with st.expander("💊 Medication History"):
        if medication_history:
            st.write(medication_history)
        else:
            st.info("No medication history was provided.")

    risk_factors = report['risk_factors']
    with st.expander("⚠️ Identified Risk Factors"):
        if risk_factors:
            for rf in risk_factors:
                st.write(f"**{rf['factor']}** ({rf['severity']})")
                st.write(f"  {rf['description']}")
                rr = rf.get('relative_risk')
                if rr and rr != 'N/A':
                    st.write(f"  *Relative Risk:* {rr}")
        else:
            st.info("No significant risk factors identified.")

    genomic = report['genomic_analysis']
    with st.expander("🧬 Genomic Mutation Analysis"):
        st.metric("Total Mutations", genomic['total_mutations'])
        if genomic['mutations_detected']:
            for mut in genomic['mutations_detected']:
                st.write(f"**{mut['mutation']}**")
                st.write(f"  {mut['description']}")
                st.write(f"  *Significance:* {mut['significance']}")
        st.write(f"**Interpretation:** {genomic['interpretation']}")

    modalities = report['modality_contributions']
    with st.expander("🔍 DR-TB Risk Modality Contributions"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Clinical Data", modalities['clinical'])
        with col2:
            st.metric("Genomic Data", modalities['genomic'])
        st.write(f"**Primary Modality:** {modalities['primary_modality']}")

    recommendations = report['recommendations']
    with st.expander("💡 Clinical Recommendations", expanded=True):
        for rec in recommendations:
            priority_color = {
                'Critical': '🔴',
                'High': '🟠',
                'Medium': '🟡',
                'Low': '🟢'
            }.get(rec['priority'], '⚪')

            st.write(f"{priority_color} **[{rec['priority']} Priority]** {rec['action']}")
            st.write(f"  {rec['description']}")
            st.write("")


def show_instructions():
    """Show usage instructions."""
    st.header("📋 Instructions")

    st.markdown("""
    ### How to Use the DR-TB Prediction System

    This system runs **two independent models** and shows two separate results:

    #### Stage 1: TB Detection (chest X-ray only)
    - Upload a chest X-ray image (PNG, JPG, JPEG)
    - The image-only model predicts TB vs Normal
    - This score is **not** affected by any clinical or genomic input

    #### Stage 2: DR-TB Risk (clinical + genomic data only)
    - Fill in age, gender, region, medical history, resistance status, and comorbidities
    - Optionally check any known genomic mutations
    - The clinical/genomic-only model predicts DR-TB risk
    - This score is **not** affected by the X-ray image

    #### Why are they separate?
    Chest X-ray appearance is not a clinically validated indicator of drug
    resistance status — resistance is a genomic/phenotypic property, not a
    radiographic one. Combining the two into a single "DR-TB from X-ray"
    number would imply the image carries information it does not actually
    carry for this task.

    #### Understanding Results

    **TB Detection**: shows Tuberculosis/Normal, probability, and confidence — based only on the image.

    **DR-TB Risk**: shows DR-TB Risk/Low Risk, probability, and confidence — based only on clinical + genomic inputs.

    **Detailed Report**: check "Show Detailed Report" for both stage summaries, risk factors, genomic mutation analysis, modality contributions (clinical vs. genomic, for the DR-TB risk score only), and clinical recommendations.

    #### Important Notes

    ⚠️ **This is a research tool and should not replace:**
    - Clinical judgment
    - Standard diagnostic procedures
    - Drug susceptibility testing (DST)
    - Expert medical consultation

    ✅ **Always correlate predictions with:**
    - Patient symptoms and history
    - Physical examination findings
    - Laboratory test results
    - Imaging findings

    #### Model Information

    - **Stage 1 — TB Detector**: EfficientNet-B4, image-only, 380x380 input
    - **Stage 2 — DR-TB Risk Model**: Clinical (14 features) + Genomic (12 mutation types) encoders with attention fusion, no image input
    """)


if __name__ == "__main__":
    main()
