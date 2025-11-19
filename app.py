"""
DR-TB Prediction Web Interface
Streamlit application for Drug-Resistant Tuberculosis prediction.
"""

import streamlit as st
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from model_loader import load_model, get_model_info
from preprocessing import preprocess_image, encode_clinical_features, encode_genomic_features
from predictor import predict_drtb
from report_generator import generate_report, format_report_text

# Page configuration
st.set_page_config(
    page_title="DR-TB Prediction System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
def load_cached_model():
    """Load and cache the model."""
    try:
        model, device = load_model()
        return model, device
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None


def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">🫁 DR-TB Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
    Multimodal AI System for Drug-Resistant Tuberculosis Prediction<br>
    Using Chest X-Ray, Clinical Data, and Genomic Mutations
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Model Information
    with st.sidebar:
        st.header("📊 Model Information")
        model_info = get_model_info()
        if model_info:
            st.success(f"✅ Model: {model_info['name']}")
            st.info(f"📦 Size: {model_info['size_mb']:.1f} MB")
            if 'validation_auc' in model_info:
                st.metric("Validation AUC", f"{model_info['validation_auc']:.3f}")
        else:
            st.warning("⚠️ Model information not available")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This system uses a multimodal deep learning model to predict 
        Drug-Resistant Tuberculosis (DR-TB) from:
        - **Chest X-Ray Images**
        - **Clinical Metadata**
        - **Genomic Mutations**
        
        **Note:** This is a research tool and should not replace 
        clinical judgment or standard diagnostic procedures.
        """)
    
    # Load model (cached)
    model, device = load_cached_model()
    if model is None:
        st.error("❌ Model could not be loaded. Please check model files.")
        st.stop()
    
    # Main interface tabs
    tab1, tab2 = st.tabs(["🔍 Prediction", "📋 Instructions"])
    
    with tab1:
        prediction_interface(model, device)
    
    with tab2:
        show_instructions()


def prediction_interface(model, device):
    """Main prediction interface (single column scroll)."""
    st.header("📤 Input Data")
    
    st.subheader("1. Chest X-Ray Image")
    uploaded_file = st.file_uploader(
        "Upload CXR Image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a chest X-ray image in PNG, JPG, or JPEG format"
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
    
    st.subheader("2. Clinical Data")
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
    
    st.subheader("4. Genomic Mutations (Optional)")
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
        if uploaded_file is None or image is None:
            st.error("❌ Please upload a CXR image first.")
        else:
            try:
                with st.spinner("Processing prediction..."):
                    cxr_tensor = preprocess_image(image)
                    
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
                    
                    result = predict_drtb(
                        model, cxr_tensor, clinical_tensor, genomic_tensor,
                        device, return_attention=True
                    )
                    
                    st.session_state['prediction_result'] = result
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
                    st.session_state['attention_weights'] = result.get('attention_weights')
            
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
    
    if 'prediction_result' in st.session_state:
        result = st.session_state['prediction_result']
        
        st.markdown("### Simple Prediction")
        prediction = result['prediction']
        probability = result['probability']
        confidence = result['confidence']
        risk_level = result['risk_level']
        
        badge_class = "drtb-badge" if prediction == "DR-TB" else "normal-badge"
        st.markdown(
            f'<div class="prediction-badge {badge_class}">{prediction}</div>',
            unsafe_allow_html=True
        )
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Probability", f"{probability * 100:.2f}%")
        with col_b:
            st.metric("Confidence", f"{confidence:.1f}%")
        with col_c:
            risk_class = f"risk-{risk_level.lower()}"
            st.markdown(f'<div class="{risk_class}">Risk: {risk_level}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Detailed Report")
        show_detailed = st.checkbox("Show Detailed Report", value=False)
        
        if show_detailed:
            report = generate_report(
                result,
                st.session_state.get('clinical_data', {}),
                st.session_state.get('genomic_data', {}),
                st.session_state.get('attention_weights')
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
    # Summary
    summary = report['summary']
    with st.expander("📋 Prediction Summary", expanded=True):
        st.write(f"**Prediction:** {summary['prediction']}")
        st.write(f"**Probability:** {summary['probability_percent']}")
        st.write(f"**Confidence:** {summary['confidence_percent']}%")
        st.write(f"**Risk Level:** {summary['risk_level']}")
        st.write(f"**Interpretation:** {summary['interpretation']}")
    
    medication_history = (report.get('medication_history') or "").strip()
    with st.expander("💊 Medication History"):
        if medication_history:
            st.write(medication_history)
        else:
            st.info("No medication history was provided.")
    
    # Risk Factors
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
    
    # Genomic Analysis
    genomic = report['genomic_analysis']
    with st.expander("🧬 Genomic Mutation Analysis"):
        st.metric("Total Mutations", genomic['total_mutations'])
        if genomic['mutations_detected']:
            for mut in genomic['mutations_detected']:
                st.write(f"**{mut['mutation']}**")
                st.write(f"  {mut['description']}")
                st.write(f"  *Significance:* {mut['significance']}")
        st.write(f"**Interpretation:** {genomic['interpretation']}")
    
    # Modality Contributions
    modalities = report['modality_contributions']
    with st.expander("🔍 Modality Contributions"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("CXR Image", modalities['cxr'])
        with col2:
            st.metric("Clinical Data", modalities['clinical'])
        with col3:
            st.metric("Genomic Data", modalities['genomic'])
        st.write(f"**Primary Modality:** {modalities['primary_modality']}")
    
    # Recommendations
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
    
    #### Step 1: Upload Chest X-Ray Image
    - Click "Browse files" or drag and drop a chest X-ray image
    - Supported formats: PNG, JPG, JPEG
    - The image will be automatically preprocessed
    
    #### Step 2: Enter Clinical Data
    - **Age**: Patient's age (0-150 years)
    - **Gender**: Select Male or Female
    - **Region**: Select patient's geographic region
    - **Medical History**: Check relevant boxes
    - **Resistance Status**: Check if any resistance is confirmed
    
    #### Step 3: Enter Genomic Mutations (Optional)
    - Expand the "Select Detected Mutations" section
    - Check boxes for any detected mutations
    - If genomic data is unavailable, leave all unchecked
    
    #### Step 4: Run Prediction
    - Click the "🔬 Run Prediction" button
    - Wait for processing (usually takes a few seconds)
    - View results in the right panel
    
    #### Understanding Results
    
    **Simple Prediction:**
    - Shows the predicted class (DR-TB or Normal)
    - Displays probability and confidence scores
    - Indicates risk level (High/Medium/Low)
    
    **Detailed Report:**
    - Check "Show Detailed Report" to see:
      - Full prediction summary and interpretation
      - Identified risk factors
      - Genomic mutation analysis
      - Modality contributions (which inputs were most important)
      - Clinical recommendations
    
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
    
    - **Model Type**: Multimodal Fusion (EfficientNet-B4 + Clinical + Genomic)
    - **Optimal Threshold**: 0.638 (from validation)
    - **Input Image Size**: 380x380 pixels
    - **Clinical Features**: 14 features
    - **Genomic Features**: 12 mutation types
    """)


if __name__ == "__main__":
    main()

