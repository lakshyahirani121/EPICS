import streamlit as st
import xgboost as xgb
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

# --- PAGE CONFIG ---
st.set_page_config(page_title="Hybrid Anemia Detector", layout="wide")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    # 1. Feature Extractor
    fe_model = tf.keras.applications.EfficientNetB3(
        weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3)
    )
    # 2. Vision Classifier
    v_model = xgb.XGBClassifier()
    v_model.load_model('vision_anemia_model.json')
    # 3. Blood Report Classifier
    b_model = xgb.XGBClassifier()
    b_model.load_model('blood_report_model.json')
    return fe_model, v_model, b_model

fe, vision_brain, blood_brain = load_models()

st.title(" Integrated Hybrid Anemia Diagnostic System")
st.markdown("Provide both an eye image and CBC values for a unified diagnosis.")

# --- INPUT SECTION ---
col1, col2 = st.columns(2)

with col1:
    st.header("Conjunctiva Scan")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB').resize((224, 224))
        st.image(img, caption="Eye Image Ready", width=300)

with col2:
    st.header("Blood Report Values")
    # Using the key features from your CSV
    hgb = st.number_input("Hemoglobin (HGB)")
    rbc = st.number_input("Red Blood Cells (RBC)")
    hct = st.number_input("Hematocrit (HCT)")
    mcv = st.number_input("MCV")
    mch = st.number_input("MCH")
    mchc = st.number_input("MCHC")
    wbc = st.number_input("White Blood Cells (WBC)")

# --- INTEGRATED ANALYSIS ---
st.markdown("---")
if st.button("RESULTS"):
    if uploaded_file:
        # 1. Process Vision Path
        img_array = tf.keras.applications.efficientnet.preprocess_input(np.array(img))
        v_features = fe.predict(np.expand_dims(img_array, axis=0), verbose=0)
        v_prob = vision_brain.predict_proba(v_features)[0][1] # Probability of Anemia
        
        # 2. Process Blood Path 
        # Filling the 14-column array expected by your model with the inputs
        blood_input = np.array([[wbc, 30.0, 60.0, 2.0, 4.0, rbc, hgb, hct, mcv, mch, mchc, 250.0, 14.0, 0.2]])
        b_prob = blood_brain.predict_proba(blood_input)[0][1] # Probability of Anemia
        
        # 3. Hybrid Fusion Logic (Weighted Average)
        # Blood (Gold Standard) gets 70% weight, Vision (Screening) gets 30%
        final_score = (b_prob * 0.7) + (v_prob * 0.3)
        
        # --- DISPLAY RESULTS ---
        st.subheader("Integrated Diagnostic Report")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Vision Confidence", f"{v_prob*100:.1f}%")
        c2.metric("Blood Confidence", f"{b_prob*100:.1f}%")
        c3.metric("Combined Score", f"{final_score*100:.1f}%")
        
        if final_score > 0.5:
            st.error("FINAL DIAGNOSIS: ANEMIA DETECTED")
            st.write("The system has identified patterns consistent with anemia across both visual and clinical data.")
        else:
            st.success("FINAL DIAGNOSIS: HEALTHY / NORMAL")
            st.write("Visual features and blood parameters appear within normal ranges.")
            
        st.progress(final_score)
    else:
        st.warning(" Please upload an eye image first to complete the hybrid analysis.")

st.sidebar.markdown("### Project Info")
st.sidebar.write("EPICS 2026: Multimodal Anemia Detection")
