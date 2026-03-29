
import streamlit as st
import xgboost as xgb
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    fe_model = tf.keras.applications.EfficientNetB3(
        weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3)
    )
    v_model = xgb.XGBClassifier()
    v_model.load_model('vision_anemia_model.json')
    b_model = xgb.XGBClassifier()
    b_model.load_model('blood_report_model.json')
    return fe_model, v_model, b_model

fe, vision_brain, blood_brain = load_models()

# --- UI INTERFACE ---
st.title("🩸 Dual-Verification Anemia Screening")

col1, col2 = st.columns(2)

with col1:
    st.header("📷 Eye Scan")
    file = st.file_uploader("Upload Conjunctiva Image", type=["jpg", "png"])
    if file:
        img = Image.open(file).convert('RGB').resize((224, 224))
        st.image(img, width=250)
        img_arr = tf.keras.applications.efficientnet.preprocess_input(np.array(img))
        feats = fe.predict(np.expand_dims(img_arr, axis=0), verbose=0)
        res = vision_brain.predict(feats)[0]
        st.error("Anemia Detected") if res == 1 else st.success("Healthy")

with col2:
    st.header("📄 Blood Report")
    hgb = st.number_input("Hemoglobin (HGB)", value=12.0)
    rbc = st.number_input("RBC", value=4.5)
    hct = st.number_input("HCT", value=38.0)
    mcv = st.number_input("MCV", value=85.0)
    mch = st.number_input("MCH", value=30.0)
    mchc = st.number_input("MCHC", value=33.0)
    wbc = st.number_input("WBC", value=7.0)

    if st.button("Final Analysis"):
        blood_input = np.array([[wbc, 30.0, 60.0, 2.0, 4.0, rbc, hgb, hct, mcv, mch, mchc, 250.0, 14.0, 0.2]])
        b_res = blood_brain.predict(blood_input)[0]
        st.error("Result: ANEMIC") if b_res == 1 else st.success("Result: HEALTHY")
