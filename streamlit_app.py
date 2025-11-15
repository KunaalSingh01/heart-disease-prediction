# streamlit_app.py
import streamlit as st
import numpy as np
import pickle
import json

st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered")

@st.cache_resource
def load_artifacts():
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("logreg_model.pkl", "rb") as f:
        logreg_model = pickle.load(f)
    with open("meta.json", "r") as f:
        meta = json.load(f)
    return scaler, xgb_model, logreg_model, meta

scaler, xgb_model, logreg_model, meta = load_artifacts()
features = meta.get("features", ['age_yr','ap_hi','ap_lo','cholesterol','BMI'])
w_xgb = meta.get("ensemble_weights", {}).get("xgb", 0.7)
w_lr  = meta.get("ensemble_weights", {}).get("logreg", 0.3)

st.title("❤️ Heart Disease Risk Predictor")
st.write("Model features: " + ", ".join(features))

col1, col2 = st.columns(2)
with col1:
    age_yr = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=50.0, step=1.0)
    ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=50, max_value=250, value=120)
    ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=30, max_value=200, value=80)
with col2:
    height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0)
    weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0)
    cholesterol = st.selectbox("Cholesterol (1=normal,2=above normal,3=high)", options=[1,2,3], index=0)

bmi = weight / ((height/100)**2)
st.markdown(f"**Computed BMI:** {bmi:.2f}")

if st.button("Predict Risk"):
    X = np.array([[age_yr, ap_hi, ap_lo, cholesterol, bmi]])
    # XGBoost
    try:
        proba_xgb = xgb_model.predict_proba(X)[:,1][0]
    except Exception as e:
        st.error("XGBoost prediction error: " + str(e))
        proba_xgb = 0.0
    # Logistic Regression
    try:
        X_scaled = scaler.transform(X)
        proba_lr = logreg_model.predict_proba(X_scaled)[:,1][0]
    except Exception as e:
        st.error("LogReg prediction error: " + str(e))
        proba_lr = 0.0

    proba = w_xgb*proba_xgb + w_lr*proba_lr
    pred = int(proba >= 0.5)

    st.metric("Predicted Class (1 = heart disease)", pred)
    st.progress(min(100, int(proba*100)))
    st.write(f"Ensemble probability: **{proba:.4f}**")
    st.write(f"- XGBoost prob: {proba_xgb:.4f}")
    st.write(f"- LogisticRegression prob: {proba_lr:.4f}")

    if proba >= 0.75:
        st.warning("High risk — recommend clinical follow-up.")
    elif proba >= 0.5:
        st.info("Moderate risk — consult doctor.")
    else:
        st.success("Low risk — maintain healthy lifestyle.")
