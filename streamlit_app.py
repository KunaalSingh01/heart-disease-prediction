# streamlit_app.py
import streamlit as st
import numpy as np
import pickle
import json
import pandas as pd
import io
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Styling (simple CSS) ---
st.markdown(
    """
    <style>
    .header {
        background: linear-gradient(90deg,#ff7a7a,#ffb199);
        padding: 18px;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .card {
        border-radius:12px;
        padding:14px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.06);
        background: linear-gradient(180deg, #ffffff, #fffaf0);
    }
    .small { font-size:0.9rem; color:#444; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="header"><h1 style="margin:4px">❤️ Heart Disease Risk Predictor</h1>'
            '<div class="small">Fast, simple risk check — not a medical diagnosis</div></div>', unsafe_allow_html=True)
st.write("")  # spacing

# Load models and meta
@st.cache_resource
def load_artifacts():
    artifacts = {}
    try:
        with open("meta.json", "r") as f:
            artifacts["meta"] = json.load(f)
    except Exception:
        artifacts["meta"] = {"features": ['age_yr','ap_hi','ap_lo','cholesterol','BMI'],
                             "ensemble_weights": {"xgb": 0.7, "logreg": 0.3}}

    try:
        with open("scaler.pkl", "rb") as f:
            artifacts["scaler"] = pickle.load(f)
    except Exception:
        artifacts["scaler"] = None

    try:
        with open("xgb_model.pkl", "rb") as f:
            artifacts["xgb_model"] = pickle.load(f)
    except Exception:
        artifacts["xgb_model"] = None

    try:
        with open("logreg_model.pkl", "rb") as f:
            artifacts["logreg_model"] = pickle.load(f)
    except Exception:
        artifacts["logreg_model"] = None

    return artifacts

art = load_artifacts()
features = art["meta"].get("features", ['age_yr','ap_hi','ap_lo','cholesterol','BMI'])
w_xgb = art["meta"].get("ensemble_weights", {}).get("xgb", 0.7)
w_lr = art["meta"].get("ensemble_weights", {}).get("logreg", 0.3)

# Sidebar: quick info and guidelines link
with st.sidebar:
    st.markdown("## About this app")
    st.write("This app uses an ensemble (XGBoost + Logistic Regression) trained on a public cardio dataset.")
    st.write("⚠️ Not a medical device. For medical advice, consult a doctor.")
    st.markdown("---")
    st.markdown("### Quick tips")
    st.write("• Enter realistic values\n• Use BMI from measured height/weight\n• Share results with your physician if risk is moderate/high")
    st.markdown("---")
    st.markdown("Made for learning and demo — keep data private.")

# Input card
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Patient details")
    col1, col2 = st.columns([1,1])
    with col1:
        age_yr = st.number_input("Age (years)", min_value=1, max_value=120, value=50, step=1,
                                 help="Patient age in whole years")
        ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=50, max_value=300, value=120, step=1,
                                help="Systolic blood pressure (mm Hg)")
        ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=30, max_value=200, value=80, step=1,
                                help="Diastolic blood pressure (mm Hg)")
    with col2:
        height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0, step=0.5)
        weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.1)
        cholesterol = st.selectbox("Cholesterol level", options=[1,2,3],
                                   format_func=lambda x: {1:"1 — Normal", 2:"2 — Above normal", 3:"3 — High"}[x])

    # compute BMI
    try:
        bmi = weight / ((height / 100) ** 2)
    except Exception:
        bmi = 0.0
    st.markdown(f"**Computed BMI:** `{bmi:.2f}`")

    # small validation
    validation_msgs = []
    if ap_lo >= ap_hi:
        validation_msgs.append("Diastolic pressure should be lower than systolic.")
    if bmi < 10 or bmi > 60:
        validation_msgs.append("BMI looks unusual — check height/weight.")
    if age_yr < 1 or age_yr > 120:
        validation_msgs.append("Age unusually out of range.")

    if validation_msgs:
        for m in validation_msgs:
            st.warning(m)

    # predict button
    st.markdown("---")
    predict_btn = st.button("Predict Risk")
    st.markdown('</div>', unsafe_allow_html=True)

# When user clicks predict
if predict_btn:
    X = np.array([[age_yr, ap_hi, ap_lo, cholesterol, bmi]])
    # XGBoost
    proba_xgb = None
    proba_lr = None

    if art.get("xgb_model") is not None:
        try:
            proba_xgb = art["xgb_model"].predict_proba(X)[:,1][0]
        except Exception as e:
            st.error("XGBoost prediction failed: " + str(e))
            proba_xgb = None
    if art.get("logreg_model") is not None and art.get("scaler") is not None:
        try:
            X_scaled = art["scaler"].transform(X)
            proba_lr = art["logreg_model"].predict_proba(X_scaled)[:,1][0]
        except Exception as e:
            st.error("Logistic Regression prediction failed: " + str(e))
            proba_lr = None

    # Fallbacks
    if proba_xgb is None and proba_lr is None:
        st.error("No predictive models available. Ensure model files (xgb_model.pkl, logreg_model.pkl, scaler.pkl) are present.")
    else:
        # Use sensible defaults if one model missing
        if proba_xgb is None:
            w_lr = 1.0
            w_xgb = 0.0
        if proba_lr is None:
            w_xgb = 1.0
            w_lr = 0.0

        # ensemble
        prob = w_xgb * (proba_xgb if proba_xgb is not None else 0.0) + w_lr * (proba_lr if proba_lr is not None else 0.0)
        prob = float(np.clip(prob, 0.0, 1.0))
        pred_class = int(prob >= 0.5)

        # Results display
        st.markdown("## Prediction")
        left, right = st.columns([1,2])
        with left:
            st.metric("Predicted class (1 = heart disease)", pred_class)
            st.write("Ensemble score:")
            st.progress(int(prob * 100))
            st.write(f"**Probability:** {prob:.3f}")
        with right:
            # Interpretation card
            if prob >= 0.75:
                st.warning("High risk — seek clinical evaluation and urgent care if symptomatic.")
            elif prob >= 0.5:
                st.info("Moderate risk — consult a doctor and review lifestyle & meds.")
            else:
                st.success("Low risk — good, maintain healthy habits and routine checks.")

            # Show model breakdown
            st.markdown("**Model breakdown:**")
            if proba_xgb is not None:
                st.write(f"- XGBoost: {proba_xgb:.3f}")
            if proba_lr is not None:
                st.write(f"- Logistic Regression: {proba_lr:.3f}")
            st.write(f"- Ensemble weights used: XGBoost={w_xgb:.2f}, LogReg={w_lr:.2f}")

        # Feature importance (if XGBoost model present)
        if art.get("xgb_model") is not None:
            try:
                import matplotlib.pyplot as plt
                from matplotlib.ticker import MaxNLocator

                booster = art["xgb_model"].get_booster() if hasattr(art["xgb_model"], "get_booster") else None
                # attempt to get feature importance from xgb
                fmap = art["xgb_model"].get_booster().get_score(importance_type='weight') if booster else {}
                # convert to a simple importance dict using model's feature_names if available
                importance = {}
                if hasattr(art["xgb_model"], "feature_names") and art["xgb_model"].feature_names is not None:
                    # newer versions
                    for i, name in enumerate(art["xgb_model"].feature_names):
                        name_key = f"f{i}"
                        importance[name] = booster.get_score(importance_type='weight').get(name_key, 0) if booster else 0
                else:
                    # fallback: try model.get_booster().get_score()
                    raw = booster.get_score(importance_type='weight') if booster else {}
                    # keys like f0, f1 -> map to features
                    for k, v in raw.items():
                        if k.startswith("f"):
                            idx = int(k[1:])
                            if idx < len(features):
                                importance[features[idx]] = v
                # If no importance computed, skip
                if importance:
                    # create bar chart
                    imp_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                    names = [x[0] for x in imp_items]
                    vals = [x[1] for x in imp_items]
                    fig, ax = plt.subplots(figsize=(4, 2.8))
                    ax.barh(names[::-1], vals[::-1])
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                    plt.title("Feature importance (XGBoost)")
                    st.pyplot(fig)
            except Exception:
                # quietly ignore plotting failures
                pass

        # Allow user to download a small CSV summary of this input + prediction
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "age_yr": age_yr, "ap_hi": ap_hi, "ap_lo": ap_lo,
            "height_cm": height, "weight_kg": weight, "BMI": round(bmi, 2),
            "cholesterol": cholesterol, "probability": round(prob, 4), "predicted_class": pred_class
        }
        df_summary = pd.DataFrame([summary])
        csv = df_summary.to_csv(index=False).encode('utf-8')
        st.download_button("Download prediction summary (CSV)", csv, file_name="prediction_summary.csv", mime="text/csv")

# Always show a helpful guidelines / suggestions panel
st.markdown("---")
st.markdown("## ❤️ Heart-Healthy Guidelines")
st.markdown(
    """
- **Keep active.** Aim for at least **150 minutes** of moderate aerobic activity per week (e.g., brisk walking).
- **Eat balanced.** Prefer whole grains, vegetables, fruits, lean proteins and healthy fats (olive oil, nuts). Reduce processed foods & trans fats.
- **Limit salt & sugar.** High sodium increases blood pressure; limit sugary drinks and sweets.
- **Maintain healthy weight.** Even modest weight loss (5–10%) can lower risk.
- **Monitor blood pressure & cholesterol.** Get checked regularly — early detection helps.
- **Avoid tobacco & limit alcohol.** Smoking drastically raises cardiovascular risk.
- **Manage stress & sleep.** Aim for 7–9 hours sleep; use relaxation techniques if stressed.
- **Follow medical advice.** If you have conditions (diabetes, hypertension), take prescribed medicines and follow up with your doctor.
"""
)

st.markdown("**Disclaimer:** This app provides an estimate based on a trained model. It is NOT a medical diagnosis. If you're concerned about your health, please consult a healthcare professional.")
