
# Modern Cancer Risk Assessment Dashboard
# Updated Streamlit UI with responsive layout, tabs, Plotly charts,
# professional styling, and downloadable reports.

import os
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Cancer Risk Assessment Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = "best_model.keras"
SCALER_PATH = "scaler.pkl"

CLASS_NAMES = ["Low", "Medium", "High"]
CLASS_COLORS = {"Low": "#2E7D32", "Medium": "#ED6C02", "High": "#D32F2F"}

FEATURES = {
    "Age": {"min": 14, "max": 73, "default": 37},
    "Gender": {"min": 1, "max": 2, "default": 1},
    "Air Pollution": {"min": 1, "max": 8, "default": 4},
    "Alcohol use": {"min": 1, "max": 8, "default": 4},
    "Dust Allergy": {"min": 1, "max": 8, "default": 5},
    "OccuPational Hazards": {"min": 1, "max": 8, "default": 5},
    "Genetic Risk": {"min": 1, "max": 7, "default": 4},
    "chronic Lung Disease": {"min": 1, "max": 7, "default": 4},
    "Balanced Diet": {"min": 1, "max": 7, "default": 4},
    "Obesity": {"min": 1, "max": 7, "default": 4},
    "Smoking": {"min": 1, "max": 8, "default": 4},
    "Passive Smoker": {"min": 1, "max": 8, "default": 4},
    "Chest Pain": {"min": 1, "max": 9, "default": 4},
    "Coughing of Blood": {"min": 1, "max": 9, "default": 4},
    "Fatigue": {"min": 1, "max": 9, "default": 4},
    "Weight Loss": {"min": 1, "max": 8, "default": 4},
    "Shortness of Breath": {"min": 1, "max": 9, "default": 4},
    "Wheezing": {"min": 1, "max": 8, "default": 4},
    "Swallowing Difficulty": {"min": 1, "max": 8, "default": 4},
    "Clubbing of Finger Nails": {"min": 1, "max": 9, "default": 4},
    "Frequent Cold": {"min": 1, "max": 7, "default": 3},
    "Dry Cough": {"min": 1, "max": 7, "default": 4},
    "Snoring": {"min": 1, "max": 7, "default": 3},
}

st.markdown("""
<style>
.block-container{max-width:1400px;padding-top:1.5rem;}
[data-testid="stMetric"]{
 border:1px solid #E5E7EB;
 border-radius:12px;
 padding:1rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    return (
        tf.keras.models.load_model(MODEL_PATH),
        joblib.load(SCALER_PATH),
    )

def predict_risk(values, model, scaler):
    scaled = scaler.transform(values.reshape(1, -1))
    probs = model.predict(scaled, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], probs

st.title("Cancer Risk Assessment Dashboard")
st.caption("Clinical decision support demonstration using a trained Artificial Neural Network")

model, scaler = load_model_and_scaler()

if model is None:
    st.error("best_model.keras or scaler.pkl not found.")
    st.stop()

tab1, tab2, tab3 = st.tabs(
    ["Patient Information", "Risk Factors", "Clinical Symptoms"]
)

values = {}

with tab1:
    values["Age"] = st.number_input("Age", 14, 100, 37)
    gender = st.selectbox("Gender", ["Male", "Female"])
    values["Gender"] = 1 if gender == "Male" else 2

with tab2:
    risk_fields = [
        "Air Pollution","Alcohol use","Dust Allergy",
        "OccuPational Hazards","Genetic Risk",
        "chronic Lung Disease","Balanced Diet",
        "Obesity","Smoking","Passive Smoker"
    ]
    cols = st.columns(2)
    for i, f in enumerate(risk_fields):
        with cols[i % 2]:
            values[f] = st.slider(
                f,
                FEATURES[f]["min"],
                FEATURES[f]["max"],
                FEATURES[f]["default"]
            )

with tab3:
    symptom_fields = [
        "Chest Pain","Coughing of Blood","Fatigue",
        "Weight Loss","Shortness of Breath",
        "Wheezing","Swallowing Difficulty",
        "Clubbing of Finger Nails","Frequent Cold",
        "Dry Cough","Snoring"
    ]
    cols = st.columns(2)
    for i, f in enumerate(symptom_fields):
        with cols[i % 2]:
            values[f] = st.slider(
                f,
                FEATURES[f]["min"],
                FEATURES[f]["max"],
                FEATURES[f]["default"]
            )

st.divider()

if st.button("Run Assessment", use_container_width=True, type="primary"):
    ordered = np.array(
        [values.get(k, FEATURES[k]["default"]) for k in FEATURES.keys()],
        dtype=float
    )

    pred_class, probs = predict_risk(ordered, model, scaler)
    confidence = float(np.max(probs) * 100)

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Risk Category", pred_class)
    with c2:
        st.metric("Confidence", f"{confidence:.1f}%")

    chart_df = pd.DataFrame({
        "Risk Level": CLASS_NAMES,
        "Probability": probs * 100
    })

    fig = px.bar(
        chart_df,
        x="Probability",
        y="Risk Level",
        orientation="h",
        text="Probability"
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "prediction": pred_class,
        "confidence": round(confidence, 2),
        "inputs": values,
    }

    st.download_button(
        "Download Assessment Report (JSON)",
        data=json.dumps(report, indent=2),
        file_name="assessment_report.json",
        mime="application/json",
    )

    st.dataframe(
        pd.DataFrame({
            "Feature": list(values.keys()),
            "Value": list(values.values())
        }),
        use_container_width=True
    )

st.divider()
st.caption(
    "BAN6440 | Nexford University | Oluwanifemi Favour Odukoya | TensorFlow/Keras ANN"
)
