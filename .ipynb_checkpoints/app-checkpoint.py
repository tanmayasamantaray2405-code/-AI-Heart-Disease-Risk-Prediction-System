import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Heart Disease Predictor",
    page_icon="🫀",
    layout="wide"
)

# =========================================================
# LOAD MODEL & SCALER
# =========================================================
@st.cache_resource
def load_artifacts():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    MODEL_PATH = os.path.join(BASE_DIR, "heart_disease_model.keras")
    SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

    return model, scaler


model, scaler = load_artifacts()

# =========================================================
# TITLE
# =========================================================
st.title("🫀 AI Heart Disease Predictor")
st.markdown(
    """
    This AI system analyzes patient health information and estimates
    the likelihood of heart disease.

    ⚠️ This tool is for educational purposes only and is not a medical diagnosis.
    """
)

# =========================================================
# DROPDOWN OPTIONS
# =========================================================
cp_options = {
    0: "Typical Angina",
    1: "Atypical Angina",
    2: "Non-anginal Pain",
    3: "No Symptoms"
}

ecg_options = {
    0: "Normal",
    1: "Minor ECG Abnormality",
    2: "Major ECG Abnormality"
}

slope_options = {
    0: "Upsloping",
    1: "Flat",
    2: "Downsloping"
}

thal_options = {
    1: "Normal",
    2: "Fixed Defect",
    3: "Reversible Defect"
}

yes_no_options = {
    0: "No",
    1: "Yes"
}

# =========================================================
# INPUT FORM
# =========================================================
col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=20,
        max_value=100,
        value=50,
        help="Enter the patient's age."
    )

    sex = st.selectbox(
        "Gender",
        [1, 0],
        format_func=lambda x: "Male" if x == 1 else "Female",
        help="Select the patient's gender."
    )

    cp = st.selectbox(
        "Type of Chest Pain",
        options=list(cp_options.keys()),
        format_func=lambda x: cp_options[x],
        help="Select the type of chest pain experienced."
    )

    trestbps = st.number_input(
        "Blood Pressure (mm Hg)",
        min_value=90,
        max_value=200,
        value=120,
        help="Normal resting blood pressure is around 120 mm Hg."
    )

    chol = st.number_input(
        "Cholesterol Level (mg/dl)",
        min_value=100,
        max_value=600,
        value=200,
        help="Enter the patient's cholesterol level."
    )

with col2:

    fbs = st.selectbox(
        "High Blood Sugar?",
        [0, 1],
        format_func=lambda x: yes_no_options[x],
        help="Select Yes if fasting blood sugar exceeds 120 mg/dl."
    )

    restecg = st.selectbox(
        "ECG Test Result",
        options=list(ecg_options.keys()),
        format_func=lambda x: ecg_options[x],
        help="Result of the ECG test while resting."
    )

    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=60,
        max_value=220,
        value=150,
        help="Highest heart rate achieved during exercise."
    )

    exang = st.selectbox(
        "Chest Pain During Exercise?",
        [0, 1],
        format_func=lambda x: yes_no_options[x],
        help="Did chest pain occur during exercise?"
    )

    oldpeak = st.number_input(
        "Exercise ECG Abnormality",
        min_value=0.0,
        max_value=6.0,
        value=1.0,
        step=0.1,
        help="Higher values may indicate increased heart stress."
    )

with col3:

    slope = st.selectbox(
        "ECG Pattern During Exercise",
        options=list(slope_options.keys()),
        format_func=lambda x: slope_options[x],
        help="Pattern observed in ECG during exercise."
    )

    ca = st.selectbox(
        "Blocked Blood Vessels",
        [0, 1, 2, 3],
        help="Number of major blood vessels showing blockage."
    )

    thal = st.selectbox(
        "Heart Test Result",
        options=list(thal_options.keys()),
        format_func=lambda x: thal_options[x],
        help="Result of a specialized heart diagnostic test."
    )

# =========================================================
# PREDICTION BUTTON
# =========================================================
if st.button("🔍 Predict Heart Disease Risk", use_container_width=True):

    patient_data = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    processed_input = scaler.transform(patient_data)

    probability = float(
        model.predict(processed_input, verbose=0).ravel()[0]
    )

    st.markdown("---")

    st.subheader("Prediction Result")

    if probability >= 0.5:

        st.error(
            f"⚠️ Higher Risk Detected ({probability * 100:.1f}%)"
        )

        st.write(
            """
            The entered values show patterns commonly associated
            with heart disease in the training dataset.
            """
        )

        st.info(
            "Consult a qualified healthcare professional for a proper medical evaluation."
        )

        st.markdown("### ❤️ Heart Care Recommendation")

        st.warning(
            "High risk detected. Please consult a cardiologist or visit a specialized heart hospital."
        )

        st.markdown(
            "[🏥 Find Best Heart Hospitals Near Me](https://www.google.com/maps/search/best+heart+hospital+near+me)"
        )

    else:

        st.success(
            f"✅ Lower Risk Detected ({probability * 100:.1f}%)"
        )

        st.write(
            """
            The entered values indicate a lower likelihood
            of heart disease according to the AI model.
            """
        )

    st.progress(min(int(probability * 100), 100))

    st.caption(
        "This prediction is generated by a trained deep learning model and should not be considered a medical diagnosis."
    )