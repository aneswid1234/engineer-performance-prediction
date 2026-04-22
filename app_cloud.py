from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "Models"

SCALER_CLASS_PATH = MODELS_DIR / "scaler_class.pkl"
SCALER_REG_PATH = MODELS_DIR / "scaler_reg.pkl"
SCALER_ANN_CLASS_PATH = MODELS_DIR / "scaler_ann_class.pkl"
SCALER_ANN_REG_PATH = MODELS_DIR / "scaler_ann_reg.pkl"

CLASS_MODEL_PATH = MODELS_DIR / "best_classification_model.pkl"
REG_MODEL_PATH = MODELS_DIR / "deploy_regression_model.pkl"
ANN_CLASS_MODEL_PATH = MODELS_DIR / "best_ann_classification.keras"
ANN_REG_MODEL_PATH = MODELS_DIR / "best_ann_regression.keras"

FULL_FEATURES = [
    "Certificates",
    "Years of Experience",
    "age",
    "Time Arrival Strafe",
    "Project Cost",
    "Project Proximity",
    "Violation Risk Index",
    "Company PCAB Score",
    "Weekly Overtime Hours",
    "Salary Bracket",
    "Experience_Ratio",
    "Punctuality_Score",
    "Burnout_Risk",
    "Salary_Experience_Ratio",
]

EFFICIENCY_MIN = 0.0
EFFICIENCY_MAX = 14.0


st.set_page_config(
    page_title="Engineer Performance Prediction",
    layout="wide",
)

st.markdown(
    """
    <style>
    .card {
        padding: 24px;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 6px 14px rgba(0,0,0,0.08);
        margin-bottom: 10px;
    }

    .green { background-color: #E8F8F5; }
    .blue { background-color: #EAF2FF; }
    .purple { background-color: #F3E8FF; }
    .orange { background-color: #FFF4E6; }

    .metric-title {
        font-size: 14px;
        color: #555;
    }

    .metric-value {
        font-size: 30px;
        font-weight: bold;
    }

    .streamlit-expanderHeader {
        font-size: 20px !important;
        font-weight: bold !important;
    }

    div.stButton > button {
        display: block;
        margin: 0 auto;
        padding: 12px 28px;
        font-size: 18px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏗️ Engineer Performance Prediction")
st.caption("Multi-Model Prediction: Classification, Regression, ANN Classification, ANN Regression")


def get_color(score: float) -> str:
    if score > 0.8:
        return "#1ABC9C"
    if score >= 0.4:
        return "#FACC15"
    return "#EF4444"


def normalize_value(value: float, min_value: float, max_value: float) -> float:
    if max_value == min_value:
        return 0.0
    normalized = (value - min_value) / (max_value - min_value)
    return max(0.0, min(1.0, normalized))


def get_efficiency_label(score: float) -> str:
    if score > 0.8:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"


def build_feature_dataframe(
    certificates,
    years_experience,
    age,
    time_arrival_strafe,
    project_cost,
    project_proximity,
    violation_risk_index,
    company_pcab_score,
    weekly_overtime_hours,
    salary_bracket,
):
    experience_ratio = years_experience / (age + 1)
    punctuality_score = time_arrival_strafe * -1
    burnout_risk = project_proximity * weekly_overtime_hours * violation_risk_index
    salary_experience_ratio = salary_bracket / (years_experience + 1)

    data = {
        "Certificates": [certificates],
        "Years of Experience": [years_experience],
        "age": [age],
        "Time Arrival Strafe": [time_arrival_strafe],
        "Project Cost": [project_cost],
        "Project Proximity": [project_proximity],
        "Violation Risk Index": [violation_risk_index],
        "Company PCAB Score": [company_pcab_score],
        "Weekly Overtime Hours": [weekly_overtime_hours],
        "Salary Bracket": [salary_bracket],
        "Experience_Ratio": [experience_ratio],
        "Punctuality_Score": [punctuality_score],
        "Burnout_Risk": [burnout_risk],
        "Salary_Experience_Ratio": [salary_experience_ratio],
    }

    return pd.DataFrame(data, columns=FULL_FEATURES)


@st.cache_resource
def load_assets():
    scalers = {
        "class": joblib.load(SCALER_CLASS_PATH),
        "reg": joblib.load(SCALER_REG_PATH),
        "ann_class": joblib.load(SCALER_ANN_CLASS_PATH),
        "ann_reg": joblib.load(SCALER_ANN_REG_PATH),
    }

    models = {
        "class_model": joblib.load(CLASS_MODEL_PATH),
        "reg_model": joblib.load(REG_MODEL_PATH),
        "ann_class_model": tf.keras.models.load_model(ANN_CLASS_MODEL_PATH),
        "ann_reg_model": tf.keras.models.load_model(ANN_REG_MODEL_PATH),
    }

    return scalers, models


with st.expander("👷🏻‍♀️👷🏻‍♂️ Input Engineer Profile", expanded=True):
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        certificates = st.number_input("Certificates", min_value=0, max_value=50, value=5, step=1)
        years_experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5, step=1)

    with c2:
        age = st.number_input("Age", min_value=18, max_value=80, value=30, step=1)
        weekly_overtime_hours = st.number_input("Weekly Overtime Hours", min_value=0.0, max_value=80.0, value=10.0, step=0.5)

    with c3:
        time_arrival_strafe = st.number_input("Time Arrival Strafe", min_value=0.0, max_value=120.0, value=5.0, step=0.5)
        project_proximity = st.slider("Project Proximity", min_value=0.0, max_value=500.0, value=50.0, step=1.0)

    with c4:
        violation_risk_index = st.slider("Violation Risk Index", min_value=0.0, max_value=1.0, value=0.30, step=0.01)
        company_pcab_score = st.number_input("Company PCAB Score", min_value=0, max_value=100, value=75, step=1)

    with c5:
        project_cost = st.number_input("Project Cost", min_value=0.0, value=150000.0, step=1000.0)
        salary_bracket = st.slider("Salary Bracket", min_value=1, max_value=10, value=5, step=1)


input_df = build_feature_dataframe(
    certificates=certificates,
    years_experience=years_experience,
    age=age,
    time_arrival_strafe=time_arrival_strafe,
    project_cost=project_cost,
    project_proximity=project_proximity,
    violation_risk_index=violation_risk_index,
    company_pcab_score=company_pcab_score,
    weekly_overtime_hours=weekly_overtime_hours,
    salary_bracket=salary_bracket,
)

st.subheader("Processed Features")
st.dataframe(input_df, use_container_width=True)

if st.button("🚀 Run Analysis"):
    try:
        scalers, models = load_assets()

        class_model = models["class_model"]
        reg_model = models["reg_model"]
        ann_class_model = models["ann_class_model"]
        ann_reg_model = models["ann_reg_model"]

        class_features = list(class_model.feature_names_in_) if hasattr(class_model, "feature_names_in_") else FULL_FEATURES
        reg_features = list(reg_model.feature_names_in_) if hasattr(reg_model, "feature_names_in_") else FULL_FEATURES

        input_scaled_class_all = pd.DataFrame(
            scalers["class"].transform(input_df[FULL_FEATURES]),
            columns=FULL_FEATURES,
        )
        input_scaled_reg_all = pd.DataFrame(
            scalers["reg"].transform(input_df[FULL_FEATURES]),
            columns=FULL_FEATURES,
        )
        input_scaled_ann_class = pd.DataFrame(
            scalers["ann_class"].transform(input_df[FULL_FEATURES]),
            columns=FULL_FEATURES,
        )
        input_scaled_ann_reg = pd.DataFrame(
            scalers["ann_reg"].transform(input_df[FULL_FEATURES]),
            columns=FULL_FEATURES,
        )

        class_input_final = input_scaled_class_all[class_features]
        reg_input_final = input_scaled_reg_all[reg_features]

        class_pred = int(class_model.predict(class_input_final)[0])
        class_prob = (
            float(class_model.predict_proba(class_input_final)[0][1])
            if hasattr(class_model, "predict_proba")
            else float(class_pred)
        )

        reg_pred = float(reg_model.predict(reg_input_final)[0])

        ann_class_prob = float(ann_class_model.predict(input_scaled_ann_class, verbose=0).ravel()[0])
        ann_class_pred = 1 if ann_class_prob >= 0.5 else 0

        ann_reg_pred = float(ann_reg_model.predict(input_scaled_ann_reg, verbose=0).ravel()[0])

        label_class = "Good" if class_pred == 1 else "Not Good"
        label_ann = "Good" if ann_class_pred == 1 else "Not Good"

        reg_score = normalize_value(reg_pred, EFFICIENCY_MIN, EFFICIENCY_MAX)
        ann_reg_score = normalize_value(ann_reg_pred, EFFICIENCY_MIN, EFFICIENCY_MAX)

        reg_percent = reg_score * 100
        ann_reg_percent = ann_reg_score * 100

        reg_label = get_efficiency_label(reg_score)
        ann_reg_label = get_efficiency_label(ann_reg_score)

        color_class = get_color(class_prob)
        color_ann = get_color(ann_class_prob)
        color_reg = get_color(reg_score)
        color_reg_ann = get_color(ann_reg_score)

        st.markdown("## 📊 Model Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"""
                <div class="card green">
                    <div class="metric-title">Classification</div>
                    <div style="color:{color_class}; font-size:32px; font-weight:bold;">
                        {label_class}
                    </div>
                    <div class="metric-title">Confidence: {class_prob:.2%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="card blue">
                    <div class="metric-title">Regression</div>
                    <div style="color:{color_reg}; font-size:30px; font-weight:bold;">
                        {reg_percent:.1f}%
                    </div>
                    <div class="metric-title">Efficiency Level: {reg_label}</div>
                    <div class="metric-title">Raw Efficiency: {reg_pred:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown(
                f"""
                <div class="card purple">
                    <div class="metric-title">ANN Classification</div>
                    <div style="color:{color_ann}; font-size:32px; font-weight:bold;">
                        {label_ann}
                    </div>
                    <div class="metric-title">Confidence: {ann_class_prob:.2%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col4:
            st.markdown(
                f"""
                <div class="card orange">
                    <div class="metric-title">ANN Regression</div>
                    <div style="color:{color_reg_ann}; font-size:30px; font-weight:bold;">
                        {ann_reg_percent:.1f}%
                    </div>
                    <div class="metric-title">Efficiency Level: {ann_reg_label}</div>
                    <div class="metric-title">Raw Efficiency: {ann_reg_pred:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.success("Analysis completed 🚀")

        with st.expander("Selected Features"):
            st.write("Classification features:", class_features)
            st.write("Regression features:", reg_features)

    except Exception as e:
        st.error(f"Terjadi error saat memuat model atau menjalankan prediksi: {e}")
