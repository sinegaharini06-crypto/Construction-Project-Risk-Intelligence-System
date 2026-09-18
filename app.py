import os
import joblib
import pandas as pd
import streamlit as st


# -----------------------------
# Page setup
# -----------------------------

st.set_page_config(
    page_title="Construction Risk Intelligence",
    page_icon="🏗️",
    layout="centered"
)


# -----------------------------
# Paths
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "final_risk_model.joblib"
)

ANOMALY_PATH = os.path.join(
    BASE_DIR,
    "models",
    "isolation_forest.joblib"
)


# -----------------------------
# Load models
# -----------------------------

@st.cache_resource
def load_models():

    risk_model = joblib.load(MODEL_PATH)
    anomaly_model = joblib.load(ANOMALY_PATH)

    return risk_model, anomaly_model


risk_model, anomaly_model = load_models()


# -----------------------------
# Feature engineering
# -----------------------------

def engineer_features(df):

    df = df.copy()

    df["budget_per_team_member"] = (
        df["planned_budget_usd"] / df["team_size"]
    )

    df["team_density"] = (
        df["team_size"] /
        df["planned_duration_days"]
    )

    df["subcontractor_ratio"] = (
        df["num_subcontractors"] /
        df["team_size"]
    )

    df["permit_complexity"] = (
        df["num_permits_required"] /
        df["complexity_rating"]
    )

    df["experience_complexity_ratio"] = (
        df["contractor_experience_years"] /
        df["complexity_rating"]
    )

    df["budget_duration_ratio"] = (
        df["planned_budget_usd"] /
        df["planned_duration_days"]
    )

    df["team_complexity_interaction"] = (
        df["team_size"] *
        df["complexity_rating"]
    )

    return df


# -----------------------------
# Prediction
# -----------------------------

def predict_project(project):

    df = pd.DataFrame([project])

    df = engineer_features(df)

    prediction = risk_model.predict(df)[0]

    probabilities = risk_model.predict_proba(df)[0]

    classes = risk_model.classes_

    probability_dict = {
        str(cls): float(prob)
        for cls, prob in zip(classes, probabilities)
    }

    anomaly_prediction = anomaly_model.predict(df)[0]

    anomaly_score = anomaly_model.decision_function(df)[0]

    return (
        prediction,
        probability_dict,
        anomaly_prediction,
        anomaly_score
    )


# -----------------------------
# Minimal styling
# -----------------------------

st.markdown(
    """
    <style>

    .stApp {
        background-color: #ffffff;
    }

    .block-container {
        max-width: 850px;
        padding-top: 45px;
        padding-bottom: 60px;
    }

    h1 {
        font-size: 32px !important;
        font-weight: 650 !important;
        color: #111827 !important;
    }

    h2 {
        font-size: 22px !important;
        color: #111827 !important;
    }

    h3 {
        font-size: 17px !important;
        color: #111827 !important;
    }

    p {
        color: #667085;
    }

    .subtitle {
        color: #667085;
        font-size: 15px;
        margin-bottom: 35px;
    }

    .section {
        font-size: 12px;
        font-weight: 700;
        color: #667085;
        letter-spacing: 1px;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .result {
        padding: 25px 0;
        border-top: 1px solid #e5e7eb;
        border-bottom: 1px solid #e5e7eb;
        margin-top: 30px;
    }

    .high {
        color: #dc2626;
        font-size: 36px;
        font-weight: 750;
    }

    .medium {
        color: #d97706;
        font-size: 36px;
        font-weight: 750;
    }

    .low {
        color: #059669;
        font-size: 36px;
        font-weight: 750;
    }

    .small-label {
        color: #667085;
        font-size: 12px;
        margin-bottom: 5px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Header
# -----------------------------

st.title(
    "Construction Project Risk Intelligence"
)

st.markdown(
    '<div class="subtitle">'
    'Planning-stage construction risk assessment using machine learning.'
    '</div>',
    unsafe_allow_html=True
)


# -----------------------------
# Inputs
# -----------------------------

st.markdown(
    '<div class="section">PROJECT DETAILS</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    project_type = st.selectbox(
        "Project type",
        [
            "Residential",
            "Commercial",
            "Industrial",
            "Infrastructure",
            "Renovation"
        ]
    )

    planned_budget = st.number_input(
        "Planned budget (USD)",
        min_value=10000.0,
        value=800000.0,
        step=10000.0
    )

    team_size = st.number_input(
        "Team size",
        min_value=1,
        value=15
    )

    complexity = st.slider(
        "Complexity rating",
        1,
        10,
        7
    )

    permits = st.number_input(
        "Permits required",
        min_value=0,
        value=4
    )


with col2:

    region = st.selectbox(
        "Region",
        [
            "North",
            "South",
            "East",
            "West",
            "Central"
        ]
    )

    planned_duration = st.number_input(
        "Planned duration (days)",
        min_value=1,
        value=90
    )

    contractor_experience = st.number_input(
        "Contractor experience (years)",
        min_value=0.0,
        value=8.0,
        step=0.5
    )

    subcontractors = st.number_input(
        "Subcontractors",
        min_value=0,
        value=5
    )


# -----------------------------
# Button
# -----------------------------

st.write("")

if st.button(
    "Assess Risk",
    type="primary",
    use_container_width=True
):

    project = {

        "project_type": project_type,

        "region": region,

        "planned_budget_usd":
            planned_budget,

        "planned_duration_days":
            planned_duration,

        "team_size":
            team_size,

        "contractor_experience_years":
            contractor_experience,

        "complexity_rating":
            complexity,

        "num_subcontractors":
            subcontractors,

        "num_permits_required":
            permits
    }

    (
        prediction,
        probabilities,
        anomaly_prediction,
        anomaly_score
    ) = predict_project(project)


    # -------------------------
    # Result
    # -------------------------

    st.markdown(
        '<div class="result">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section">RESULT</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="small-label">RISK LEVEL</div>',
        unsafe_allow_html=True
    )

    if prediction == "High":

        st.markdown(
            '<div class="high">HIGH</div>',
            unsafe_allow_html=True
        )

    elif prediction == "Medium":

        st.markdown(
            '<div class="medium">MEDIUM</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="low">LOW</div>',
            unsafe_allow_html=True
        )


    # -------------------------
    # Probabilities
    # -------------------------

    st.markdown(
        "### Risk probabilities"
    )

    probability_df = pd.DataFrame(
        {
            "Risk": probabilities.keys(),
            "Probability (%)": [
                round(value * 100, 1)
                for value in probabilities.values()
            ]
        }
    )

    st.dataframe(
        probability_df,
        hide_index=True,
        use_container_width=True
    )


    # -------------------------
    # Anomaly
    # -------------------------

    st.markdown(
        "### Anomaly"
    )

    if anomaly_prediction == -1:

        st.warning(
            "Anomalous planning profile"
        )

    else:

        st.success(
            "Normal planning profile"
        )

    st.caption(
        f"Anomaly score: {anomaly_score:.4f}"
    )

    st.markdown(
        """
        <div class="subtitle">
        Risk classification and anomaly detection are separate signals.
        </div>
        """,
        unsafe_allow_html=True
    )