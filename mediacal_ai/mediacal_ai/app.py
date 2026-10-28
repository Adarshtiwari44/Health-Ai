
from pathlib import Path

import pandas as pd
import streamlit as st

from src.mediacal_ai.predict import (
    get_model_status,
    predict_insurance_charges,
    predict_hospital_risk,
    predict_pharma_side_effect,
    predict_readmission,
    HEALTH_OUTCOME_FEATURES,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="MediCal AI",
    page_icon="🏥",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetic_data.csv"


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

@st.cache_data
def load_readmission_data(path):
    return pd.read_csv(path)


def show_disclaimer():
    st.caption(
        "Research prototype only. Predictions are estimates, "
        "not medical advice, insurance decisions, or guarantees."
    )


def show_model_status():
    statuses = get_model_status()

    labels = {
        "insurance": "Insurance Charges",
        "hospital_risk": "Hospital Financial Risk",
        "pharma_side_effect": "Pharma Side Effects",
        "health_outcome": "30-Day Readmission",
    }

    cols = st.columns(4)

    for col, (key, label) in zip(cols, labels.items()):
        with col:
            if statuses.get(key, False):
                st.success(f"{label}: Loaded")
            else:
                st.error(f"{label}: Not loaded")


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🏥 MediCal AI")
st.sidebar.caption("Healthcare Risk Analytics Platform")

page = st.sidebar.radio(
    "Choose a module",
    [
        "Dashboard",
        "Insurance Charges",
        "Hospital Financial Risk",
        "Pharma Side Effects",
        "30-Day Readmission Risk",
    ],
)

st.sidebar.divider()
st.sidebar.info(
    "This is an experimental ML project. "
    "Do not use its predictions to make clinical decisions."
)


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":
    st.title("🏥 MediCal AI Dashboard")

    st.write(
        "An experimental healthcare analytics platform using "
        "machine learning to explore insurance costs, hospital "
        "financial performance, drug side-effect patterns, and "
        "readmission risk."
    )

    show_model_status()

    st.divider()
    st.subheader("Available modules")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💰 Insurance Charges")
        st.write(
            "Estimate insurance charges from demographic and "
            "lifestyle features."
        )

        st.markdown("### 🏥 Hospital Financial Risk")
        st.write(
            "Estimate whether a hospital falls into the "
            "reported negative operating-margin category."
        )

    with col2:
        st.markdown("### 💊 Pharma Side Effects")
        st.write(
            "Explore side-effect patterns learned from the "
            "provided research dataset."
        )

        st.markdown("### 📊 30-Day Readmission Risk")
        st.write(
            "Explore a model's prediction of readmission within "
            "30 days using historical patient records."
        )

    st.divider()
    st.subheader("Important model limitations")

    st.warning(
        "These models are experimental and have not been "
        "independently clinically validated. The pharma model "
        "has weak evaluation results, and the readmission model "
        "has limited predictive performance. Use only for "
        "educational and research demonstrations."
    )

    show_disclaimer()


# --------------------------------------------------
# INSURANCE CHARGES
# --------------------------------------------------

elif page == "Insurance Charges":
    st.title("💰 Insurance Charges Prediction")

    st.write(
        "Estimate insurance charges using the trained "
        "regression model."
    )

    with st.form("insurance_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=30,
            )

            sex = st.selectbox(
                "Sex",
                ["male", "female"],
            )

            bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=80.0,
                value=25.5,
                step=0.1,
            )

        with col2:
            children = st.number_input(
                "Number of children",
                min_value=0,
                max_value=10,
                value=1,
            )

            smoker = st.selectbox(
                "Smoker",
                ["no", "yes"],
            )

            region = st.selectbox(
                "Region",
                ["southwest", "southeast", "northwest", "northeast"],
            )

        submitted = st.form_submit_button(
            "Estimate Charges",
            type="primary",
        )

    if submitted:
        try:
            prediction = predict_insurance_charges(
                age=int(age),
                sex=sex,
                bmi=float(bmi),
                children=int(children),
                smoker=smoker,
                region=region,
            )

            st.metric(
                "Estimated Insurance Charges",
                f"${prediction:,.2f}",
            )

            st.caption(
                "Estimate based on the training dataset. "
                "It is not an insurance quote."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

    show_disclaimer()


# --------------------------------------------------
# HOSPITAL FINANCIAL RISK
# --------------------------------------------------

elif page == "Hospital Financial Risk":
    st.title("🏥 Hospital Financial Risk")

    st.write(
        "Estimate the likelihood of a hospital belonging to "
        "the dataset's negative operating-margin category."
    )

    with st.form("hospital_form"):
        col1, col2 = st.columns(2)

        with col1:
            city = st.text_input("City", value="Houston")
            state = st.text_input("State code", value="TX")
            control_type = st.selectbox(
                "Control type",
                [
                    "Voluntary non-profit - Private",
                    "Proprietary",
                    "Government - Hospital District or Authority",
                    "Government - State",
                    "Government - Local",
                    "Government - Federal",
                    "Other",
                ],
            )

            fy_end_year = st.number_input(
                "Fiscal year",
                min_value=2000,
                max_value=2035,
                value=2023,
            )

        with col2:
            total_beds = st.number_input(
                "Total beds",
                min_value=0,
                max_value=100000,
                value=200,
            )

            total_revenues = st.number_input(
                "Total revenues ($)",
                min_value=0.0,
                value=50_000_000.0,
                step=1_000_000.0,
            )

            net_income = st.number_input(
                "Net income ($)",
                value=1_000_000.0,
                step=100_000.0,
            )

        submitted = st.form_submit_button(
            "Estimate Financial Risk",
            type="primary",
        )

    if submitted:
        try:
            result = predict_hospital_risk(
                city=city,
                state=state,
                control_type=control_type,
                fy_end_year=int(fy_end_year),
                total_beds=int(total_beds),
                total_revenues=float(total_revenues),
                net_income=float(net_income),
            )

            probability = float(result["risk_probability"])

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Model Risk Score",
                    f"{probability:.1%}",
                )

            with col2:
                st.metric(
                    "Predicted Category",
                    str(result["predicted_class"]),
                )

            st.progress(max(0.0, min(1.0, probability)))

            st.write(result["interpretation"])
            st.warning(result["disclaimer"])

        except Exception as error:
            st.error(f"Prediction failed: {error}")

    show_disclaimer()


# --------------------------------------------------
# PHARMA SIDE EFFECTS
# --------------------------------------------------

elif page == "Pharma Side Effects":
    st.title("💊 Pharma Side-Effect Explorer")

    st.write(
        "Explore predictions from the experimental "
        "drug side-effect classification model."
    )

    with st.form("pharma_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age",
                min_value=1,
                max_value=110,
                value=45,
            )

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"],
            )

            condition = st.selectbox(
                "Condition",
                [
                    "Hypertension",
                    "Diabetes",
                    "Asthma",
                    "Arthritis",
                    "Migraine",
                    "Other",
                ],
            )

        with col2:
            drug_name = st.text_input(
                "Drug name",
                value="Amlodipine",
                help="Use a drug name present in your training dataset.",
            )

            dosage_mg = st.number_input(
                "Dosage (mg)",
                min_value=0.1,
                max_value=5000.0,
                value=5.0,
                step=0.5,
            )

            treatment_duration_days = st.number_input(
                "Treatment duration (days)",
                min_value=1,
                max_value=3650,
                value=30,
            )

        submitted = st.form_submit_button(
            "Explore Side-Effect Prediction",
            type="primary",
        )

    if submitted:
        try:
            result = predict_pharma_side_effect(
                age=int(age),
                gender=gender,
                condition=condition,
                drug_name=drug_name.strip(),
                dosage_mg=float(dosage_mg),
                treatment_duration_days=int(treatment_duration_days),
            )

            st.subheader("Experimental model output")

            st.metric(
                "Predicted side effect",
                str(result["predicted_side_effect"]),
            )

            if result.get("model_probability") is not None:
                st.metric(
                    "Model score",
                    f"{float(result['model_probability']):.1%}",
                )

            st.warning(result["disclaimer"])

        except Exception as error:
            st.error(f"Prediction failed: {error}")

    st.info(
        "The side-effect model performed poorly in evaluation. "
        "Its output is not a reliable estimate of an individual's "
        "chance of experiencing an adverse drug reaction."
    )

    show_disclaimer()


# --------------------------------------------------
# 30-DAY READMISSION RISK
# --------------------------------------------------

elif page == "30-Day Readmission Risk":
    st.title("📊 30-Day Readmission Risk")

    st.write(
        "Explore the model's estimate of readmission within "
        "30 days using features from the historical diabetes dataset."
    )

    st.warning(
        "Research demo only. This model is not clinically validated "
        "and must not guide treatment, discharge, or patient-care decisions."
    )

    if not DATA_PATH.exists():
        st.error(
            f"Training dataset not found at: {DATA_PATH}\n\n"
            "Check that data/raw/diabetic_data.csv exists."
        )
        st.stop()

    try:
        dataset = load_readmission_data(str(DATA_PATH))

        missing_features = [
            feature
            for feature in HEALTH_OUTCOME_FEATURES
            if feature not in dataset.columns
        ]

        if missing_features:
            st.error(
                "Dataset is missing model features: "
                + ", ".join(missing_features)
            )
            st.stop()

        st.caption(
            "For this prototype, fields are populated from valid "
            "values in the dataset. These defaults do not represent "
            "a real individual patient."
        )

        with st.form("readmission_form"):
            patient_data = {}

            st.subheader("Patient and encounter features")

            # Three columns keep the long form more compact.
            columns = st.columns(3)

            for index, feature in enumerate(HEALTH_OUTCOME_FEATURES):
                series = dataset[feature]
                numeric = pd.to_numeric(series, errors="coerce")

                with columns[index % 3]:
                    label = feature.replace("_", " ").title()

                    # Use numeric input for genuinely numeric columns.
                    if numeric.notna().mean() > 0.95:
                        values = numeric.dropna()

                        if values.empty:
                            patient_data[feature] = 0.0
                        else:
                            minimum = float(values.min())
                            maximum = float(values.max())
                            default = float(values.median())

                            patient_data[feature] = st.number_input(
                                label,
                                min_value=minimum,
                                max_value=maximum,
                                value=min(max(default, minimum), maximum),
                                key=f"readmission_{feature}",
                            )

                    else:
                        options = (
                            series.dropna()
                            .astype(str)
                            .loc[lambda values: values != "?"]
                            .unique()
                            .tolist()
                        )

                        options = sorted(options)

                        if not options:
                            patient_data[feature] = "Unknown"
                            st.text_input(
                                label,
                                value="Unknown",
                                disabled=True,
                                key=f"readmission_{feature}",
                            )
                        else:
                            patient_data[feature] = st.selectbox(
                                label,
                                options,
                                key=f"readmission_{feature}",
                            )

            submitted = st.form_submit_button(
                "Estimate Readmission Risk",
                type="primary",
            )

        if submitted:
            try:
                result = predict_readmission(patient_data)

                probability = float(result["risk_probability"])
                probability = max(0.0, min(1.0, probability))

                st.subheader("Prediction result")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Model-predicted probability",
                        f"{probability:.1%}",
                    )

                with col2:
                    st.metric(
                        "Model class",
                        str(result["predicted_class"]),
                    )

                st.progress(probability)

                st.write(result["interpretation"])
                st.warning(result["disclaimer"])

            except Exception as error:
                st.error(f"Readmission prediction failed: {error}")

    except Exception as error:
        st.error(f"Could not load the readmission dataset: {error}")

    show_disclaimer()
