from pathlib import Path
import zipfile
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Dashboard", layout="centered")

BASE_DIR = Path(__file__).resolve().parent

ZIP_PATH = BASE_DIR / "archive.zip"
MODEL_PATH = BASE_DIR / "logistic_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
KMEANS_PATH = BASE_DIR / "kmeans_model.pkl"

st.title("Customer Churn Prediction Dashboard")
st.write(
    "Enter customer information below to predict churn risk and assign the customer to a behavioral segment."
)


@st.cache_resource
def load_artifacts():
    log_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    return log_model, scaler, kmeans


@st.cache_data
def load_reference_data():
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        with z.open("WA_Fn-UseC_-Telco-Customer-Churn.csv") as f:
            df = pd.read_csv(f)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    model_df = df.drop(columns=["customerID"]).copy()
    model_df["Churn"] = model_df["Churn"].map({"No": 0, "Yes": 1})

    reference_raw = model_df.drop(columns=["Churn"]).copy()

    model_df_encoded = pd.get_dummies(model_df, drop_first=True)
    X = model_df_encoded.drop(columns=["Churn"])

    cluster_ref = model_df[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]].copy()
    cluster_mean = cluster_ref.mean()
    cluster_std = cluster_ref.std(ddof=0).replace(0, 1)

    return X.columns.tolist(), cluster_mean, cluster_std, reference_raw


log_model, scaler, kmeans = load_artifacts()
feature_columns, cluster_mean, cluster_std, reference_raw = load_reference_data()

st.subheader("Customer Input")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1], help="0 = No, 1 = Yes")
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", min_value=1, max_value=72, value=12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    if phone_service == "No":
        multiple_lines = "No phone service"
    else:
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])

    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col2:
    if internet_service == "No":
        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"
    else:
        online_security = st.selectbox("Online Security", ["No", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])

    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=18.25,
        max_value=118.75,
        value=70.00,
        step=0.01,
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=18.80,
        max_value=9000.00,
        value=500.00,
        step=0.01,
    )


def build_input_row():
    return pd.DataFrame(
        [
            {
                "SeniorCitizen": senior,
                "tenure": tenure,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
                "gender": gender,
                "Partner": partner,
                "Dependents": dependents,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment_method,
            }
        ]
    )


def assign_segment_name(cluster_id):
    names = {
        0: "Cluster 0 - Stable long-term customers",
        1: "Cluster 1 - Lower-charge, lower-total-charge customers",
        2: "Cluster 2 - Higher-charge customers with highest historical churn",
    }
    return names.get(cluster_id, f"Cluster {cluster_id}")


def get_segment_description(cluster_id):
    descriptions = {
        0: "This segment historically includes customers with long tenure, high total charges, and low churn rates.",
        1: "This segment historically includes customers with lower monthly and total charges, and may include a mix of newer or less costly customer profiles.",
        2: "This segment historically includes customers with moderate tenure, higher monthly charges, and the highest churn rate.",
    }
    return descriptions.get(cluster_id, "No segment description available.")


def get_risk_level(probability):
    if probability < 0.20:
        return "Low"
    elif probability < 0.40:
        return "Moderate"
    else:
        return "High"


if st.button("Predict Churn"):
    input_df = build_input_row()

    combined = pd.concat([reference_raw, input_df], ignore_index=True)
    combined_encoded = pd.get_dummies(combined, drop_first=True)
    encoded = combined_encoded.reindex(columns=feature_columns, fill_value=0).tail(1)

    scaled_features = scaler.transform(encoded)
    prob = log_model.predict_proba(scaled_features)[0][1]

    # Custom threshold for demo/business flagging
    pred = 1 if prob >= 0.40 else 0

    risk_level = get_risk_level(prob)

    cluster_df = input_df[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]].copy()
    cluster_scaled = (cluster_df - cluster_mean) / cluster_std
    cluster_id = int(kmeans.predict(cluster_scaled)[0])

    st.subheader("Prediction Result")
    st.metric("Predicted Churn", "Yes" if pred == 1 else "No")
    st.metric("Churn Probability", f"{prob:.2%}")
    st.metric("Risk Level", risk_level)
    st.metric("Customer Segment", assign_segment_name(cluster_id))

    st.caption(
        "Customer segment is based on similarity grouping, while churn prediction comes from the logistic regression model."
    )
    st.write(get_segment_description(cluster_id))

    reasons = []
    if tenure <= 12:
        reasons.append("short tenure")
    if contract == "Month-to-month":
        reasons.append("month-to-month contract")
    if monthly_charges >= 80:
        reasons.append("higher monthly charges")
    if payment_method == "Electronic check":
        reasons.append("electronic check payment")
    if internet_service == "Fiber optic":
        reasons.append("fiber optic service")
    if senior == 1:
        reasons.append("senior-citizen customer profile")
    if online_security == "No":
        reasons.append("lack of online security")
    if tech_support == "No":
        reasons.append("lack of tech support")

    st.subheader("Interpretation")
    if reasons:
        st.write("This result may be influenced by: " + ", ".join(reasons) + ".")
    else:
        st.write(
            "This customer profile does not show the strongest common risk signals from the training results."
        )

st.markdown("---")
st.caption("Model used: Logistic Regression | Segmentation used: K-Means clustering")