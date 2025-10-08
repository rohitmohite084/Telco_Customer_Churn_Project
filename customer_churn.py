import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# Load Model and Dataset
# ----------------------------
# Make sure these files are in the same folder as this script
model = pickle.load(open("customer_churn.pkl", "rb"))
df = pd.read_csv("customer_churn.csv")

# Preprocess dataset for label encoding
df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Identify categorical columns
categorical_cols = df.select_dtypes(include='object').columns.tolist()

# Fit LabelEncoders
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ----------------------------
# Streamlit App
# ----------------------------
st.title("📊 Telco Customer Churn Prediction")

# Sidebar Inputs
st.sidebar.header("Customer Information")
def user_input_features():
    data = {
        "gender": st.sidebar.selectbox("Gender", ("Male", "Female")),
        "SeniorCitizen": st.sidebar.selectbox("Senior Citizen", (0, 1)),
        "Partner": st.sidebar.selectbox("Partner", ("Yes", "No")),
        "Dependents": st.sidebar.selectbox("Dependents", ("Yes", "No")),
        "tenure": st.sidebar.slider("Tenure (months)", 0, 72, 12),
        "PhoneService": st.sidebar.selectbox("Phone Service", ("Yes", "No")),
        "MultipleLines": st.sidebar.selectbox("Multiple Lines", ("Yes", "No", "No phone service")),
        "InternetService": st.sidebar.selectbox("Internet Service", ("DSL", "Fiber optic", "No")),
        "OnlineSecurity": st.sidebar.selectbox("Online Security", ("Yes", "No", "No internet service")),
        "OnlineBackup": st.sidebar.selectbox("Online Backup", ("Yes", "No", "No internet service")),
        "DeviceProtection": st.sidebar.selectbox("Device Protection", ("Yes", "No", "No internet service")),
        "TechSupport": st.sidebar.selectbox("Tech Support", ("Yes", "No", "No internet service")),
        "StreamingTV": st.sidebar.selectbox("Streaming TV", ("Yes", "No", "No internet service")),
        "StreamingMovies": st.sidebar.selectbox("Streaming Movies", ("Yes", "No", "No internet service")),
        "Contract": st.sidebar.selectbox("Contract", ("Month-to-month", "One year", "Two year")),
        "PaperlessBilling": st.sidebar.selectbox("Paperless Billing", ("Yes", "No")),
        "PaymentMethod": st.sidebar.selectbox(
            "Payment Method", 
            ("Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)")
        ),
        "MonthlyCharges": st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0),
        "TotalCharges": st.sidebar.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1500.0)
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Encode categorical user input
for col in categorical_cols:
    if col in input_df.columns:
        le = encoders[col]
        input_df[col] = le.transform(input_df[col])

# ----------------------------
# Prediction
# ----------------------------
st.subheader("Predict Customer Churn")
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ The customer is **likely to churn** (Probability: {prediction_proba:.2f})")
    else:
        st.success(f"✅ The customer is **not likely to churn** (Probability: {prediction_proba:.2f})")

st.markdown("---")
st.caption("Created with ❤️ using Streamlit and Scikit-learn")
