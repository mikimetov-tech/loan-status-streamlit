import streamlit as st
import numpy as np
import pandas as pd
import joblib
from textblob import TextBlob

# =========================
# 1. Load model and settings
# =========================
model = joblib.load('loan_logreg_model.pkl')
feature_cols = joblib.load('loan_logreg_features.pkl')
default_values = joblib.load('loan_logreg_defaults.pkl')

# Mappings (same as in preprocessing)
grade_mapping = {g: i for i, g in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G'], start=1)}
sentiment_mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
home_ownership_mapping = {
    'OWN': 1,
    'MORTGAGE': 2,
    'RENT': 3,
    'OTHER': 4,
    'NONE': 5,
    'ANY': 5
}

st.set_page_config(page_title="Loan Status Prediction", page_icon="💳")

st.title("💳 Loan Status Prediction App")
st.markdown(
    "Enter a few key loan and borrower details and a short description. "
    "The model will use additional typical values in the background and "
    "predict whether the loan is likely to be **Fully Paid (1)** or **Default (0)**."
)

# =========================
# 2. Sidebar inputs (only 5 fields)
# =========================

st.sidebar.header("Input Loan Information")

loan_amnt = st.sidebar.number_input("Loan Amount", min_value=0.0, step=1000.0, value=10000.0)
term = st.sidebar.selectbox("Term (months)", [36, 60])
emp_length_years = st.sidebar.number_input("Employment Length (years)", min_value=0.0, max_value=50.0, step=1.0, value=5.0)
annual_inc = st.sidebar.number_input("Annual Income", min_value=0.0, step=1000.0, value=50000.0)

home_ownership_text = st.sidebar.selectbox(
    "Home Ownership",
    ['OWN', 'MORTGAGE', 'RENT', 'OTHER', 'NONE', 'ANY']
)

st.sidebar.markdown("---")
desc_text = st.sidebar.text_area(
    "Loan description / comment for sentiment analysis",
    "I need this loan to consolidate my debts and improve my financial situation."
)

# =========================
# 3. Convert inputs to numeric features
# =========================

def get_sentiment_label(text: str) -> int:
    """Compute sentiment label (-1, 0, 1) from free text."""
    if not text or not text.strip():
        return 0
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.05:
        return 1
    elif polarity < -0.05:
        return -1
    else:
        return 0

sentiment_label_num = get_sentiment_label(desc_text)
home_ownership_risk = home_ownership_mapping[home_ownership_text]

# Start from dataset medians (background defaults)
input_data = default_values.copy()

# Override with user-provided values
input_data['id'] = 0  # ID not important, just placeholder
input_data['loan_amnt'] = loan_amnt
input_data['term'] = float(term)
input_data['emp_length'] = emp_length_years
input_data['annual_inc'] = annual_inc
input_data['home_ownership_risk'] = home_ownership_risk
input_data['sentiment_label'] = sentiment_label_num

# Create DataFrame in correct column order
input_df = pd.DataFrame([input_data])[feature_cols]

st.subheader("Input Summary (model-ready features)")
st.write(input_df)

st.markdown(
    f"**Detected sentiment from description:** "
    f"{'positive' if sentiment_label_num == 1 else 'negative' if sentiment_label_num == -1 else 'neutral'} "
    f" (label = {sentiment_label_num})"
)

# =========================
# 4. Prediction
# =========================

if st.button("Predict Loan Status"):
    pred_class = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0, 1]

    st.subheader("Prediction Result")
    if pred_class == 1:
        st.success("✅ The model predicts this loan is likely to be **Fully Paid (1)**.")
    else:
        st.error("⚠️ The model predicts this loan is likely to **Default / Charged Off (0)**.")

    st.write(f"**Probability of Fully Paid (class 1):** {prob:.3f}")
