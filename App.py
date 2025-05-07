import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
clf = joblib.load('water_risk_model.pkl')
encoders = joblib.load('encoders.pkl')

st.title("💧 Campus Water Crisis: Risk Predictor")
st.markdown("Predict illness risk & see community patterns from water survey data.")

# Create input fields dynamically based on encoders
input_data = {}
for col, encoder in encoders.items():
    if col == 'illness_experience':  # exclude the target label
        continue
    options = list(encoder.classes_)
    choice = st.selectbox(f"{col.replace('_', ' ').capitalize()}:", options)
    input_data[col] = choice

input_df = pd.DataFrame([input_data])
for col in input_df.columns:
    input_df[col] = encoders[col].transform(input_df[col].astype(str))

# Prediction
if st.button("Predict Illness Risk"):
    prediction = clf.predict(input_df)[0]
    prob = clf.predict_proba(input_df)[0][prediction]
    result = "⚠️ High Risk" if prediction == 1 else "✅ Low Risk"
    st.subheader(f"Prediction: {result}")
    st.caption(f"Confidence: {prob:.2%}")

