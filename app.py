import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import pandas as pd
from google import genai
from google.genai import types

# ✅ Load preprocessing + best model (Change model here if needed)
scaler = pickle.load(open("models/insurance_scaler.pkl", "rb"))
model = pickle.load(open("models/insurance_lgb_model.pkl", "rb"))

# ✅ Load feature columns exactly in training order
with open("models/feature_columns.txt", "r") as f:
    feature_cols = [line.strip() for line in f.readlines()]

# ✅ Initialize Gemini client
client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

st.title("💡 Insurance Cost Prediction using Gemini AI")

st.write("Enter the details to estimate the medical insurance cost.")

# -------------------------------
# 🧍 User Inputs
# -------------------------------
age = st.slider("Age", 18, 80, 30)
bmi = st.slider("BMI", 10.0, 50.0, 25.0, 0.1)
children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])
sex = st.selectbox("Sex", ["male", "female"])

user_data = pd.DataFrame({
    "age": [age],
    "bmi": [bmi],
    "children": [children],
    "smoker_yes": [1 if smoker == "yes" else 0],
    "sex_male": [1 if sex == "male" else 0],
    "region_northeast": [1 if region == "northeast" else 0],
    "region_northwest": [1 if region == "northwest" else 0],
    "region_southeast": [1 if region == "southeast" else 0],
    "region_southwest": [1 if region == "southwest" else 0],
})

# Ensure feature alignment
user_data = user_data.reindex(columns=feature_cols, fill_value=0)

scaled_input = scaler.transform(user_data)

# -------------------------------
# ✅ Prediction + Gemini Explanation
# -------------------------------
if st.button("Predict Insurance Cost"):
    prediction = model.predict(scaled_input)[0]
    st.success(f"💰 Estimated Insurance Cost: **${prediction:,.2f}**")

    with st.spinner("Generating AI explanation..."):
        prompt = f"""
        A person has insurance attributes:
        Age={age}, BMI={bmi}, Children={children}, Smoker={smoker}, Region={region}, Sex={sex}.
        The predicted cost is ${prediction:,.2f}.
        Explain why this cost might be high or low in simple terms.
        """
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=prompt
        )
        st.info(response.text)

st.caption("⚙️ Models: LGBM / Neural Network / XGBoost / Random Forest")
