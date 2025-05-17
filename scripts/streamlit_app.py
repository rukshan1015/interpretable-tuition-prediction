import streamlit as st
import joblib
import pandas as pd
import os

# Import custom FrequencyEncoder class so joblib can unpickle it

from custom_transformers.customencoder import FrequencyEncoder # Get the script from the folder custom_transformers


# Load the trained model pipeline
model_path = "outputs/model_pipeline.joblib"
if not os.path.exists(model_path):
    st.error("Model file not found. Please train the model first.")
    st.stop()

model = joblib.load(model_path)

# === UI Title ===
st.title("🎓 Tuition Cost Estimator (v2)")
st.write("Enter student and cost-related information to predict tuition.")

# === Categorical Inputs ===
known_countries = ["USA", "UK", "Canada", "Australia"]  # Replace with actual list
known_cities = ["New York", "London", "Toronto", "Sydney"]  # Replace with actual list
levels = ["Bachelor", "Master", "PhD"]  # Replace with what your model uses

country = st.selectbox("Country", known_countries)
city = st.selectbox("City", known_cities)
level = st.selectbox("Program Level", levels)

# === Numerical Inputs ===
duration = st.number_input("Program Duration (Years)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
living_cost = st.number_input("Living Cost Index", min_value=0.0, max_value=200.0, value=85.0)
rent = st.number_input("Rent (USD)", min_value=0.0, value=500.0)
visa_fee = st.number_input("Visa Fee (USD)", min_value=0.0, value=160.0)
insurance = st.number_input("Insurance (USD)", min_value=0.0, value=100.0)
exchange_rate = st.number_input("Exchange Rate", min_value=0.01, value=1.0)

# === Predict Button ===
if st.button("Predict Tuition Cost"):
    input_df = pd.DataFrame([{
        "Country": country,
        "City": city,
        "Level": level,
        "Duration_Years": duration,
        "Living_Cost_Index": living_cost,
        "Rent_USD": rent,
        "Visa_Fee_USD": visa_fee,
        "Insurance_USD": insurance,
        "Exchange_Rate": exchange_rate
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"📘 Estimated Tuition Cost: **${{prediction:,.2f}}**")
