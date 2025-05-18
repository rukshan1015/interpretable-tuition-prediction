import gradio as gr
import numpy as np
import joblib
import sys
import os

# Add custom_transformers to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../custom_transformers")))

# Import custom FrequencyEncoder
from customencoder import FrequencyEncoder

# Load model from outputs folder
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../outputs/model_pipeline.joblib"))
model = joblib.load(model_path)

# Define prediction function
def predict_tuition(country, city, level, duration, living_cost, rent, visa_fee, insurance, exchange_rate):
    import pandas as pd
    X = pd.DataFrame([[
        country, city, level, duration, living_cost, rent, visa_fee, insurance, exchange_rate
    ]], columns=[
        "Country", "City", "Level", "Duration_Years", "Living_Cost_Index",
        "Rent_USD", "Visa_Fee_USD", "Insurance_USD", "Exchange_Rate"
    ])
    prediction = model.predict(X)[0]
    return f"Estimated Tuition: ${prediction:,.2f}"


# Define Gradio inputs
inputs = [
    gr.Dropdown(choices=["USA", "UK", "Canada", "Australia", "Other"], label="Country"),
    gr.Dropdown(choices=["New York", "London", "Toronto", "Sydney", "Other"], label="City"),
    gr.Dropdown(choices=["Bachelor", "Master", "PhD"], label="Level"),
    gr.Slider(minimum=0.5, maximum=6.0, step=0.5, label="Duration (Years)"),
    gr.Slider(minimum=30.0, maximum=150.0, step=1.0, label="Living Cost Index"),
    gr.Number(label="Rent (USD)"),
    gr.Number(label="Visa Fee (USD)"),
    gr.Number(label="Insurance (USD)"),
    gr.Number(label="Exchange Rate")
]

# Launch app
gr.Interface(
    fn=predict_tuition,
    inputs=inputs,
    outputs=gr.Textbox(label="Predicted Tuition"),
    title="University Tuition Cost Estimator",
    description="Enter student information and costs to estimate total tuition."
).launch()
