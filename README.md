# Interpretable Tuition Prediction

This project builds a machine learning model to predict university tuition costs based on features like country, living costs, and duration. The pipeline uses XGBoost along with SHAP values and Partial Dependence Plots (PDP) to ensure interpretability of the model's decisions. It is designed with best practices using Scikit-learn pipelines and includes both global and local explanation tools.

## 🧠 Key Features
- Clean preprocessing pipeline with scaling and frequency encoding
- XGBoost regression model with hyperparameter control
- SHAP values for feature contribution (global + local explainability)
- Partial Dependence Plots (PDP) for trend interpretation
- Model exported using `joblib` for reuse/inference

## 📂 Project Structure
- `scripts/` — Python training and explanation script (`tuition_estimater.py`)
- `notebooks/` — Optional visual analysis and summary report notebook
- `outputs/` — Folder to store trained model or plots
- `requirements.txt` — Python dependencies

## 🚀 Setup Instructions

1. Install dependencies:

```bash
pip install -r requirements.txt

2. python scripts/tuition_estimater.py


Insights Summary

SHAP identified cat__Country as the top driver of predicted tuition.
Living_Cost_Index was less impactful when controlling for multicollinearity with Rent and Insurance.
PDP provided general trend direction but was limited by feature correlation.
