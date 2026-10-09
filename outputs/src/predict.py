"""Score a customer with the saved model. Run: python src/predict.py"""
from pathlib import Path
import joblib
import pandas as pd

model = joblib.load(Path(__file__).resolve().parents[1] / "outputs" / "churn_model.joblib")

customer = {
    "gender": "Female", "seniorcitizen": 0, "partner": "No", "dependents": "No",
    "tenure": 3, "phoneservice": "Yes", "multiplelines": "No",
    "internetservice": "Fiber optic", "onlinesecurity": "No", "onlinebackup": "No",
    "deviceprotection": "No", "techsupport": "No", "streamingtv": "Yes",
    "streamingmovies": "Yes", "contract": "Month-to-month", "paperlessbilling": "Yes",
    "paymentmethod": "Electronic check", "monthlycharges": 95.0, "totalcharges": 285.0,
}
customer["expected_total"] = customer["monthlycharges"] * customer["tenure"]

p = model.predict_proba(pd.DataFrame([customer]))[0, 1]
print(f"Churn probability: {p:.1%}  ->  {'HIGH RISK' if p >= 0.5 else 'low risk'}")
