import joblib
import pandas as pd

from config import MODEL_PATH


sample_customer = pd.DataFrame(
    [
        {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "No",
            "MultipleLines": "No phone service",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85,
            "TotalCharges": 29.85,
        }
    ]
)


def main():
    model_pipeline = joblib.load(MODEL_PATH)

    prediction = model_pipeline.predict(sample_customer)[0]
    churn_probability = model_pipeline.predict_proba(sample_customer)[0][1]

    result = "Customer is likely to churn" if prediction == 1 else "Customer is not likely to churn"

    print("Prediction:", prediction)
    print("Churn probability:", f"{churn_probability * 100:.2f}%")
    print("Result:", result)


if __name__ == "__main__":
    main()
