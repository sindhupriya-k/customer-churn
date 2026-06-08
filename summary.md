# Customer Churn Prediction - Project Summary

## Problem Statement

The goal of this project is to predict whether a customer is likely to churn based on customer account and service details.

Churn means the customer leaves the company or stops using the service.

## Dataset

The project uses the Telco Customer Churn dataset.

- Total rows: 7043
- Total columns: 21
- Target column: Churn

## Target Distribution

- No churn: 73.46%
- Churn: 26.54%

This shows that the dataset is imbalanced.

## Key EDA Insights

1. Month-to-month contract customers have the highest churn rate.
2. Electronic check payment customers have the highest churn rate.
3. Fiber optic customers show higher churn than DSL customers.
4. Customers with shorter tenure are more likely to churn.
5. Customers with higher monthly charges are more likely to churn.
6. Customers without tech support are more likely to churn.

## Data Cleaning

The `TotalCharges` column was originally read as text because it had blank values.

I converted it into numeric using:

```python
pd.to_numeric(df["TotalCharges"], errors="coerce")





Blank values became missing values and were handled in the preprocessing pipeline.

Model Used
The first model used was Logistic Regression.

Reason:

suitable for binary classification
easy to explain
good baseline model
interview-friendly
Preprocessing
Numerical features:

missing values filled using median
values scaled using StandardScaler
Categorical features:

missing values filled using most frequent value
categories converted using OneHotEncoder
Evaluation Results
Accuracy: 73.8%
ROC-AUC: 84.1%
Recall for churn class: 78%
Business Interpretation
The model is useful because it identifies many customers who are likely to churn. The company can use this to target risky customers with retention offers, discounts, support calls, or better service plans.

Future Improvements
Try Random Forest
Try Gradient Boosting
Tune hyperparameters
Tune prediction threshold
Build a Streamlit app
Deploy the model
