# Customer Churn Prediction - ML Project Summary

## Objective

Predict whether a telecom customer is likely to churn using machine learning.

## Dataset

- Dataset: Telco Customer Churn
- Rows: 7,043
- Columns: 21
- Target: Churn
- Churn rate: 26.54%

## ML Workflow

1. Loaded and inspected the dataset.
2. Converted `TotalCharges` to numeric.
3. Split data into training and testing sets.
4. Built preprocessing pipelines for numeric and categorical features.
5. Trained Logistic Regression with balanced class weights.
6. Evaluated the model with accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix.
7. Saved the model with joblib.
8. Created a sample prediction script.

## Key Results

- Accuracy: 73.8%
- ROC-AUC: 84.1%
- Churn recall: 78%

## Business Value

The model helps identify customers who are likely to churn, allowing the business to prioritize retention campaigns and reduce customer loss.

## Resume Positioning

This project is suitable for a fresher Data Analyst resume because it demonstrates Python, EDA, data preprocessing, model evaluation, business interpretation, and reproducible project structure.
