# Customer Churn Prediction

This project predicts whether a telecom customer is likely to churn using machine learning.

## Problem Statement

Customer churn means a customer stops using a company's service. The goal of this project is to predict churn so that a company can identify risky customers early and take retention actions.

## Dataset

Dataset used: Telco Customer Churn dataset.

- Rows: 7043
- Columns: 21
- Target column: `Churn`

The dataset file is not included in this repository. Place it here before running the project:

```text
data/raw/customer_churn.csv
```

## Technologies Used

- Python
- pandas
- NumPy
- matplotlib
- scikit-learn
- joblib
- Git and GitHub

## Project Workflow

1. Loaded the customer churn dataset.
2. Checked shape, columns, missing values, and data types.
3. Converted `TotalCharges` from text to numeric.
4. Performed exploratory data analysis.
5. Split data into training and testing sets.
6. Built preprocessing pipelines for numeric and categorical columns.
7. Trained a Logistic Regression model.
8. Evaluated the model using accuracy, ROC-AUC, confusion matrix, precision, recall, and F1-score.
9. Saved the trained model using joblib.
10. Created a prediction script for new customer data.

## EDA Insights

- Churn rate is around 26.54%.
- Month-to-month contract customers have the highest churn.
- Electronic check customers have the highest churn by payment method.
- Fiber optic customers show higher churn than DSL customers.
- Customers with shorter tenure are more likely to churn.
- Customers with higher monthly charges are more likely to churn.
- Customers without tech support are more likely to churn.

## Model Results

Logistic Regression results:

- Accuracy: 73.8%
- ROC-AUC: 84.1%
- Churn recall: 78%

The model is useful because it identifies many customers who are likely to churn.

## How to Run

Clone the repository:

```powershell
git clone https://github.com/sindhupriya-k/customer-churn.git
cd customer-churn
```

Create virtual environment:

```powershell
py -3.11 -m venv .venv
```

Activate environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Add the dataset file at:

```text
data/raw/customer_churn.csv
```

Run EDA:

```powershell
python src/eda.py
```

Train model:

```powershell
python src/train.py
```

Run prediction:

```powershell
python src/predict.py
```

## Future Improvements

- Try Random Forest
- Try Gradient Boosting
- Tune hyperparameters
- Tune prediction threshold
- Build a Streamlit app
- Deploy the model

