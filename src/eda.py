import matplotlib.pyplot as plt
import pandas as pd

from config import RAW_DATA_PATH, TARGET_COLUMN


df = pd.read_csv(RAW_DATA_PATH)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("EDA started")
print("Dataset shape:", df.shape)

print("\nTarget distribution:")
print(df[TARGET_COLUMN].value_counts())

print("\nTarget percentage:")
print(df[TARGET_COLUMN].value_counts(normalize=True) * 100)

contract_churn = pd.crosstab(
    df["Contract"],
    df[TARGET_COLUMN],
    normalize="index",
) * 100

print("\nChurn percentage by Contract:")
print(contract_churn)

payment_churn = pd.crosstab(
    df["PaymentMethod"],
    df[TARGET_COLUMN],
    normalize="index",
) * 100

print("\nChurn percentage by Payment Method:")
print(payment_churn)

internet_churn = pd.crosstab(
    df["InternetService"],
    df[TARGET_COLUMN],
    normalize="index",
) * 100

print("\nChurn percentage by Internet Service:")
print(internet_churn)

print("\nAverage tenure by Churn:")
print(df.groupby(TARGET_COLUMN)["tenure"].mean())

print("\nMedian tenure by Churn:")
print(df.groupby(TARGET_COLUMN)["tenure"].median())

print("\nAverage MonthlyCharges by Churn:")
print(df.groupby(TARGET_COLUMN)["MonthlyCharges"].mean())

print("\nMedian MonthlyCharges by Churn:")
print(df.groupby(TARGET_COLUMN)["MonthlyCharges"].median())

tech_support_churn = pd.crosstab(
    df["TechSupport"],
    df[TARGET_COLUMN],
    normalize="index",
) * 100

print("\nChurn percentage by Tech Support:")
print(tech_support_churn)

print("\nEDA Summary:")
print("1. Dataset has 7043 customers and 21 columns.")
print("2. Churn rate is about 26.54%.")
print("3. Month-to-month contract customers have the highest churn.")
print("4. Electronic check customers have the highest churn by payment method.")
print("5. Fiber optic customers have higher churn than DSL customers.")
print("6. Customers with shorter tenure are more likely to churn.")
print("7. Customers with higher monthly charges are more likely to churn.")
print("8. Customers without tech support are more likely to churn.")


plt.figure(figsize=(6, 4))
df[TARGET_COLUMN].value_counts().plot(kind="bar")
plt.title("Customer Churn Count")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("reports/figures/churn_count.png")
plt.close()

print("\nChart saved: reports/figures/churn_count.png")


contract_churn["Yes"].sort_values(ascending=False).plot(kind="bar", figsize=(7, 4))
plt.title("Churn Percentage by Contract")
plt.xlabel("Contract Type")
plt.ylabel("Churn Percentage")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("reports/figures/churn_by_contract.png")
plt.close()

print("Chart saved: reports/figures/churn_by_contract.png")

payment_churn["Yes"].sort_values(ascending=False).plot(kind="bar", figsize=(8, 4))
plt.title("Churn Percentage by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Churn Percentage")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig("reports/figures/churn_by_payment_method.png")
plt.close()

print("Chart saved: reports/figures/churn_by_payment_method.png")


plt.figure(figsize=(7, 4))
df[df[TARGET_COLUMN] == "No"]["tenure"].plot(kind="hist", alpha=0.6, bins=30, label="No Churn")
df[df[TARGET_COLUMN] == "Yes"]["tenure"].plot(kind="hist", alpha=0.6, bins=30, label="Churn")
plt.title("Tenure Distribution by Churn")
plt.xlabel("Tenure in Months")
plt.ylabel("Number of Customers")
plt.legend()
plt.tight_layout()
plt.savefig("reports/figures/tenure_distribution_by_churn.png")
plt.close()

print("Chart saved: reports/figures/tenure_distribution_by_churn.png")