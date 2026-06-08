import pandas as pd

from config import RAW_DATA_PATH, TARGET_COLUMN


df = pd.read_csv(RAW_DATA_PATH)

print("Dataset loaded successfully")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget column value counts:")
print(df[TARGET_COLUMN].value_counts())

print("\nColumn data types:")
print(df.dtypes)

blank_total_charges = df[df["TotalCharges"].str.strip() == ""]

print("\nBlank TotalCharges rows:")
print(blank_total_charges.shape)

print("\nRows with blank TotalCharges:")
print(blank_total_charges[["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]])

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("\nTotalCharges after conversion:")
print(df["TotalCharges"].dtype)

print("\nMissing values after TotalCharges conversion:")
print(df.isnull().sum())