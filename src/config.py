from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "customer_churn.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"

TARGET_COLUMN = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2