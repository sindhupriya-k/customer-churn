import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import MODEL_PATH, PROJECT_ROOT, RANDOM_STATE, RAW_DATA_PATH, TARGET_COLUMN, TEST_SIZE


REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "model_metrics.json"
EVALUATION_REPORT_PATH = REPORTS_DIR / "model_evaluation.md"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"


def load_data():
    df = pd.read_csv(RAW_DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def build_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def save_feature_importance(model_pipeline):
    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": model.coef_[0],
        }
    )
    feature_importance["absolute_coefficient"] = feature_importance["coefficient"].abs()
    feature_importance = feature_importance.sort_values(
        "absolute_coefficient", ascending=False
    )
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)


def save_evaluation_report(metrics, report_text):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    with open(EVALUATION_REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("# Model Evaluation\n\n")
        file.write("## Model Used\n\n")
        file.write("Logistic Regression with balanced class weights.\n\n")
        file.write("## Metrics\n\n")
        file.write(f"- Accuracy: {metrics['accuracy']:.4f}\n")
        file.write(f"- Precision: {metrics['precision']:.4f}\n")
        file.write(f"- Recall: {metrics['recall']:.4f}\n")
        file.write(f"- F1-score: {metrics['f1_score']:.4f}\n")
        file.write(f"- ROC-AUC: {metrics['roc_auc']:.4f}\n\n")
        file.write("## Confusion Matrix\n\n")
        file.write("```text\n")
        file.write(str(metrics["confusion_matrix"]))
        file.write("\n```\n\n")
        file.write("## Classification Report\n\n")
        file.write("```text\n")
        file.write(report_text)
        file.write("\n```\n")


def main():
    df = load_data()

    X = df.drop(columns=[TARGET_COLUMN, "customerID"])
    y = df[TARGET_COLUMN].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "str"]).columns.tolist()

    model_pipeline = build_pipeline(numeric_features, categorical_features)
    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]

    report_text = classification_report(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(report_dict["1"]["f1-score"]),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_pipeline, MODEL_PATH)
    save_feature_importance(model_pipeline)
    save_evaluation_report(metrics, report_text)

    print("Model training completed successfully")
    print("Model saved to:", MODEL_PATH)
    print("Evaluation report saved to:", EVALUATION_REPORT_PATH)
    print("Feature importance saved to:", FEATURE_IMPORTANCE_PATH)
    print("ROC-AUC:", round(metrics["roc_auc"], 4))
    print("Churn recall:", round(metrics["recall"], 4))


if __name__ == "__main__":
    main()

