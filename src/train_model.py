"""
Model Training Script
Project: Legal Document Importance Prediction
Objective: Train CatBoost regression model and generate predictions
"""

import json
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor


# --------------------------------------------------
# Paths
# --------------------------------------------------
PROCESSED_DIR = Path("../data/processed")
MODEL_DIR = Path("../models")
OUTPUT_DIR = Path("../outputs")

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_PATH = PROCESSED_DIR / "train_features.csv"
TEST_PATH = PROCESSED_DIR / "test_features.csv"


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def str_to_list(x):
    if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return x if isinstance(x, list) else []


def list_to_text(x):
    if isinstance(x, list):
        return "; ".join(x)
    return "" if pd.isna(x) else str(x)


# --------------------------------------------------
# Main Training Pipeline
# --------------------------------------------------
def train_model():
    print("Loading feature datasets...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # --------------------------------------------------
    # Convert List-like Columns
    # --------------------------------------------------
    list_cols = ["Lead_Types", "Power_Mentions", "Agencies", "Tags"]

    for col in list_cols:
        train_df[col] = train_df[col].apply(str_to_list)
        test_df[col] = test_df[col].apply(str_to_list)

    for col in list_cols:
        train_df[col] = train_df[col].apply(list_to_text)
        test_df[col] = test_df[col].apply(list_to_text)

    # --------------------------------------------------
    # Target Preparation
    # --------------------------------------------------
    train_df["target"] = train_df["Importance_Score"] / 100

    X = train_df.drop(columns=["Importance_Score", "target", "id"])
    y = train_df["target"]

    # --------------------------------------------------
    # Train / Validation Split
    # --------------------------------------------------
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    text_cols = ["Headline", "Reasoning", "Key_Insights"]

    for col in text_cols:
        X_train[col] = X_train[col].fillna("").astype(str)
        X_valid[col] = X_valid[col].fillna("").astype(str)
        test_df[col] = test_df[col].fillna("").astype(str)

    # --------------------------------------------------
    # Train Validation Model
    # --------------------------------------------------
    print("Training validation model...")

    model = CatBoostRegressor(
        loss_function="RMSE",
        depth=8,
        learning_rate=0.03,
        n_estimators=3000,
        random_seed=42,
        verbose=200,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        text_features=text_cols,
        use_best_model=True,
    )

    model.save_model(MODEL_DIR / "catboost_validation.cbm")

    # Save feature list
    feature_cols = X_train.columns.tolist()
    with open(MODEL_DIR / "features.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # --------------------------------------------------
    # Train Final Model on Full Data
    # --------------------------------------------------
    print("Training final model on full dataset...")

    for col in text_cols:
        train_df[col] = train_df[col].fillna("").astype(str)
        test_df[col] = test_df[col].fillna("").astype(str)

    X_full = train_df.drop(columns=["Importance_Score", "target", "id"])
    y_full = train_df["target"]

    model_final = CatBoostRegressor(
        loss_function="RMSE",
        depth=8,
        learning_rate=0.03,
        n_estimators=3000,
        random_seed=42,
        verbose=200,
    )

    model_final.fit(
        X_full,
        y_full,
        text_features=text_cols,
    )

    model_final.save_model(MODEL_DIR / "catboost_full.cbm")

    # --------------------------------------------------
    # Predict on Test Set
    # --------------------------------------------------
    print("Generating predictions...")

    test_features = pd.read_csv(TEST_PATH)

    for col in text_cols:
        test_features[col] = test_features[col].fillna("").astype(str)

    X_test = test_features.drop(columns=["id"])

    for col in X_test.columns:
        if col not in text_cols:
            X_test[col] = (
                X_test[col]
                .replace("[]", 0)
                .replace("", 0)
            )
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce").fillna(0)

    test_preds_scaled = model_final.predict(X_test)
    test_preds = np.clip(test_preds_scaled * 100, 0, 100)

    submission = pd.DataFrame({
        "id": test_features["id"],
        "Importance_Score": test_preds
    })

    submission_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)

    print(f"Submission file saved at: {submission_path}")
    print("Training pipeline completed successfully.")

if __name__ == "__main__":
    train_model()