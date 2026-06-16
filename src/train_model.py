"""
Model Training Script
Project: Legal Document Importance Prediction
Objective: Train CatBoost regression model, evaluate rigorously, generate predictions
"""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor


# Paths
_SRC_DIR      = Path(__file__).resolve().parent
_ROOT         = _SRC_DIR.parent

PROCESSED_DIR = _ROOT / "data" / "processed"
MODEL_DIR     = _ROOT / "models"
OUTPUT_DIR    = _ROOT / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_PATH = PROCESSED_DIR / "train_features.csv"
TEST_PATH  = PROCESSED_DIR / "test_features.csv"


def read_csv_safe(path: Path) -> pd.DataFrame:
    """
    Read CSV using Python csv module, streaming directly into
    per-column lists to avoid allocating one large list-of-dicts.
    """
    import csv
    print(f"    Reading {path.name} with Python csv reader...")
    columns = None
    col_data = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                columns = row
                col_data = {c: [] for c in columns}
            else:
                for c, val in zip(columns, row):
                    col_data[c].append(val)
    df = pd.DataFrame(col_data)
    # Restore numeric columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass
    print(f"    Loaded {len(df):,} rows x {len(df.columns)} cols")
    return df

def parse_list_col(x):
    if isinstance(x, list):
        return x
    if not isinstance(x, str) or not x.strip():
        return []
    s = x.strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            return ast.literal_eval(s)
        except Exception:
            pass
    return [t.strip() for t in s.split(";") if t.strip()]


def lists_to_text(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for col in cols:
        df[col] = df[col].apply(parse_list_col).apply(
            lambda v: "; ".join(v) if isinstance(v, list) else (str(v) if v else "")
        )
    return df


def assign_priority_tier(score: float) -> str:
    if score >= 70:   return "Critical"
    elif score >= 45: return "High"
    elif score >= 20: return "Medium"
    else:             return "Low"


# Evaluation
def evaluate_and_report(model, X_valid, y_valid, feature_cols, report_path: Path):
    """Compute metrics, print them, save a text report."""
    y_pred_scaled = model.predict(X_valid)
    y_pred  = np.clip(y_pred_scaled * 100, 0, 100)
    y_true  = y_valid.values * 100

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    # Score-distribution buckets
    buckets = {"Low (0–20)": 0, "Medium (20–45)": 0,
               "High (45–70)": 0, "Critical (70–100)": 0}
    for s in y_pred:
        if s < 20:        buckets["Low (0–20)"]        += 1
        elif s < 45:      buckets["Medium (20–45)"]    += 1
        elif s < 70:      buckets["High (45–70)"]      += 1
        else:             buckets["Critical (70–100)"] += 1

    # Top-10 feature importances
    fi = pd.Series(model.get_feature_importance(),
                   index=feature_cols).sort_values(ascending=False).head(10)

    lines = [
        "=" * 55,
        "  MODEL EVALUATION REPORT",
        "  Legal Document Importance Prediction",
        "=" * 55,
        f"  Validation set size : {len(y_true):,} documents",
        f"  RMSE                : {rmse:.4f}",
        f"  MAE                 : {mae:.4f}",
        f"  R²                  : {r2:.4f}",
        "",
        "  Predicted Score Distribution:",
    ]
    for tier, count in buckets.items():
        pct = 100 * count / len(y_pred)
        lines.append(f"    {tier:<22} {count:>5} docs  ({pct:.1f}%)")
    lines += [
        "",
        "  Top-10 Feature Importances:",
    ]
    for feat, imp in fi.items():
        lines.append(f"    {feat:<35} {imp:.2f}")
    lines.append("=" * 55)

    report = "\n".join(lines)
    print("\n" + report)

    report_path.write_text(report, encoding='utf-8')
    print(f"\n  Evaluation report saved → {report_path}")

    return rmse, mae, r2


# Main Training Pipeline
def train_model():
    print("\n[3/3] MODEL TRAINING & EVALUATION")
    print("=" * 50)

    print("  Loading feature datasets (chunked)...")
    train_df = read_csv_safe(TRAIN_PATH)
    test_df  = read_csv_safe(TEST_PATH)

    # Parse & flatten list columns
    list_cols = ["Lead_Types", "Power_Mentions", "Agencies", "Tags"]
    train_df = lists_to_text(train_df, list_cols)
    test_df  = lists_to_text(test_df,  list_cols)

    # Target
    drop_cols = ["Headline", "Reasoning", "Key_Insights",
                 "Lead_Types", "Power_Mentions", "Agencies", "Tags"]

    train_df["target"] = train_df["Importance_Score"] / 100

    X = train_df.drop(columns=["Importance_Score", "target", "id"] + drop_cols)
    y = train_df["target"]

    # Train / Validation Split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    # All text columns dropped before training — no cleanup needed

    # Train Validation Model
    print("\n  Training validation model (80/20 split)...")
    model_val = CatBoostRegressor(
        loss_function="RMSE",
        depth=5,            # was 8 — halves memory per tree
        learning_rate=0.05, # faster convergence at lower n_estimators
        n_estimators=500,   # was 3000 — biggest RAM saving
        random_seed=42,
        verbose=100,
        thread_count=1,     # single thread = lower peak RAM on Windows
        max_ctr_complexity=1,  # reduces text feature memory usage
    )
    model_val.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
    )
    model_val.save_model(MODEL_DIR / "catboost_validation.cbm")

    feature_cols = X_train.columns.tolist()
    with open(MODEL_DIR / "features.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Evaluation
    evaluate_and_report(
        model_val, X_valid, y_valid,
        feature_cols,
        OUTPUT_DIR / "evaluation_report.txt"
    )

    # Train Final Model on Full Data
    print("\n  Training final model on full dataset...")


    X_full = train_df.drop(columns=["Importance_Score", "target", "id"] + drop_cols)
    y_full = train_df["target"]

    model_final = CatBoostRegressor(
        loss_function="RMSE",
        depth=5,
        learning_rate=0.05,
        n_estimators=500,
        random_seed=42,
        verbose=100,
        thread_count=1,
    )
    model_final.fit(X_full, y_full)
    model_final.save_model(MODEL_DIR / "catboost_full.cbm")

    # Predict on Test Set
    print("\n  Generating predictions...")
    test_features = read_csv_safe(TEST_PATH)
    test_features = lists_to_text(test_features, list_cols)

    X_test = test_features.drop(columns=["id"] + drop_cols)
    for col in X_test.columns:
        X_test[col] = pd.to_numeric(
            X_test[col].replace({"[]": 0, "": 0}), errors="coerce"
        ).fillna(0)

    test_preds = np.clip(model_final.predict(X_test) * 100, 0, 100)

    # Submission - (raw scores)
    submission = pd.DataFrame({
        "id": test_features["id"],
        "Importance_Score": test_preds,
    })
    submission_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"  submission.csv saved → {submission_path}")

    # Prioritized Output
    # Ranked, tiered output for actionable document review
    prioritized = submission.copy()
    prioritized["Priority_Rank"] = (
        prioritized["Importance_Score"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    prioritized["Priority_Tier"] = prioritized["Importance_Score"].apply(
        assign_priority_tier
    )
    prioritized = prioritized.sort_values("Priority_Rank").reset_index(drop=True)

    prio_path = OUTPUT_DIR / "prioritized_documents.csv"
    prioritized.to_csv(prio_path, index=False)
    print(f"  prioritized_documents.csv saved → {prio_path}")

    # Print tier summary
    tier_summary = prioritized["Priority_Tier"].value_counts()
    print("\n  Document Priority Breakdown:")
    for tier in ["Critical", "High", "Medium", "Low"]:
        count = tier_summary.get(tier, 0)
        print(f"    {tier:<10} {count:>5} documents")

    print("\n  Training pipeline completed successfully.")


if __name__ == "__main__":
    train_model()
