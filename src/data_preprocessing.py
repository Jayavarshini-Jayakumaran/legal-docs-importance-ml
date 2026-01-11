"""
Data Preprocessing Script
Project: Legal Document Importance Prediction
Objective: Prepare clean, model-ready features from raw legal documents
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------
RAW_DATA_DIR = Path("../data/raw")
PROCESSED_DATA_DIR = Path("../data/processed")

TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"

OUTPUT_TRAIN_PATH = PROCESSED_DATA_DIR / "train_clean.csv"
OUTPUT_TEST_PATH = PROCESSED_DATA_DIR / "test_clean.csv"


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def fix_encoding(text: str) -> str:
    """Fix common text encoding issues."""
    if not isinstance(text, str):
        return text
    return re.sub(r"â€‘|â€“|â€”", "-", text)


def split_to_list(text: str) -> list:
    """Convert semicolon-separated strings into lists."""
    if not text or not isinstance(text, str):
        return []
    return [t.strip() for t in text.split(";")]


# --------------------------------------------------
# Main Preprocessing Function
# --------------------------------------------------
def preprocess_data():
    print("Loading raw datasets...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH) 

    # Standardize column names
    train_df.columns = train_df.columns.str.strip().str.replace(" ", "_")
    test_df.columns = test_df.columns.str.strip().str.replace(" ", "_")

    # Create working copies
    train_clean = train_df.copy()
    test_clean = test_df.copy()

    # -----------------------------
    # Handle Missing Values
    # -----------------------------
    text_cols = ["Headline", "Reasoning", "Key_Insights", "Tags"]
    list_cols = ["Lead_Types", "Power_Mentions", "Agencies"]

    train_clean[text_cols] = train_clean[text_cols].fillna("")
    test_clean[text_cols] = test_clean[text_cols].fillna("")

    train_clean[list_cols] = train_clean[list_cols].fillna("")
    test_clean[list_cols] = test_clean[list_cols].fillna("")

    # -----------------------------
    # Fix Encoding Issues
    # -----------------------------
    for col in ["Headline", "Key_Insights", "Reasoning"]:
        train_clean[col] = train_clean[col].apply(fix_encoding)
        test_clean[col] = test_clean[col].apply(fix_encoding)

    # -----------------------------
    # Normalize Text
    # -----------------------------
    for col in ["Headline", "Key_Insights", "Reasoning"]:
        train_clean[col] = train_clean[col].str.lower().str.strip()
        test_clean[col] = test_clean[col].str.lower().str.strip()

    # -----------------------------
    # Parse List-like Columns
    # -----------------------------
    for col in list_cols:
        train_clean[col] = train_clean[col].apply(split_to_list)
        test_clean[col] = test_clean[col].apply(split_to_list)

    # -----------------------------
    # Feature Engineering
    # -----------------------------
    # Text length features
    train_clean["headline_len"] = train_clean["Headline"].str.len()
    train_clean["insight_len"] = train_clean["Key_Insights"].str.len()

    test_clean["headline_len"] = test_clean["Headline"].str.len()
    test_clean["insight_len"] = test_clean["Key_Insights"].str.len()

    # Metadata count features
    train_clean["num_lead_types"] = train_clean["Lead_Types"].apply(len)
    train_clean["num_power_mentions"] = train_clean["Power_Mentions"].apply(len)
    train_clean["num_agencies"] = train_clean["Agencies"].apply(len)
    train_clean["num_tags"] = train_clean["Tags"].apply(
        lambda x: len(x.split(";")) if x else 0
    )

    test_clean["num_lead_types"] = test_clean["Lead_Types"].apply(len)
    test_clean["num_power_mentions"] = test_clean["Power_Mentions"].apply(len)
    test_clean["num_agencies"] = test_clean["Agencies"].apply(len)
    test_clean["num_tags"] = test_clean["Tags"].apply(
        lambda x: len(x.split(";")) if x else 0
    )

    # -----------------------------
    # Drop Non-Predictive Columns
    # -----------------------------
    for df in [train_clean, test_clean]:
        if "Source_File" in df.columns:
            df.drop(columns=["Source_File"], inplace=True)

    # -----------------------------
    # Save Processed Data
    # -----------------------------
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_clean.to_csv(OUTPUT_TRAIN_PATH, index=False)
    test_clean.to_csv(OUTPUT_TEST_PATH, index=False)

    print("Preprocessing complete.")
    print(f"Saved: {OUTPUT_TRAIN_PATH}")
    print(f"Saved: {OUTPUT_TEST_PATH}")

if __name__ == "__main__":
    preprocess_data()