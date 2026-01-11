"""
Feature Engineering Script
Project: Legal Document Importance Prediction
Objective: Create advanced, model-ready features from cleaned data
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Paths
# --------------------------------------------------
PROCESSED_DATA_DIR = Path("../data/processed")

TRAIN_INPUT = PROCESSED_DATA_DIR / "train_clean.csv"
TEST_INPUT = PROCESSED_DATA_DIR / "test_clean.csv"

TRAIN_OUTPUT = PROCESSED_DATA_DIR / "train_features.csv"
TEST_OUTPUT = PROCESSED_DATA_DIR / "test_features.csv"


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def compute_intent_score(row):
    score = 0
    score += row["num_lead_types"] * 2
    score += row["num_agencies"] * 3
    score += row["num_power_mentions"]
    score += row["insight_len"] / 150
    return score


def map_intent_category(score):
    if score <= 1:
        return "contextual"
    elif score <= 3:
        return "informational"
    elif score <= 6:
        return "disclosure"
    elif score <= 9:
        return "analytical"
    else:
        return "allegational"


# --------------------------------------------------
# Main Feature Engineering Function
# --------------------------------------------------
def feature_engineering():
    print("Loading cleaned datasets...")
    train_df = pd.read_csv(TRAIN_INPUT)
    test_df = pd.read_csv(TEST_INPUT)

    # --------------------------------------------------
    # Base Text Length Metrics
    # --------------------------------------------------
    for df in [train_df, test_df]:
        df["headline_len"] = df["Headline"].str.len()
        df["insight_len"] = df["Key_Insights"].str.len()
        df["reason_len"] = df["Reasoning"].str.len()

    # --------------------------------------------------
    # Density & Ratio Features
    # --------------------------------------------------
    train_df["power_density"] = train_df["Power_Mentions"].apply(len) / (train_df["insight_len"] + 1)
    test_df["power_density"] = test_df["Power_Mentions"].apply(len) / (test_df["insight_len"] + 1)

    train_df["agency_density"] = train_df["Agencies"].apply(len) / (train_df["insight_len"] + 1)
    test_df["agency_density"] = test_df["Agencies"].apply(len) / (test_df["insight_len"] + 1)

    train_df["institutional_index"] = (
        train_df["Agencies"].apply(len) + train_df["Lead_Types"].apply(len)
    )
    test_df["institutional_index"] = (
        test_df["Agencies"].apply(len) + test_df["Lead_Types"].apply(len)
    )

    train_df["lead_complexity_ratio"] = (
        train_df["Lead_Types"].apply(len) / (train_df["Power_Mentions"].apply(len) + 1)
    )
    test_df["lead_complexity_ratio"] = (
        test_df["Lead_Types"].apply(len) / (test_df["Power_Mentions"].apply(len) + 1)
    )

    train_df["insight_concentration"] = train_df["insight_len"] / (train_df["reason_len"] + 1)
    test_df["insight_concentration"] = test_df["insight_len"] / (test_df["reason_len"] + 1)

    train_df["entity_insight_ratio"] = train_df["Power_Mentions"].apply(len) / (train_df["insight_len"] + 1)
    test_df["entity_insight_ratio"] = test_df["Power_Mentions"].apply(len) / (test_df["insight_len"] + 1)

    train_df["tag_lead_alignment"] = train_df["num_tags"] / (train_df["Lead_Types"].apply(len) + 1)
    test_df["tag_lead_alignment"] = test_df["num_tags"] / (test_df["Lead_Types"].apply(len) + 1)

    train_df["actionability_proxy"] = (
        train_df["Power_Mentions"].apply(len)
        + train_df["Agencies"].apply(len)
        + train_df["Lead_Types"].apply(len)
    )
    test_df["actionability_proxy"] = (
        test_df["Power_Mentions"].apply(len)
        + test_df["Agencies"].apply(len)
        + test_df["Lead_Types"].apply(len)
    )

    train_df["information_density"] = (
        train_df["insight_len"] + train_df["reason_len"]
    ) / (train_df["headline_len"] + 1)

    test_df["information_density"] = (
        test_df["insight_len"] + test_df["reason_len"]
    ) / (test_df["headline_len"] + 1)

    # --------------------------------------------------
    # Importance Prior
    # --------------------------------------------------
    train_df["importance_prior"] = train_df.groupby(
        "num_power_mentions"
    )["Importance_Score"].transform("mean")

    global_prior = train_df["Importance_Score"].mean()
    test_df["importance_prior"] = global_prior

    # --------------------------------------------------
    # Investigative Intent Features
    # --------------------------------------------------
    intent_encoding = {
        "contextual": 0,
        "informational": 1,
        "disclosure": 2,
        "analytical": 3,
        "allegational": 4,
    }

    train_df["intent_score"] = train_df.apply(compute_intent_score, axis=1)
    test_df["intent_score"] = test_df.apply(compute_intent_score, axis=1)

    train_df["intent_category"] = train_df["intent_score"].apply(map_intent_category)
    test_df["intent_category"] = test_df["intent_score"].apply(map_intent_category)

    train_df["intent_level"] = train_df["intent_category"].map(intent_encoding)
    test_df["intent_level"] = test_df["intent_category"].map(intent_encoding)

    # --------------------------------------------------
    # SBERT
    # --------------------------------------------------
    print("Computing SBERT semantic alignment...")
    model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

    for df in [train_df, test_df]:
        df["Headline"] = df["Headline"].fillna("").astype(str)
        df["Key_Insights"] = df["Key_Insights"].fillna("").astype(str)

        headline_emb = model.encode(df["Headline"].tolist(), show_progress_bar=True)
        insight_emb = model.encode(df["Key_Insights"].tolist(), show_progress_bar=True)

        df["semantic_alignment"] = [
            cosine_similarity(
                headline_emb[i].reshape(1, -1),
                insight_emb[i].reshape(1, -1),
            )[0][0]
            for i in range(len(df))
        ]

    # --------------------------------------------------
    # Temporal & Legal Signal Features
    # --------------------------------------------------
    date_pattern = r"\b(19|20)\d{2}\b"

    train_df["temporal_density"] = train_df["Reasoning"].str.count(
        date_pattern
    ) / (train_df["reason_len"] + 1)

    test_df["temporal_density"] = test_df["Reasoning"].str.count(
        date_pattern
    ) / (test_df["reason_len"] + 1)

    legal_terms = ["foia", "indict", "prosecut", "sanction", "oversight", "plea", "probe"]

    train_df["legal_trigger_count"] = train_df["Reasoning"].apply(
        lambda t: sum(term in t for term in legal_terms)
    )

    test_df["legal_trigger_count"] = test_df["Reasoning"].apply(
        lambda t: sum(term in t for term in legal_terms)
    )

    # --------------------------------------------------
    # Final Power Density (override)
    # --------------------------------------------------
    train_df["power_density"] = train_df["num_power_mentions"] / np.log1p(
        train_df["headline_len"] + train_df["insight_len"]
    )

    test_df["power_density"] = test_df["num_power_mentions"] / np.log1p(
        test_df["headline_len"] + test_df["insight_len"]
    )

    # --------------------------------------------------
    # Cleanup & Save
    # --------------------------------------------------
    train_df.drop(columns=["intent_category"], inplace=True)
    test_df.drop(columns=["intent_category"], inplace=True)

    train_df.to_csv(TRAIN_OUTPUT, index=False)
    test_df.to_csv(TEST_OUTPUT, index=False)

    print("Feature engineering complete.")
    print(f"Saved: {TRAIN_OUTPUT}")
    print(f"Saved: {TEST_OUTPUT}")


if __name__ == "__main__":
    feature_engineering()