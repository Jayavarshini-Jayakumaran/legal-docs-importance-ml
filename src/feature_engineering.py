"""
Feature Engineering Script
Project: Legal Document Importance Prediction
Objective: Create advanced, model-ready features from cleaned data
"""

import ast
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Paths
_SRC_DIR = Path(__file__).resolve().parent
_ROOT    = _SRC_DIR.parent

PROCESSED_DATA_DIR = _ROOT / "data" / "processed"
CACHE_DIR          = _ROOT / "data" / "processed" / "embedding_cache"

TRAIN_INPUT  = PROCESSED_DATA_DIR / "train_clean.csv"
TEST_INPUT   = PROCESSED_DATA_DIR / "test_clean.csv"
TRAIN_OUTPUT = PROCESSED_DATA_DIR / "train_features.csv"
TEST_OUTPUT  = PROCESSED_DATA_DIR / "test_features.csv"


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
    """'['FBI', 'DOJ']'  →  ['FBI', 'DOJ']  |  ''  →  []"""
    if isinstance(x, list):
        return x
    if not isinstance(x, str) or not x.strip():
        return []
    stripped = x.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            return ast.literal_eval(stripped)
        except Exception:
            pass
    # semicolon-separated fallback (raw preprocessing output)
    return [t.strip() for t in stripped.split(";") if t.strip()]


def compute_intent_score(row):
    score  = row["num_lead_types"]     * 2
    score += row["num_agencies"]       * 3
    score += row["num_power_mentions"]
    score += row["insight_len"]        / 150
    return score


def map_intent_category(score):
    if score <= 1:   return "contextual"
    elif score <= 3: return "informational"
    elif score <= 6: return "disclosure"
    elif score <= 9: return "analytical"
    else:            return "allegational"


# Semantic alignment — chunked word-overlap cosine (minimal RAM)
# Fits on headlines only, transforms in chunks of 1000 rows,
# falls back to token Jaccard if still OOM.
def _jaccard_row(h: str, i: str) -> float:
    h_tok = set(h.lower().split())
    i_tok = set(i.lower().split())
    if not h_tok or not i_tok:
        return 0.0
    return len(h_tok & i_tok) / len(h_tok | i_tok)


def add_semantic_alignment(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    headlines = df["Headline"].fillna("").astype(str).tolist()
    insights  = df["Key_Insights"].fillna("").astype(str).tolist()

    try:
        tfidf = TfidfVectorizer(
            max_features=2000,
            sublinear_tf=True,
            analyzer="word",
            ngram_range=(1, 1),
            dtype=np.float32,
        )
        tfidf.fit(headlines)   # fit only on headlines — small and fast

        scores = []
        chunk_size = 1000
        for s in range(0, len(df), chunk_size):
            h_chunk = headlines[s : s + chunk_size]
            i_chunk = insights[s  : s + chunk_size]
            hv = tfidf.transform(h_chunk)
            iv = tfidf.transform(i_chunk)
            dot     = np.array(hv.multiply(iv).sum(axis=1)).flatten()
            h_norms = np.sqrt(np.array(hv.multiply(hv).sum(axis=1)).flatten())
            i_norms = np.sqrt(np.array(iv.multiply(iv).sum(axis=1)).flatten())
            denom   = np.maximum(h_norms * i_norms, 1e-10)
            scores.extend((dot / denom).tolist())
            del hv, iv, dot, h_norms, i_norms, denom

        df["semantic_alignment"] = scores

    except MemoryError:
        print("    [fallback] TF-IDF OOM — using token Jaccard")
        df["semantic_alignment"] = [
            _jaccard_row(h, i) for h, i in zip(headlines, insights)
        ]

    print(f"    semantic_alignment: {len(df):,} rows  "
          f"mean={df['semantic_alignment'].mean():.3f}")
    return df


# Main
def feature_engineering():
    print("\n[2/3] FEATURE ENGINEERING")
    print("=" * 50)

    print("  Loading cleaned datasets (chunked)...")
    train_df = read_csv_safe(TRAIN_INPUT)
    test_df  = read_csv_safe(TEST_INPUT)

    # Parse list columns BEFORE any .apply(len)
    list_cols = ["Lead_Types", "Power_Mentions", "Agencies"]
    for col in list_cols:
        train_df[col] = train_df[col].apply(parse_list_col)
        test_df[col]  = test_df[col].apply(parse_list_col)

    # Recompute counts (preprocessing may have saved string counts from
    # before list parsing; recompute here on real lists to be safe)
    for df in [train_df, test_df]:
        df["num_lead_types"]     = df["Lead_Types"].apply(len)
        df["num_power_mentions"] = df["Power_Mentions"].apply(len)
        df["num_agencies"]       = df["Agencies"].apply(len)

    # Base text-length features
    for df in [train_df, test_df]:
        df["headline_len"] = df["Headline"].fillna("").str.len()
        df["insight_len"]  = df["Key_Insights"].fillna("").str.len()
        df["reason_len"]   = df["Reasoning"].fillna("").str.len()

    # Density & ratio features
    for df in [train_df, test_df]:
        n_pm = df["num_power_mentions"]
        n_ag = df["num_agencies"]
        n_lt = df["num_lead_types"]
        il   = df["insight_len"]
        rl   = df["reason_len"]
        hl   = df["headline_len"]

        df["agency_density"]        = n_ag / (il + 1)
        df["institutional_index"]   = n_ag + n_lt
        df["lead_complexity_ratio"] = n_lt / (n_pm + 1)
        df["insight_concentration"] = il   / (rl + 1)
        df["entity_insight_ratio"]  = n_pm / (il + 1)
        df["tag_lead_alignment"]    = df["num_tags"] / (n_lt + 1)
        df["actionability_proxy"]   = n_pm + n_ag + n_lt
        df["information_density"]   = (il + rl) / (hl + 1)
        df["power_density"]         = n_pm / np.log1p(hl + il)

    # Importance prior
    # Guard with column-existence check; test set uses global mean
    if "Importance_Score" in train_df.columns:
        prior_map    = train_df.groupby("num_power_mentions")["Importance_Score"].mean()
        global_prior = train_df["Importance_Score"].mean()
        train_df["importance_prior"] = (
            train_df["num_power_mentions"].map(prior_map).fillna(global_prior)
        )
    else:
        raise ValueError("Importance_Score missing from train_clean.csv — check preprocessing.")

    test_df["importance_prior"] = global_prior

    # Investigative intent
    intent_encoding = {"contextual": 0, "informational": 1,
                       "disclosure": 2, "analytical": 3, "allegational": 4}

    for df in [train_df, test_df]:
        df["intent_score"]  = df.apply(compute_intent_score, axis=1)
        df["intent_level"]  = df["intent_score"].apply(map_intent_category).map(intent_encoding)

    # Semantic alignment (TF-IDF cosine, no model needed)
    print("  Computing semantic alignment (TF-IDF)...")
    train_df = add_semantic_alignment(train_df, "train")
    test_df  = add_semantic_alignment(test_df,  "test")

    # Temporal & legal signal features
    date_pattern = r"\b(19|20)\d{2}\b"
    legal_terms  = ["foia", "indict", "prosecut", "sanction", "oversight", "plea", "probe"]

    for df in [train_df, test_df]:
        reasoning = df["Reasoning"].fillna("").astype(str)
        df["temporal_density"]    = reasoning.str.count(date_pattern) / (df["reason_len"] + 1)
        df["legal_trigger_count"] = reasoning.apply(
            lambda t: sum(term in t for term in legal_terms)
        )

    # Save
    train_df.to_csv(TRAIN_OUTPUT, index=False)
    test_df.to_csv(TEST_OUTPUT,  index=False)

    print(f"  Saved: {TRAIN_OUTPUT}")
    print(f"  Saved: {TEST_OUTPUT}")
    print("  Feature engineering complete.\n")


if __name__ == "__main__":
    feature_engineering()
