# Legal Document Importance Prediction

Predicts the importance score of legal documents (0–100) using structural
and semantic features, enabling automated prioritisation for faster
investigations and more efficient document review.

## Results

| Metric | Value |
|--------|-------|
| RMSE   | 6.44  |
| MAE    | 3.83  |
| R²     | 0.87  |
| Runtime | ~20s |

## Project Structure

```
├── data/
│   ├── raw/                  — Original train.csv / test.csv
│   └── processed/            — Cleaned data and engineered features
├── models/                   — Saved CatBoost models + feature list
├── notebooks/                — Exploratory and experimental notebooks
├── outputs/
│   ├── submission.csv              Raw predicted importance scores
│   ├── prioritized_documents.csv   Ranked + tiered for document review
│   └── evaluation_report.txt       RMSE, MAE, R², feature importances
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── train_model.py
├── main.py
└── requirements.txt
```

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

## Pipeline Steps

### 1. Data Preprocessing (`src/data_preprocessing.py`)
- Reads CSVs using Python's built-in `csv` module (low RAM footprint)
- Fixes text encoding artifacts
- Normalises text fields (lowercase, strip)
- Parses semicolon-separated list columns into real Python lists
- Engineers base count and length features

### 2. Feature Engineering (`src/feature_engineering.py`)
- Recomputes list-column counts from parsed lists (not string lengths)
- Computes 15+ engineered features across four categories:

  **Text length:** `headline_len`, `insight_len`, `reason_len`

  **Density & ratio:** `power_density`, `agency_density`,
  `institutional_index`, `lead_complexity_ratio`,
  `insight_concentration`, `entity_insight_ratio`,
  `tag_lead_alignment`, `actionability_proxy`, `information_density`

  **Investigative intent:** `intent_score`, `intent_level`
  (ordinal: contextual → informational → disclosure → analytical → allegational)

  **Semantic & legal signals:** `semantic_alignment` (TF-IDF cosine between
  Headline and Key_Insights, computed in 1,000-row chunks),
  `temporal_density`, `legal_trigger_count`

- `importance_prior`: group mean of Importance_Score by `num_power_mentions`
  (train only; test receives global mean — no leakage)

### 3. Model Training & Evaluation (`src/train_model.py`)
- All raw text columns dropped before training; signal is fully captured
  in engineered numeric features
- 80/20 train/validation split with `use_best_model=True`
- CatBoost Regressor: depth=5, lr=0.05, 500 trees, single thread
- Evaluation report printed and saved to `outputs/evaluation_report.txt`
- Final model retrained on 100% of training data
- Predictions clipped to [0, 100] and saved to two output files

## Outputs

| File | Description |
|------|-------------|
| `outputs/submission.csv` | `id`, `Importance_Score` for all test documents |
| `outputs/prioritized_documents.csv` | + `Priority_Rank` + `Priority_Tier` |
| `outputs/evaluation_report.txt` | Validation metrics and top feature importances |
| `models/catboost_validation.cbm` | Model trained on 80% split |
| `models/catboost_full.cbm` | Final model trained on 100% of train data |
| `models/features.json` | Ordered feature list used at training time |

## Priority Tiers

| Tier | Score | Meaning |
|------|-------|---------|
| Critical | 70–100 | Immediate investigative action |
| High | 45–70 | Priority review |
| Medium | 20–45 | Scheduled review |
| Low | 0–20 | Background / contextual |

## Top Features (by importance)

| Feature | Importance |
|---------|-----------|
| `num_lead_types` | 44.4% |
| `reason_len` | 19.5% |
| `num_tags` | 3.8% |
| `lead_complexity_ratio` | 3.6% |
| `intent_score` | 2.7% |

`num_lead_types` dominates — the number of distinct investigative lead
categories in a document is the strongest single signal of importance.
`reason_len` is second — longer reasoning sections correlate strongly
with higher-value documents.

## Requirements

```
pandas
numpy
scikit-learn
catboost
```
