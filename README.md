# Legal Document Importance Prediction

This project predicts the importance score of legal documents using
textual, structural, and semantic features.

## Project Structure
- `data/raw/` – Original datasets
- `data/processed/` – Cleaned & feature-engineered data
- `notebooks/` – Exploratory and experimental notebooks
- `src/` – Production-ready pipeline scripts
- `models/` – Trained CatBoost models
- `outputs/` – Prediction outputs

## Pipeline Steps
1. Data cleaning & normalization
2. Feature engineering (text, metadata, semantic)
3. Model training using CatBoost
4. Prediction

## How to Run
```bash
python main.py
```

## Model
- Algorithm: CatBoost Regressor
- Objective: RMSE
- Text features handled natively

## Output
`outputs/submission.csv` contains predicted importance scores
