"""
Main Pipeline Runner
Project: Legal Document Importance Prediction

Pipeline order:
1. Data Preprocessing
2. Feature Engineering
3. Model Training & Submission Generation
"""

import sys
import traceback


def main():
    try:
        print("\nStarting Legal Document Importance Prediction Pipeline")

        # Import pipeline steps
        from src.data_preprocessing import preprocess_data
        from src.feature_engineering import feature_engineering
        from src.train_model import train_model

        preprocess_data()
        feature_engineering()
        train_model()
        print("\nPipeline executed successfully!")

    except Exception:
        print("\n❌ Pipeline execution failed")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()