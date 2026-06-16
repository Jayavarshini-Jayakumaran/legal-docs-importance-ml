"""
Main Pipeline Runner
Project: Legal Document Importance Prediction
"""
import sys
import os
import io
import time
import traceback

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
else:
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, line_buffering=True, encoding=sys.stdout.encoding
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, line_buffering=True, encoding=sys.stderr.encoding
    )

def _print(msg=""):
    print(msg, flush=True)
    sys.stdout.flush()

def main():
    start = time.time()
    _print("\n" + "=" * 55)
    _print("  Legal Document Importance Prediction Pipeline")
    _print("=" * 55)

    _print("\nImporting pipeline modules...")
    try:
        from src.data_preprocessing import preprocess_data
        from src.feature_engineering import feature_engineering
        from src.train_model import train_model
        _print("Imports OK\n")
    except Exception:
        _print("\nIMPORT ERROR:")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        return 1

    steps = [
        ("Data Preprocessing",  preprocess_data),
        ("Feature Engineering", feature_engineering),
        ("Model Training",      train_model),
    ]

    for name, fn in steps:
        _print(f"--- {name} ---")
        try:
            fn()
        except Exception:
            _print(f"\nFAILED at: {name}")
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            return 1

    elapsed = time.time() - start
    _print("\n" + "=" * 55)
    _print(f"  Pipeline completed in {elapsed:.1f}s")
    _print("  Outputs:")
    _print("    outputs/submission.csv")
    _print("    outputs/prioritized_documents.csv")
    _print("    outputs/evaluation_report.txt")
    _print("=" * 55 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
