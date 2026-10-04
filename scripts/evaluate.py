import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import numpy as np

from config.settings import settings
from src.logger import logger
from src.models.evaluator import evaluate_predictions, save_evaluation_plots
from src.models.explainer import ChurnExplainer

def main():
    parser = argparse.ArgumentParser(
        description="Customer Churn Evaluation Script - Standalone Model Testing & Slice Analysis"
    )
    parser.add_argument(
        "--pipeline-path",
        type=str,
        default=str(settings.MODEL_ARTIFACTS_DIR / "best_churn_pipeline.joblib"),
        help="Path to serialized ChurnPipeline joblib file"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=str(settings.DATA_PROCESSED_DIR / "test.csv"),
        help="Path to held-out test split CSV"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for classification (default: 0.5)"
    )

    args = parser.parse_args()

    pipeline_file = Path(args.pipeline_path)
    test_file = Path(args.test_data)

    if not pipeline_file.exists():
        print(f"[!] Error: Model pipeline artifact not found at {pipeline_file}")
        print("    Please run 'python scripts/train.py' first.")
        sys.exit(1)

    if not test_file.exists():
        print(f"[!] Error: Test data split not found at {test_file}")
        print("    Please run 'python scripts/train.py' first to generate processed splits.")
        sys.exit(1)

    print("=" * 80)
    print("CUSTOMER CHURN PLATFORM - STANDALONE MODEL EVALUATION & SLICE AUDIT")
    print("=" * 80)
    print(f"Pipeline Artifact : {pipeline_file}")
    print(f"Test Split Path   : {test_file}")
    print(f"Decision Cutoff   : {args.threshold}")
    print("-" * 80)

    # 1. Load pipeline and test dataset
    pipeline = joblib.load(pipeline_file)
    test_df = pd.read_csv(test_file)
    print(f"[+] Loaded test dataset with {len(test_df):,} customer records.")

    # 2. Extract inputs and ground truth
    X_test = test_df.drop(columns=["customerID", "Churn"], errors="ignore")
    y_test = pd.to_numeric(test_df["Churn"], errors="coerce").fillna(0).to_numpy().astype(int)

    # 3. Compute Probabilities
    probs = pipeline.predict_proba(X_test)[:, 1]

    # 4. Rigorous Evaluation
    metrics = evaluate_predictions(
        model_name=pipeline.model_type,
        y_true=y_test,
        y_proba=probs,
        threshold=args.threshold,
        original_df_slice=test_df
    )

    # 5. Display Performance Summary
    print("\n" + "=" * 80)
    print("OVERALL PERFORMANCE SUMMARY (Held-Out Test Set)")
    print("=" * 80)
    print(f"- Model Type          : {pipeline.model_type.upper()}")
    print(f"- ROC-AUC             : {metrics.roc_auc:.4f}")
    print(f"- PR-AUC (Avg Prec)   : {metrics.pr_auc:.4f}")
    print(f"- F1-Score            : {metrics.f1_score:.4f}")
    print(f"- Recall (Sensitivity): {metrics.recall * 100:.2f}%")
    print(f"- Precision           : {metrics.precision * 100:.2f}%")
    print(f"- Accuracy            : {metrics.accuracy * 100:.2f}%")
    print("-" * 80)

    cm = metrics.confusion_matrix
    print("CONFUSION MATRIX BREAKDOWN:")
    print(f"  True Positives  (Correctly detected churn) : {cm['true_positive']:,}")
    print(f"  False Positives (False churn alarms)       : {cm['false_positive']:,}")
    print(f"  False Negatives (Missed churners)          : {cm['false_negative']:,}")
    print(f"  True Negatives  (Retained correctly)       : {cm['true_negative']:,}")

    # 6. Slice Analysis
    print("\n" + "=" * 80)
    print("SLICE-BASED ANALYSIS (Performance by Contract Cohort)")
    print("=" * 80)
    print(f"{'Contract Type':<20} | {'Count':<8} | {'Churn Rate':<12} | {'Recall':<10} | {'Avg Predicted Risk':<18}")
    print("-" * 80)
    by_contract = metrics.slice_analysis.get("by_contract", {})
    for contract, stats in by_contract.items():
        print(f"{contract:<20} | {stats['count']:<8} | {stats['actual_churn_rate']*100:<10.1f}% | {stats['recall']*100:<8.1f}% | {stats['avg_predicted_risk']*100:<16.1f}%")

    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
