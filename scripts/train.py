import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from config.settings import settings
from src.logger import logger
from src.data.loader import load_raw_data, ensure_raw_data
from src.data.validator import validate_raw_dataset
from src.data.preprocessor import clean_data, split_data
from src.models.trainer import train_all_models

def main():
    parser = argparse.ArgumentParser(
        description="Customer Churn ML Training Pipeline - End-to-End Reproducible Model Training"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(settings.DATA_RAW_PATH),
        help="Path to raw customer churn CSV dataset"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Proportion of dataset held out for final test evaluation (default: 0.20)"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Proportion of dataset held out for validation tuning (default: 0.15)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for deterministic, reproducible splitting and training"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CUSTOMER CHURN INTELLIGENCE & PREDICTION PLATFORM - ML TRAINING PIPELINE")
    print("=" * 80)
    print(f"Data Source       : {args.data_path}")
    print(f"Test Split Ratio  : {args.test_size * 100:.1f}%")
    print(f"Val Split Ratio   : {args.val_size * 100:.1f}%")
    print(f"Random State      : {args.random_state}")
    print("-" * 80)

    # 1. Ingest Data
    raw_path = Path(args.data_path)
    if not raw_path.exists():
        print(f"[*] Dataset not found at {raw_path}. Downloading official IBM Telco dataset...")
        raw_path = ensure_raw_data(destination=raw_path)

    df_raw = pd.read_csv(raw_path)
    print(f"[+] Loaded raw dataset: {len(df_raw):,} rows, {len(df_raw.columns)} columns.")

    # 2. Validate & Clean
    print("[*] Validating schema constraints, data types, and handling whitespace...")
    cleaned_df, report = validate_raw_dataset(df_raw, strict=True)
    cleaned_df = clean_data(cleaned_df)
    print(f"[+] Validation passed: {report.duplicates_removed} duplicates removed, {report.whitespace_fixed_count} empty records fixed.")
    print(f"    Class distribution: {report.target_distribution}")

    # 3. Leakage-Free Stratified Split
    print("[*] Performing leakage-free stratified split into Train, Validation, and Test...")
    bundle = split_data(
        cleaned_df,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
    )
    print(f"[+] Splits created: Train={len(bundle.train_df):,}, Val={len(bundle.val_df):,}, Test={len(bundle.test_df):,}")

    # 4. Train All Candidates & Select Champion
    print("\n[*] Training model candidates (Logistic Regression, Random Forest, Gradient Boosting, XGBoost)...")
    result = train_all_models(bundle)

    # 5. Output Summary Table
    print("\n" + "=" * 80)
    print("OUT-OF-SAMPLE TEST EVALUATION BENCHMARK (1,409 Held-Out Customers)")
    print("=" * 80)
    print(f"{'Model Candidate':<32} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'F1':<6} | {'Recall':<8} | {'Precision':<9} | {'Accuracy':<8}")
    print("-" * 80)

    for name, m in result.all_metrics.items():
        champ_flag = " * [CHAMPION]" if name == result.best_model_name else ""
        print(f"{name + champ_flag:<32} | {m.roc_auc:<8.4f} | {m.pr_auc:<8.4f} | {m.f1_score:<6.4f} | {m.recall*100:<7.2f}% | {m.precision*100:<8.2f}% | {m.accuracy*100:<7.2f}%")

    print("=" * 80)
    print(f"[OK] Champion Model Selected : {result.best_model_name.upper()}")
    print(f"[OK] Saved Model Pipeline    : {settings.MODEL_ARTIFACTS_DIR / 'best_churn_pipeline.joblib'}")
    print(f"[OK] Saved Model Metadata    : {settings.MODEL_ARTIFACTS_DIR / 'model_metadata.json'}")
    print(f"[OK] Saved Evaluation Report : {settings.MODEL_ARTIFACTS_DIR / 'evaluation_report.md'}")
    print(f"[OK] Saved Visual SVG Plots  : {settings.MODEL_ARTIFACTS_DIR / 'plots'}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
