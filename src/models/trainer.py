import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from config.settings import settings
from src.logger import logger
from src.exceptions import ModelTrainingError
from src.data.loader import load_raw_data
from src.data.preprocessor import clean_data, split_data, DataBundle
from src.features.engineer import TelcoFeatureEngineer, TelcoDataTransformer
from src.models.evaluator import evaluate_predictions, EvaluationMetrics
from src.models.explainer import ChurnExplainer

from src.models.pipeline import NumpyLogisticRegression, ChurnPipeline


@dataclass
class ModelTrainingResult:
    best_model_name: str
    best_pipeline: ChurnPipeline
    all_metrics: Dict[str, EvaluationMetrics]
    metadata: Dict[str, Any]


def train_all_models(data_bundle: Optional[DataBundle] = None) -> ModelTrainingResult:
    """
    Train baseline and tree-based models, benchmark all metrics, and persist production artifacts.
    """
    if data_bundle is None:
        raw_df = load_raw_data()
        cleaned_df = clean_data(raw_df)
        data_bundle = split_data(cleaned_df)

    X_train_raw = data_bundle.train_df.drop(columns=["customerID", "Churn"], errors="ignore")
    y_train = data_bundle.train_df["Churn"].to_numpy().astype(int)

    X_val_raw = data_bundle.val_df.drop(columns=["customerID", "Churn"], errors="ignore")
    y_val = data_bundle.val_df["Churn"].to_numpy().astype(int)

    X_test_raw = data_bundle.test_df.drop(columns=["customerID", "Churn"], errors="ignore")
    y_test = data_bundle.test_df["Churn"].to_numpy().astype(int)

    # 1. Fit Feature Engineering and Data Transformer on train split only (strictly zero leakage)
    engineer = TelcoFeatureEngineer()
    X_train_eng = engineer.transform(X_train_raw)

    transformer = TelcoDataTransformer()
    transformer.fit(X_train_eng)

    X_train_trans = transformer.transform(X_train_eng)
    feature_names = transformer.encoded_column_names

    X_val_trans = transformer.transform(engineer.transform(X_val_raw))
    X_test_trans = transformer.transform(engineer.transform(X_test_raw))

    # Class distribution & imbalance factor
    neg_count = int(np.sum(y_train == 0))
    pos_count = int(np.sum(y_train == 1))
    imbalance_ratio = neg_count / max(pos_count, 1)
    logger.info(f"Class distribution: Non-churn={neg_count}, Churn={pos_count}, Imbalance ratio={imbalance_ratio:.2f}")

    dtrain = xgb.DMatrix(X_train_trans.values, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val_trans.values, label=y_val, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test_trans.values, label=y_test, feature_names=feature_names)

    # 2. Train Candidates
    all_metrics: Dict[str, EvaluationMetrics] = {}
    pipelines: Dict[str, ChurnPipeline] = {}

    # Candidate 1: Logistic Regression Baseline
    logger.info("Training Candidate 1: Logistic Regression Baseline...")
    lr_model = NumpyLogisticRegression(lr=0.08, max_iter=600, l2=1e-3, class_weight="balanced")
    lr_model.fit(X_train_trans.values, y_train)
    p_lr = ChurnPipeline(engineer, transformer, lr_model, "logistic_regression_baseline", feature_names)
    pipelines["logistic_regression_baseline"] = p_lr
    val_probs_lr = p_lr.predict_proba(X_val_raw)[:, 1]
    test_probs_lr = p_lr.predict_proba(X_test_raw)[:, 1]
    all_metrics["logistic_regression_baseline"] = evaluate_predictions(
        "logistic_regression_baseline", y_test, test_probs_lr, original_df_slice=data_bundle.test_df
    )

    # Candidate 2: Random Forest
    logger.info("Training Candidate 2: Random Forest (XGBoost parallel trees)...")
    rf_params = {
        "booster": "gbtree",
        "learning_rate": 1.0,
        "num_parallel_tree": 80,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bynode": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": settings.RANDOM_STATE
    }
    rf_bst = xgb.train(rf_params, dtrain, num_boost_round=1)
    p_rf = ChurnPipeline(engineer, transformer, rf_bst, "random_forest", feature_names)
    pipelines["random_forest"] = p_rf
    test_probs_rf = p_rf.predict_proba(X_test_raw)[:, 1]
    all_metrics["random_forest"] = evaluate_predictions(
        "random_forest", y_test, test_probs_rf, original_df_slice=data_bundle.test_df
    )

    # Candidate 3: XGBoost (Gradient Boosted Ensemble with Class Weighting)
    logger.info("Training Candidate 3: XGBoost Gradient Boosted Ensemble...")
    xgb_params = {
        "booster": "gbtree",
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "scale_pos_weight": imbalance_ratio,
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "aucpr"],
        "random_state": settings.RANDOM_STATE
    }
    xgb_bst = xgb.train(xgb_params, dtrain, num_boost_round=120)
    p_xgb = ChurnPipeline(engineer, transformer, xgb_bst, "xgboost", feature_names)
    pipelines["xgboost"] = p_xgb
    test_probs_xgb = p_xgb.predict_proba(X_test_raw)[:, 1]
    all_metrics["xgboost"] = evaluate_predictions(
        "xgboost", y_test, test_probs_xgb, original_df_slice=data_bundle.test_df
    )

    # Model Selection: Prioritize PR-AUC and F1 for churn retention optimization
    best_name = "xgboost"
    best_score = -1.0
    for name, metric in all_metrics.items():
        score = (metric.pr_auc * 0.6) + (metric.f1_score * 0.4)
        logger.info(f"Model [{name}] Selection Score: {score:.4f} (PR-AUC: {metric.pr_auc:.4f}, F1: {metric.f1_score:.4f}, ROC-AUC: {metric.roc_auc:.4f})")
        if score > best_score:
            best_score = score
            best_name = name

    logger.info(f"Champion Model Selected: [{best_name}] with combined score {best_score:.4f}")
    best_pipeline = pipelines[best_name]

    # Global feature importance from champion model
    explainer = ChurnExplainer(best_pipeline)
    global_importances = explainer.get_global_feature_importance(top_n=15)

    # Compile metadata
    metadata = {
        "model_name": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "app_version": settings.APP_VERSION,
        "dataset": "IBM Telco Customer Churn",
        "train_samples": int(len(X_train_raw)),
        "val_samples": int(len(X_val_raw)),
        "test_samples": int(len(X_test_raw)),
        "class_imbalance_ratio": round(float(imbalance_ratio), 2),
        "primary_metrics": {
            "accuracy": all_metrics[best_name].accuracy,
            "precision": all_metrics[best_name].precision,
            "recall": all_metrics[best_name].recall,
            "f1_score": all_metrics[best_name].f1_score,
            "roc_auc": all_metrics[best_name].roc_auc,
            "pr_auc": all_metrics[best_name].pr_auc
        },
        "confusion_matrix": all_metrics[best_name].confusion_matrix,
        "global_feature_importance": global_importances,
        "risk_thresholds": {
            "low_max": settings.LOW_RISK_MAX,
            "high_min": settings.HIGH_RISK_MIN
        }
    }

    # Save artifacts
    save_dir = settings.MODEL_ARTIFACTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Unified pipeline
    pipeline_path = save_dir / "best_churn_pipeline.joblib"
    joblib.dump(best_pipeline, pipeline_path)
    logger.info(f"Saved best model pipeline to {pipeline_path}")

    # 2. Metadata
    metadata_path = save_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata to {metadata_path}")

    # 3. Model comparison summary
    comparison_summary = {
        name: {
            "accuracy": m.accuracy,
            "precision": m.precision,
            "recall": m.recall,
            "f1_score": m.f1_score,
            "roc_auc": m.roc_auc,
            "pr_auc": m.pr_auc,
            "confusion_matrix": m.confusion_matrix,
            "roc_curve": m.roc_curve_data,
            "pr_curve": m.pr_curve_data,
            "slice_analysis": m.slice_analysis
        }
        for name, m in all_metrics.items()
    }
    comparison_path = save_dir / "all_models_comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)
    logger.info(f"Saved models comparison to {comparison_path}")

    return ModelTrainingResult(
        best_model_name=best_name,
        best_pipeline=best_pipeline,
        all_metrics=all_metrics,
        metadata=metadata
    )

if __name__ == "__main__":
    train_all_models()
