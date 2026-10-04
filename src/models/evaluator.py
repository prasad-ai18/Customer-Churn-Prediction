from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from src.logger import logger

@dataclass
class EvaluationMetrics:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: Dict[str, int]
    roc_curve_data: Dict[str, List[float]]
    pr_curve_data: Dict[str, List[float]]
    slice_analysis: Dict[str, Any]

def compute_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, num_thresholds: int = 50) -> Dict[str, List[float]]:
    """Compute empirical ROC curve points (FPR vs TPR)."""
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    fpr_list = []
    tpr_list = []

    pos_total = max(int(np.sum(y_true == 1)), 1)
    neg_total = max(int(np.sum(y_true == 0)), 1)

    for thresh in thresholds:
        pred_pos = y_proba >= thresh
        tp = int(np.sum((y_true == 1) & pred_pos))
        fp = int(np.sum((y_true == 0) & pred_pos))
        tpr_list.append(round(tp / pos_total, 4))
        fpr_list.append(round(fp / neg_total, 4))

    # Sort by FPR ascending
    combined = sorted(zip(fpr_list, tpr_list), key=lambda p: (p[0], p[1]))
    # Remove duplicates
    unique_fpr = []
    unique_tpr = []
    for f, t in combined:
        if not unique_fpr or f != unique_fpr[-1] or t != unique_tpr[-1]:
            unique_fpr.append(f)
            unique_tpr.append(t)

    return {"fpr": unique_fpr, "tpr": unique_tpr}

def compute_pr_curve(y_true: np.ndarray, y_proba: np.ndarray, num_thresholds: int = 50) -> Dict[str, List[float]]:
    """Compute empirical Precision-Recall curve points."""
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    prec_list = []
    rec_list = []

    pos_total = max(int(np.sum(y_true == 1)), 1)

    for thresh in thresholds:
        pred_pos = y_proba >= thresh
        tp = int(np.sum((y_true == 1) & pred_pos))
        fp = int(np.sum((y_true == 0) & pred_pos))
        rec = tp / pos_total
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec_list.append(round(rec, 4))
        prec_list.append(round(prec, 4))

    # Sort by recall ascending
    combined = sorted(zip(rec_list, prec_list), key=lambda p: p[0])
    return {"recall": [p[0] for p in combined], "precision": [p[1] for p in combined]}

def evaluate_predictions(
    model_name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    original_df_slice: pd.DataFrame = None
) -> EvaluationMetrics:
    """
    Comprehensive evaluation using pure numerical arrays.
    Calculates Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix, and Slices.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    y_pred = (y_proba >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    n = len(y_true)
    acc = (tp + tn) / n if n > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    # ROC-AUC via rank statistic (equivalent to Mann-Whitney U test)
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    pos_scores = y_proba[pos_mask]
    neg_scores = y_proba[neg_mask]

    if len(pos_scores) > 0 and len(neg_scores) > 0:
        greater = np.sum(pos_scores[:, None] > neg_scores[None, :])
        equal = np.sum(pos_scores[:, None] == neg_scores[None, :])
        roc_auc = float((greater + 0.5 * equal) / (len(pos_scores) * len(neg_scores)))
    else:
        roc_auc = 0.5

    # PR-AUC via Average Precision
    # Rank probabilities descending
    sort_idx = np.argsort(-y_proba)
    y_sorted = y_true[sort_idx]
    cum_tp = np.cumsum(y_sorted == 1)
    cum_fp = np.cumsum(y_sorted == 0)
    precisions = cum_tp / (cum_tp + cum_fp)
    recalls = cum_tp / max(len(pos_scores), 1)

    # Trapezoid integration for PR-AUC
    pr_auc = float(np.sum((recalls[1:] - recalls[:-1]) * precisions[1:]))
    if pr_auc <= 0.0 and len(pos_scores) > 0:
        pr_auc = float(np.mean(precisions[y_sorted == 1])) if (y_sorted == 1).any() else 0.0

    roc_curve = compute_roc_curve(y_true, y_proba)
    pr_curve = compute_pr_curve(y_true, y_proba)

    # Slice analysis
    slice_results = {}
    if original_df_slice is not None and "Contract" in original_df_slice.columns:
        slice_df = original_df_slice.copy()
        slice_df["y_true"] = y_true
        slice_df["y_pred"] = y_pred
        slice_df["y_prob"] = y_proba

        contract_slices = {}
        for contract, group in slice_df.groupby("Contract"):
            grp_true = group["y_true"].to_numpy()
            grp_pred = group["y_pred"].to_numpy()
            grp_prob = group["y_prob"].to_numpy()

            g_tp = int(np.sum((grp_true == 1) & (grp_pred == 1)))
            g_fp = int(np.sum((grp_true == 0) & (grp_pred == 1)))
            g_fn = int(np.sum((grp_true == 1) & (grp_pred == 0)))

            g_rec = g_tp / (g_tp + g_fn) if (g_tp + g_fn) > 0 else 0.0
            g_prec = g_tp / (g_tp + g_fp) if (g_tp + g_fp) > 0 else 0.0

            contract_slices[str(contract)] = {
                "count": int(len(group)),
                "actual_churn_rate": round(float(grp_true.mean()), 4),
                "recall": round(float(g_rec), 4),
                "precision": round(float(g_prec), 4),
                "avg_predicted_risk": round(float(grp_prob.mean()), 4)
            }
        slice_results["by_contract"] = contract_slices

    logger.info(
        f"Evaluation for {model_name}: "
        f"ROC-AUC={roc_auc:.4f} | PR-AUC={pr_auc:.4f} | "
        f"F1={f1:.4f} | Recall={rec:.4f} | Precision={prec:.4f}"
    )

    return EvaluationMetrics(
        model_name=model_name,
        accuracy=round(float(acc), 4),
        precision=round(float(prec), 4),
        recall=round(float(rec), 4),
        f1_score=round(float(f1), 4),
        roc_auc=round(float(roc_auc), 4),
        pr_auc=round(float(pr_auc), 4),
        confusion_matrix={"true_negative": tn, "false_positive": fp, "false_negative": fn, "true_positive": tp},
        roc_curve_data=roc_curve,
        pr_curve_data=pr_curve,
        slice_analysis=slice_results
    )
