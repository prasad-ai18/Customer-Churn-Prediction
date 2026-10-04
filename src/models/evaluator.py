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


def save_evaluation_plots(metrics: EvaluationMetrics, global_importances: List[Dict[str, Any]], output_dir: Any) -> None:
    """Save clean, standalone SVG visual artifacts for ROC, PR, Confusion Matrix, and Feature Importance."""
    from pathlib import Path
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. ROC Curve SVG
    fprs = metrics.roc_curve_data.get("fpr", [])
    tprs = metrics.roc_curve_data.get("tpr", [])
    if fprs and tprs:
        pts = " ".join([f"{40 + f * 320:.1f},{340 - t * 300:.1f}" for f, t in zip(fprs, tprs)])
        roc_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400" width="100%" height="100%" style="background:#0f172a; font-family:sans-serif;">
  <text x="200" y="30" text-anchor="middle" fill="#f8fafc" font-size="16" font-weight="bold">ROC Curve - {metrics.model_name}</text>
  <text x="200" y="50" text-anchor="middle" fill="#94a3b8" font-size="12">ROC-AUC: {metrics.roc_auc:.4f}</text>
  <!-- Grid -->
  <line x1="40" y1="40" x2="40" y2="340" stroke="#334155" stroke-width="1"/>
  <line x1="40" y1="340" x2="360" y2="340" stroke="#334155" stroke-width="1"/>
  <!-- Diagonal Baseline -->
  <line x1="40" y1="340" x2="360" y2="40" stroke="#475569" stroke-dasharray="4,4" stroke-width="1.5"/>
  <!-- ROC Polyline -->
  <polyline fill="none" stroke="#6366f1" stroke-width="3" points="{pts}" />
  <!-- Labels -->
  <text x="40" y="360" fill="#64748b" font-size="11">0.0</text>
  <text x="360" y="360" fill="#64748b" font-size="11" text-anchor="end">1.0</text>
  <text x="25" y="45" fill="#64748b" font-size="11">1.0</text>
  <text x="200" y="380" text-anchor="middle" fill="#94a3b8" font-size="12">False Positive Rate (FPR)</text>
  <text x="15" y="190" text-anchor="middle" fill="#94a3b8" font-size="12" transform="rotate(-90 15,190)">True Positive Rate (TPR)</text>
</svg>"""
        (out / "roc_curve.svg").write_text(roc_svg, encoding="utf-8")

    # 2. Precision-Recall Curve SVG
    recs = metrics.pr_curve_data.get("recall", [])
    precs = metrics.pr_curve_data.get("precision", [])
    if recs and precs:
        pts = " ".join([f"{40 + r * 320:.1f},{340 - p * 300:.1f}" for r, p in zip(recs, precs)])
        pr_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400" width="100%" height="100%" style="background:#0f172a; font-family:sans-serif;">
  <text x="200" y="30" text-anchor="middle" fill="#f8fafc" font-size="16" font-weight="bold">Precision-Recall Curve - {metrics.model_name}</text>
  <text x="200" y="50" text-anchor="middle" fill="#94a3b8" font-size="12">PR-AUC (Avg Precision): {metrics.pr_auc:.4f}</text>
  <!-- Grid -->
  <line x1="40" y1="40" x2="40" y2="340" stroke="#334155" stroke-width="1"/>
  <line x1="40" y1="340" x2="360" y2="340" stroke="#334155" stroke-width="1"/>
  <!-- PR Polyline -->
  <polyline fill="none" stroke="#10b981" stroke-width="3" points="{pts}" />
  <!-- Labels -->
  <text x="40" y="360" fill="#64748b" font-size="11">0.0</text>
  <text x="360" y="360" fill="#64748b" font-size="11" text-anchor="end">1.0</text>
  <text x="25" y="45" fill="#64748b" font-size="11">1.0</text>
  <text x="200" y="380" text-anchor="middle" fill="#94a3b8" font-size="12">Recall (Sensitivity)</text>
  <text x="15" y="190" text-anchor="middle" fill="#94a3b8" font-size="12" transform="rotate(-90 15,190)">Precision</text>
</svg>"""
        (out / "precision_recall_curve.svg").write_text(pr_svg, encoding="utf-8")

    # 3. Confusion Matrix SVG
    cm = metrics.confusion_matrix
    tp = cm.get("true_positive", 0)
    fp = cm.get("false_positive", 0)
    fn = cm.get("false_negative", 0)
    tn = cm.get("true_negative", 0)
    cm_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 440 380" width="100%" height="100%" style="background:#0f172a; font-family:sans-serif;">
  <text x="220" y="35" text-anchor="middle" fill="#f8fafc" font-size="16" font-weight="bold">Confusion Matrix - {metrics.model_name}</text>
  <text x="220" y="55" text-anchor="middle" fill="#94a3b8" font-size="12">Total Test Samples: {tp + fp + fn + tn:,}</text>
  <!-- Headers -->
  <text x="170" y="90" text-anchor="middle" fill="#cbd5e1" font-size="13" font-weight="bold">Pred: Retain</text>
  <text x="310" y="90" text-anchor="middle" fill="#cbd5e1" font-size="13" font-weight="bold">Pred: Churn</text>
  <text x="25" y="165" text-anchor="middle" fill="#cbd5e1" font-size="13" font-weight="bold" transform="rotate(-90 25,165)">Actual: Retain</text>
  <text x="25" y="275" text-anchor="middle" fill="#cbd5e1" font-size="13" font-weight="bold" transform="rotate(-90 25,275)">Actual: Churn</text>
  <!-- Cells -->
  <!-- TN -->
  <rect x="100" y="110" width="140" height="100" fill="rgba(16, 185, 129, 0.15)" stroke="#10b981" rx="8"/>
  <text x="170" y="160" text-anchor="middle" fill="#10b981" font-size="24" font-weight="bold">{tn}</text>
  <text x="170" y="185" text-anchor="middle" fill="#94a3b8" font-size="11">True Negative</text>
  <!-- FP -->
  <rect x="250" y="110" width="140" height="100" fill="rgba(244, 63, 94, 0.12)" stroke="#f43f5e" rx="8"/>
  <text x="320" y="160" text-anchor="middle" fill="#f43f5e" font-size="24" font-weight="bold">{fp}</text>
  <text x="320" y="185" text-anchor="middle" fill="#94a3b8" font-size="11">False Positive</text>
  <!-- FN -->
  <rect x="100" y="220" width="140" height="100" fill="rgba(244, 63, 94, 0.12)" stroke="#f43f5e" rx="8"/>
  <text x="170" y="270" text-anchor="middle" fill="#f43f5e" font-size="24" font-weight="bold">{fn}</text>
  <text x="170" y="295" text-anchor="middle" fill="#94a3b8" font-size="11">False Negative</text>
  <!-- TP -->
  <rect x="250" y="220" width="140" height="100" fill="rgba(16, 185, 129, 0.15)" stroke="#10b981" rx="8"/>
  <text x="320" y="270" text-anchor="middle" fill="#10b981" font-size="24" font-weight="bold">{tp}</text>
  <text x="320" y="295" text-anchor="middle" fill="#94a3b8" font-size="11">True Positive</text>
</svg>"""
    (out / "confusion_matrix.svg").write_text(cm_svg, encoding="utf-8")

    # 4. Feature Importance SVG
    if global_importances:
        top10 = global_importances[:10]
        max_v = max([i["importance"] for i in top10], default=0.1)
        bars_svg = []
        for idx, item in enumerate(top10):
            y = 70 + idx * 28
            w = int((item["importance"] / max_v) * 200)
            bars_svg.append(f"""
  <text x="180" y="{y + 14}" text-anchor="end" fill="#e2e8f0" font-size="11">{item.get('title', item['feature'])[:24]}</text>
  <rect x="190" y="{y}" width="{w}" height="18" fill="url(#grad)" rx="4"/>
  <text x="{195 + w}" y="{y + 14}" fill="#38bdf8" font-size="11" font-weight="bold">{(item['importance']*100):.1f}%</text>
""")
        feat_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 450 370" width="100%" height="100%" style="background:#0f172a; font-family:sans-serif;">
  <defs>
    <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#6366f1"/>
      <stop offset="100%" stop-color="#a855f7"/>
    </linearGradient>
  </defs>
  <text x="225" y="35" text-anchor="middle" fill="#f8fafc" font-size="16" font-weight="bold">Top 10 Feature Importances</text>
  {"".join(bars_svg)}
</svg>"""
        (out / "feature_importance.svg").write_text(feat_svg, encoding="utf-8")

