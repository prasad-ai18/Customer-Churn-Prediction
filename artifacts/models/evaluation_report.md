# Model Training & Evaluation Report

**Trained At**: `2026-10-04T14:13:57.523927+00:00`
**Dataset**: `IBM Telco Customer Churn` (Total Evaluated: 7,043 samples)
**Class Imbalance Ratio**: Non-churn / Churn = `2.77:1`

## 1. Candidate Comparison Benchmark (Held-Out Test Set: 1,409 samples)

| Model Architecture | ROC-AUC | PR-AUC | F1-Score | Recall | Precision | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **logistic_regression_baseline ★ Champion** | **0.8464** | **0.6753** | **0.6265** | **79.14%** | 51.84% | 74.95% |
| **random_forest** | **0.8405** | **0.6566** | **0.5957** | **55.35%** | 64.49% | 80.06% |
| **gradient_boosting** | **0.8442** | **0.6705** | **0.5985** | **53.21%** | 68.38% | 81.05% |
| **xgboost** | **0.8429** | **0.6717** | **0.6100** | **78.61%** | 49.83% | 73.31% |

## 2. Champion Model Confusion Matrix (logistic_regression_baseline)
- **True Positives (Churn Detected)**: 296
- **False Positives (False Alarms)**: 275
- **False Negatives (Missed Churn)**: 78
- **True Negatives (Retained Correctly)**: 760

## 3. Top Risk Drivers (Global SHAP Feature Importance)
1. **Two-Year Long-Term Contract** (`Contract_Two year`): 7.43%
2. **Month-to-Month Contract** (`Contract_Month-to-month`): 6.94%
3. **Customer Tenure (0 months)** (`tenure`): 6.65%
4. **Recent Monthly Bill Spike** (`bill_shock_indicator`): 3.93%
5. **Fiber Optic High-Speed Internet** (`InternetService_Fiber optic`): 3.37%
6. **Paper Billing** (`PaperlessBilling_No`): 3.34%
7. **DSL Internet Service** (`InternetService_DSL`): 3.22%
8. **Monthly Charges ($0.00)** (`MonthlyCharges`): 3.08%
9. **Early Lifecycle Cohort (0-12m)** (`tenure_cohort_0-12m`): 3.03%
10. **Multiplelines No** (`MultipleLines_No`): 2.66%
