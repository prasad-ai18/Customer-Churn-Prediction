from .trainer import train_all_models, ModelTrainingResult, ChurnPipeline
from .evaluator import evaluate_predictions, EvaluationMetrics
from .explainer import ChurnExplainer

evaluate_model = evaluate_predictions

__all__ = [
    "train_all_models",
    "ModelTrainingResult",
    "ChurnPipeline",
    "evaluate_predictions",
    "evaluate_model",
    "EvaluationMetrics",
    "ChurnExplainer"
]
