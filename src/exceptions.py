class ChurnPredictionException(Exception):
    """Base exception for the Churn Intelligence Platform."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

class DataLoadError(ChurnPredictionException):
    """Raised when data loading fails."""
    pass

class DataValidationError(ChurnPredictionException):
    """Raised when dataset fails schema, type, or constraint validation."""
    pass

class ModelTrainingError(ChurnPredictionException):
    """Raised when model training or pipeline construction fails."""
    pass

class ModelNotLoadedError(ChurnPredictionException):
    """Raised when requested model artifacts cannot be loaded."""
    pass

class InferenceError(ChurnPredictionException):
    """Raised when prediction or explanation generation fails."""
    pass
