import logging
import sys
from pathlib import Path
from config.settings import settings

def setup_logger(name: str = "churn_intelligence") -> logging.Logger:
    """Set up and configure a structured logger for the application."""
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    try:
        settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = settings.LOGS_DIR / "app.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not initialize file log handler: {e}")

    logger.propagate = False
    return logger

logger = setup_logger()
