"""
Logging utilities
"""
import logging
import sys
from pathlib import Path
from typing import Optional

try:
    from config import LOG_LEVEL, LOG_TO_FILE, LOG_FILE, DEBUG
except ImportError:
    # Fallback if config not imported
    LOG_LEVEL = "INFO"
    LOG_TO_FILE = False
    LOG_FILE = "app.log"
    DEBUG = False

def setup_logging(name: str = "video_prompt_pipeline", log_file: Optional[str] = None) -> logging.Logger:
    """
    Sets up structured logging

    Args:
        name: Logger name
        log_file: Path to log file (optional)

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # If already configured, return existing
    if logger.handlers:
        return logger

    # Log format
    if DEBUG:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if enabled)
    if LOG_TO_FILE or log_file:
        file_path = log_file or Path(__file__).parent.parent / LOG_FILE
        file_handler = logging.FileHandler(file_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Global logger for use in project
logger = setup_logging()

