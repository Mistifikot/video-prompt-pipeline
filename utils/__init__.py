"""
Утилиты для логирования
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from config import LOG_LEVEL, LOG_TO_FILE, LOG_FILE, DEBUG

def setup_logging(name: str = "video_prompt_pipeline", log_file: Optional[str] = None) -> logging.Logger:
    """
    Настраивает структурированное логирование

    Args:
        name: Имя логгера
        log_file: Путь к файлу для логирования (опционально)

    Returns:
        Настроенный логгер
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Если уже настроен, возвращаем существующий
    if logger.handlers:
        return logger

    # Формат логов
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

    # Консольный handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Файловый handler (если включен)
    if LOG_TO_FILE or log_file:
        file_path = log_file or Path(__file__).parent / LOG_FILE
        file_handler = logging.FileHandler(file_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Глобальный логгер для использования в проекте
logger = setup_logging()

