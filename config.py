"""
Конфигурация проекта
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Загружаем .env файл
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
else:
    load_dotenv()

# API Keys
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
PPLX_API_KEY: Optional[str] = os.getenv("PPLX_API_KEY")
KIE_API_KEY: Optional[str] = os.getenv("KIE_API_KEY")

# Настройки сервера
PORT: int = int(os.getenv("PORT", "8000"))
HOST: str = os.getenv("HOST", "0.0.0.0")
DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

# Настройки логирования
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_TO_FILE: bool = os.getenv("LOG_TO_FILE", "false").lower() == "true"
LOG_FILE: Optional[str] = os.getenv("LOG_FILE", "app.log")

# Настройки анализа
USE_GEMINI: bool = os.getenv("USE_GEMINI", "false").lower() == "true"
MAX_FRAMES_FOR_ANALYSIS: int = int(os.getenv("MAX_FRAMES_FOR_ANALYSIS", "8"))

# Настройки генерации промптов
DEFAULT_PLATFORM: str = os.getenv("DEFAULT_PLATFORM", "veo3")
DEFAULT_USE_CASE: str = os.getenv("DEFAULT_USE_CASE", "product_video")
AUTO_PERPLEXITY_POLISH: Optional[bool] = None
if os.getenv("AUTO_PERPLEXITY_POLISH"):
    AUTO_PERPLEXITY_POLISH = os.getenv("AUTO_PERPLEXITY_POLISH").lower() in ("true", "1", "yes")

# Настройки Veo 3.1
PREFER_KIE_API: Optional[bool] = None
if os.getenv("PREFER_KIE_API"):
    PREFER_KIE_API = os.getenv("PREFER_KIE_API").lower() in ("true", "1", "yes")
KIE_VEO_MODEL: str = os.getenv("KIE_VEO_MODEL", "veo3")

# Настройки S3/Media Uploader
S3_ENDPOINT: Optional[str] = os.getenv("S3_ENDPOINT")
S3_ACCESS_KEY: Optional[str] = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY: Optional[str] = os.getenv("S3_SECRET_KEY")
S3_BUCKET: Optional[str] = os.getenv("S3_BUCKET")
S3_REGION: Optional[str] = os.getenv("S3_REGION", "us-east-1")

# Валидация обязательных ключей
def validate_config() -> tuple[bool, list[str]]:
    """Проверяет конфигурацию и возвращает (is_valid, errors)"""
    errors = []

    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY не найден в .env файле")

    # GEMINI_API_KEY и KIE_API_KEY опциональны, но хотя бы один нужен для Veo 3.1
    if not GEMINI_API_KEY and not KIE_API_KEY:
        errors.append("Предупреждение: ни GEMINI_API_KEY, ни KIE_API_KEY не найдены - Veo 3.1 будет недоступен")

    return len(errors) == 0, errors

