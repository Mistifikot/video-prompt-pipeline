"""
Project configuration
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env file
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

# Server settings
PORT: int = int(os.getenv("PORT", "8000"))
HOST: str = os.getenv("HOST", "0.0.0.0")
DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

# Logging settings
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_TO_FILE: bool = os.getenv("LOG_TO_FILE", "false").lower() == "true"
LOG_FILE: Optional[str] = os.getenv("LOG_FILE", "app.log")

# Analysis settings
USE_GEMINI: bool = os.getenv("USE_GEMINI", "false").lower() == "true"
MAX_FRAMES_FOR_ANALYSIS: int = int(os.getenv("MAX_FRAMES_FOR_ANALYSIS", "8"))

# Prompt generation settings
DEFAULT_PLATFORM: str = os.getenv("DEFAULT_PLATFORM", "veo3")
DEFAULT_USE_CASE: str = os.getenv("DEFAULT_USE_CASE", "product_video")
AUTO_PERPLEXITY_POLISH: Optional[bool] = None
if os.getenv("AUTO_PERPLEXITY_POLISH"):
    AUTO_PERPLEXITY_POLISH = os.getenv("AUTO_PERPLEXITY_POLISH").lower() in ("true", "1", "yes")

# Veo 3.1 settings
PREFER_KIE_API: Optional[bool] = None
if os.getenv("PREFER_KIE_API"):
    PREFER_KIE_API = os.getenv("PREFER_KIE_API").lower() in ("true", "1", "yes")
KIE_VEO_MODEL: str = os.getenv("KIE_VEO_MODEL", "veo3")

# S3/Media Uploader settings
S3_ENDPOINT: Optional[str] = os.getenv("S3_ENDPOINT")
S3_ACCESS_KEY: Optional[str] = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY: Optional[str] = os.getenv("S3_SECRET_KEY")
S3_BUCKET: Optional[str] = os.getenv("S3_BUCKET")
S3_REGION: Optional[str] = os.getenv("S3_REGION", "us-east-1")

# Required keys validation
def validate_config() -> tuple[bool, list[str]]:
    """Checks configuration and returns (is_valid, errors)"""
    errors = []

    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY not found in .env file")

    # GEMINI_API_KEY and KIE_API_KEY are optional, but at least one is needed for Veo 3.1
    if not GEMINI_API_KEY and not KIE_API_KEY:
        errors.append("Warning: neither GEMINI_API_KEY nor KIE_API_KEY found - Veo 3.1 will be unavailable")

    return len(errors) == 0, errors

