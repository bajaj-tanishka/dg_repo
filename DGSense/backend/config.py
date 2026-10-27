"""
DGSense Backend Configuration — V2 Two-Way Translation
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

class Settings:
    HOST: str = os.getenv("DGSENSE_BACKEND_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("DGSENSE_BACKEND_PORT", "8000"))
    DEBUG: bool = os.getenv("DGSENSE_DEBUG", "False").lower() in ("true", "1", "t")

    # Artifacts directory path
    ARTIFACTS_DIR: Path = Path(os.getenv("DGSENSE_ARTIFACTS_DIR", str(BASE_DIR / "artifacts"))).resolve()

    # Multi-language extensibility configuration
    SIGN_LANGUAGE_CODE: str = os.getenv("DGSENSE_SIGN_LANG", "dgs-DE")
    SPOKEN_LANGUAGE_CODE: str = os.getenv("DGSENSE_SPOKEN_LANG", "de-DE")
    SECONDARY_SPOKEN_LANG: str = "en-US"

    # Dataset download size enforcement (under 5 GB default)
    DATASET_SIZE_LIMIT_MB: int = int(os.getenv("DGSENSE_DATASET_LIMIT_MB", "5120"))

    # Inference parameters
    MIN_CONFIDENCE: float = float(os.getenv("DGSENSE_MIN_CONFIDENCE", "0.5"))
    FEATURE_SCHEMA_VERSION: str = os.getenv("DGSENSE_FEATURE_SCHEMA_VERSION", "dgsense-v1")
    SEQUENCE_LENGTH: int = int(os.getenv("DGSENSE_SEQUENCE_LENGTH", "30"))
    FEATURE_DIM: int = int(os.getenv("DGSENSE_FEATURE_DIM", "144"))
    DEVICE: str = os.getenv("DGSENSE_DEVICE", "cpu")

    # Allowed CORS origins
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

settings = Settings()
