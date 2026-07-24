"""
backend/core/config.py
Application settings for CrisisGrid FastAPI server using Pydantic settings.
"""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "CrisisGrid AI Operations Platform"
    VERSION: str = "2.0.0"
    API_PREFIX: str = "/api"
    
    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # Paths
    REPO_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    LOGS_DIR: str = os.path.join(REPO_ROOT, "logs")
    CACHE_FILE_PATH: str = os.path.join(REPO_ROOT, "data", "cached_comparison_runs.json")
    DEFAULT_CHECKPOINT_PATH: str = os.path.join(REPO_ROOT, "checkpoints", "checkpoint-180")
    HF_LORA_REPO: str = "thebosskt/crisisgrid-lora"
    
    # CORS Allowed Origins
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://localhost:8000",
        "*"
    ]

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
