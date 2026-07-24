"""
backend/core/logging.py
Structured logging system for CrisisGrid FastAPI server.
Logs formatted messages to console and persists output to logs/backend.log.
"""

import os
import sys
import logging
from backend.core.config import settings

os.makedirs(settings.LOGS_DIR, exist_ok=True)
log_file_path = os.path.join(settings.LOGS_DIR, "backend.log")

# Define formatter
FORMATTER = logging.Formatter(
    fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(FORMATTER)
        logger.addHandler(console_handler)
        
        # File Handler
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(FORMATTER)
        logger.addHandler(file_handler)
        
    return logger
