"""
backend/services/inference_service.py
Service for model initialization, checkpoint loading, and text generation.
Loads model ONCE during server startup and reuses it globally.
"""

import os
import torch
from typing import Tuple, Optional, Any
from backend.core.config import settings
from backend.core.logging import get_logger
from utils.agent_utils import load_model_and_tokenizer, generate_one

logger = get_logger("InferenceService")


class InferenceService:
    def __init__(self):
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._is_loaded: bool = False
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, checkpoint_path_override: Optional[str] = None) -> bool:
        """Load base Qwen2 model and LoRA adapter weights into memory."""
        if self._is_loaded and self._model is not None:
            logger.info("Model already loaded in memory.")
            return True

        lora_path = checkpoint_path_override or os.getenv("CRISISGRID_LORA_REPO", "").strip() or os.getenv("CRISISGRID_CHECKPOINT_PATH", "").strip()
        if not lora_path:
            local_ckpt = settings.DEFAULT_CHECKPOINT_PATH
            lora_path = local_ckpt if os.path.exists(local_ckpt) else settings.HF_LORA_REPO

        logger.info(f"Loading Qwen2 model and LoRA adapter from: {lora_path} (device: {self._device})")
        try:
            self._model, self._tokenizer = load_model_and_tokenizer(lora_path)
            self._is_loaded = True
            logger.info("Model and tokenizer loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load model from {lora_path}: {str(e)}", exc_info=True)
            self._is_loaded = False
            return False

    def generate(self, prompt: str, max_new_tokens: int = 600) -> str:
        """Run text generation given a prompt."""
        if not self._is_loaded or self._model is None or self._tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        return generate_one(
            model=self._model,
            tokenizer=self._tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens
        )

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def device(self) -> str:
        return self._device


# Global singleton instance
inference_service = InferenceService()
