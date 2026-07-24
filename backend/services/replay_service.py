"""
backend/services/replay_service.py
Service for retrieving pre-cached comparison runs for instant zero-latency replay.
"""

import os
import json
from typing import Dict, Any, Optional, List
from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger("ReplayService")


class ReplayService:
    def __init__(self):
        self._cache_path = settings.CACHE_FILE_PATH
        self._cache_data: Optional[Dict[str, Any]] = None
        self._load_cache()

    def _load_cache(self):
        if not os.path.exists(self._cache_path):
            logger.warning(f"Cached runs file not found at: {self._cache_path}")
            return

        try:
            with open(self._cache_path, "r", encoding="utf-8") as f:
                self._cache_data = json.load(f)
                logger.info(f"Loaded trajectory cache for seeds: {list(self._cache_data.keys())}")
        except Exception as e:
            logger.error(f"Failed to load cache file: {str(e)}", exc_info=True)

    def get_cached_trajectory(self, seed: int) -> Optional[Dict[str, Any]]:
        """Return the pre-cached trajectory data for a given seed."""
        if self._cache_data is None:
            self._load_cache()

        if self._cache_data and str(seed) in self._cache_data:
            return self._cache_data[str(seed)]
        return None

    def get_available_seeds(self) -> List[int]:
        """Return the list of seeds available in the trajectory cache."""
        if self._cache_data is None:
            self._load_cache()

        if self._cache_data:
            return [int(s) for s in self._cache_data.keys()]
        return [123, 42, 999]


# Global singleton instance
replay_service = ReplayService()
