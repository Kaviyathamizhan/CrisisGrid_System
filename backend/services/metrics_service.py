"""
backend/services/metrics_service.py
Service for computing executive summary metrics and KPI data from environment runs.
"""

import numpy as np
from typing import Dict, Any, List


class MetricsService:
    @staticmethod
    def compute_summary(
        env_grid: np.ndarray,
        initial_pop: float,
        total_reward: float,
        info: dict,
        valid_count: int,
        total_count: int
    ) -> Dict[str, Any]:
        """Compute structured summary metrics dictionary."""
        survival_rate = float(info.get("survival_rate", 0.0))
        saved_pop = int(np.sum(env_grid[:, :, 0]))
        
        # Calculate active emergencies (severity > 0.7)
        emergencies = int(np.sum(env_grid[:, :, 1] > 0.7))
        
        # Calculate resource deployment efficiency
        total_deploys = np.sum(env_grid[:, :, 2] > 0)
        effective_deploys = np.sum((env_grid[:, :, 2] > 0) & (env_grid[:, :, 1] > 0.4))
        resource_eff = float(effective_deploys / max(1, total_deploys))

        return {
            "final_survival": survival_rate,
            "population_saved": saved_pop,
            "initial_population": int(initial_pop),
            "total_reward": float(total_reward),
            "agent_reliability": float(valid_count / total_count if total_count > 0 else 1.0),
            "active_emergencies": emergencies,
            "resource_efficiency": resource_eff
        }


metrics_service = MetricsService()
