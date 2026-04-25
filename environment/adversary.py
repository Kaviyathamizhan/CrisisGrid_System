"""
adversary.py
MinimalAdversary — injects severity spikes into the grid on a limited budget.
Simulates unexpected disaster escalations to test agent resilience.
"""

import random
from typing import List, Tuple


class MinimalAdversary:
    """
    Budget-limited adversarial injector.
    Every `inject_interval` steps, spikes one random cell's severity.
    Total spikes limited by `budget`.
    """

    def __init__(self, budget: int = 5, severity_boost: float = 0.3,
                 inject_interval: int = 10):
        """
        Args:
            budget:          Maximum number of severity spikes across the episode.
            severity_boost:  How much severity to add per spike (0.0–1.0 scale).
            inject_interval: Inject every N timesteps (e.g., 10 → steps 10, 20, 30, 40).
        """
        self.budget = budget
        self.severity_boost = severity_boost
        self.inject_interval = inject_interval
        self.spent = 0
        self.injection_log: List[dict] = []

    def inject(self, grid, timestep: int):
        """
        Possibly inject a severity spike into the grid.
        Mutates grid in-place. Returns the grid.

        Args:
            grid:     numpy array of shape [5][5][4] — the environment state grid.
            timestep: Current step number (1-indexed).

        Returns:
            grid (mutated in place)
        """
        if self.spent >= self.budget:
            return grid

        if timestep % self.inject_interval != 0:
            return grid

        if timestep == 0:
            return grid

        # Pick a random cell to spike
        i = random.randint(0, 4)
        j = random.randint(0, 4)

        old_severity = float(grid[i][j][1])
        new_severity = min(1.0, old_severity + self.severity_boost)
        grid[i][j][1] = new_severity

        self.spent += 1
        self.injection_log.append({
            "step": timestep,
            "cell": (i, j),
            "old_severity": round(old_severity, 3),
            "new_severity": round(new_severity, 3),
            "boost": self.severity_boost
        })

        return grid

    def get_injections(self) -> List[dict]:
        """Return the full log of adversarial injections this episode."""
        return self.injection_log.copy()

    def remaining_budget(self) -> int:
        return max(0, self.budget - self.spent)

    def reset(self):
        """Reset for a new episode."""
        self.spent = 0
        self.injection_log = []

    def summary(self) -> str:
        return (
            f"[Adversary] Budget={self.budget} | "
            f"Spent={self.spent} | "
            f"Injections={len(self.injection_log)}"
        )
