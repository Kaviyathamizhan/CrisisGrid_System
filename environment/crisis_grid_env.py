"""
crisis_grid_env.py
Main CrisisGrid Environment — 5×5 disaster response grid, 50-timestep episodes.
Two agents: Command (RL-trained) + Resource (rule-based) coordinate under constraints.

Key mechanics:
  - Severity increases +0.02/step/cell (flood spreading)
  - Resource allocation reduces severity: N units → severity -= N * 0.08
  - Population loss: severity > 0.9 for 3 consecutive steps → -10% population (irreversible)
  - Message truncation: Command Agent messages hard-capped at 50 tokens
  - Schema drift: at step 25, POST /allocate → PATCH /distribution (deterministic)
  - Episode ends at step 50 or when all cells severity < 0.1 (rare full stabilisation)
"""

import numpy as np
import json
from typing import Tuple, Optional, Dict, Any

from environment.resource_agent import ResourceAgent
from environment.schema_drift import SchemaDrift
from environment.oversight import OversightLayer
from environment.adversary import MinimalAdversary
from utils.message_utils import (
    validate_message, count_tokens, truncate_to_tokens
)


class CrisisGridEnv:
    """
    CrisisGrid Multi-Agent Disaster Response Environment.

    State space: float[5][5][4] per cell:
        [0] population  — int, survivors (0–100). Permanent loss when severity > 0.9 for 3+ steps.
        [1] severity    — float 0.0–1.0. Disaster intensity. +0.02/step. Reduced by allocation.
        [2] resources   — int. Units currently assigned to cell.
        [3] zone_id     — int. 0 = Command Agent zone (rows 0–1), 1 = Resource Agent zone (rows 2–4).
    """

    # ─── Constants ────────────────────────────────────────────────────────────
    GRID_SIZE = 5
    MAX_STEPS = 50
    CHANNELS = 4       # population, severity, resources, zone_id

    # Physics
    SEVERITY_INCREASE_PER_STEP = 0.012   # Adjusted to lift survival > 15%
    SEVERITY_REDUCTION_PER_UNIT = 0.06
    POPULATION_LOSS_THRESHOLD = 0.9
    CONSECUTIVE_STEPS_FOR_LOSS = 3
    POPULATION_LOSS_RATE = 0.10
    DAMAGE_THRESHOLD = 0.7              # CHANGE 2: severity above this increments damage
    DAMAGE_STEPS_FOR_LOSS = 2           # CHANGE 2: 2 accumulated damage steps → pop loss
    DAMAGE_POP_MULTIPLIER = 0.70        # CHANGE 2: population *= 0.7 per damage event

    # Initial ranges
    INIT_POPULATION_RANGE = (40, 100)
    INIT_SEVERITY_RANGE = (0.3, 0.7)    # CHANGE 1: was (0.1,0.4)

    def __init__(self, adversary_budget: int = 8, seed: Optional[int] = None):
        """
        Args:
            adversary_budget: Max adversarial severity spikes per episode.
            seed:             RNG seed for reproducibility.
        """
        self.rng = np.random.RandomState(seed)

        # Components
        self.resource_agent = ResourceAgent()
        self.schema_drift = SchemaDrift()
        self.oversight = OversightLayer()
        self.adversary = MinimalAdversary(budget=5,
                                         severity_boost=0.25,
                                         inject_interval=10)

        # State
        self.grid = None
        self.timestep = 0
        self.done = False
        self.consecutive_critical = None  # [5][5] counter for severity > 0.9
        self.initial_total_population = 0
        self.total_reward = 0.0
        self.episode_trajectory = []      # For demo recording

        # Metrics
        self.last_survival_rate = 0.0

    # ─── Reset ────────────────────────────────────────────────────────────────
    def reset(self) -> Tuple[dict, dict]:
        """
        Reset environment to initial state.

        Returns:
            (obs_cmd, obs_res) — observations for Command Agent and Resource Agent.
        """
        self.timestep = 0
        self.done = False
        self.total_reward = 0.0
        self.episode_trajectory = []

        # Initialize grid: [5][5][4]
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE, self.CHANNELS),
                             dtype=np.float64)

        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                self.grid[i][j][0] = self.rng.randint(
                    self.INIT_POPULATION_RANGE[0], self.INIT_POPULATION_RANGE[1] + 1
                )
                self.grid[i][j][1] = self.rng.uniform(
                    self.INIT_SEVERITY_RANGE[0], self.INIT_SEVERITY_RANGE[1]
                )
                self.grid[i][j][2] = 0   # No resources initially
                self.grid[i][j][3] = 0 if i < 2 else 1   # Zone assignment

        self.initial_total_population = float(np.sum(self.grid[:, :, 0]))
        self.consecutive_critical = np.zeros((self.GRID_SIZE, self.GRID_SIZE),
                                             dtype=np.int32)
        self.damage_counter = np.zeros((self.GRID_SIZE, self.GRID_SIZE),
                                       dtype=np.int32)  # CHANGE 2: cumulative damage

        # Reset components
        self.resource_agent.reset()
        self.schema_drift.reset()
        self.oversight.reset()
        self.adversary.reset()
        self.damage_counter[:] = 0

        return self._observe_cmd(), self._observe_res()

    # ─── Step ─────────────────────────────────────────────────────────────────
    def step(self, cmd_msg: Optional[dict]) -> Tuple[dict, float, bool, dict]:
        """
        Process one timestep.

        Args:
            cmd_msg: Command Agent's message dict (or None).

        Returns:
            (obs_cmd, reward, done, info)
        """
        if self.done:
            raise RuntimeError("Episode already done. Call reset().")

        self.timestep += 1
        step_reward = 0.0

        # ── 1. Truncate Command Agent message to 50 tokens ────────────────
        truncated_msg = None
        tokens_used = 0
        if cmd_msg is not None:
            msg_str = json.dumps(cmd_msg)
            tokens_used = count_tokens(msg_str)
            truncated_str = truncate_to_tokens(msg_str)

            # Try to parse truncated message back
            try:
                truncated_msg = json.loads(truncated_str)
            except json.JSONDecodeError:
                truncated_msg = None  # Truncation broke JSON — treated as malformed

            self.oversight.log_message(cmd_msg, tokens_used, self.timestep)

        # ── 2. Validate message ───────────────────────────────────────────
        is_valid = False
        if truncated_msg is not None:
            is_valid, error = validate_message(truncated_msg)
            if not is_valid:
                self.oversight.log_malformed(error, self.timestep)
                step_reward -= 0.5   # Malformed penalty
        else:
            if cmd_msg is not None:
                self.oversight.log_malformed("truncation_broke_json", self.timestep)
                step_reward -= 0.5

        # ── 3. Resource Agent decides ─────────────────────────────────────
        res_action = self.resource_agent.act(
            truncated_msg if is_valid else None,
            self.timestep
        )

        if res_action["action"] == "refuse":
            self.oversight.log_refuse(
                res_action.get("reason", "unknown"), self.timestep
            )
            # Malformed penalty already applied above

        elif res_action["action"] == "default":
            self.oversight.log_default_action(self.timestep)
            step_reward -= 0.3   # Default action penalty

        elif res_action["action"] == "allocate":
            step_reward += 0.2   # Valid communication reward

        # ── 4. Apply resource allocation ──────────────────────────────────
        if res_action["action"] in ("allocate", "default"):
            zone_flat = res_action.get("zone", 0)
            units = res_action.get("units", 1)
            step_reward += self._apply_allocation(zone_flat, units)

        # ── 5. Update severity across entire grid (+0.02 per cell) ────────
        self._update_severity()

        # ── 6. Adversary injection ────────────────────────────────────────
        self.grid = self.adversary.inject(self.grid, self.timestep)

        # ── 7. Check population loss (severity > 0.9 for 3+ steps) ───────
        self._check_population_loss()

        # ── 7b. Cumulative damage loss (severity > 0.7 for 2+ steps) ─────
        self._check_damage_loss()

        # ── 8. Schema drift at step 25 ────────────────────────────────────
        drift_fired = self.schema_drift.tick(self.timestep)
        if drift_fired:
            self.oversight.log_schema_drift(self.timestep)

        # Check schema recovery (if agent adapts after drift)
        if self.schema_drift.is_drifted() and truncated_msg is not None:
            self.schema_drift.validate_api_call(truncated_msg, self.timestep)
            if self.schema_drift.is_recovered():
                self.oversight.log_schema_recovery(self.timestep)
                step_reward += 2.0   # One-time schema recovery bonus

        # ── 9. Population preservation reward ─────────────────────────────
        current_pop = float(np.sum(self.grid[:, :, 0]))
        if current_pop >= self.initial_total_population * 0.99:
            step_reward += 0.3   # No significant loss this step

        # ── 10. Token efficiency penalty (Mercor) ─────────────────────────
        total_tokens_ep = self.oversight.total_tokens_used_this_episode
        step_reward = step_reward * (1.0 / (1.0 + 0.001 * total_tokens_ep))

        # ── 11. Check termination ─────────────────────────────────────────
        terminal_reward = 0.0
        all_stable = np.all(self.grid[:, :, 1] < 0.1)

        if self.timestep >= self.MAX_STEPS:
            self.done = True
        elif all_stable:
            self.done = True
            terminal_reward = 5.0   # Full stabilisation bonus (rare)

        step_reward += terminal_reward
        self.total_reward += step_reward

        # ── 12. Record trajectory step ────────────────────────────────────
        self.episode_trajectory.append({
            "step": self.timestep,
            "cmd_msg": cmd_msg,
            "res_action": res_action,
            "reward": round(step_reward, 4),
            "grid_snapshot": self.grid.copy().tolist(),
        })

        # ── 13. Compute info dict ────────────────────────────────────────
        current_total_pop = float(np.sum(self.grid[:, :, 0]))
        survival_rate = current_total_pop / self.initial_total_population if self.initial_total_population > 0 else 0.0
        self.last_survival_rate = survival_rate

        info = {
            "survival_rate": survival_rate,
            "total_reward": self.total_reward,
            "total_population": current_total_pop,
            "initial_population": self.initial_total_population,
            "total_population_lost": self.initial_total_population - current_total_pop,
            "mean_severity": float(np.mean(self.grid[:, :, 1])),
            "max_severity": float(np.max(self.grid[:, :, 1])),
            "all_stable": all_stable,
            "timestep": self.timestep,
            **self.oversight.get_episode_metrics()
        }

        return self._observe_cmd(), step_reward, self.done, info

    # ─── Internal Mechanics ───────────────────────────────────────────────────

    def _update_severity(self):
        """Severity increases +0.02 per cell per step (flood spreading)."""
        self.grid[:, :, 1] += self.SEVERITY_INCREASE_PER_STEP

        # Clear resources from previous step (resources consumed each step)
        self.grid[:, :, 2] = 0

        self.grid[:, :, 1] = np.clip(self.grid[:, :, 1], 0.0, 1.0)

    def _apply_allocation(self, zone_flat: int, units: int) -> float:
        """
        Apply resource allocation to a cell. Reduces severity.
        zone_flat is 0–24 mapping to grid[row][col].

        Returns severity reduction reward component.
        """
        row = zone_flat // self.GRID_SIZE
        col = zone_flat % self.GRID_SIZE

        # Clamp to valid range
        row = max(0, min(row, self.GRID_SIZE - 1))
        col = max(0, min(col, self.GRID_SIZE - 1))
        units = max(0, min(units, 10))  # Cap at 10 units

        old_severity = float(self.grid[row][col][1])
        reduction = units * self.SEVERITY_REDUCTION_PER_UNIT
        new_severity = old_severity - reduction
        
        # FIX 3: Clamp Severity Bounds
        new_severity = max(new_severity, 0.0)
        new_severity = min(new_severity, 1.0)
        
        self.grid[row][col][1] = new_severity
        self.grid[row][col][2] = units  # Set current resources (not accumulate)

        # Reward for severity reduction: +0.1 per 0.1 reduced
        severity_reduced = old_severity - new_severity
        return severity_reduced * 1.0   # +0.1 per 0.1 reduced → scale factor 1.0

    def _check_population_loss(self):
        """
        If severity > 0.9 for 3 consecutive steps → population -= 10%.
        Counter resets after each loss event. Permanent and irreversible.
        """
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                if self.grid[i][j][1] > self.POPULATION_LOSS_THRESHOLD:
                    self.consecutive_critical[i][j] += 1
                else:
                    self.consecutive_critical[i][j] = 0

                if self.consecutive_critical[i][j] >= self.CONSECUTIVE_STEPS_FOR_LOSS:
                    pop = self.grid[i][j][0]
                    loss = pop * self.POPULATION_LOSS_RATE
                    self.grid[i][j][0] = max(0.0, pop - loss)
                    self.consecutive_critical[i][j] = 0

                    self.oversight.log_population_loss(
                        cell=(i, j),
                        amount=self.POPULATION_LOSS_RATE,
                        timestep=self.timestep
                    )

    def _check_damage_loss(self):
        """
        CHANGE 2: Cumulative damage counter.
        If severity > 0.7, damage_counter[i][j] += 1.
        If damage_counter >= 2, population *= 0.7.
        Counter does NOT reset — accumulates across steps.
        """
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                if self.grid[i][j][1] > self.DAMAGE_THRESHOLD:
                    self.damage_counter[i][j] += 1
                else:
                    self.damage_counter[i][j] = max(0, self.damage_counter[i][j] - 1)

                if self.damage_counter[i][j] >= self.DAMAGE_STEPS_FOR_LOSS:
                    self.grid[i][j][0] = max(
                        0.0,
                        self.grid[i][j][0] * self.DAMAGE_POP_MULTIPLIER
                    )
                    self.damage_counter[i][j] = 0  # Reset after damage applied

    # ─── Observations ────────────────────────────────────────────────────────

    def _observe_cmd(self) -> dict:
        """
        Full observation for the Command Agent.
        Includes entire grid + schema drift status fields.
        """
        obs = {
            "grid": self.grid.tolist(),
            "timestep": self.timestep,
            "max_steps": self.MAX_STEPS,
            "zone": "command",
            "visible_rows": list(range(self.GRID_SIZE)),  # Full visibility
        }
        # Add schema drift observation fields
        obs.update(self.schema_drift.get_obs_fields())
        return obs

    def _observe_res(self) -> dict:
        """
        Partial observation for the Resource Agent (rows 2–4 only).
        """
        partial_grid = self.grid[2:, :, :].tolist()  # Rows 2, 3, 4 only
        return {
            "grid": partial_grid,
            "timestep": self.timestep,
            "max_steps": self.MAX_STEPS,
            "zone": "resource",
            "visible_rows": [2, 3, 4],
        }

    # ─── Utilities ────────────────────────────────────────────────────────────

    def get_trajectory(self) -> list:
        """Return the full trajectory of this episode (for demo recording)."""
        return self.episode_trajectory

    def get_docs(self) -> dict:
        """Simulate GET /docs — agent can call this at any timestep."""
        return self.schema_drift.get_docs()

    def get_grid_summary(self) -> dict:
        """Quick summary of grid state for debugging."""
        return {
            "mean_severity": float(np.mean(self.grid[:, :, 1])),
            "max_severity": float(np.max(self.grid[:, :, 1])),
            "total_population": float(np.sum(self.grid[:, :, 0])),
            "cells_critical": int(np.sum(self.grid[:, :, 1] > 0.9)),
            "cells_stable": int(np.sum(self.grid[:, :, 1] < 0.1)),
        }

    def render(self) -> str:
        """Render grid as ASCII string for terminal debugging."""
        from utils.grid_viz import render_grid
        return render_grid(self.grid, self.timestep, title="CrisisGrid")
