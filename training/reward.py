"""
reward.py
Complete reward model for CrisisGrid — 7 components + token efficiency penalty.

Components:
  1. Severity Reduction   — +1.0 per unit severity reduced (scaled from actual reduction)
  2. Population Preserved — +0.3 if no significant population loss this step
  3. Comm Valid           — +0.2 when Resource Agent executes (not refuse/default)
  4. Schema Recovery      — +2.0 one-time bonus when agent first uses new schema post-drift
  5. Malformed Message    — -0.5 per occurrence (missing required field)
  6. Default Action       — -0.3 when Resource Agent resorts to fallback
  7. Full Stabilisation   — +5.0 terminal bonus if all cells severity < 0.1

Token efficiency (Mercor):
  final_reward = base_reward * (1 / (1 + 0.001 * total_tokens_used_this_episode))

Note: Rewards are computed inline in crisis_grid_env.py step(). This module provides
standalone computation for GRPO training and evaluation where reward needs to be
computed separately from the environment step.
"""

import numpy as np
import json
from typing import Optional, Dict


# ─── Reward Constants ─────────────────────────────────────────────────────────
SEVERITY_REDUCTION_SCALE = 1.0      # +1.0 per unit severity reduced
POPULATION_PRESERVED_REWARD = 0.3   # +0.3 if pop >= 99% of initial
COMM_VALID_REWARD = 0.2             # +0.2 for valid communication
SCHEMA_RECOVERY_BONUS = 2.0        # +2.0 one-time for adapting to new schema
MALFORMED_PENALTY = -0.5           # -0.5 per malformed message
DEFAULT_PENALTY = -0.3             # -0.3 per default action
FULL_STABILISATION_BONUS = 5.0     # +5.0 terminal if all severity < 0.1
TOKEN_EFFICIENCY_FACTOR = 0.001    # Mercor token decay coefficient


def compute_reward(
    step_info: dict,
    severity_reduced: float = 0.0,
    comm_valid: bool = False,
    comm_malformed: bool = False,
    default_action: bool = False,
    schema_recovered_this_step: bool = False,
    population_preserved: bool = True,
    all_stable: bool = False,
    total_tokens_episode: int = 0,
    is_terminal: bool = False,
) -> dict:
    """
    Compute the full reward for a single timestep.

    Returns:
        dict with keys: total, components (breakdown), token_penalty_factor
    """
    components = {}

    # 1. Severity Reduction
    components["severity_reduction"] = severity_reduced * SEVERITY_REDUCTION_SCALE

    # 2. Population Preserved
    components["population_preserved"] = (
        POPULATION_PRESERVED_REWARD if population_preserved else 0.0
    )

    # 3. Communication Valid
    components["comm_valid"] = COMM_VALID_REWARD if comm_valid else 0.0

    # 4. Schema Recovery (one-time)
    components["schema_recovery"] = (
        SCHEMA_RECOVERY_BONUS if schema_recovered_this_step else 0.0
    )

    # 5. Malformed Message Penalty
    components["malformed_penalty"] = MALFORMED_PENALTY if comm_malformed else 0.0

    # 6. Default Action Penalty
    components["default_penalty"] = DEFAULT_PENALTY if default_action else 0.0

    # 7. Terminal — Full Stabilisation Bonus
    components["full_stabilisation"] = (
        FULL_STABILISATION_BONUS if (is_terminal and all_stable) else 0.0
    )

    # Base reward (sum of all components)
    base_reward = sum(components.values())

    # Token efficiency penalty (Mercor)
    token_factor = 1.0 / (1.0 + TOKEN_EFFICIENCY_FACTOR * total_tokens_episode)
    final_reward = base_reward * token_factor

    return {
        "total": round(final_reward, 6),
        "base_reward": round(base_reward, 6),
        "components": {k: round(v, 6) for k, v in components.items()},
        "token_penalty_factor": round(token_factor, 6),
        "total_tokens": total_tokens_episode,
    }


def compute_grpo_reward(response_str: str, env, timestep_info: dict = None) -> float:
    """
    Simplified reward function for GRPO training.
    Called by grpo_train.py for each LLM response.

    Args:
        response_str: Raw string output from the LLM.
        env:          CrisisGridEnv instance (mid-episode).
        timestep_info: Optional info dict from previous step.

    Returns:
        Scalar reward value.
    """
    try:
        msg = json.loads(response_str)
        if not isinstance(msg, dict):
            return -1.0

        _, reward, done, info = env.step(msg)
        return float(reward)

    except json.JSONDecodeError:
        return -1.0  # Unparseable JSON — hard penalty

    except Exception:
        return -0.5  # Other errors — soft penalty


def format_reward_summary(reward_info: dict) -> str:
    """Pretty-print a reward breakdown for debugging."""
    lines = ["┌── Reward Breakdown ──"]
    for name, value in reward_info["components"].items():
        sign = "+" if value >= 0 else ""
        lines.append(f"│  {name:25s} {sign}{value:.4f}")
    lines.append(f"│  {'─' * 35}")
    lines.append(f"│  {'base_reward':25s} = {reward_info['base_reward']:.4f}")
    lines.append(f"│  {'token_factor':25s} × {reward_info['token_penalty_factor']:.4f}")
    lines.append(f"│  {'FINAL':25s} = {reward_info['total']:.4f}")
    lines.append("└──────────────────────")
    return "\n".join(lines)
