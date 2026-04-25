"""
oversight.py
OversightLayer — tracks communication metrics, schema recovery, and logs all events.
Metrics logged to wandb: comm_error_rate, schema_recovery_step, token_usage_per_episode.
"""

from typing import Optional, List, Dict
import time


class OversightLayer:
    """
    Monitors all agent communication and environment events.
    Maintains per-episode and cumulative metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all per-episode metrics. Called at the start of each episode."""
        self.step_count: int = 0

        # Communication metrics
        self.total_messages: int = 0
        self.malformed_messages: int = 0
        self.refused_messages: int = 0
        self.default_actions: int = 0
        self.comm_error_rate: float = 0.0

        # Token tracking
        self.total_tokens_used_this_episode: int = 0
        self.tokens_per_step: List[int] = []

        # Schema recovery
        self.schema_drifted: bool = False
        self.schema_recovery_step: Optional[int] = None   # step at which agent first recovered

        # Event log
        self.flags: List[Dict] = []
        self.episode_start_time: float = time.time()

    def log_message(self, msg: dict, tokens_used: int, timestep: int):
        """Record a Command Agent message and its token count."""
        self.total_messages += 1
        self.step_count = timestep
        self.total_tokens_used_this_episode += tokens_used
        self.tokens_per_step.append(tokens_used)

    def log_malformed(self, reason: str, timestep: int):
        """Record a malformed message from the Command Agent."""
        self.malformed_messages += 1
        self._update_comm_error_rate()
        self.flags.append({
            "type": "malformed_message",
            "reason": reason,
            "step": timestep,
            "penalty": -0.5
        })

    def log_refuse(self, reason: str, timestep: int):
        """Record a Resource Agent refusal."""
        self.refused_messages += 1
        self.flags.append({
            "type": "resource_refused",
            "reason": reason,
            "step": timestep
        })

    def log_default_action(self, timestep: int):
        """Record when Resource Agent falls back to default (no command received)."""
        self.default_actions += 1
        self.flags.append({
            "type": "default_action",
            "step": timestep,
            "penalty": -0.3
        })

    def log_schema_drift(self, timestep: int):
        """Record that schema drift occurred at this step."""
        self.schema_drifted = True
        self.flags.append({
            "type": "schema_drift",
            "step": timestep,
            "endpoint_change": "POST /allocate → PATCH /distribution"
        })

    def log_schema_recovery(self, timestep: int):
        """Record the first step at which the agent successfully uses the new schema."""
        if self.schema_recovery_step is None:
            self.schema_recovery_step = timestep
            self.flags.append({
                "type": "schema_recovery",
                "step": timestep,
                "steps_to_recover": timestep - 25
            })

    def log_population_loss(self, cell: tuple, amount: float, timestep: int):
        """Log irreversible population loss events."""
        self.flags.append({
            "type": "population_loss",
            "cell": cell,
            "loss_pct": amount,
            "step": timestep
        })

    def _update_comm_error_rate(self):
        """Recompute comm_error_rate as ratio of malformed to total messages."""
        if self.total_messages > 0:
            self.comm_error_rate = self.malformed_messages / self.total_messages
        else:
            self.comm_error_rate = 0.0

    def get_episode_metrics(self) -> dict:
        """Return all metrics for this episode — passed to wandb and info dict."""
        self._update_comm_error_rate()
        return {
            "comm_error_rate":          self.comm_error_rate,
            "schema_recovery_step":     self.schema_recovery_step,
            "total_tokens":             self.total_tokens_used_this_episode,
            "malformed_messages":       self.malformed_messages,
            "refused_messages":         self.refused_messages,
            "default_actions":          self.default_actions,
            "total_messages":           self.total_messages,
            "schema_drifted":           self.schema_drifted,
            "flags_count":              len(self.flags),
            "episode_duration_s":       round(time.time() - self.episode_start_time, 2)
        }

    def get_flags(self) -> List[Dict]:
        return self.flags.copy()

    def summary(self) -> str:
        m = self.get_episode_metrics()
        return (
            f"[Oversight] Steps={self.step_count} | "
            f"MsgErrors={m['malformed_messages']} | "
            f"CommErrRate={m['comm_error_rate']:.2%} | "
            f"Tokens={m['total_tokens']} | "
            f"SchemaRecovery={m['schema_recovery_step']}"
        )
