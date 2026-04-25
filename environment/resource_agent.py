"""
resource_agent.py
Rule-based Resource Agent. Receives (possibly truncated) Command Agent message.
Executes valid commands. Refuses malformed ones. Falls back to default if no command.
"""

import random
from typing import Optional
from utils.message_utils import validate_message, VALID_RESOURCES


class ResourceAgent:
    """
    Rule-based Resource Agent covering rows 2–4 of the 5×5 grid (zones 10–24).
    Does NOT learn. Executes, refuses, or defaults each timestep.
    """

    ZONE_RANGE = list(range(10, 25))   # rows 2, 3, 4 → zones 10..24
    DEFAULT_RESOURCE = "food"
    DEFAULT_UNITS = 1

    def __init__(self):
        self.last_action = None
        self.last_reason = None
        self.action_history = []

    def act(self, cmd_msg: Optional[dict], timestep: int) -> dict:
        """
        Given a Command Agent message (or None), return an action dict.

        Returns one of:
          {"action": "allocate", "zone": int, "units": int, "resource": str}
          {"action": "refuse",   "reason": str}
          {"action": "default",  "zone": int, "units": int, "resource": str}
        """
        action = self._decide(cmd_msg, timestep)
        self.last_action = action
        self.action_history.append({"step": timestep, "action": action})
        return action

    def _decide(self, cmd_msg: Optional[dict], timestep: int) -> dict:
        # Case 1: No message received — default action
        if cmd_msg is None:
            return self._default_action()

        # Case 2: Message received — validate it
        is_valid, error_reason = validate_message(cmd_msg)

        if is_valid:
            return {
                "action":   "allocate",
                "zone":     cmd_msg["zone"],
                "units":    cmd_msg.get("units", 3),
                "resource": cmd_msg["resource"]
            }
        else:
            self.last_reason = error_reason
            return {
                "action": "refuse",
                "reason": "malformed_command",
                "detail": error_reason
            }

    def _default_action(self) -> dict:
        """Fallback when no command received. Penalised in reward."""
        zone = random.choice(self.ZONE_RANGE)
        return {
            "action":   "default",
            "zone":     zone,
            "units":    self.DEFAULT_UNITS,
            "resource": self.DEFAULT_RESOURCE
        }

    def reset(self):
        self.last_action = None
        self.last_reason = None
        self.action_history = []
