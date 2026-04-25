"""
schema_drift.py
Defines the deterministic schema drift event that fires at step 25.
At step 25: POST /allocate → 404. Agent must discover PATCH /distribution via GET /docs.
"""

from typing import Optional

# ─── Drift Event Registry ─────────────────────────────────────────────────────
SCHEMA_DRIFT_EVENTS = {
    25: {
        "old_endpoint":   "POST /allocate",
        "old_fields":     {"zone_id": int, "resource_type": str, "quantity": int},
        "new_endpoint":   "PATCH /distribution",
        "new_fields":     {"geo_hash": str, "priority_code": str, "units": int},
        "field_map":      {
            "zone_id":       "geo_hash",
            "resource_type": "priority_code",
            "quantity":      "units"
        },
        "discovery":      "GET /docs returns updated schema at any time"
    }
}

# Observation fields injected into Command Agent's obs from step 25 onward
DRIFT_OBS_FIELDS = {
    "api_status": "deprecated",
    "last_error": "POST /allocate returned 404",
    "docs_available": True,
    "current_schema_version": 2
}

PRE_DRIFT_OBS_FIELDS = {
    "api_status": "active",
    "last_error": None,
    "docs_available": True,
    "current_schema_version": 1
}

# ─── SchemaDrift Class ─────────────────────────────────────────────────────────
class SchemaDrift:
    """
    Manages the deterministic API schema mutation at step 25.
    Maintains state: which schema version is currently active.
    """

    def __init__(self):
        self.current_version: int = 1
        self.drifted: bool = False
        self.drift_step: int = 25
        self._recovery_detected: bool = False

    def tick(self, timestep: int) -> bool:
        """
        Called every step. Returns True if drift just fired this step.
        """
        if timestep == self.drift_step and not self.drifted:
            self.drifted = True
            self.current_version = 2
            return True
        return False

    def is_drifted(self) -> bool:
        return self.drifted

    def get_obs_fields(self) -> dict:
        """Returns the schema-related observation fields for the Command Agent."""
        return DRIFT_OBS_FIELDS.copy() if self.drifted else PRE_DRIFT_OBS_FIELDS.copy()

    def get_docs(self) -> dict:
        """Simulates GET /docs — returns the current active schema."""
        if not self.drifted:
            event = SCHEMA_DRIFT_EVENTS[25]
            return {
                "endpoint": event["old_endpoint"],
                "fields": event["old_fields"],
                "version": 1
            }
        else:
            event = SCHEMA_DRIFT_EVENTS[25]
            return {
                "endpoint": event["new_endpoint"],
                "fields": event["new_fields"],
                "version": 2
            }

    def validate_api_call(self, msg: dict, timestep: int) -> tuple:
        """
        Check if the agent is using the correct API endpoint for the current schema.
        Returns (is_valid, error_reason).

        Before step 25: expects old field names (zone_id, resource_type, quantity) OR
                        the standard command fields (intent, zone, resource, priority).
        After step 25:  expects new fields (geo_hash, priority_code, units) for
                        the distribution endpoint — OR standard command fields still work.
        """
        # Standard command message validation always passes through resource_agent
        # This is for tracking schema_recovery specifically
        if not self.drifted:
            return True, None

        # After drift: check if agent correctly uses new schema in any explicit API call
        new_fields = SCHEMA_DRIFT_EVENTS[25]["new_fields"]
        if any(f in msg for f in new_fields.keys()):
            self._recovery_detected = True
            return True, "new_schema_used"

        return True, None  # Standard messages still routed correctly

    def mark_recovery(self):
        """Explicitly mark that the agent has recovered from schema drift."""
        self._recovery_detected = True

    def is_recovered(self) -> bool:
        return self._recovery_detected

    def reset(self):
        self.current_version = 1
        self.drifted = False
        self._recovery_detected = False
