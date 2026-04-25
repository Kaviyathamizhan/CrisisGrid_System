"""
message_utils.py
Utilities for JSON message validation, token counting, and truncation.
Used by CommandAgent, OversightLayer, and the Environment.
"""

import json
import re
from typing import Tuple, Optional

# Required fields every Command Agent message must contain
REQUIRED_FIELDS = ["intent", "zone", "resource", "priority"]
OPTIONAL_FIELDS = ["units", "reason", "confidence"]
VALID_RESOURCES = ["medicine", "food", "rescue", "water", "shelter"]
VALID_PRIORITIES = ["high", "medium", "low"]
VALID_INTENTS = ["allocate", "redirect", "hold"]

MAX_TOKENS = 50


def validate_message(msg: dict) -> Tuple[bool, Optional[str]]:
    """
    Validate a Command Agent message. Returns (is_valid, error_reason).
    A message is valid if all required fields are present and have correct types.

    Returns:
        (True, None)               — valid
        (False, "missing_fields")  — one or more required fields missing
        (False, "type_error")      — field present but wrong type
        (False, "invalid_value")   — field value not in allowed set
    """
    if not isinstance(msg, dict):
        return False, "not_a_dict"

    # Check required fields
    missing = [f for f in REQUIRED_FIELDS if f not in msg]
    if missing:
        return False, f"missing_fields:{','.join(missing)}"

    # Type checks
    if not isinstance(msg.get("zone"), int):
        return False, "type_error:zone must be int"

    if not isinstance(msg.get("units", 3), int):
        return False, "type_error:units must be int"

    # Value checks
    if msg.get("intent") not in VALID_INTENTS:
        return False, f"invalid_value:intent must be one of {VALID_INTENTS}"

    if msg.get("resource") not in VALID_RESOURCES:
        return False, f"invalid_value:resource must be one of {VALID_RESOURCES}"

    if msg.get("priority") not in VALID_PRIORITIES:
        return False, f"invalid_value:priority must be one of {VALID_PRIORITIES}"

    zone = msg.get("zone")
    if not (0 <= zone <= 24):
        return False, "invalid_value:zone must be 0-24"

    return True, None


def count_tokens(msg) -> int:
    """
    Count tokens in a message. Accepts dict or string.
    Uses whitespace-split tokenization (no heavy model needed).
    Each JSON key:value pair = ~2 tokens, punctuation counted separately.
    """
    if isinstance(msg, dict):
        msg_str = json.dumps(msg)
    elif isinstance(msg, str):
        msg_str = msg
    else:
        return 0

    # Tokenize: split on whitespace and common delimiters
    tokens = re.findall(r'[a-zA-Z0-9_\-\.]+|[{}:,"\[\]]', msg_str)
    return len(tokens)


def truncate_to_tokens(msg, max_tokens: int = MAX_TOKENS) -> str:
    """
    Hard-truncate a message string to max_tokens.
    The environment applies this before the Resource Agent sees the message.
    Returns the truncated JSON string (may be invalid JSON after truncation).
    """
    if isinstance(msg, dict):
        msg_str = json.dumps(msg)
    else:
        msg_str = str(msg)

    tokens = re.findall(r'[a-zA-Z0-9_\-\.]+|[{}:,"\[\]]', msg_str)
    if len(tokens) <= max_tokens:
        return msg_str

    # Reconstruct from truncated tokens
    truncated_tokens = tokens[:max_tokens]
    return " ".join(truncated_tokens)


def parse_message_safe(msg_str: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Safely parse a JSON string from the Command Agent.
    Returns (parsed_dict, None) on success or (None, error_reason) on failure.
    """
    try:
        parsed = json.loads(msg_str)
        if not isinstance(parsed, dict):
            return None, "not_a_dict"
        return parsed, None
    except json.JSONDecodeError as e:
        return None, f"json_decode_error:{str(e)}"


def make_valid_message(zone: int, resource: str = "medicine",
                       priority: str = "high", units: int = 3) -> dict:
    """Helper to construct a well-formed message. Used in baseline and tests."""
    return {
        "intent": "allocate",
        "zone": int(zone),
        "resource": resource,
        "priority": priority,
        "units": int(units)
    }
