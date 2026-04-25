"""
grid_viz.py
ASCII visualiser for the 5×5 CrisisGrid. Used for debugging and demo notebook.
Displays severity levels as colour-coded characters in the terminal.
"""

import numpy as np
from typing import Optional


# Severity thresholds → display characters
SEVERITY_CHARS = [
    (0.0, "·"),    # No severity
    (0.2, "░"),    # Low
    (0.5, "▒"),    # Medium
    (0.8, "▓"),    # High
    (0.9, "█"),    # Critical — population loss imminent
]

# ANSI colour codes for terminal output
COLORS = {
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "orange": "\033[38;5;208m",
    "red":    "\033[91m",
    "bright_red": "\033[1;91m",
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "dim":    "\033[2m",
}


def severity_to_char(severity: float) -> str:
    """Convert a severity value to a display character."""
    char = "·"
    for threshold, c in SEVERITY_CHARS:
        if severity >= threshold:
            char = c
    return char


def severity_to_color(severity: float) -> str:
    """Get ANSI colour code based on severity level."""
    if severity >= 0.9:
        return COLORS["bright_red"]
    elif severity >= 0.8:
        return COLORS["red"]
    elif severity >= 0.5:
        return COLORS["orange"]
    elif severity >= 0.2:
        return COLORS["yellow"]
    else:
        return COLORS["green"]


def render_grid(grid: np.ndarray, timestep: int = 0,
                show_population: bool = True,
                show_resources: bool = False,
                title: Optional[str] = None) -> str:
    """
    Render the 5×5 grid as a formatted ASCII string.

    Args:
        grid:            numpy array of shape [5][5][4], channels:
                         [0]=population, [1]=severity, [2]=resources, [3]=zone_id
        timestep:        Current step number for display.
        show_population: If True, show population count below each cell.
        show_resources:  If True, show resource allocation below each cell.
        title:           Optional title line above the grid.

    Returns:
        Formatted string ready for print().
    """
    lines = []

    # Header
    if title:
        lines.append(f"\n{COLORS['bold']}═══ {title} ═══{COLORS['reset']}")
    lines.append(f"{COLORS['dim']}Step: {timestep}/50{COLORS['reset']}")
    lines.append("")

    # Column headers
    col_header = "     " + "".join(f"  {j}  " for j in range(5))
    lines.append(col_header)
    lines.append("    " + "─" * 27)

    for i in range(5):
        # Zone label
        zone_label = "CMD" if i < 2 else "RES"
        row_chars = f" {i} {zone_label}│"

        for j in range(5):
            severity = float(grid[i][j][1])
            population = int(grid[i][j][0])
            resources = int(grid[i][j][2])

            color = severity_to_color(severity)
            char = severity_to_char(severity)

            # Cell display: severity character + population
            if show_population:
                cell = f"{color}{char}{population:3d}{COLORS['reset']}│"
            elif show_resources:
                cell = f"{color}{char} r{resources}{COLORS['reset']}│"
            else:
                cell = f"{color} {char}  {COLORS['reset']}│"

            row_chars += cell

        lines.append(row_chars)
        lines.append("    " + "─" * 27)

    # Legend
    lines.append("")
    lines.append(f"{COLORS['dim']}Legend: · safe  ░ low  ▒ medium  ▓ high  █ CRITICAL{COLORS['reset']}")
    lines.append(f"{COLORS['dim']}Zones:  CMD = Command Agent (rows 0-1) | RES = Resource Agent (rows 2-4){COLORS['reset']}")

    return "\n".join(lines)


def render_grid_simple(grid: np.ndarray, timestep: int = 0) -> str:
    """
    Simple no-colour grid render (for log files and non-ANSI terminals).
    Shows severity values as 2-decimal floats.
    """
    lines = [f"Step {timestep}/50", ""]
    lines.append("     " + "  ".join(f" C{j} " for j in range(5)))
    lines.append("    " + "-" * 30)

    for i in range(5):
        zone = "CMD" if i < 2 else "RES"
        cells = []
        for j in range(5):
            sev = grid[i][j][1]
            pop = int(grid[i][j][0])
            cells.append(f"{sev:.2f}")
        lines.append(f" {i} {zone}| " + " | ".join(cells) + " |")
        lines.append("    " + "-" * 30)

    return "\n".join(lines)


def print_episode_summary(info: dict):
    """Print a formatted episode summary from the info dict."""
    print(f"\n{'='*50}")
    print(f"  EPISODE SUMMARY")
    print(f"{'='*50}")
    print(f"  Survival Rate:     {info.get('survival_rate', 0):.1%}")
    print(f"  Total Reward:      {info.get('total_reward', 0):.3f}")
    print(f"  Population Lost:   {info.get('total_population_lost', 0)}")
    print(f"  Comm Error Rate:   {info.get('comm_error_rate', 0):.2%}")
    print(f"  Schema Recovery:   Step {info.get('schema_recovery_step', 'N/A')}")
    print(f"  Total Tokens:      {info.get('total_tokens', 0)}")
    print(f"{'='*50}\n")
