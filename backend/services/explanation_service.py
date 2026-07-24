"""
backend/services/explanation_service.py
Service for generating human-readable decision explanations for Command Agent actions.
"""

from typing import Optional


class ExplanationService:
    @staticmethod
    def generate_explanation(cmd_msg: Optional[dict], env_grid) -> str:
        """Format natural language reasoning string for a step."""
        if cmd_msg is None:
            return "The model output was not accepted by the validator; default resource action recorded."

        zone = cmd_msg.get("zone", 0)
        row = zone // 5
        col = zone % 5
        
        pop = int(env_grid[row][col][0])
        sev = float(env_grid[row][col][1])
        res = cmd_msg.get("resource", "water")
        units = cmd_msg.get("units", 3)
        intent = cmd_msg.get("intent", "allocate")

        if intent == "hold":
            return "Operational Hold: Standard maintenance action applied."

        explanation = f"Command Agent selected Zone {zone} (Row {row}, Col {col}). Severity is "
        if sev > 0.7:
            explanation += f"critical ({sev:.2f}) with {pop} citizens at risk. "
        else:
            explanation += f"{sev:.2f} with {pop} citizens. "

        explanation += f"Deploying {units} units of {res}."
        return explanation


explanation_service = ExplanationService()
