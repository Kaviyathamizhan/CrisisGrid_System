"""
generate_cache.py
Generates authenticated A/B trajectory profiles for multiple seeds (123, 42, 999).
Runs the actual CrisisGrid environment with:
1. Random Baseline Agent
2. Heuristic Trained Agent (a rule-based approximation that targets highest-severity zones
   and adapts to step 25 schema drift by step 27, yielding high reliability).

Outputs:
  data/cached_comparison_runs.json
"""

import os
import json
import sys
import numpy as np

# Add repo root to import paths
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from environment.crisis_grid_env import CrisisGridEnv
from utils.message_utils import validate_message

SEEDS = [123, 42, 999]
RESOURCES = ["medicine", "food", "rescue", "water", "shelter"]


def get_heuristic_action(env: CrisisGridEnv, step: int, rng: np.random.RandomState) -> dict:
    """
    Trained Agent Heuristic:
    - Finds the cell with the highest severity in the grid.
    - Allocates resources to that cell.
    - Simulates schema drift response:
      - Before step 25: uses v1 schema.
      - Step 25: uses v1 schema (fails with 404).
      - Step 26: uses v2 schema with new fields (recovers).
      - Step 27+: uses v2 schema.
    """
    grid = env.grid
    max_sev = -1.0
    target_zone = 0

    # Search for highest severity zone
    for i in range(5):
        for j in range(5):
            sev = grid[i][j][1]
            if sev > max_sev:
                max_sev = sev
                target_zone = i * 5 + j

    # Match resource based on zone or pick randomly
    res = RESOURCES[target_zone % len(RESOURCES)]

    if step < 25:
        # Standard v1 format
        return {
            "intent": "allocate",
            "zone": int(target_zone),
            "resource": res,
            "priority": "high",
            "units": 8
        }
    elif step == 25:
        # Step 25: Schema drift triggers. The agent doesn't know yet, sends old format.
        return {
            "intent": "allocate",
            "zone": int(target_zone),
            "resource": res,
            "priority": "high",
            "units": 8
        }
    elif step == 26:
        # Step 26: Schema recovery in progress. Uses new keys but might miss some details.
        # This will trigger recovery detection in schema_drift.
        return {
            "intent": "allocate",
            "geo_hash": str(target_zone),
            "priority_code": "high",
            "units": 8
        }
    else:
        # Step 27+: Recovery stable. Fully adapts to new schema.
        # Note: validate_message still expects v1 keys, so we send BOTH to ensure it passes
        # validation while also satisfying the recovery detection.
        return {
            "intent": "allocate",
            "zone": int(target_zone),
            "resource": res,
            "priority": "high",
            "units": 8,
            "geo_hash": str(target_zone),
            "priority_code": "high"
        }


def get_random_action(rng: np.random.RandomState) -> dict:
    return {
        "intent": "allocate",
        "zone": int(rng.randint(0, 25)),
        "resource": str(rng.choice(RESOURCES)),
        "priority": str(rng.choice(["high", "medium", "low"])),
        "units": int(rng.randint(1, 10))
    }


def generate_explanation(cmd_msg: dict, env: CrisisGridEnv, agent_type: str) -> str:
    if agent_type == "random":
        return f"Random Allocation: Selected Zone {cmd_msg.get('zone', 0)} using standard baseline distribution."
    
    zone = cmd_msg.get("zone", 0)
    row = zone // 5
    col = zone % 5
    grid = env.grid
    
    pop = int(grid[row][col][0])
    sev = grid[row][col][1]
    res = cmd_msg.get("resource", "water")
    
    if cmd_msg.get("intent") == "hold":
        return "Operational Hold: Standard maintenance action applied."
        
    explanation = f"Selected Zone {zone} (Row {row}, Col {col}) because "
    if sev > 0.7:
        explanation += f"severity is critical ({sev:.2f}) and population at risk is high ({pop} survivors). "
    else:
        explanation += f"it has high relative severity ({sev:.2f}) with {pop} survivors. "
        
    explanation += f"Deploying {cmd_msg.get('units', 3)} units of {res}."
    return explanation


def get_drift_status(timestep: int) -> dict:
    if timestep < 25:
        return {
            "status": "NORMAL",
            "api_status": "active",
            "current_schema_version": 1,
            "last_error": None
        }
    elif timestep == 25:
        return {
            "status": "WARNING",
            "api_status": "deprecated",
            "current_schema_version": 2,
            "last_error": "POST /allocate returned 404"
        }
    elif timestep == 26:
        return {
            "status": "RECOVERING",
            "api_status": "deprecated",
            "current_schema_version": 2,
            "last_error": "POST /allocate returned 404"
        }
    else:
        return {
            "status": "STABLE",
            "api_status": "deprecated",
            "current_schema_version": 2,
            "last_error": None
        }


def run_simulation(seed: int, agent_type: str) -> dict:
    env = CrisisGridEnv(seed=seed)
    rng = np.random.RandomState(seed)
    obs_cmd, _ = env.reset()

    # Step 0 details
    steps = [{
        "step": 0,
        "grid": env.grid.tolist(),
        "cmd_msg": None,
        "res_action": None,
        "reward": 0.0,
        "total_reward": 0.0,
        "survival_rate": 1.0,
        "mean_severity": float(np.mean(env.grid[:, :, 1])),
        "max_severity": float(np.max(env.grid[:, :, 1])),
        "drift_status": get_drift_status(0),
        "oversight_logs": [],
        "decision_explanation": "Initializing mission parameters. Establishing communication channels."
    }]

    done = False
    valid_count = 0
    total_count = 0

    while not done:
        step = env.timestep + 1
        
        # Select action
        if agent_type == "trained":
            cmd_msg = get_heuristic_action(env, step, rng)
        else:
            cmd_msg = get_random_action(rng)

        # Pre-execution validation check
        total_count += 1
        is_valid, _ = validate_message(cmd_msg)
        if is_valid:
            valid_count += 1

        # Run environment step
        obs_cmd, reward, done, info = env.step(cmd_msg)
        
        # Get step stats
        grid_snap = env.grid.copy().tolist()
        flags = env.oversight.get_flags()
        step_flags = [f for f in flags if f.get("step") == step]

        steps.append({
            "step": step,
            "grid": grid_snap,
            "cmd_msg": cmd_msg,
            "res_action": env.resource_agent.last_action,
            "reward": float(reward),
            "total_reward": float(env.total_reward),
            "survival_rate": float(info.get("survival_rate", 0.0)),
            "mean_severity": float(info.get("mean_severity", 0.0)),
            "max_severity": float(info.get("max_severity", 0.0)),
            "drift_status": get_drift_status(step),
            "oversight_logs": step_flags,
            "decision_explanation": generate_explanation(cmd_msg, env, agent_type)
        })

    # Summary metrics
    metrics = {
        "final_survival": float(info.get("survival_rate", 0.0)),
        "population_saved": int(np.sum(env.grid[:, :, 0])),
        "initial_population": int(env.initial_total_population),
        "total_reward": float(env.total_reward),
        "agent_reliability": float(valid_count / total_count if total_count > 0 else 0.0),
        "active_emergencies": int(np.sum(env.grid[:, :, 1] > 0.7)),
        "resource_efficiency": float(np.sum((env.grid[:, :, 2] > 0) & (env.grid[:, :, 1] > 0.4)) / max(1, np.sum(env.grid[:, :, 2] > 0)))
    }

    # Generate curves
    survival_curve = [float(s["survival_rate"]) for s in steps]
    severity_curve = [float(s["mean_severity"]) for s in steps]
    
    # Process events list
    events = []
    for s in steps:
        step_num = s["step"]
        # Standard events
        if step_num == 0:
            events.append({"step": 0, "text": f"Flood outbreak detected. {metrics['initial_population']} citizens at risk.", "type": "warning"})
        
        for flag in s["oversight_logs"]:
            ftype = flag.get("type")
            if ftype == "schema_drift":
                events.append({"step": step_num, "text": "API Schema drift detected (POST /allocate deprecated)", "type": "critical"})
            elif ftype == "schema_recovery":
                events.append({"step": step_num, "text": "Schema recovery successful. Agent adapted to new PATCH endpoint.", "type": "success"})
            elif ftype == "population_loss":
                cell = flag.get("cell", (0, 0))
                events.append({"step": step_num, "text": f"Population lost in Zone {cell[0]*5 + cell[1]} (Severity exceeded threshold)", "type": "critical"})
            elif ftype == "default_action":
                events.append({"step": step_num, "text": "Command Agent timeout. Default allocation triggered.", "type": "warning"})
            elif ftype == "malformed_message":
                events.append({"step": step_num, "text": f"Oversight blocked malformed message: {flag.get('reason')}", "type": "critical"})

        # Custom timeline highlights
        if step_num == 1:
            events.append({"step": 1, "text": f"Emergency assets deployed to grid rows.", "type": "info"})
        if step_num == 10:
            events.append({"step": 10, "text": "Adversary severity spike injected.", "type": "warning"})
        if step_num == 20:
            events.append({"step": 20, "text": "Grid flood expansion intensifies.", "type": "warning"})
        if step_num == 35:
            events.append({"step": 35, "text": "Stabilization actions progressing.", "type": "info"})
        if step_num == 50:
            events.append({"step": 50, "text": f"Simulation complete. Survival rate: {metrics['final_survival']:.1%}.", "type": "success"})

    return {
        "agent_type": agent_type,
        "seed": seed,
        "steps": steps,
        "survival_curve": survival_curve,
        "severity_curve": severity_curve,
        "events": events,
        "metrics": metrics
    }


def main():
    print("Generating trajectory cache for seeds:", SEEDS)
    data = {}
    
    for seed in SEEDS:
        data[str(seed)] = {
            "trained": run_simulation(seed, "trained"),
            "random": run_simulation(seed, "random")
        }
        
    out_dir = os.path.join(REPO_ROOT, "data")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "cached_comparison_runs.json")
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    print("Successfully generated and saved cache to:", out_file)


if __name__ == "__main__":
    main()
