"""
evaluate.py
Evaluation script — generate comparison episode data for demo and pitch.
Produces demo_random.json and demo_trained.json for demo.ipynb A/B player.

Usage:
    python -m training.evaluate
    python -m training.evaluate --checkpoint checkpoints/ep_500
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.crisis_grid_env import CrisisGridEnv

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=None, help="Path to trained model checkpoint")
parser.add_argument("--episodes",   type=int, default=10)
parser.add_argument("--seed",       type=int, default=123)
args = parser.parse_args()


def record_episode(agent_type: str, checkpoint: str = None, seed: int = 0) -> dict:
    """
    Record a full episode trajectory.

    Args:
        agent_type:  "random" or "trained"
        checkpoint:  Path to trained model checkpoint (only for "trained")
        seed:        RNG seed for reproducibility.

    Returns:
        dict with full episode metadata and step-by-step trajectory.
    """
    env = CrisisGridEnv(seed=seed)
    rng = np.random.RandomState(seed)

    model, tokenizer = None, None
    if agent_type == "trained" and checkpoint:
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=checkpoint,
                max_seq_length=512,
                load_in_4bit=True
            )
            print(f"  [eval] Loaded trained model from {checkpoint}")
        except Exception as e:
            print(f"  [eval] Could not load model: {e}. Using random agent.")
            agent_type = "random"

    obs_cmd, obs_res = env.reset()
    done = False
    trajectory = []
    step_num = 0

    while not done:
        step_num += 1
        if agent_type == "trained" and model:
            from training.grpo_train import build_prompt
            import torch
            prompt = build_prompt(obs_cmd)
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=80, do_sample=False)
            resp = tokenizer.decode(output[0], skip_special_tokens=True)
            resp = resp[len(prompt):].strip()
            try:
                msg = json.loads(resp)
            except json.JSONDecodeError:
                msg = _random_msg(rng)
        else:
            msg = _random_msg(rng)

        obs_cmd, reward, done, info = env.step(msg)

        trajectory.append({
            "step":          step_num,
            "msg":           msg,
            "reward":        round(reward, 4),
            "survival_rate": round(info["survival_rate"], 4),
            "mean_severity": round(info["mean_severity"], 4),
            "max_severity":  round(info["max_severity"], 4),
            "schema_status": obs_cmd.get("api_status", "active"),
            "population_lost": round(info["total_population_lost"], 1),
        })

    return {
        "agent_type":        agent_type,
        "checkpoint":        checkpoint,
        "seed":              seed,
        "total_steps":       step_num,
        "final_survival":    round(info["survival_rate"], 4),
        "final_reward":      round(info["total_reward"], 4),
        "comm_error_rate":   round(info.get("comm_error_rate", 0), 4),
        "schema_recovery_step": info.get("schema_recovery_step"),
        "total_tokens":      info.get("total_tokens", 0),
        "trajectory":        trajectory,
    }


def _random_msg(rng):
    return {
        "intent":   rng.choice(["allocate", "redirect", "hold"]),
        "zone":     int(rng.randint(0, 25)),
        "resource": rng.choice(["medicine", "food", "rescue", "water", "shelter"]),
        "priority": rng.choice(["high", "medium", "low"]),
        "units":    int(rng.randint(1, 6))
    }


def main():
    os.makedirs("data", exist_ok=True)

    print("=" * 55)
    print("  CrisisGrid v2 — Episode Evaluation")
    print("=" * 55)

    # Episode A: Random agent
    print("\n[1/2] Recording RANDOM agent episode...")
    ep_random = record_episode("random", seed=args.seed)
    out_a = "data/demo_random.json"
    with open(out_a, "w") as f:
        json.dump(ep_random, f, indent=2)
    print(f"  Survival: {ep_random['final_survival']:.1%} | "
          f"Tokens: {ep_random['total_tokens']} | "
          f"Schema Recovery: {ep_random['schema_recovery_step']}")
    print(f"  Saved: {out_a}")

    # Episode B: Trained agent
    print("\n[2/2] Recording TRAINED agent episode...")
    ep_trained = record_episode("trained", checkpoint=args.checkpoint, seed=args.seed)
    out_b = "data/demo_trained.json"
    with open(out_b, "w") as f:
        json.dump(ep_trained, f, indent=2)
    print(f"  Survival: {ep_trained['final_survival']:.1%} | "
          f"Tokens: {ep_trained['total_tokens']} | "
          f"Schema Recovery: {ep_trained['schema_recovery_step']}")
    print(f"  Saved: {out_b}")

    # Delta
    delta = ep_trained["final_survival"] - ep_random["final_survival"]
    print(f"\n  ✅ IMPROVEMENT: {delta:+.1%} survival vs random baseline")
    print("=" * 55)


if __name__ == "__main__":
    main()
