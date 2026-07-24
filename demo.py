"""
demo.py
A/B demo episode generator: random agent vs trained agent.

Saves:
  - data/demo_random.json
  - data/demo_trained.json

Env vars:
  - CRISISGRID_CHECKPOINT_PATH: directory containing the LoRA adapter (PEFT)
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from environment.crisis_grid_env import CrisisGridEnv
from utils.message_utils import validate_message
from utils.agent_utils import (
    BASE_MODEL,
    repair_json,
    decode_action,
    build_prompt,
    get_clean_checkpoint_path,
    load_model_and_tokenizer,
    generate_one,
    random_valid_message
)


def run_episode_random(env: CrisisGridEnv, rng: np.random.RandomState) -> dict:
    obs_cmd, _ = env.reset()
    done = False
    traj: List[dict] = []
    while not done:
        msg = random_valid_message(rng)
        obs_cmd, reward, done, info = env.step(msg)
        traj.append(
            {
                "step": int(info.get("timestep", len(traj) + 1)),
                "msg": msg,
                "reward": float(reward),
                "survival_rate": float(info.get("survival_rate", 0.0)),
                "mean_severity": float(info.get("mean_severity", 0.0)),
                "max_severity": float(info.get("max_severity", 0.0)),
            }
        )
    return {
        "agent_type": "random",
        "final_survival": float(info.get("survival_rate", 0.0)),
        "final_reward": float(info.get("total_reward", 0.0)),
        "trajectory": traj,
    }


def run_episode_trained(env: CrisisGridEnv, model, tokenizer, rng: np.random.RandomState, max_new_tokens: int) -> dict:
    obs_cmd, _ = env.reset()
    done = False
    traj: List[dict] = []
    repair_triggers = 0
    fallbacks = 0

    while not done:
        prompt = build_prompt(obs_cmd)
        comp = generate_one(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        msg, diag = decode_action(comp, rng)
        repair_triggers += 1 if diag.get("json_repair_triggered") else 0
        fallbacks += 1 if diag.get("decode_fallback") else 0

        obs_cmd, reward, done, info = env.step(msg)
        traj.append(
            {
                "step": int(info.get("timestep", len(traj) + 1)),
                "msg": msg,
                "reward": float(reward),
                "survival_rate": float(info.get("survival_rate", 0.0)),
                "mean_severity": float(info.get("mean_severity", 0.0)),
                "max_severity": float(info.get("max_severity", 0.0)),
                "json_repair_triggered": bool(diag.get("json_repair_triggered")),
                "decode_fallback": bool(diag.get("decode_fallback")),
            }
        )

    return {
        "agent_type": "trained",
        "final_survival": float(info.get("survival_rate", 0.0)),
        "final_reward": float(info.get("total_reward", 0.0)),
        "json_repair_triggers": int(repair_triggers),
        "decode_fallbacks": int(fallbacks),
        "trajectory": traj,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", default=os.getenv("CRISISGRID_CHECKPOINT_PATH", ""))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-new-tokens", type=int, default=700)
    args = parser.parse_args()

    if not args.checkpoint_path:
        raise SystemExit(
            "Missing checkpoint path. Set CRISISGRID_CHECKPOINT_PATH or pass --checkpoint-path."
        )

    os.makedirs(os.path.join(REPO_ROOT, "data"), exist_ok=True)

    rng = np.random.RandomState(args.seed)

    env_a = CrisisGridEnv(seed=args.seed)
    random_ep = run_episode_random(env_a, rng)

    env_b = CrisisGridEnv(seed=args.seed)
    model, tokenizer = load_model_and_tokenizer(args.checkpoint_path)
    trained_ep = run_episode_trained(env_b, model, tokenizer, rng, max_new_tokens=args.max_new_tokens)

    out_a = os.path.join(REPO_ROOT, "data", "demo_random.json")
    out_b = os.path.join(REPO_ROOT, "data", "demo_trained.json")
    with open(out_a, "w", encoding="utf-8") as f:
        json.dump(random_ep, f, indent=2)
    with open(out_b, "w", encoding="utf-8") as f:
        json.dump(trained_ep, f, indent=2)

    delta = trained_ep["final_survival"] - random_ep["final_survival"]
    print(f"Random survival:  {random_ep['final_survival']:.1%}")
    print(f"Trained survival: {trained_ep['final_survival']:.1%}")
    print(f"Δ survival:       {delta:+.1%}")
    print(f"Saved: {out_a}")
    print(f"Saved: {out_b}")


if __name__ == "__main__":
    main()

