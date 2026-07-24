"""
evaluate.py
Run evaluation episodes for CrisisGrid using the same decode/generation config as training.

Outputs:
  - survival rate summary over N episodes (default 50)
  - best/worst episode indices + survival
  - baseline comparison (from logs/baseline_results.json if present)

Env vars:
  - CRISISGRID_CHECKPOINT_PATH: directory containing the LoRA adapter (PEFT)
  - CRISISGRID_BASELINE_PATH: optional override for baseline json path
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


def load_baseline(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", default=os.getenv("CRISISGRID_CHECKPOINT_PATH", ""))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--log-json-repairs", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint_path:
        raise SystemExit(
            "Missing checkpoint path. Set CRISISGRID_CHECKPOINT_PATH or pass --checkpoint-path."
        )

    env = CrisisGridEnv(seed=args.seed)
    rng = np.random.RandomState(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.checkpoint_path)

    survival_rates: List[float] = []
    episode_summaries: List[dict] = []
    repair_count = 0
    fallback_count = 0

    for ep in range(1, args.episodes + 1):
        obs_cmd, _ = env.reset()
        done = False
        steps = 0
        last_info = None

        while not done:
            prompt = build_prompt(obs_cmd)
            comp = generate_one(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            msg, diag = decode_action(comp, rng, log_repair=args.log_json_repairs)
            repair_count += 1 if diag.get("json_repair_triggered") else 0
            fallback_count += 1 if diag.get("decode_fallback") else 0

            obs_cmd, reward, done, info = env.step(msg)
            last_info = info
            steps += 1

        sr = float(last_info["survival_rate"]) if last_info else 0.0
        survival_rates.append(sr)
        episode_summaries.append(
            {
                "episode": ep,
                "steps": steps,
                "survival_rate": sr,
                "total_reward": float(last_info.get("total_reward", 0.0)) if last_info else 0.0,
                "schema_recovery_step": last_info.get("schema_recovery_step") if last_info else None,
                "total_tokens": int(last_info.get("total_tokens", 0)) if last_info else 0,
            }
        )
        print(f"Ep {ep:02d}/{args.episodes}: survival={sr:.1%} steps={steps}")

    mean_sr = float(np.mean(survival_rates)) if survival_rates else 0.0
    std_sr = float(np.std(survival_rates)) if survival_rates else 0.0
    best_i = int(np.argmax(survival_rates)) if survival_rates else -1
    worst_i = int(np.argmin(survival_rates)) if survival_rates else -1

    baseline_path = os.getenv(
        "CRISISGRID_BASELINE_PATH", os.path.join(REPO_ROOT, "logs", "baseline_results.json")
    )
    baseline = load_baseline(baseline_path)
    baseline_mean = float(baseline.get("survival_rate_mean")) if baseline else None

    print("\n" + "=" * 60)
    print("CrisisGrid v2 — EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Episodes:   {args.episodes}")
    print(f"Survival:   {mean_sr:.1%} ± {std_sr:.1%}")
    if best_i >= 0:
        print(f"Best ep:    #{best_i+1} ({survival_rates[best_i]:.1%})")
        print(f"Worst ep:   #{worst_i+1} ({survival_rates[worst_i]:.1%})")
    if baseline_mean is not None:
        delta = mean_sr - baseline_mean
        print(f"Baseline:   {baseline_mean:.1%} (from {baseline_path})")
        print(f"Δ vs base:  {delta:+.1%}")
    print(f"JSON repair triggers: {repair_count}")
    print(f"Decode fallbacks:     {fallback_count}")
    print("=" * 60)

    out_dir = os.path.join(REPO_ROOT, "logs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint_path,
                "episodes": args.episodes,
                "mean_survival": mean_sr,
                "std_survival": std_sr,
                "best_episode": episode_summaries[best_i] if best_i >= 0 else None,
                "worst_episode": episode_summaries[worst_i] if worst_i >= 0 else None,
                "baseline_mean": baseline_mean,
                "delta_vs_baseline": (mean_sr - baseline_mean) if baseline_mean is not None else None,
                "json_repair_triggers": repair_count,
                "decode_fallbacks": fallback_count,
                "episodes_detail": episode_summaries,
            },
            f,
            indent=2,
        )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

