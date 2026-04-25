"""
baseline.py
Run 1000 episodes with random (but structurally valid) messages.
Produces the baseline survival rate — your pitch anchor number.

Usage:
    python -m training.baseline
"""

import sys
import os
import numpy as np
import json
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.crisis_grid_env import CrisisGridEnv


# ─── Random Valid Message Generator ──────────────────────────────────────────
RESOURCES = ["medicine", "food", "rescue", "water", "shelter"]
PRIORITIES = ["high", "medium", "low"]
INTENTS = ["allocate", "redirect", "hold"]


def random_valid_message(rng: np.random.RandomState = None) -> dict:
    """Generate a random but structurally valid Command Agent message."""
    if rng is None:
        rng = np.random.RandomState()
    return {
        "intent": rng.choice(INTENTS),
        "zone": int(rng.randint(0, 25)),
        "resource": rng.choice(RESOURCES),
        "priority": rng.choice(PRIORITIES),
        "units": int(rng.randint(1, 6))
    }


def run_baseline(num_episodes: int = 1000, seed: int = 42, verbose: bool = True):
    """
    Run baseline evaluation.

    Args:
        num_episodes: Number of episodes to run.
        seed:         RNG seed for reproducibility.
        verbose:      Print progress every 100 episodes.

    Returns:
        dict with mean, std, min, max survival rate and full results array.
    """
    rng = np.random.RandomState(seed)
    env = CrisisGridEnv(seed=seed)

    survival_rates = []
    total_rewards = []
    comm_error_rates = []
    mean_severities = []

    t_start = time.time()

    for ep in range(num_episodes):
        obs_cmd, obs_res = env.reset()
        done = False

        while not done:
            msg = random_valid_message(rng)
            obs_cmd, reward, done, info = env.step(msg)

        survival_rates.append(info['survival_rate'])
        total_rewards.append(info['total_reward'])
        comm_error_rates.append(info.get('comm_error_rate', 0.0))
        mean_severities.append(info.get('mean_severity', 0.0))

        if verbose and (ep + 1) % 100 == 0:
            elapsed = time.time() - t_start
            mean_so_far = np.mean(survival_rates)
            print(
                f"  Episode {ep + 1:4d}/{num_episodes} | "
                f"Survival: {mean_so_far:.1%} | "
                f"Time: {elapsed:.1f}s"
            )

    elapsed = time.time() - t_start

    results = {
        "num_episodes": num_episodes,
        "seed": seed,
        "survival_rate_mean": float(np.mean(survival_rates)),
        "survival_rate_std": float(np.std(survival_rates)),
        "survival_rate_min": float(np.min(survival_rates)),
        "survival_rate_max": float(np.max(survival_rates)),
        "total_reward_mean": float(np.mean(total_rewards)),
        "total_reward_std": float(np.std(total_rewards)),
        "comm_error_rate_mean": float(np.mean(comm_error_rates)),
        "mean_severity_at_end": float(np.mean(mean_severities)),
        "elapsed_seconds": round(elapsed, 2),
    }

    return results


def main():
    print("=" * 60)
    print("  CrisisGrid v2 — BASELINE EVALUATION")
    print("  Running 1000 episodes with random valid messages...")
    print("=" * 60)
    print()

    results = run_baseline(num_episodes=1000, seed=42, verbose=True)

    print()
    print("=" * 60)
    print("  BASELINE RESULTS")
    print("=" * 60)
    print(f"  Survival Rate:  {results['survival_rate_mean']:.1%} ± {results['survival_rate_std']:.1%}")
    print(f"  Range:          [{results['survival_rate_min']:.1%}, {results['survival_rate_max']:.1%}]")
    print(f"  Total Reward:   {results['total_reward_mean']:.3f} ± {results['total_reward_std']:.3f}")
    print(f"  Comm Error:     {results['comm_error_rate_mean']:.2%}")
    print(f"  End Severity:   {results['mean_severity_at_end']:.3f}")
    print(f"  Time:           {results['elapsed_seconds']:.1f}s")
    print("=" * 60)
    print()
    print("  *** WRITE THIS NUMBER DOWN. IT IS YOUR PITCH ANCHOR. ***")
    print(f"  >>> BASELINE: {results['survival_rate_mean']:.1%} ± {results['survival_rate_std']:.1%}")
    print()

    # Save results to file
    results_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "logs", "baseline_results.json"
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
