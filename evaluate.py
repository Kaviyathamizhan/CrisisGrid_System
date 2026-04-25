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

BASE_MODEL = "Qwen/Qwen2-1.5B-Instruct"


def _extract_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def repair_json(text: str) -> Tuple[Optional[Dict[str, Any]], bool, Optional[str]]:
    raw = (text or "").strip()

    # Strip markdown fences
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

    # Repair truncated JSON arrays (50-action lists cut mid-output)
    if raw.startswith("[") and not raw.endswith("]"):
        last_brace = raw.rfind("}")
        if last_brace != -1:
            raw = raw[:last_brace + 1] + "]"

    candidate = _extract_json_object(raw) or raw
    try:
        obj = json.loads(candidate)
        return (obj if isinstance(obj, dict) else None), False, None
    except Exception:
        pass

    for reason, cand in [
        ("single_to_double_quotes", candidate.replace("'", '"')),
        ("removed_trailing_commas", candidate.replace("'", '"').replace(",}", "}").replace(",]", "]")),
    ]:
        try:
            obj = json.loads(cand)
            return (obj if isinstance(obj, dict) else None), True, reason
        except Exception:
            continue
    return None, True, "unparseable_after_repairs"


def random_valid_message(rng: np.random.RandomState) -> Dict[str, Any]:
    return {
        "intent": "allocate",
        "zone": int(rng.randint(0, 25)),
        "resource": str(rng.choice(["medicine", "food", "rescue", "water", "shelter"])),
        "priority": str(rng.choice(["high", "medium", "low"])),
        "units": int(rng.randint(1, 6)),
    }


def decode_action(
    llm_text: str, rng: np.random.RandomState, log_repair: bool
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    diag: Dict[str, Any] = {
        "json_repair_triggered": False,
        "json_repair_reason": None,
        "decode_fallback": False,
        "validate_error": None,
    }

    msg, repaired, reason = repair_json(llm_text)
    if repaired:
        diag["json_repair_triggered"] = True
        diag["json_repair_reason"] = reason
        if log_repair:
            print(f"[json-repair] reason={reason}")

    if not isinstance(msg, dict):
        diag["decode_fallback"] = True
        return random_valid_message(rng), diag

    ok, err = validate_message(msg)
    if not ok:
        diag["validate_error"] = err
        diag["decode_fallback"] = True
        return random_valid_message(rng), diag

    return msg, diag


def build_prompt(obs: dict) -> str:
    try:
        from training.grpo_train import build_prompt as _bp  # type: ignore

        return _bp(obs)
    except Exception:
        timestep = obs.get("timestep", 0)
        grid = obs.get("grid", [])
        worst = []
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                sev = float(cell[1]) if len(cell) > 1 else 0.0
                worst.append((sev, i * 5 + j))
        worst.sort(reverse=True)
        top = [z for _, z in worst[:3]]
        return (
            "Output ONLY one valid JSON command with fields intent, zone, resource, priority, units.\n"
            f"Step={timestep} critical_zones={top}\nYour JSON command:"
        )


def load_model_and_tokenizer(checkpoint_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import os

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype="auto",
        device_map="auto"
    )
    if os.path.exists(checkpoint_path):
        print(f"Loaded checkpoint from: {checkpoint_path} (local)")
    else:
        print(f"Loaded checkpoint from: {checkpoint_path} (HuggingFace Hub)")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()
    return model, tokenizer


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return decoded[len(prompt) :].strip() if decoded.startswith(prompt) else decoded.strip()


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
    parser.add_argument("--max-new-tokens", type=int, default=1200)
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

