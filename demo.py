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

BASE_MODEL = "unsloth/Qwen2-1.5B-Instruct-bnb-4bit"


def random_valid_message(rng: np.random.RandomState) -> Dict[str, Any]:
    return {
        "intent": "allocate",
        "zone": int(rng.randint(0, 25)),
        "resource": str(rng.choice(["medicine", "food", "rescue", "water", "shelter"])),
        "priority": str(rng.choice(["high", "medium", "low"])),
        "units": int(rng.randint(1, 6)),
    }


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

    candidate2 = raw.replace("'", '"').replace(",}", "}").replace(",]", "]")
    try:
        obj = json.loads(raw)
        return (obj if isinstance(obj, dict) else None), False, None
    except Exception:
        pass
    try:
        obj = json.loads(candidate2)
        return (obj if isinstance(obj, dict) else None), True, "repaired_basic"
    except Exception:
        return None, True, "unparseable_after_repairs"


def decode_action(llm_text: str, rng: np.random.RandomState) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
        return "Output ONLY one valid JSON command with keys intent, zone, resource, priority, units:"


def load_model_and_tokenizer(checkpoint_path: str):
    from unsloth import FastLanguageModel
    from peft import PeftModel
    import os

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )
    if os.path.exists(checkpoint_path):
        print(f"Loaded checkpoint from: {checkpoint_path} (local)")
    else:
        print(f"Loaded checkpoint from: {checkpoint_path} (HuggingFace Hub)")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    FastLanguageModel.for_inference(model)
    model.eval()
    return model, tokenizer


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return decoded[len(prompt) :].strip() if decoded.startswith(prompt) else decoded.strip()


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

