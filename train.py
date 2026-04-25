"""
train.py
Scale/resume CrisisGrid GRPO training on HuggingFace A100.

Constraints:
  - Do not change environment/reward logic (reward comes from env.step()).
  - Load base model + LoRA adapter from checkpoint directory.
  - Use TRL GRPOTrainer.
  - Resume training from checkpoint.
  - max_completion_length = 700.
  - Enable Weights & Biases logging.

Env vars:
  - CRISISGRID_CHECKPOINT_PATH: directory containing the LoRA adapter (PEFT)
  - WANDB_PROJECT, WANDB_RUN_NAME (optional)
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Ensure local imports work when executed from repo root
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from environment.crisis_grid_env import CrisisGridEnv
from utils.message_utils import validate_message


BASE_MODEL = "unsloth/Qwen2-1.5B-Instruct-bnb-4bit"
REQUIRED_FIELDS = ("intent", "zone", "resource", "priority")


def _extract_json_object(text: str) -> Optional[str]:
    """Return the first {...} substring if present."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def repair_json(text: str) -> Tuple[Optional[Dict[str, Any]], bool, Optional[str]]:
    """
    Attempt to parse/repair JSON. Returns (dict_or_none, repaired?, reason).
    Minimal + safe: only a few deterministic transforms.
    """
    raw = text.strip()
    candidate = _extract_json_object(raw) or raw

    # 1) direct
    try:
        obj = json.loads(candidate)
        return (obj if isinstance(obj, dict) else None), False, None
    except Exception:
        pass

    # 2) strip code fences
    if candidate.startswith("```"):
        parts = candidate.split("```")
        if len(parts) >= 2:
            candidate2 = parts[1].strip()
            if candidate2.startswith("json"):
                candidate2 = candidate2[4:].strip()
            try:
                obj = json.loads(candidate2)
                return (obj if isinstance(obj, dict) else None), True, "stripped_code_fence"
            except Exception:
                candidate = candidate2

    # 3) common repairs: single quotes -> double quotes
    candidate3 = candidate.replace("'", '"')
    try:
        obj = json.loads(candidate3)
        return (obj if isinstance(obj, dict) else None), True, "single_to_double_quotes"
    except Exception:
        pass

    # 4) remove trailing commas before } or ]
    candidate4 = candidate3.replace(",}", "}").replace(",]", "]")
    try:
        obj = json.loads(candidate4)
        return (obj if isinstance(obj, dict) else None), True, "removed_trailing_commas"
    except Exception:
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
    llm_text: str,
    rng: np.random.RandomState,
    log_repair: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Safety check: always return a valid Command message dict.
    Also returns a small diagnostics dict for logging.
    """
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

    # Validate using existing project logic
    ok, err = validate_message(msg)
    if not ok:
        diag["validate_error"] = err
        diag["decode_fallback"] = True
        return random_valid_message(rng), diag

    # Ensure required structure (defensive)
    for k in REQUIRED_FIELDS:
        if k not in msg:
            diag["decode_fallback"] = True
            return random_valid_message(rng), diag

    return msg, diag


def build_prompt(obs: dict) -> str:
    """
    Build a compact prompt locally for the current environment step.
    """
    timestep = obs.get("timestep", 0)
    api_status = obs.get("api_status", "active")
    schema_version = obs.get("current_schema_version", 1)
    last_error = obs.get("last_error", None)
    grid = obs.get("grid", [])

    worst = []
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            sev = float(cell[1]) if len(cell) > 1 else 0.0
            pop = float(cell[0]) if len(cell) > 0 else 0.0
            worst.append((sev, i * 5 + j, pop))
    worst.sort(reverse=True)
    top3 = worst[:3]

    header = f"Step {timestep}/50 | API v{schema_version} | status={api_status}"
    if api_status == "deprecated" and last_error:
        header += f" | last_error={last_error}"
    critical = " | ".join([f"z{z}(sev={s:.2f},pop={int(p)})" for s, z, p in top3])

    return (
        "You are the Command Agent in a disaster response simulation.\n"
        "Output ONLY one valid JSON command.\n"
        "Required fields: intent, zone, resource, priority. Optional: units.\n"
        "Valid values: intent=allocate|redirect|hold, zone=0-24, "
        "resource=medicine|food|rescue|water|shelter, priority=high|medium|low.\n\n"
        f"{header}\nCritical zones: {critical}\n\nYour JSON command:"
    )


@dataclass
class TrainConfig:
    checkpoint_path: str
    episodes: int
    seed: int
    max_completion_length: int
    max_prompt_length: int
    lr: float
    batch_size: int
    grad_accum: int
    logging_steps: int
    save_steps: int
    output_dir: str
    sample_generation: bool
    log_json_repairs: bool


def _checkpoint_kind(checkpoint_path: str) -> str:
    """
    Detect whether checkpoint_path looks like:
      - a TRL/HF Trainer checkpoint (contains trainer_state.json), OR
      - an adapter-only LoRA checkpoint (contains adapter_config.json), OR
      - unknown.
    """
    if os.path.isdir(checkpoint_path):
        if os.path.exists(os.path.join(checkpoint_path, "trainer_state.json")):
            return "trl_checkpoint"
        if os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
            return "lora_adapter"
    return "unknown"


def get_clean_checkpoint_path(checkpoint_path: str):
    import os
    import json
    from huggingface_hub import snapshot_download
    
    # If it's already a local directory, assume it's clean
    if os.path.exists(checkpoint_path):
        print(f"[Checkpoint] Using local path: {checkpoint_path}")
        return checkpoint_path

    # Use unique cache dir to avoid corruption
    local_dir = "./patched_checkpoint_cache"

    if not os.path.exists(local_dir):
        print(f"[Checkpoint] Downloading from HF: {checkpoint_path}")
        snapshot_download(repo_id=checkpoint_path, local_dir=local_dir)

        config_path = os.path.join(local_dir, "adapter_config.json")

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                adapter_cfg = json.load(f)

            # Dynamically find all valid keys for pure PEFT
            import inspect
            from peft import LoraConfig
            valid_keys = set(inspect.signature(LoraConfig.__init__).parameters.keys())
            valid_keys.update(["peft_type", "auto_mapping", "base_model_name_or_path", "revision", "task_type", "inference_mode"])

            removed_keys = []
            for k in list(adapter_cfg.keys()):
                if k not in valid_keys:
                    adapter_cfg.pop(k)
                    removed_keys.append(k)

            with open(config_path, "w") as f:
                json.dump(adapter_cfg, f, indent=2)

            print(f"[Checkpoint] Cleaned keys: {removed_keys}")
        else:
            raise FileNotFoundError("adapter_config.json not found in checkpoint")
    else:
        print(f"[Checkpoint] Using cached patched checkpoint")

    return local_dir

def _load_model_and_tokenizer(checkpoint_path: str):
    # Base model via Unsloth (4-bit to avoid OOM)
    from unsloth import FastLanguageModel
    from peft import PeftModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )

    clean_path = get_clean_checkpoint_path(checkpoint_path)
    model = PeftModel.from_pretrained(model, clean_path, is_trainable=True)
    print("Loaded LoRA adapter successfully")
    model.eval()
    return model, tokenizer


def _sample_generate(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model = model.to(device)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", default=os.getenv("CRISISGRID_CHECKPOINT_PATH", ""))
    parser.add_argument("--episodes", type=int, default=int(os.getenv("CRISISGRID_EPISODES", "500")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("CRISISGRID_SEED", "42")))
    parser.add_argument("--max-completion-length", type=int, default=700)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=float(os.getenv("CRISISGRID_LR", "5e-5")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("CRISISGRID_BATCH_SIZE", "1")))
    parser.add_argument("--grad-accum", type=int, default=int(os.getenv("CRISISGRID_GRAD_ACCUM", "4")))
    parser.add_argument("--logging-steps", type=int, default=int(os.getenv("CRISISGRID_LOGGING_STEPS", "10")))
    parser.add_argument("--save-steps", type=int, default=int(os.getenv("CRISISGRID_SAVE_STEPS", "50")))
    parser.add_argument("--output-dir", default=os.getenv("CRISISGRID_OUTPUT_DIR", "checkpoints_a100"))
    parser.add_argument("--no-sample-generation", action="store_true")
    parser.add_argument("--log-json-repairs", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint_path:
        raise SystemExit(
            "Missing checkpoint path. Set CRISISGRID_CHECKPOINT_PATH or pass --checkpoint-path."
        )
    # We no longer strictly check os.path.exists here because it could be a HuggingFace Hub repo path (e.g., 'thebosskt/crisisgrid-lora')

    cfg = TrainConfig(
        checkpoint_path=args.checkpoint_path,
        episodes=args.episodes,
        seed=args.seed,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        sample_generation=not args.no_sample_generation,
        log_json_repairs=bool(args.log_json_repairs),
    )

    # WandB enabled by default (can be disabled via WANDB_MODE=disabled)
    import wandb

    wandb_project = os.getenv("WANDB_PROJECT", "crisisgrid")
    wandb_run_name = os.getenv("WANDB_RUN_NAME", "a100-resume")
    wandb.init(project=wandb_project, name=wandb_run_name)

    print(f"[train] base_model={BASE_MODEL}")
    print(f"[train] checkpoint_path={cfg.checkpoint_path}")
    print(f"[train] episodes={cfg.episodes} seed={cfg.seed} max_completion_length={cfg.max_completion_length}")
    ckpt_kind = _checkpoint_kind(cfg.checkpoint_path)
    print(f"[train] checkpoint_kind={ckpt_kind}")

    env = CrisisGridEnv(seed=cfg.seed)
    rng = np.random.RandomState(cfg.seed)

    model, tokenizer = _load_model_and_tokenizer(cfg.checkpoint_path)

    # Validate LoRA loading with one sample generation before training
    if cfg.sample_generation:
        obs_cmd, _ = env.reset()
        prompt = build_prompt(obs_cmd)
        sample = _sample_generate(model, tokenizer, prompt, max_new_tokens=160)
        msg, diag = decode_action(sample, rng, log_repair=cfg.log_json_repairs)
        print("[validate] sample_generation_ok=True")
        print(f"[validate] decoded_msg={msg}")
        if diag.get("json_repair_triggered"):
            print(f"[validate] json_repair_triggered reason={diag.get('json_repair_reason')}")
        if diag.get("decode_fallback"):
            print(f"[validate] decode_fallback=True validate_error={diag.get('validate_error')}")

    # TRL GRPOTrainer
    from trl import GRPOConfig, GRPOTrainer

    def reward_func(prompts, completions, **kwargs):
        rewards = []
        for comp in completions:
            msg, diag = decode_action(comp, rng, log_repair=cfg.log_json_repairs)
            _, r, done, info = env.step(msg)
            rewards.append(float(r))

            # lightweight logging
            wandb.log(
                {
                    "step_survival_rate": info.get("survival_rate", 0.0),
                    "step_comm_error_rate": info.get("comm_error_rate", 0.0),
                    "step_total_tokens": info.get("total_tokens", 0),
                    "json_repair_triggered": 1.0 if diag.get("json_repair_triggered") else 0.0,
                    "decode_fallback": 1.0 if diag.get("decode_fallback") else 0.0,
                }
            )

            if done:
                env.reset()
        return rewards

    grpo_cfg = GRPOConfig(
        output_dir=cfg.output_dir,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        report_to=["wandb"],
        max_completion_length=cfg.max_completion_length,
        # TRL safety: generation_batch_size must be divisible by num_generations.
        # Spaces/A100 defaults can mismatch (e.g. 4 vs 8) and hard-fail.
        num_generations=4,
        generation_batch_size=4,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_cfg,
        tokenizer=tokenizer,
        reward_funcs=reward_func,
    )

    # Resume training from checkpoint directory
    try:
        # If this is a full TRL checkpoint, this will resume optimizer/scheduler state too.
        # If it's adapter-only, TRL may raise; we fall back to starting a new Trainer run
        # but still "resume" model weights via the LoRA adapter load above.
        trainer.train(resume_from_checkpoint=cfg.checkpoint_path)
    except Exception as e:
        if ckpt_kind == "lora_adapter":
            print(f"[train] resume_from_checkpoint failed for adapter-only checkpoint: {e}")
            print("[train] continuing with LoRA-loaded weights (no trainer state to resume).")
            trainer.train()
        else:
            raise

    wandb.finish()
    print(f"[train] done. outputs in {cfg.output_dir}")


if __name__ == "__main__":
    main()

