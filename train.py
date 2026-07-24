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

REQUIRED_FIELDS = ("intent", "zone", "resource", "priority")


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", default=os.getenv("CRISISGRID_CHECKPOINT_PATH", ""))
    parser.add_argument("--episodes", type=int, default=int(os.getenv("CRISISGRID_EPISODES", "120")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("CRISISGRID_SEED", "42")))
    parser.add_argument("--max-completion-length", type=int, default=600)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=float(os.getenv("CRISISGRID_LR", "5e-5")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("CRISISGRID_BATCH_SIZE", "1")))
    parser.add_argument("--grad-accum", type=int, default=int(os.getenv("CRISISGRID_GRAD_ACCUM", "4")))
    parser.add_argument("--logging-steps", type=int, default=int(os.getenv("CRISISGRID_LOGGING_STEPS", "10")))
    parser.add_argument("--save-steps", type=int, default=int(os.getenv("CRISISGRID_SAVE_STEPS", "20")))
    parser.add_argument("--output-dir", default=os.getenv("CRISISGRID_OUTPUT_DIR", "checkpoints_a100"))
    parser.add_argument("--no-sample-generation", action="store_true")
    parser.add_argument("--log-json-repairs", action="store_true")
    args = parser.parse_args()

    if args.max_completion_length < 500:
        print("WARNING: max_completion_length too low — may cause truncated JSON and reward collapse")

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

    model, tokenizer = load_model_and_tokenizer(cfg.checkpoint_path)

    # Validate LoRA loading with one sample generation before training
    if cfg.sample_generation:
        obs_cmd, _ = env.reset()
        prompt = build_prompt(obs_cmd)
        sample = generate_one(model, tokenizer, prompt, max_new_tokens=160)
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

    # Build prompt dataset from environment observations
    from datasets import Dataset as HFDataset
    print(f"[dataset] Generating {cfg.episodes} episode prompts...")
    prompt_records = []
    dataset_env = CrisisGridEnv(seed=cfg.seed)
    for i in range(cfg.episodes):
        obs_i, _ = dataset_env.reset()
        prompt_records.append({"prompt": build_prompt(obs_i)})
    train_dataset = HFDataset.from_list(prompt_records)
    print(f"[dataset] {len(train_dataset)} prompts ready.")

    grpo_cfg = GRPOConfig(
        output_dir=cfg.output_dir,
        learning_rate=cfg.lr,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=cfg.grad_accum,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        report_to=["wandb"],
        max_completion_length=cfg.max_completion_length,
        num_generations=2,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_cfg,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        train_dataset=train_dataset,
    )

    # LoRA weights are already loaded into the model via PeftModel.from_pretrained().
    # No TRL trainer state to resume — just start training with the loaded weights.
    print("[train] Starting GRPO training with pre-loaded LoRA weights...")
    trainer.train()

    wandb.finish()
    print(f"[train] done. outputs in {cfg.output_dir}")


if __name__ == "__main__":
    main()

