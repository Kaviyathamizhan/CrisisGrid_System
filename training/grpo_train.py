"""
grpo_train.py  —  CrisisGrid Real GRPO Training
=================================================
Architecture:
  ONE FULL EPISODE  =  ONE TRAINING SAMPLE
  Model generates a JSON array of up to 50 actions in a single completion.
  Environment runs the full trajectory.
  Reward  =  final episode survival_rate.

Usage (Colab / A100):
    python -m training.grpo_train
    python -m training.grpo_train --model unsloth/Qwen2-7B-Instruct-bnb-8bit --continue-from /path/to/ep_500

Usage (dry-run, no GPU):
    python -m training.grpo_train --dry-run
"""

import os, sys, json, argparse, csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",            default="Qwen/Qwen2-1.5B-Instruct")
parser.add_argument("--episodes",         type=int, default=200)
parser.add_argument("--max-seq-len",      type=int, default=2048)
parser.add_argument("--lora-r",           type=int, default=16)
parser.add_argument("--lora-alpha",       type=int, default=16)
parser.add_argument("--lr",               type=float, default=2e-5)
parser.add_argument("--batch-size",       type=int, default=2)
parser.add_argument("--grad-accum",       type=int, default=4)
parser.add_argument("--checkpoint-every", type=int, default=10)
parser.add_argument("--continue-from",    default=None)
parser.add_argument("--ckpt-dir",         default="checkpoints")
parser.add_argument("--wandb-project",    default="crisisgrid")
parser.add_argument("--wandb-run",        default="colab-grpo-run")
parser.add_argument("--dry-run",          action="store_true",
                    help="Validate env + reward pipeline only (no GPU needed)")
args = parser.parse_args()

# ── Environment ───────────────────────────────────────────────────────────────
from environment.crisis_grid_env import CrisisGridEnv

# ── DRY-RUN: validate env + reward pipeline without GPU ──────────────────────
if args.dry_run:
    print("[dry-run] Validating environment + reward pipeline...")
    rng = np.random.RandomState(0)
    env = CrisisGridEnv(seed=0)
    for ep in range(3):
        obs, _ = env.reset()
        done, step, total_r = False, 0, 0.0
        while not done:
            msg = {"intent": "allocate",
                   "zone": int(rng.randint(0, 25)),
                   "resource": rng.choice(["medicine", "food", "rescue"]),
                   "priority": "high",
                   "units": int(rng.randint(1, 6))}
            obs, r, done, info = env.step(msg)
            total_r += r; step += 1
        print(f"  ep {ep+1}: steps={step} total_reward={total_r:.3f} "
              f"survival={info['survival_rate']:.1%}")
    print("[dry-run] PASSED — environment and reward are functional.")
    sys.exit(0)

# ── WandB ─────────────────────────────────────────────────────────────────────
try:
    import wandb
    wandb.init(project=args.wandb_project, name=args.wandb_run)
    USE_WANDB = True
    print(f"[wandb] project={args.wandb_project}  run={args.wandb_run}")
except Exception:
    print("[wandb] Not available — logging to CSV only.")
    USE_WANDB = False

# ── Model (Pure HF + PEFT) ───────────────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

load_path = args.continue_from or args.model
tokenizer = AutoTokenizer.from_pretrained(load_path)
model = AutoModelForCausalLM.from_pretrained(
    load_path,
    torch_dtype="auto",
    device_map="auto"
)

if not args.continue_from:
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()

print(f"[model] Loaded: {load_path}")

# ── Prompt builder ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are the Command Agent in CrisisGrid, a 5×5 disaster response grid.
Each cell has: [population, severity (0–1), resources, zone_id].
You MUST output EXACTLY a JSON array of 50 action objects — one per timestep.

Each action object MUST have these exact fields:
  "intent"   : "allocate" | "redirect" | "hold"
  "zone"     : integer 0–24  (row-major: row*5+col)
  "resource" : "medicine" | "food" | "rescue" | "water" | "shelter"
  "priority" : "high" | "medium" | "low"
  "units"    : integer 1–10

STRATEGY:
- Target zones with severity > 0.7 first (they cause population loss)
- At step 25: POST /allocate returns 404. Use PATCH /distribution instead.
  Remap fields: zone→geo_hash, resource→priority_code, quantity→units
- Keep messages under 50 tokens or they are truncated
- Valid JSON only — invalid messages get -0.5 penalty

Output ONLY a valid JSON array. No markdown. No explanation.
Example (first 2 of 50):
[{"intent":"allocate","zone":12,"resource":"medicine","priority":"high","units":5},
 {"intent":"allocate","zone":7,"resource":"rescue","priority":"high","units":4}, ...]
"""

def build_episode_prompt(grid_state: np.ndarray) -> str:
    """Build a single-prompt encoding of the full initial grid state."""
    top_cells = []
    for i in range(5):
        for j in range(5):
            sev = grid_state[i][j][1]
            pop = grid_state[i][j][0]
            top_cells.append((sev, i * 5 + j, int(pop)))
    top_cells.sort(reverse=True)

    critical = " | ".join(
        f"zone{z}(sev={s:.2f},pop={p})"
        for s, z, p in top_cells[:5]
    )
    avg_sev = float(np.mean(grid_state[:, :, 1]))

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Initial grid state:\n"
        f"  Mean severity: {avg_sev:.3f}\n"
        f"  Top critical zones: {critical}\n\n"
        f"Output your 50-action JSON array now:"
    )

# ── Action decoder ───────────────────────────────────────────────────────────
def decode_action(act):
    """Safely decode a compressed or full action dict into env-compatible form."""
    if not isinstance(act, dict):
        return {"intent": "hold", "zone": 0, "resource": "food", "priority": "low", "units": 1}

    intent_map   = {"a": "allocate", "r": "redirect", "h": "hold"}
    resource_map = {"m": "medicine", "f": "food", "re": "rescue",
                    "w": "water",    "s": "shelter", "t": "transport"}
    priority_map = {"h": "high", "m": "medium", "l": "low"}

    # Support both compressed keys (i/z/r/p/u) and full keys (intent/zone/…)
    intent   = act.get("intent")   or intent_map.get(act.get("i", "a"), "allocate")
    zone     = act.get("zone")     if act.get("zone") is not None else int(act.get("z", 0))
    resource = act.get("resource") or resource_map.get(act.get("r", "f"), "food")
    priority = act.get("priority") or priority_map.get(act.get("p", "l"), "low")
    units    = act.get("units")    if act.get("units") is not None else int(act.get("u", 1))

    return {
        "intent":   intent   if intent   in ("allocate", "redirect", "hold") else "allocate",
        "zone":     int(zone),
        "resource": resource if resource in ("medicine", "food", "rescue", "water", "shelter") else "food",
        "priority": priority if priority in ("high", "medium", "low") else "low",
        "units":    max(1, min(10, int(units))),
    }

# ── Reward function for GRPOTrainer ──────────────────────────────────────────
def grpo_reward_fn(prompts, completions, **kwargs):
    """
    Each completion = JSON array of up to 50 actions.
    Returns scalar reward = final episode survival_rate.
    """
    rewards = []
    for completion in completions:
        env = CrisisGridEnv()          # fresh env per completion
        env.reset()

        # Parse
        try:
            text = completion.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            actions = json.loads(text)
            if not isinstance(actions, list):
                actions = [actions]
        except (json.JSONDecodeError, ValueError):
            rewards.append(-5.0)       # Unparseable — hard penalty
            continue

        # Run trajectory — every action goes through decode_action
        done, step = False, 0
        info = {}
        while not done and step < 50:
            raw = actions[step] if step < len(actions) else {}
            # Handle the (unusual) case where a single completion item is itself a list
            if isinstance(raw, list):
                raw = raw[0] if raw else {}
            decoded = decode_action(raw)
            try:
                _, _, done, info = env.step(decoded)
            except Exception:
                done = True
            step += 1

        rewards.append(float(info.get("survival_rate", 0.0)))

    return rewards

# ── Build Dataset ─────────────────────────────────────────────────────────────
from datasets import Dataset

print(f"\n[dataset] Generating {args.episodes} episode prompts...")
records = []
seed_env = CrisisGridEnv(seed=42)
for i in range(args.episodes):
    obs, _ = seed_env.reset()
    grid = seed_env.grid.copy()
    records.append({"prompt": build_episode_prompt(grid)})

dataset = Dataset.from_list(records)
print(f"[dataset] {len(dataset)} samples ready.")

# ── GRPO Config + Trainer ─────────────────────────────────────────────────────
from trl import GRPOConfig, GRPOTrainer

os.makedirs(args.ckpt_dir, exist_ok=True)
os.makedirs("logs", exist_ok=True)

config = GRPOConfig(
    output_dir=args.ckpt_dir,
    num_train_epochs=1,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.lr,
    logging_steps=10,
    save_steps=args.checkpoint_every,
    save_total_limit=3,
    report_to="wandb" if USE_WANDB else "none",
    max_completion_length=1200,        # 50 actions × ~24 tokens each; 1200 is safe headroom
    temperature=0.7,
    num_generations=4,                 # GRPO group size
    generation_kwargs={
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 1200,
    },
)
print("Using max_completion_length=1200")
if 1200 > 1400:
    print("WARNING: max_completion_length too high")

trainer = GRPOTrainer(
    model=model,
    args=config,
    reward_funcs=[grpo_reward_fn],
    train_dataset=dataset,
    tokenizer=tokenizer,
)
print("[trainer] GRPOTrainer initialized.")

# ── TRAIN ─────────────────────────────────────────────────────────────────────
print("\n[train] Starting trainer.train() — REAL RL TRAINING\n")
trainer.train()
print("\n[train] Training complete.")

# ── Save final model ──────────────────────────────────────────────────────────
final_ckpt = os.path.join(args.ckpt_dir, "final")
model.save_pretrained(final_ckpt)
tokenizer.save_pretrained(final_ckpt)
print(f"[ckpt] Final model saved to {final_ckpt}")

# ── WandB finish ─────────────────────────────────────────────────────────────
if USE_WANDB:
    wandb.finish()
    print("[wandb] Run finished. Check your dashboard for 3 curves.")
