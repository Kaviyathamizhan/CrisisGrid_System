"""
prepare_deploy.py
Prepares the project for deployment into separate target directories:
  1. GitHub_Repo (clean source code + requirements)
  2. HF_Space (Gradio app + web_ui.html + cached runs + requirements)
  3. HF_Model (adapter checkpoints only)
"""

import os
import shutil

import stat

ROOT = os.path.dirname(os.path.abspath(__file__))
DEPLOY_DIR = os.path.join(ROOT, "deploy_ready")

# Define target directories
GITHUB_DIR = os.path.join(DEPLOY_DIR, "GitHub_Repo")
HF_MODEL_DIR = os.path.join(DEPLOY_DIR, "HF_Model")
HF_SPACE_DIR = os.path.join(DEPLOY_DIR, "HF_Space")

def rmtree_force(dirpath):
    def handle_error(func, path, exc_info):
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            pass
    if os.path.exists(dirpath):
        try:
            shutil.rmtree(dirpath, onexc=handle_error)
        except TypeError:
            shutil.rmtree(dirpath, onerror=handle_error)

# Clean up if exists and create
rmtree_force(DEPLOY_DIR)
os.makedirs(GITHUB_DIR)
os.makedirs(HF_MODEL_DIR)
os.makedirs(HF_SPACE_DIR)

# ---------------------------------------------------------
# 1. Prepare GitHub_Repo
# ---------------------------------------------------------
print("Preparing GitHub_Repo...")
for item in ["environment", "utils", "training", "notebooks", "data"]:
    src = os.path.join(ROOT, item)
    if os.path.exists(src):
        dst = os.path.join(GITHUB_DIR, item)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

for item in ["demo.py", "evaluate.py", "train.py", "app.py", "web_ui.html", "generate_cache.py", "prepare_deploy.py"]:
    src = os.path.join(ROOT, item)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(GITHUB_DIR, item))

# Create GitHub requirements.txt
github_reqs = """numpy>=1.26.0
torch>=2.2.0
transformers>=4.45.0
accelerate>=0.33.0
bitsandbytes>=0.43.0
peft>=0.12.0
trl>=0.12.0
unsloth>=2024.10
wandb>=0.17.0
gradio>=4.44.0
datasets>=2.16.0
"""
with open(os.path.join(GITHUB_DIR, "requirements.txt"), "w") as f:
    f.write(github_reqs)

# Create GitHub README.md
github_readme = """# CrisisGrid

A Multi-Agent Reinforcement Learning Environment for Disaster Response Coordination.

CrisisGrid simulates a 5x5 disaster zone where an AI agent must prioritize and allocate limited resources to maximize population survival. The environment features dynamic severity, resource constraints, and an intentional schema drift (API change) mid-episode to test agent adaptability.

## Repository Structure
- `environment/`: Core simulation logic (State, Actions, Schema Drift).
- `utils/`: Helper functions for message validation and visualization.
- `training/`: GRPO training loop, reward functions, and evaluation scripts.
- `notebooks/`: Ready-to-run Google Colab notebooks for training.
- `app.py`: Gradio app serving the CrisisGrid AI Command Center.
- `web_ui.html`: Glassmorphic operations dashboard frontend.
- `generate_cache.py`: Script to generate pre-cached trajectory runs.
- `demo.py`: Script to generate A/B test trajectories (random vs trained).

## Training
This project uses **GRPO** (Group Relative Policy Optimization) with Qwen2-1.5B via Unsloth.
To run the training pipeline, use the provided Colab notebook in the `notebooks/` directory.

## HuggingFace Links
- **Demo Space**: [Link to your Space]
- **Trained LoRA Weights**: [Link to your Model Repo]
"""
with open(os.path.join(GITHUB_DIR, "README.md"), "w") as f:
    f.write(github_readme)


# ---------------------------------------------------------
# 2. Prepare HF_Model
# ---------------------------------------------------------
print("Preparing HF_Model...")
ckpt_dir = os.path.join(ROOT, "checkpoints", "checkpoint-180")
if os.path.exists(ckpt_dir):
    for f in ["adapter_config.json", "adapter_model.safetensors"]:
        src = os.path.join(ckpt_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(HF_MODEL_DIR, f))
else:
    print(f"Warning: Checkpoint dir not found at {ckpt_dir}")


# ---------------------------------------------------------
# 3. Prepare HF_Space
# ---------------------------------------------------------
print("Preparing HF_Space...")
for item in ["environment", "utils"]:
    src = os.path.join(ROOT, item)
    if os.path.exists(src):
        dst = os.path.join(HF_SPACE_DIR, item)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

# Copy data cache
data_src = os.path.join(ROOT, "data")
data_dst = os.path.join(HF_SPACE_DIR, "data")
os.makedirs(data_dst, exist_ok=True)
if os.path.exists(data_src):
    for f in ["cached_comparison_runs.json"]:
        src_file = os.path.join(data_src, f)
        if os.path.exists(src_file):
            shutil.copy2(src_file, os.path.join(data_dst, f))

# Copy backend files
for item in ["app.py", "web_ui.html"]:
    src = os.path.join(ROOT, item)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(HF_SPACE_DIR, item))

# Create Space requirements.txt
space_reqs = """gradio>=4.44.0
transformers>=4.45.0
peft>=0.12.0
torch>=2.2.0
numpy>=1.26.0
bitsandbytes>=0.43.0
accelerate>=0.33.0
huggingface_hub>=0.23.0,<0.25.0
"""
with open(os.path.join(HF_SPACE_DIR, "requirements.txt"), "w") as f:
    f.write(space_reqs)

# Create Space README.md with required frontmatter
space_readme = """---
title: CrisisGrid - Premium AI Command Center
emoji: 🚀
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# CrisisGrid AI Command Center
A professional disaster response simulation dashboard powered by Qwen2-1.5B and Reinforcement Learning.
"""
with open(os.path.join(HF_SPACE_DIR, "README.md"), "w", encoding="utf-8") as f:
    f.write(space_readme)

print("Deployment preparation complete.")
