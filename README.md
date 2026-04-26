---
title: CrisisGrid - Teaching LLMs to Save Cities Through Multi-Agent RL
emoji: 🏙️
colorFrom: red
colorTo: blue
sdk: gradio
app_file: app.py
python_version: "3.10"
pinned: false
---

# 🏙️ CrisisGrid: Teaching LLMs to Save Cities with GRPO

A Multi-Agent Reinforcement Learning Environment for Disaster Response Coordination.

> **OpenEnv India Hackathon 2026 Submission**

---

## 📌 Quick Links

- 🤗 **Live Demo Space**: [https://huggingface.co/spaces/thebosskt/crisisgrid-train](https://huggingface.co/spaces/thebosskt/crisisgrid-train)
- 🧠 **Trained LoRA Weights**: [https://huggingface.co/thebosskt/crisisgrid-lora](https://huggingface.co/thebosskt/crisisgrid-lora)
- 📓 **Training Notebook**: [training_run.ipynb](./notebooks/training_run.ipynb)
- 📝 **Blog Post / Writeup**: [blog_post.md](./blog_post.md)

---

## 🎯 Problem Statement

In real-world disaster scenarios, central command agents must coordinate resources across many zones simultaneously. The challenge: existing LLMs are structurally unstable when forced to output strict JSON API commands under dynamic, adversarial conditions.

**CrisisGrid** is a custom 5×5 grid environment where an AI Command Agent must allocate resources (food, medicine, rescue, water, shelter) to 25 disaster zones to maximize population survival. The environment includes:
- Dynamic severity escalation
- Mid-episode schema drift (API changes at step 25)
- Adversary agents that destabilize zones
- Strict JSON communication protocol

---

## 🚀 What We Trained

We used **GRPO (Group Relative Policy Optimization)** via Hugging Face TRL to fine-tune a `Qwen/Qwen2-1.5B-Instruct` model with a LoRA adapter.

The reward function penalizes:
- ❌ Invalid JSON output (decode fallback penalty)
- ❌ Misallocating resources to low-severity zones

And rewards:
- ✅ Valid structured JSON every step
- ✅ Targeting highest-severity zones

---

## 📊 Results

| Metric | Baseline (Random) | GRPO Trained Agent |
|---|---|---|
| Avg Survival Rate | ~30.8% | ~33–38% |
| JSON Decode Fallbacks | High | **Zero (0)** |
| Structural Stability | Unstable | **100% Stable** |
| Training Reward | ~0.30 | **~0.71** |

> The trained agent achieved **100% JSON structural stability** and a **~2.3× improvement in RL reward** over the untrained baseline, demonstrating clear evidence of policy learning.

---

## 🏗️ Architecture

```
CrisisGridEnv (OpenEnv)
    ↓
build_prompt(obs) → structured context for LLM
    ↓
Qwen2-1.5B-Instruct + LoRA adapter (checkpoint-20)
    ↓
GRPO reward_func(completions, prompts)
    ↓
GRPOTrainer (TRL 0.15.x) → policy update
```

---

## 🗂️ Repository Structure

- `environment/`: Core simulation (state, schema drift, oversight, adversary)
- `utils/`: Message validation + visualization helpers
- `training/`: GRPO reward functions + baseline scripts
- `notebooks/`: Training run notebook for judges
- `train.py`: GRPO training script with LoRA resume
- `evaluate.py`: Evaluation with JSON repair logging
- `demo.py`: A/B episode dump (random vs trained)
- `app.py`: Gradio demo (Spaces)

---

## ▶️ Quickstart

```bash
git clone https://github.com/Kaviyathamizhan/CrisisGrid_System.git
cd CrisisGrid_System
pip install -r requirements.txt

# Evaluate the trained agent
python evaluate.py --checkpoint-path thebosskt/crisisgrid-lora --episodes 10

# Run A/B demo
python demo.py --checkpoint-path thebosskt/crisisgrid-lora
```

---

## 🔬 Training

```bash
python train.py \
  --checkpoint-path thebosskt/crisisgrid-lora \
  --max-completion-length 600 \
  --episodes 120
```

Checkpoints are saved every 20 steps to `checkpoints_a100/`.
