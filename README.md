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

| Resource | Link |
|---|---|
| 🤗 **Live Demo Space** | [huggingface.co/spaces/thebosskt/crisisgrid-demo](https://huggingface.co/spaces/thebosskt/crisisgrid-demo) |
| 🧠 **Trained LoRA Weights** | [huggingface.co/thebosskt/crisisgrid-lora](https://huggingface.co/thebosskt/crisisgrid-lora) |
| 📓 **Training Notebook** | [training_run.ipynb](https://huggingface.co/spaces/thebosskt/crisisgrid-demo/blob/main/notebooks/training_run.ipynb) |
| 📝 **Blog Post / Writeup** | [Blog.md](https://huggingface.co/spaces/thebosskt/crisisgrid-demo/blob/main/Blog.md) |
| 📈 **WandB Training Logs** | [wandb.ai/kaviyathamizhan37-/crisisgrid](https://wandb.ai/kaviyathamizhan37-/crisisgrid/workspace?nw=nwuserkaviyathamizhan37) |

---

## 🎯 The Problem

In real-world disaster scenarios, central command agents must coordinate resources across many zones simultaneously. Existing LLMs are structurally unstable when forced to output strict JSON API commands under dynamic, adversarial conditions. They hallucinate free-form text, ignore high-severity zones, and fail when APIs change mid-crisis.

**CrisisGrid** asks: *Can we use reinforcement learning to teach a small LLM to reliably coordinate disaster response?*

---

## 🌍 The Environment

CrisisGrid is a custom 5×5 grid environment where an AI Command Agent must allocate resources (food, medicine, rescue, water, shelter) to 25 disaster zones to maximize population survival.

**What makes it genuinely challenging:**
- 🔥 **Dynamic Severity Escalation** — Unaddressed zones get worse every step
- 🔄 **Mid-Episode Schema Drift** — The API schema changes at step 25; the agent must adapt or its commands fail
- 👹 **Adversary Agent** — Randomly destabilizes zones, forcing reprioritization
- 📋 **Strict JSON Protocol** — All commands must be valid JSON with exact field names

**This is NOT a toy gridworld.** The agent must handle real-world challenges: API versioning, structured output under pressure, and adversarial conditions.

---

## 🏗️ Reward Signal Design (7 Components)

Our reward is composable, informative, and hard to game:

| Component | Value | What it Teaches |
|---|---|---|
| Severity Reduction | +1.0/unit | Allocate to high-severity zones |
| Population Preserved | +0.3 | Keep people alive |
| Valid Communication | +0.2 | Output proper JSON |
| Schema Recovery | +2.0 (one-time) | Adapt to API changes |
| Malformed Message | -0.5 | Don't hallucinate |
| Default Action | -0.3 | Don't force fallback |
| Full Stabilisation | +5.0 (terminal) | Solve the crisis completely |

Plus a **token efficiency penalty** that prevents verbose outputs.

---

## 🚀 Training

We used **GRPO (Group Relative Policy Optimization)** via Hugging Face TRL to fine-tune `Qwen/Qwen2-1.5B-Instruct` with a LoRA adapter (rank 16, alpha 32).

### Training Progression

| Step | Reward | Reward Std | Observation |
|------|--------|------------|-------------|
| 4 | 0.593 | 0.053 | Agent starts producing valid JSON |
| 17 | 0.593 | 0.053 | Stable baseline established |
| **20** | **0.711** | **0.035** | **Peak — checkpoint saved** |
| 30 | 0.667 | 0.038 | Slight regression |
| 40 | 0.608 | 0.037 | Over-exploration |

> **Peak reward: 0.711 at step 20** — a **~2.3× improvement** over the baseline reward of ~0.30.
> `decode_fallback=False` from step 17 onwards = **100% JSON structural stability**.

### Training Plots

![Reward Curves](https://raw.githubusercontent.com/Kaviyathamizhan/CrisisGrid_System/main/assets/reward_curves.png)
*GRPO reward, loss, and KL divergence over training steps. Reward peaks at 0.711 (step 20).*

![Training Charts](https://raw.githubusercontent.com/Kaviyathamizhan/CrisisGrid_System/main/assets/training_charts.png)
*Survival rate, decode fallback, and JSON repair metrics. Fallback drops to zero at step 20.*

> 📈 **Live WandB Dashboard**: [View all training metrics here](https://wandb.ai/kaviyathamizhan37-/crisisgrid/workspace?nw=nwuserkaviyathamizhan37)

---

## 📊 Results

| Metric | Baseline (Random) | GRPO Trained Agent |
|---|---|---|
| Avg Survival Rate | ~30.8% | **~33–38%** |
| JSON Decode Fallbacks | Frequent | **Zero (0)** |
| Structural Stability | Unstable | **100% Stable** |
| Training Reward | ~0.30 | **~0.71** |
| Reward Std Dev | High | **0.035 (stable)** |

### What the Agent Learned
1. **Structural Mastery**: Perfect JSON output 100% of the time — zero decode fallbacks
2. **Zone Prioritization**: Directs resources to highest-severity zones first
3. **Stable Policy**: Low variance (0.035) = consistent, repeatable strategy

### Key Insight
The agent's reward spiked primarily from solving the structural problem (perfect JSON). The strategic survival improvement was ~2-7%. This reveals that in strict-protocol environments, **structural compliance dominates the reward landscape** — a genuine RL alignment finding.

---

## 🏗️ Architecture

```
CrisisGridEnv (OpenEnv)
    ↓ observation (grid state, schema, API status)
build_prompt(obs) → structured context for LLM
    ↓
Qwen2-1.5B-Instruct + LoRA adapter (checkpoint-20)
    ↓ JSON command
GRPO reward_func → 7-component reward signal
    ↓
GRPOTrainer (TRL 0.15.x) → policy gradient update
```

---

## 🗂️ Repository Structure

| Path | Purpose |
|---|---|
| `environment/` | Core simulation (state, schema drift, oversight, adversary) |
| `training/` | GRPO reward functions + baseline scripts |
| `utils/` | Message validation + visualization helpers |
| `notebooks/` | Training notebook for judges |
| `train.py` | GRPO training with LoRA resume |
| `evaluate.py` | Multi-episode evaluation with JSON repair logging |
| `demo.py` | A/B episode dump (random vs trained) |
| `app.py` | Gradio live demo (Spaces) |

---

## ▶️ Quickstart

```bash
git clone https://github.com/Kaviyathamizhan/CrisisGrid_System.git
cd CrisisGrid_System
pip install -r requirements.txt

# Train (~15 min on A100)
python train.py --checkpoint-path thebosskt/crisisgrid-lora --max-completion-length 600 --episodes 120

# Evaluate
python evaluate.py --checkpoint-path thebosskt/crisisgrid-lora --episodes 10

# Run A/B demo
python demo.py --checkpoint-path thebosskt/crisisgrid-lora
```

---

## 🙏 Acknowledgements

Built with [OpenEnv](https://github.com/openenv-ai/openenv), [Hugging Face TRL](https://github.com/huggingface/trl), [PEFT](https://github.com/huggingface/peft), and [Qwen2](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct).
