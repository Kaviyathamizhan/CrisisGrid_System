# 🏙️ Stabilizing LLM Agents in CrisisGrid with GRPO

> **OpenEnv India Hackathon 2026** — Team Submission

---

## The Problem: Why LLMs Fail at Disaster Response

Imagine a real disaster scenario: an earthquake has devastated a city divided into 25 zones. A central Command Agent must coordinate food, medicine, rescue teams, water, and shelter across all zones simultaneously. Every decision must be transmitted as a structured JSON API command to field responders.

We deployed a base `Qwen/Qwen2-1.5B-Instruct` model as the Command Agent. The result? **Catastrophic failure.** The model:
- Hallucinated free-form text instead of valid JSON commands
- Ignored high-severity zones in favor of random allocations
- Failed to adapt when the communication API changed mid-crisis (schema drift at step 25)

The baseline survival rate? **~30.8%** — barely better than random.

---

## The Environment: CrisisGrid v2

CrisisGrid is a custom OpenEnv environment that simulates a realistic multi-agent disaster scenario:

- **5×5 Grid**: 25 zones, each with a population, severity level, and resource needs
- **Dynamic Severity**: Crisis severity escalates every step if unaddressed
- **Schema Drift**: At step 25, the API communication schema changes — the agent must adapt or its commands get rejected
- **Adversary Agent**: Randomly destabilizes zones, forcing the Command Agent to reprioritize
- **Oversight Module**: Validates all JSON commands against strict schemas

### What the Agent Sees (Observation)
Each step, the agent receives:
- Full grid state (severity per zone)
- Current API schema version
- API status (active/degraded)
- Last error message (if any)

### What the Agent Does (Action)
The agent must output a single JSON command:
```json
{"intent": "allocate", "zone": 12, "resource": "medicine", "priority": "high", "units": 3}
```

### The Reward Signal (7 Components)
Our reward function is composable, not monolithic:

| Component | Value | Purpose |
|---|---|---|
| Severity Reduction | +1.0 per unit | Rewards effective resource allocation |
| Population Preserved | +0.3 | Rewards keeping people alive |
| Valid Communication | +0.2 | Rewards proper JSON formatting |
| Schema Recovery | +2.0 (one-time) | Rewards adapting to API changes |
| Malformed Message | -0.5 | Penalizes broken JSON |
| Default Action | -0.3 | Penalizes fallback to random |
| Full Stabilisation | +5.0 (terminal) | Bonus if all zones stabilized |

Plus a **token efficiency penalty** that discourages verbose outputs.

---

## Training: GRPO with LoRA

We used **Group Relative Policy Optimization (GRPO)** via Hugging Face TRL to train the agent. GRPO compares groups of completions and updates the policy based on relative performance — no critic network needed.

### Configuration
- **Base Model**: `Qwen/Qwen2-1.5B-Instruct`
- **Adapter**: LoRA (rank 16, alpha 32) via PEFT
- **RL Algorithm**: GRPO (TRL 0.15.x)
- **Max Completion Length**: 600 tokens
- **Episodes**: 120 prompts per training run
- **Save Frequency**: Every 20 steps

### Training Progression

| Step | Reward | Reward Std | Key Observation |
|------|--------|------------|-----------------|
| 4 | 0.593 | 0.053 | Early learning — agent starts producing valid JSON |
| 17 | 0.593 | 0.053 | Stable baseline established |
| 20 | **0.711** | **0.035** | **Peak performance — checkpoint saved** |
| 30 | 0.667 | 0.038 | Slight regression from over-exploration |
| 40 | 0.608 | 0.037 | Continued exploration, lower reward |

The agent peaked at **step 20** with a reward of **0.711**, representing a **~2.3× improvement** over the baseline reward of ~0.30.

Key training signal: `decode_fallback=False` was consistently logged from step 17 onwards, proving the agent learned **100% JSON structural stability**.

---

## Results

| Metric | Baseline (Random/Untrained) | GRPO Trained Agent (checkpoint-20) |
|---|---|---|
| Avg Survival Rate | ~30.8% | ~33–38% |
| JSON Decode Fallbacks | Frequent | **Zero (0)** |
| Structural Stability | Unstable | **100% Stable** |
| Training Reward | ~0.30 | **~0.71** |
| Reward Std Dev | High variance | **0.035 (stable policy)** |

### What the Agent Learned

1. **Structural Mastery**: The most dramatic improvement. The base model frequently hallucinated free-form text. After GRPO training, the agent outputs perfectly formatted JSON **100% of the time**. Zero decode fallbacks.

2. **Zone Prioritization**: The agent learned to read the `critical_zones` from its observation and direct resources to high-severity areas first.

3. **Consistent Policy**: The low reward standard deviation (0.035) indicates the agent developed a stable, repeatable strategy rather than random guessing.

---

## Key Technical Insight

We discovered an interesting RL alignment observation: the agent's internal reward spiked dramatically (from 0.30 to 0.71) primarily because it solved the **structural problem** (perfect JSON formatting). The strategic improvement (survival rate) was more modest (~2-7% gain).

This reveals that in environments with strict communication protocols, **structural compliance dominates the reward landscape**. The agent prioritized "speak correctly" over "speak wisely" — a classic reward shaping challenge that future work could address by increasing the weight of survival-based reward components.

---

## Architecture

```
CrisisGridEnv (OpenEnv)
    ↓ observation (grid state, schema, API status)
build_prompt(obs) → structured context
    ↓
Qwen2-1.5B-Instruct + LoRA adapter
    ↓ JSON command
GRPO reward_func → 7-component reward signal
    ↓
GRPOTrainer (TRL) → policy gradient update
```

---

## How to Reproduce

```bash
# Clone and install
git clone https://github.com/Kaviyathamizhan/CrisisGrid_System.git
cd CrisisGrid_System
pip install -r requirements.txt

# Train (takes ~15 min on A100)
python train.py --checkpoint-path thebosskt/crisisgrid-lora --max-completion-length 600 --episodes 120

# Evaluate
python evaluate.py --checkpoint-path checkpoints_a100/checkpoint-20 --episodes 10

# Run A/B demo
python demo.py --checkpoint-path checkpoints_a100/checkpoint-20
```

---

## Links

- **Live Demo Space**: [https://huggingface.co/spaces/thebosskt/crisisgrid-train](https://huggingface.co/spaces/thebosskt/crisisgrid-train)
- **Trained LoRA Weights**: [https://huggingface.co/thebosskt/crisisgrid-lora](https://huggingface.co/thebosskt/crisisgrid-lora)
- **Training Notebook**: [training_run.ipynb](./notebooks/training_run.ipynb)
- **GitHub Repository**: [https://github.com/Kaviyathamizhan/CrisisGrid_System](https://github.com/Kaviyathamizhan/CrisisGrid_System)
