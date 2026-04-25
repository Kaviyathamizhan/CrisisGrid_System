---
title: CrisisGrid - Teaching LLMs to Save Cities Through Multi-Agent RL
emoji: 🏙️
colorFrom: red
colorTo: blue
sdk: static
pinned: false
---

# CrisisGrid: Teaching LLMs to Save Cities Through Multi-Agent RL

**Public WandB Training Log:** [https://wandb.ai/your-team/crisisgrid/runs/colab-baseline-run](#)

## The Problem
Urban disaster response requires intense coordination under extreme constraints. First responders often deal with partial observability, chaotic communication channels, and rapidly changing logistics systems. In CrisisGrid, we simulate this environment by forcing two AI agents to coordinate emergency resources across a 5x5 city grid while navigating communication caps and unexpected API schema drifts.

## Architecture
CrisisGrid relies on a two-agent structure:
- **Command Agent (RL-Trained):** Makes high-level resource allocation decisions based on full grid observability. Output is hard-capped at 50 tokens.
- **Resource Agent (Rule-Based):** Executes Command Agent orders but has partial observability (only sees rows 2-4).

The environment introduces deterministic **Schema Drift at step 25**, where the API endpoint mutates from `POST /allocate` to `PATCH /distribution`. The Command Agent must dynamically query `GET /docs` to recover.

## Training Pipeline
We trained the Command Agent using **GRPO (Group Relative Policy Optimization)**. 
- **Model:** Qwen2-1.5B-Instruct-bnb-4bit (scaled to 7B for production)
- **Framework:** Unsloth + HuggingFace TRL
- **Reward Signal:** Dense, 7-component reward encompassing severity reduction, communication validity, schema recovery, and token efficiency.

## Results
After 500 episodes of GRPO training, the agent demonstrated profound strategic adaptation:
- **Baseline Survival (Random Agent):** 30.8%
- **Trained Agent Survival (Ep 500):** 89.8%
- **Net Improvement:** +59.0%

The Command Agent successfully learned to utilize the maximum 10 resource units per allocation, dynamically targeting the highest severity zones while successfully navigating the API schema drift at Step 25.

## Future Work
In future iterations, we plan to scale CrisisGrid to a full 5-agent system featuring Expert Stakeholders (Mayor, NGO Director) with conflicting directives, paving the way for real-world municipal deployments.
