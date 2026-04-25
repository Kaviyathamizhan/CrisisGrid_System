# CrisisGrid

A Multi-Agent Reinforcement Learning Environment for Disaster Response Coordination.

CrisisGrid simulates a 5x5 disaster zone where an AI agent must prioritize and allocate limited resources to maximize population survival. The environment features dynamic severity, resource constraints, and an intentional schema drift (API change) mid-episode to test agent adaptability.

## Repository Structure
- `environment/`: Core simulation logic (State, Actions, Schema Drift).
- `utils/`: Helper functions for message validation and visualization.
- `training/`: GRPO training loop, reward functions, and evaluation scripts.
- `notebooks/`: Ready-to-run Google Colab notebooks for training.
- `app.py`: Gradio UI for interactive visualization.
- `demo.py`: Script to generate A/B test trajectories (random vs trained).

## Training
This project uses **GRPO** (Group Relative Policy Optimization) with Qwen2-1.5B via Unsloth.
To run the training pipeline, use the provided Colab notebook in the `notebooks/` directory.

## HuggingFace Links
- **Demo Space**: [Link to your Space]
- **Trained LoRA Weights**: [Link to your Model Repo]
