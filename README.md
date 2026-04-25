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

# CrisisGrid

A Multi-Agent Reinforcement Learning Environment for Disaster Response Coordination.

## Overview

CrisisGrid simulates a 5×5 disaster zone where an AI Command Agent must allocate resources to maximize population survival. The environment includes dynamic severity, strict communication structure, and a deterministic schema drift mid-episode.

## Repository Structure

- `environment/`: Core simulation logic (state, schema drift, oversight, adversary)
- `utils/`: Message validation + visualization helpers
- `training/`: GRPO training + baseline scripts
- `notebooks/`: Colab notebooks
- `train.py`: GRPO training (Unsloth + TRL) with LoRA resume
- `evaluate.py`: 50-episode evaluation with JSON repair logging
- `demo.py`: A/B episode dump (random vs trained) into `data/`
- `app.py`: Gradio demo (Spaces)

## Training (A100 / Spaces)

Set your LoRA adapter reference via env var:

- `CRISISGRID_CHECKPOINT_PATH`: local adapter directory **or** HF repo id (downloaded via `snapshot_download`)

Then run:

```bash
python train.py --checkpoint-path "$CRISISGRID_CHECKPOINT_PATH"
```

## HuggingFace Links

- **Demo Space**: (add your Space URL)
- **Trained LoRA Weights**: (add your model repo URL)
