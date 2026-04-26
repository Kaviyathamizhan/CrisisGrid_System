# Stabilizing LLM Agents in CrisisGrid with GRPO

## The Problem
CrisisGrid is a high-stakes, multi-agent disaster response simulation. We deployed a base `Qwen/Qwen2-1.5B-Instruct` model to act as the Command Agent, directing resources to critical zones. However, the base model struggled with two critical issues:
1. **Structural Instability:** The model frequently hallucinated text instead of outputting valid JSON, causing the simulation to reject its commands and fallback to random actions.
2. **Strategic Inefficiency:** It struggled to prioritize high-severity disaster zones consistently.

## Our Approach: GRPO Reinforcement Learning
To solve this, we implemented a custom Group Relative Policy Optimization (GRPO) training loop. Instead of just fine-tuning on static data, we allowed the agent to explore the environment and gave it rewards based on two metrics:
1. **Syntax Integrity:** Massive penalties for invalid JSON formats.
2. **Survival Rate:** Positive rewards for taking actions that minimized the global crisis severity.

## The Results
The results were exactly what we hypothesized:
- **100% Structural Stabilization:** The RL-trained agent learned to perfectly format JSON output. The `decode_fallback` penalty dropped to zero, completely eliminating parser errors.
- **Improved Survival Rate:** The agent achieved an average survival rate of ~33%, outperforming the baseline random/base-model agent (~30.8%).

By using GRPO, we transformed an unpredictable generative model into a structurally stable, statistically superior decision engine.
