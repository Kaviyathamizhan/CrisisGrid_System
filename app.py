"""
app.py
HuggingFace Spaces (Gradio) demo for CrisisGrid.

UI:
  - Button: "Run Episode"
  - Displays: step-by-step actions + final survival rate

Env vars:
  - CRISISGRID_CHECKPOINT_PATH: directory containing the LoRA adapter (PEFT)
"""

from __future__ import annotations

import os
import sys
import json
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from environment.crisis_grid_env import CrisisGridEnv
from utils.message_utils import validate_message


BASE_MODEL = "Qwen/Qwen2-1.5B-Instruct"


def _extract_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def repair_json(text: str) -> Tuple[Optional[Dict[str, Any]], bool, Optional[str]]:
    raw = (text or "").strip()
    candidate = _extract_json_object(raw) or raw
    try:
        obj = json.loads(candidate)
        return (obj if isinstance(obj, dict) else None), False, None
    except Exception:
        pass

    # basic safe transforms
    candidate2 = candidate.replace("'", '"').replace(",}", "}").replace(",]", "]")
    try:
        obj = json.loads(candidate2)
        return (obj if isinstance(obj, dict) else None), True, "repaired_basic"
    except Exception:
        return None, True, "unparseable_after_repairs"


def random_valid_message(rng: np.random.RandomState) -> Dict[str, Any]:
    return {
        "intent": "allocate",
        "zone": int(rng.randint(0, 25)),
        "resource": str(rng.choice(["medicine", "food", "rescue", "water", "shelter"])),
        "priority": str(rng.choice(["high", "medium", "low"])),
        "units": int(rng.randint(1, 6)),
    }


def decode_action(llm_text: str, rng: np.random.RandomState) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    diag: Dict[str, Any] = {
        "json_repair_triggered": False,
        "json_repair_reason": None,
        "decode_fallback": False,
        "validate_error": None,
    }
    msg, repaired, reason = repair_json(llm_text)
    if repaired:
        diag["json_repair_triggered"] = True
        diag["json_repair_reason"] = reason

    if not isinstance(msg, dict):
        diag["decode_fallback"] = True
        return random_valid_message(rng), diag

    ok, err = validate_message(msg)
    if not ok:
        diag["validate_error"] = err
        diag["decode_fallback"] = True
        return random_valid_message(rng), diag

    return msg, diag


def build_prompt(obs: dict) -> str:
    timestep = obs.get("timestep", 0)
    api_status = obs.get("api_status", "active")
    schema_version = obs.get("current_schema_version", 1)
    last_error = obs.get("last_error", None)
    grid = obs.get("grid", [])

    worst = []
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            sev = float(cell[1]) if len(cell) > 1 else 0.0
            worst.append((sev, i * 5 + j))
    worst.sort(reverse=True)
    top = [z for _, z in worst[:3]]

    prompt = (
        "You are the Command Agent for CrisisGrid.\n"
        "Output ONLY one valid JSON command with keys: intent, zone, resource, priority, units.\n"
        f"Schema={schema_version} API={api_status}\n"
    )
    if last_error:
        prompt += f"LAST ERROR: {last_error}\n"
    prompt += f"Step={timestep} critical_zones={top}\nYour JSON command:"
    return prompt


def load_model_and_tokenizer(lora_path_or_repo: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype="auto",
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, lora_path_or_repo)
    model.eval()
    return model, tokenizer


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int = 600) -> str:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return decoded[len(prompt) :].strip() if decoded.startswith(prompt) else decoded.strip()


# Global, lazy-loaded to avoid slow import-time failures in Spaces
_MODEL = None
_TOKENIZER = None


def run_episode() -> Tuple[str, str]:
    lora_path_or_repo = os.getenv("CRISISGRID_LORA_REPO", "").strip() or os.getenv("CRISISGRID_CHECKPOINT_PATH", "").strip() or "thebosskt/crisisgrid-lora"
    if not lora_path_or_repo:
        return (
            "Missing LoRA reference.",
            "Set CRISISGRID_LORA_REPO (HF repo id) or CRISISGRID_CHECKPOINT_PATH (local path).",
        )

    global _MODEL, _TOKENIZER
    if _MODEL is None or _TOKENIZER is None:
        _MODEL, _TOKENIZER = load_model_and_tokenizer(lora_path_or_repo)

    env = CrisisGridEnv(seed=123)
    rng = np.random.RandomState(123)
    obs_cmd, _ = env.reset()

    done = False
    steps: List[str] = []
    repair_logs: List[str] = []

    while not done:
        prompt = build_prompt(obs_cmd)
        comp = generate_one(_MODEL, _TOKENIZER, prompt, max_new_tokens=600)
        msg, diag = decode_action(comp, rng)
        if diag.get("json_repair_triggered"):
            repair_logs.append(f"step={obs_cmd.get('timestep', '?')}: json_repair={diag.get('json_repair_reason')}")
        if diag.get("decode_fallback"):
            repair_logs.append(f"step={obs_cmd.get('timestep', '?')}: fallback validate_error={diag.get('validate_error')}")

        obs_cmd, reward, done, info = env.step(msg)
        steps.append(
            f"t={info.get('timestep')} msg={msg} reward={float(reward):.3f} survival={float(info.get('survival_rate',0.0)):.3f}"
        )

    final_survival = float(info.get("survival_rate", 0.0))
    steps_text = "\n".join(steps)
    summary = f"Final survival rate: {final_survival:.1%}\n\nJSON repair logs:\n" + ("\n".join(repair_logs) if repair_logs else "(none)")
    return steps_text, summary


def main():
    import gradio as gr

    with gr.Blocks(title="CrisisGrid Demo") as demo:
        gr.Markdown("## 🏙️ CrisisGrid — Live Disaster Response Demo")
        gr.Markdown("Click to run one full 50-step episode with the GRPO-trained Command Agent.")

        btn = gr.Button("▶️ Run Episode", variant="primary")
        out_steps = gr.Textbox(label="Step-by-step actions", lines=22)
        out_summary = gr.Textbox(label="Summary", lines=6)

        btn.click(fn=run_episode, inputs=[], outputs=[out_steps, out_summary])

    demo.launch()


if __name__ == "__main__":
    main()
