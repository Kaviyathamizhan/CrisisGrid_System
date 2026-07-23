"""
app.py
Gradio application serving the CrisisGrid AI Command Center.
Integrates HTML/CSS/JS frontend with the environment simulation backend.
Supports:
  1. Fast Replay Mode (zero-latency replay from pre-cached runs)
  2. Live AI Inference Mode (real-time LLM inference using Qwen2-1.5B + LoRA)
  3. Explicit execution errors when the model cannot be loaded.
"""

from __future__ import annotations

import os
import sys
import json
from typing import Any, Dict, Optional, Tuple, List
import numpy as np

# Add repo root to import paths
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from environment.crisis_grid_env import CrisisGridEnv
from utils.message_utils import validate_message

BASE_MODEL = "Qwen/Qwen2-1.5B-Instruct"

# Global lazy loaded models to avoid memory waste on CPU spaces
_MODEL = None
_TOKENIZER = None


def _extract_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def repair_json(text: str) -> Tuple[Optional[Dict[str, Any]], bool, Optional[str]]:
    raw = (text or "").strip()
    
    # Strip markdown fences
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

    # Repair truncated JSON arrays (50-action lists cut mid-output)
    if raw.startswith("[") and not raw.endswith("]"):
        last_brace = raw.rfind("}")
        if last_brace != -1:
            raw = raw[:last_brace + 1] + "]"

    candidate = _extract_json_object(raw) or raw
    try:
        obj = json.loads(candidate)
        return (obj if isinstance(obj, dict) else None), False, None
    except Exception:
        pass

    for reason, cand in [
        ("single_to_double_quotes", candidate.replace("'", '"')),
        ("removed_trailing_commas", candidate.replace("'", '"').replace(",}", "}").replace(",]", "]")),
    ]:
        try:
            obj = json.loads(cand)
            return (obj if isinstance(obj, dict) else None), True, reason
        except Exception:
            continue
    return None, True, "unparseable_after_repairs"


def decode_action(llm_text: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
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
        return None, diag

    ok, err = validate_message(msg)
    if not ok:
        diag["validate_error"] = err
        diag["decode_fallback"] = True
        return None, diag

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
    from huggingface_hub import snapshot_download
    import json

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # Check GPU availability
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True
    )

    # Construct a compatible PEFT config in memory. Do not mutate the supplied
    # checkpoint; it may be mounted read-only or shared with training artifacts.
    adapter_config = None
    try:
        local_dir = snapshot_download(lora_path_or_repo)
    except Exception:
        local_dir = lora_path_or_repo # Fallback if it's already a local path
        
    config_path = os.path.join(local_dir, "adapter_config.json")
    if os.path.exists(config_path):
        from peft import LoraConfig
        import inspect
        valid_keys = set(inspect.signature(LoraConfig.__init__).parameters.keys())
        valid_keys.discard("self")
        with open(config_path, "r") as f:
            cfg_dict = json.load(f)
        adapter_config = LoraConfig(**{k: v for k, v in cfg_dict.items() if k in valid_keys})

    model = PeftModel.from_pretrained(model, local_dir, config=adapter_config)
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


def run_live_inference(seed: int, lora_path_or_repo: str) -> dict:
    """Runs a live episode using the PyTorch Qwen2 model."""
    global _MODEL, _TOKENIZER
    if _MODEL is None or _TOKENIZER is None:
        _MODEL, _TOKENIZER = load_model_and_tokenizer(lora_path_or_repo)

    env = CrisisGridEnv(seed=seed)
    obs_cmd, _ = env.reset()

    # Step 0 details
    steps = [{
        "step": 0,
        "grid": env.grid.tolist(),
        "cmd_msg": None,
        "res_action": None,
        "reward": 0.0,
        "total_reward": 0.0,
        "survival_rate": 1.0,
        "mean_severity": float(np.mean(env.grid[:, :, 1])),
        "max_severity": float(np.max(env.grid[:, :, 1])),
        "drift_status": get_drift_status_for_step(0),
        "oversight_logs": [],
        "decision_explanation": "Initializing mission parameters. Establishing communication channels."
    }]

    done = False
    valid_count = 0
    total_count = 0

    while not done:
        step = env.timestep + 1
        prompt = build_prompt(obs_cmd)
        comp = generate_one(_MODEL, _TOKENIZER, prompt, max_new_tokens=100)
        cmd_msg, diag = decode_action(comp)

        total_count += 1
        is_valid = cmd_msg is not None
        if is_valid:
            valid_count += 1

        obs_cmd, reward, done, info = env.step(cmd_msg)
        
        flags = env.oversight.get_flags()
        step_flags = [f for f in flags if f.get("step") == step]

        steps.append({
            "step": step,
            "grid": env.grid.tolist(),
            "cmd_msg": cmd_msg,
            "model_output": comp,
            "decode_diagnostics": diag,
            "command_accepted": is_valid,
            "res_action": env.resource_agent.last_action,
            "reward": float(reward),
            "total_reward": float(env.total_reward),
            "survival_rate": float(info.get("survival_rate", 0.0)),
            "mean_severity": float(info.get("mean_severity", 0.0)),
            "max_severity": float(info.get("max_severity", 0.0)),
            "drift_status": get_drift_status_for_step(step),
            "oversight_logs": step_flags,
            "decision_explanation": generate_explanation_for_cell(cmd_msg, env)
        })

    # Summary metrics
    metrics = {
        "final_survival": float(info.get("survival_rate", 0.0)),
        "population_saved": int(np.sum(env.grid[:, :, 0])),
        "initial_population": int(env.initial_total_population),
        "total_reward": float(env.total_reward),
        "agent_reliability": float(valid_count / total_count if total_count > 0 else 0.0),
        "active_emergencies": int(np.sum(env.grid[:, :, 1] > 0.7)),
        "resource_efficiency": float(np.sum((env.grid[:, :, 2] > 0) & (env.grid[:, :, 1] > 0.4)) / max(1, np.sum(env.grid[:, :, 2] > 0)))
    }

    # Generate curves
    survival_curve = [float(s["survival_rate"]) for s in steps]
    severity_curve = [float(s["mean_severity"]) for s in steps]
    
    # Process events list
    events = []
    for s in steps:
        step_num = s["step"]
        if step_num == 0:
            events.append({"step": 0, "text": f"Flood outbreak detected. {metrics['initial_population']} citizens at risk.", "type": "warning"})
        
        for flag in s["oversight_logs"]:
            ftype = flag.get("type")
            if ftype == "schema_drift":
                events.append({"step": step_num, "text": "API Schema drift detected (POST /allocate deprecated)", "type": "critical"})
            elif ftype == "schema_recovery":
                events.append({"step": step_num, "text": "Schema recovery successful. Agent adapted to new PATCH endpoint.", "type": "success"})
            elif ftype == "population_loss":
                cell = flag.get("cell", (0, 0))
                events.append({"step": step_num, "text": f"Population lost in Zone {cell[0]*5 + cell[1]} (Severity exceeded threshold)", "type": "critical"})
            elif ftype == "default_action":
                events.append({"step": step_num, "text": "Command Agent timeout. Default allocation triggered.", "type": "warning"})
            elif ftype == "malformed_message":
                events.append({"step": step_num, "text": f"Oversight blocked malformed message: {flag.get('reason')}", "type": "critical"})

        if step_num == 1:
            events.append({"step": 1, "text": f"Emergency assets deployed to grid rows.", "type": "info"})
        if step_num == 10:
            events.append({"step": 10, "text": "Adversary severity spike injected.", "type": "warning"})
        if step_num == 20:
            events.append({"step": 20, "text": "Grid flood expansion intensifies.", "type": "warning"})
        if step_num == 35:
            events.append({"step": 35, "text": "Stabilization actions progressing.", "type": "info"})
        if step_num == 50:
            events.append({"step": 50, "text": f"Simulation complete. Survival rate: {metrics['final_survival']:.1%}.", "type": "success"})

    return {
        "agent_type": "trained",
        "seed": seed,
        "steps": steps,
        "survival_curve": survival_curve,
        "severity_curve": severity_curve,
        "events": events,
        "metrics": metrics
    }


def get_drift_status_for_step(timestep: int) -> dict:
    if timestep < 25:
        return {
            "status": "NORMAL",
            "api_status": "active",
            "current_schema_version": 1,
            "last_error": None
        }
    elif timestep == 25:
        return {
            "status": "WARNING",
            "api_status": "deprecated",
            "current_schema_version": 2,
            "last_error": "POST /allocate returned 404"
        }
    elif timestep == 26:
        return {
            "status": "RECOVERING",
            "api_status": "deprecated",
            "current_schema_version": 2,
            "last_error": "POST /allocate returned 404"
        }
    else:
        return {
            "status": "STABLE",
            "api_status": "deprecated",
            "current_schema_version": 2,
            "last_error": None
        }


def generate_explanation_for_cell(cmd_msg: Optional[dict], env: CrisisGridEnv) -> str:
    if cmd_msg is None:
        return "The model output was not accepted by the command validator; the environment recorded its configured default resource action."
    zone = cmd_msg.get("zone", 0)
    row = zone // 5
    col = zone % 5
    grid = env.grid
    
    pop = int(grid[row][col][0])
    sev = grid[row][col][1]
    res = cmd_msg.get("resource", "water")
    
    if cmd_msg.get("intent") == "hold":
        return "Operational Hold: Standard maintenance action applied."
        
    explanation = f"Executed model command for Zone {zone} (Row {row}, Col {col}). Post-step severity is "
    if sev > 0.7:
        explanation += f"critical ({sev:.2f}) with {pop} survivors. "
    else:
        explanation += f"{sev:.2f} with {pop} survivors. "
        
    explanation += f"Deploying {cmd_msg.get('units', 3)} units of {res}."
    return explanation


def get_cached_run(seed: int) -> Optional[Dict[str, Any]]:
    cache_path = os.path.join(REPO_ROOT, "data", "cached_comparison_runs.json")
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get(str(seed))
    except Exception:
        return None


def run_simulation_api(seed_val: float, mode_val: str) -> str:
    """
    Main API callback.
    Checks mode:
      - 'replay': loads from cached_comparison_runs.json.
      - 'live': loads Qwen + LoRA and executes the environment episode.
    """
    seed = int(seed_val)
    
    # 1. Replay Mode
    if mode_val == "replay":
        cached = get_cached_run(seed)
        if cached:
            return json.dumps({
                "fallback_triggered": False,
                "trained": cached["trained"],
                "random": cached["random"]
            })
        return json.dumps({"error": f"No cached trajectory exists for seed {seed}."})

    # 2. Live AI Inference Mode
    lora_path_or_repo = os.getenv("CRISISGRID_LORA_REPO", "").strip() or os.getenv("CRISISGRID_CHECKPOINT_PATH", "").strip()
    if not lora_path_or_repo:
        local_ckpt = os.path.join(REPO_ROOT, "checkpoints", "checkpoint-180")
        lora_path_or_repo = local_ckpt if os.path.exists(local_ckpt) else "thebosskt/crisisgrid-lora"

    try:
        trained_traj = run_live_inference(seed, lora_path_or_repo)
        return json.dumps({
            "fallback_triggered": False,
            "trained": trained_traj,
            "random": None
        })
    except Exception as e:
        print(f"[Inference Error] {str(e)}")
        return json.dumps({"error": f"Live inference failed: {str(e)}"})


def main():
    import gradio as gr

    # Read the custom dashboard HTML
    html_path = os.path.join(REPO_ROOT, "web_ui.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
    else:
        html_content = "<h2>Error: web_ui.html not found!</h2>"

    # Aggressive CSS to:
    # 1. Hide Gradio wrapper containers around bridge elements safely via a specific class
    # 2. Remove Gradio's own header/footer/padding chrome
    # 3. Make the HTML dashboard component fill the full viewport
    css = """
    /* Hide bridge elements safely without affecting the main layout */
    .bridge-hidden {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        overflow: hidden !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
    }

    /* Remove Gradio page chrome */
    .gradio-container {
        max-width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
        background: #0b0f19 !important;
    }
    footer {
        display: none !important;
    }
    .main.svelte-1kyws56 {
        padding: 0 !important;
    }
    .contain {
        padding: 0 !important;
    }
    .app {
        padding: 0 !important;
    }

    /* Make the HTML component full-bleed */
    .gradio-html {
        padding: 0 !important;
        max-width: 100% !important;
    }
    """

    with gr.Blocks(title="CrisisGrid AI Command Center", css=css, theme=gr.themes.Base()) as demo:
        # Bridge elements: visible=True so Gradio mounts them in DOM for JS access, hidden by CSS class
        trajectory_output = gr.Textbox(elem_id="trajectory_output", elem_classes="bridge-hidden", visible=True)
        seed_input = gr.Number(value=123, elem_id="seed_input", elem_classes="bridge-hidden", visible=True)
        mode_input = gr.Textbox(value="replay", elem_id="mode_input", elem_classes="bridge-hidden", visible=True)
        
        run_btn = gr.Button(elem_id="run_btn", elem_classes="bridge-hidden", visible=True)

        # Trigger simulation run and return JSON string to textarea
        run_btn.click(
            fn=run_simulation_api,
            inputs=[seed_input, mode_input],
            outputs=[trajectory_output]
        ).then(
            fn=None,
            inputs=[trajectory_output],
            outputs=None,
            js="""
            (json_str) => {
                if (window.loadCrisisGridTrajectory) {
                    window.loadCrisisGridTrajectory(json_str);
                } else {
                    window.dispatchEvent(new CustomEvent("crisisgrid:trajectory", { detail: json_str }));
                }
            }
            """
        )

        # Render custom full-bleed glassmorphic dashboard
        gr.HTML(html_content)
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
