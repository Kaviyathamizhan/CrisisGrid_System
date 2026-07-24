"""
agent_utils.py
Shared utilities for prompt construction, LLM response generation, JSON repair,
action decoding, and checkpoint/model loading across CrisisGrid apps and scripts.
"""

from __future__ import annotations

import os
import sys
import json
from typing import Any, Dict, Optional, Tuple
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.message_utils import validate_message, REQUIRED_FIELDS, VALID_RESOURCES, VALID_PRIORITIES, VALID_INTENTS

BASE_MODEL = "Qwen/Qwen2-1.5B-Instruct"


def random_valid_message(rng: Optional[np.random.RandomState] = None) -> Dict[str, Any]:
    """Generate a valid Command Agent message dictionary using numpy RandomState or random choice."""
    if rng is None:
        rng = np.random.RandomState()
    return {
        "intent": "allocate",
        "zone": int(rng.randint(0, 25)),
        "resource": str(rng.choice(VALID_RESOURCES)),
        "priority": str(rng.choice(VALID_PRIORITIES)),
        "units": int(rng.randint(1, 6)),
    }


def _extract_json_object(text: str) -> Optional[str]:
    """Return the first {...} substring if present."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def repair_json(text: str) -> Tuple[Optional[Dict[str, Any]], bool, Optional[str]]:
    """
    Attempt to parse and repair LLM generated JSON strings.
    Returns (dict_or_none, repaired_boolean, reason_string).
    """
    raw = (text or "").strip()
    
    # 1) Direct parse attempt
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj, False, None
    except Exception:
        pass

    # 2) Strip markdown fences (e.g. ```json ... ```)
    candidate = raw
    if candidate.startswith("```"):
        parts = candidate.split("```")
        if len(parts) >= 2:
            candidate2 = parts[1].strip()
            if candidate2.startswith("json"):
                candidate2 = candidate2[4:].strip()
            try:
                obj = json.loads(candidate2)
                if isinstance(obj, dict):
                    return obj, True, "stripped_code_fence"
            except Exception:
                candidate = candidate2

    # 3) Repair truncated JSON arrays or sub-structures
    if candidate.startswith("[") and not candidate.endswith("]"):
        last_brace = candidate.rfind("}")
        if last_brace != -1:
            candidate = candidate[:last_brace + 1] + "]"

    extracted = _extract_json_object(candidate) or candidate
    try:
        obj = json.loads(extracted)
        if isinstance(obj, dict):
            return obj, True, "extracted_json_object"
    except Exception:
        pass

    # 4) Common fixes: single to double quotes
    cand_quotes = extracted.replace("'", '"')
    try:
        obj = json.loads(cand_quotes)
        if isinstance(obj, dict):
            return obj, True, "single_to_double_quotes"
    except Exception:
        pass

    # 5) Remove trailing commas before closing braces
    cand_commas = cand_quotes.replace(",}", "}").replace(",]", "]")
    try:
        obj = json.loads(cand_commas)
        if isinstance(obj, dict):
            return obj, True, "removed_trailing_commas"
    except Exception:
        pass

    return None, True, "unparseable_after_repairs"


def decode_action(
    llm_text: str,
    rng: Optional[np.random.RandomState] = None,
    log_repair: bool = False
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Standardized LLM output decoder. Converts string completion to valid Command message.
    Returns (msg_dict, diagnostics_dict).
    Guarantees a valid message dict fallback if parsing or validation fails.
    """
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
        if log_repair and reason:
            print(f"[json-repair] reason={reason}")

    if not isinstance(msg, dict):
        diag["decode_fallback"] = True
        return random_valid_message(rng), diag

    ok, err = validate_message(msg)
    if not ok:
        diag["validate_error"] = err
        diag["decode_fallback"] = True
        return random_valid_message(rng), diag

    # Ensure required structure
    for k in REQUIRED_FIELDS:
        if k not in msg:
            diag["validate_error"] = f"missing_field:{k}"
            diag["decode_fallback"] = True
            return random_valid_message(rng), diag

    return msg, diag


def build_prompt(obs: dict) -> str:
    """
    Build a compact prompt for the Command Agent based on environment observation.
    """
    timestep = obs.get("timestep", 0)
    api_status = obs.get("api_status", "active")
    schema_version = obs.get("current_schema_version", 1)
    last_error = obs.get("last_error", None)
    grid = obs.get("grid", [])

    worst = []
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            sev = float(cell[1]) if len(cell) > 1 else 0.0
            pop = float(cell[0]) if len(cell) > 0 else 0.0
            worst.append((sev, i * 5 + j, pop))
    worst.sort(reverse=True)
    top3 = worst[:3]

    header = f"Step {timestep}/50 | API v{schema_version} | status={api_status}"
    if api_status == "deprecated" and last_error:
        header += f" | last_error={last_error}"
    critical = " | ".join([f"z{z}(sev={s:.2f},pop={int(p)})" for s, z, p in top3])

    return (
        "You are the Command Agent for CrisisGrid.\n"
        "Output ONLY one valid JSON command with keys: intent, zone, resource, priority, units.\n"
        "Valid values: intent=allocate|redirect|hold, zone=0-24, "
        "resource=medicine|food|rescue|water|shelter, priority=high|medium|low.\n\n"
        f"{header}\nCritical zones: {critical}\nYour JSON command:"
    )


def get_clean_checkpoint_path(checkpoint_path: str) -> str:
    """
    Ensure checkpoint path exists locally or download from HuggingFace Hub.
    Patches adapter_config.json to filter out Unsloth-only keys if needed.
    """
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    local_dir = os.path.join(REPO_ROOT, "patched_checkpoint_cache")

    if not os.path.exists(local_dir):
        print(f"[Checkpoint] Downloading from HF: {checkpoint_path}")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=checkpoint_path, local_dir=local_dir)

        config_path = os.path.join(local_dir, "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                adapter_cfg = json.load(f)

            import inspect
            from peft import LoraConfig
            valid_keys = set(inspect.signature(LoraConfig.__init__).parameters.keys())
            valid_keys.update([
                "peft_type", "auto_mapping", "base_model_name_or_path",
                "revision", "task_type", "inference_mode"
            ])

            removed_keys = []
            for k in list(adapter_cfg.keys()):
                if k not in valid_keys:
                    adapter_cfg.pop(k)
                    removed_keys.append(k)

            with open(config_path, "w") as f:
                json.dump(adapter_cfg, f, indent=2)

            print(f"[Checkpoint] Cleaned keys: {removed_keys}")
        else:
            raise FileNotFoundError("adapter_config.json not found in checkpoint")

    return local_dir


def load_model_and_tokenizer(checkpoint_path_or_repo: str):
    """
    Load base Qwen model and attach PeftModel LoRA adapter.
    Handles device mapping (CUDA vs CPU) automatically.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    clean_path = get_clean_checkpoint_path(checkpoint_path_or_repo)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(model, clean_path)
    model.eval()
    return model, tokenizer


def generate_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 600,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> str:
    """
    Generate LLM completion string for a given prompt.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    if decoded.startswith(prompt):
        return decoded[len(prompt):].strip()
    return decoded.strip()
