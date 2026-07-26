"""
backend/routers/websocket.py
WebSocket streaming endpoint for step-by-step mission playback.
Supports both 'replay' (cached) and 'live' (real Qwen2 inference) modes.
"""

import asyncio
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from backend.core.logging import get_logger
from backend.services.simulation_service import simulation_service
from backend.services.replay_service import replay_service
from backend.services.inference_service import inference_service
from backend.services.metrics_service import metrics_service
from backend.services.explanation_service import explanation_service
from environment.crisis_grid_env import CrisisGridEnv
from utils.agent_utils import build_prompt, decode_action

logger = get_logger("WebSocketRouter")
router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/simulate")
async def websocket_simulate(
    websocket: WebSocket,
    seed: int = Query(default=123),
    mode: str = Query(default="replay"),
):
    await websocket.accept()
    logger.info(f"WebSocket connection established: seed={seed}, mode={mode}")

    try:
        if mode == "live":
            await _stream_live_simulation(websocket, seed)
        else:
            await _stream_replay(websocket, seed)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for seed {seed}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except Exception:
            pass


async def _stream_replay(websocket: WebSocket, seed: int):
    """Stream pre-cached replay trajectory step-by-step."""
    sim_data = simulation_service.run_replay(seed)
    steps = sim_data["steps"]

    await websocket.send_json({
        "type": "init",
        "seed": seed,
        "total_steps": len(steps) - 1,
        "metrics": sim_data["metrics"]
    })

    for idx, step_data in enumerate(steps):
        await websocket.send_json({
            "type": "step",
            "step": idx,
            "data": step_data
        })
        await asyncio.sleep(0.05)

    await websocket.send_json({
        "type": "complete",
        "metrics": sim_data["metrics"],
        "events": sim_data["events"]
    })
    logger.info(f"Replay streaming complete for seed {seed}")


async def _stream_live_simulation(websocket: WebSocket, seed: int):
    """Run real Qwen2-1.5B inference and stream each step as it completes."""
    # Ensure model is loaded
    if not inference_service.is_loaded:
        await websocket.send_json({
            "type": "status",
            "message": "Loading Qwen2-1.5B + LoRA model... This may take 30-60 seconds on CPU."
        })
        # Run blocking model load in a thread so we don't freeze the event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, inference_service.load_model)

    env = CrisisGridEnv(seed=seed)
    obs_cmd, _ = env.reset()

    all_steps = []
    initial_step = {
        "step": 0,
        "grid": env.grid.tolist(),
        "cmd_msg": None,
        "res_action": None,
        "reward": 0.0,
        "total_reward": 0.0,
        "survival_rate": 1.0,
        "mean_severity": float(np.mean(env.grid[:, :, 1])),
        "max_severity": float(np.max(env.grid[:, :, 1])),
        "drift_status": simulation_service.get_drift_status(0),
        "oversight_logs": [],
        "decision_explanation": "Initializing mission parameters. Establishing communication channels."
    }
    all_steps.append(initial_step)

    # Send init frame
    await websocket.send_json({
        "type": "init",
        "seed": seed,
        "total_steps": 50,
        "metrics": {}
    })

    # Send step 0
    await websocket.send_json({
        "type": "step",
        "step": 0,
        "data": initial_step
    })

    done = False
    valid_count = 0
    total_count = 0
    info = {}

    while not done:
        step_num = env.timestep + 1
        prompt = build_prompt(obs_cmd)

        # Run inference in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        try:
            comp = await loop.run_in_executor(
                None,
                lambda: inference_service.generate(prompt, max_new_tokens=600)
            )
            cmd_msg, diag = decode_action(comp, rng=env.rng)
        except Exception as e:
            logger.error(f"Live inference step {step_num} failed: {str(e)}")
            cmd_msg = None
            diag = {"decode_fallback": True, "error": str(e)}

        total_count += 1
        is_valid = not diag.get("decode_fallback", False)
        if is_valid:
            valid_count += 1

        obs_cmd, reward, done, info = env.step(cmd_msg)
        flags = env.oversight.get_flags()
        step_flags = [f for f in flags if f.get("step") == step_num]

        step_data = {
            "step": step_num,
            "grid": env.grid.tolist(),
            "cmd_msg": cmd_msg,
            "res_action": env.resource_agent.last_action,
            "reward": float(reward),
            "total_reward": float(env.total_reward),
            "survival_rate": float(info.get("survival_rate", 0.0)),
            "mean_severity": float(info.get("mean_severity", 0.0)),
            "max_severity": float(info.get("max_severity", 0.0)),
            "drift_status": simulation_service.get_drift_status(step_num),
            "oversight_logs": step_flags,
            "decision_explanation": explanation_service.generate_explanation(cmd_msg, env.grid)
        }
        all_steps.append(step_data)

        # Stream this step to the client immediately
        await websocket.send_json({
            "type": "step",
            "step": step_num,
            "data": step_data
        })

    # Compute final metrics
    final_metrics = metrics_service.compute_summary(
        env.grid, env.initial_total_population, env.total_reward, info, valid_count, total_count
    )
    events = simulation_service._generate_events(all_steps, final_metrics)

    await websocket.send_json({
        "type": "complete",
        "metrics": final_metrics,
        "events": events
    })
    logger.info(f"Live simulation streaming complete for seed {seed}: {total_count} steps, {valid_count} valid")
