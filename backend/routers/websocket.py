"""
backend/routers/websocket.py
WebSocket streaming endpoint for step-by-step mission playback.
Supports both 'replay' (cached) and 'live' (real Qwen2 inference) modes.
"""

import time
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
    logger.info(f"WebSocket Connected: seed={seed}, mode={mode}")
    logger.info(f"Mission Started: seed={seed}, mode={mode}")

    try:
        if mode == "live":
            await _stream_live_simulation(websocket, seed)
        else:
            await _stream_replay(websocket, seed)

    except WebSocketDisconnect:
        logger.info(f"WebSocket Closed (Disconnect): seed={seed}")
    except Exception as e:
        logger.error(f"WebSocket Error: {str(e)}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Mission failed: {str(e)}"
            })
        except Exception:
            pass
    finally:
        logger.info(f"WebSocket Closed: seed={seed}")


async def _stream_replay(websocket: WebSocket, seed: int):
    """Stream pre-cached replay trajectory step-by-step."""
    start_time = time.perf_counter()
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

    duration = time.perf_counter() - start_time
    avg_step_ms = round((duration / max(1, len(steps))) * 1000, 2)
    logger.info(f"Mission Finished: seed={seed}, mode=replay, duration={duration:.2f}s, avg_step_time={avg_step_ms}ms")

    await websocket.send_json({
        "type": "complete",
        "metrics": sim_data["metrics"],
        "events": sim_data["events"],
        "telemetry": {
            "mission_duration_s": round(duration, 2),
            "avg_inference_time_ms": avg_step_ms,
            "fps": round(len(steps) / max(0.1, duration), 1)
        }
    })


async def _stream_live_simulation(websocket: WebSocket, seed: int):
    """Run real Qwen2-1.5B inference and stream each step with precise sub-stage profiling."""
    mission_start_time = time.perf_counter()
    
    # Ensure model is loaded
    if not inference_service.is_loaded:
        await websocket.send_json({
            "type": "status",
            "message": "Loading Qwen2-1.5B + LoRA model... This may take 30-60 seconds on CPU."
        })
        load_start = time.perf_counter()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, inference_service.load_model)
        logger.info(f"Model Load Time: {time.perf_counter() - load_start:.2f}s")

    env = CrisisGridEnv(seed=seed)
    obs_cmd, _ = env.reset()

    all_steps = []
    inf_times, env_times, ser_times, ws_times = [], [], [], []

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

        # Stage 1: LLM Inference
        t_inf_start = time.perf_counter()
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

        inf_ms = round((time.perf_counter() - t_inf_start) * 1000, 3)
        inf_times.append(inf_ms)

        total_count += 1
        is_valid = not diag.get("decode_fallback", False)
        if is_valid:
            valid_count += 1

        # Stage 2: Environment Step
        t_env_start = time.perf_counter()
        obs_cmd, reward, done, info = env.step(cmd_msg)
        flags = env.oversight.get_flags()
        step_flags = [f for f in flags if f.get("step") == step_num]
        env_ms = round((time.perf_counter() - t_env_start) * 1000, 3)
        env_times.append(env_ms)

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
            "decision_explanation": explanation_service.generate_explanation(cmd_msg, env.grid),
            "profiling": {
                "inference_ms": inf_ms,
                "env_step_ms": env_ms
            }
        }
        all_steps.append(step_data)

        # Stage 3: JSON Serialization
        t_ser_start = time.perf_counter()
        frame_payload = {
            "type": "step",
            "step": step_num,
            "data": step_data
        }
        json_bytes = json.dumps(frame_payload)
        ser_ms = round((time.perf_counter() - t_ser_start) * 1000, 3)
        ser_times.append(ser_ms)

        # Stage 4: WebSocket Transmission
        t_ws_start = time.perf_counter()
        await websocket.send_text(json_bytes)
        ws_ms = round((time.perf_counter() - t_ws_start) * 1000, 3)
        ws_times.append(ws_ms)

        logger.info(
            f"Step {step_num:02d}/50 Profile | Inf: {inf_ms:.1f}ms | Env: {env_ms:.2f}ms | Ser: {ser_ms:.2f}ms | WS: {ws_ms:.2f}ms"
        )

    total_mission_duration = round(time.perf_counter() - mission_start_time, 2)
    avg_inf = round(sum(inf_times) / max(1, len(inf_times)), 2)
    avg_env = round(sum(env_times) / max(1, len(env_times)), 3)
    avg_ser = round(sum(ser_times) / max(1, len(ser_times)), 3)
    avg_ws = round(sum(ws_times) / max(1, len(ws_times)), 3)
    fps = round(len(all_steps) / max(0.1, total_mission_duration), 2)

    logger.info("=" * 80)
    logger.info(f"PERFORMANCE PROFILING SUMMARY (Seed {seed}):")
    logger.info(f"  1. LLM Inference     : {avg_inf:>8.2f} ms")
    logger.info(f"  2. Environment Step  : {avg_env:>8.3f} ms")
    logger.info(f"  3. Serialization     : {avg_ser:>8.3f} ms")
    logger.info(f"  4. WebSocket Send    : {avg_ws:>8.3f} ms")
    logger.info(f"  Total Mission Time   : {total_mission_duration:>8.2f} s ({fps} FPS)")
    logger.info("=" * 80)

    # Compute final metrics
    final_metrics = metrics_service.compute_summary(
        env.grid, env.initial_total_population, env.total_reward, info, valid_count, total_count
    )
    events = simulation_service._generate_events(all_steps, final_metrics)

    await websocket.send_json({
        "type": "complete",
        "metrics": final_metrics,
        "events": events,
        "telemetry": {
            "mission_duration_s": total_mission_duration,
            "avg_inference_time_ms": avg_inf,
            "avg_env_step_time_ms": avg_env,
            "avg_serialization_time_ms": avg_ser,
            "avg_websocket_send_ms": avg_ws,
            "fps": fps
        }
    })
