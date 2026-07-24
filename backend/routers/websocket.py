"""
backend/routers/websocket.py
WebSocket streaming endpoint for step-by-step live mission playback.
"""

import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from backend.core.logging import get_logger
from backend.services.simulation_service import simulation_service

logger = get_logger("WebSocketRouter")
router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/simulate")
async def websocket_simulate(websocket: WebSocket, seed: int = Query(default=123)):
    await websocket.accept()
    logger.info(f"WebSocket connection established for seed {seed}")

    try:
        # Generate or load simulation payload
        sim_data = simulation_service.run_replay(seed)
        steps = sim_data["steps"]
        total_steps = len(steps)

        # Send init frame
        await websocket.send_json({
            "type": "init",
            "seed": seed,
            "total_steps": total_steps - 1,
            "metrics": sim_data["metrics"]
        })

        # Stream steps sequentially
        for idx, step_data in enumerate(steps):
            await websocket.send_json({
                "type": "step",
                "step": idx,
                "data": step_data
            })
            # Small streaming delay between frames (e.g. 100ms)
            await asyncio.sleep(0.1)

        # Send completion frame
        await websocket.send_json({
            "type": "complete",
            "metrics": sim_data["metrics"],
            "events": sim_data["events"]
        })
        logger.info(f"WebSocket simulation streaming complete for seed {seed}")

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
