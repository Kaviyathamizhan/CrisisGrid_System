"""
backend/routers/simulation.py
REST API endpoints for seeds, cached replay, comparison, and mission simulation.
"""

from fastapi import APIRouter, Query, HTTPException
from backend.services.replay_service import replay_service
from backend.services.simulation_service import simulation_service
from backend.models.schemas.simulation import (
    SeedsResponse, SimulationRequest, SimulationResponse, ComparisonResponse
)

router = APIRouter(tags=["Simulation"])


@router.get("/seeds", response_model=SeedsResponse)
async def get_available_seeds():
    seeds = replay_service.get_available_seeds()
    return SeedsResponse(seeds=seeds)


@router.get("/replay", response_model=SimulationResponse)
async def get_replay_trajectory(seed: int = Query(default=123)):
    trajectory = simulation_service.run_replay(seed)
    if not trajectory:
        raise HTTPException(status_code=404, detail=f"No trajectory available for seed {seed}")
    return trajectory


@router.get("/comparison", response_model=ComparisonResponse)
async def get_comparison_trajectories(seed: int = Query(default=123)):
    """Return both trained and random baseline trajectories for side-by-side comparison."""
    comparison = simulation_service.run_comparison_replay(seed)
    if not comparison:
        raise HTTPException(status_code=404, detail=f"No comparison data for seed {seed}")
    return comparison


@router.post("/simulate", response_model=SimulationResponse)
async def run_simulation(req: SimulationRequest):
    if req.mode == "live":
        return simulation_service.run_live_simulation(req.seed)
    else:
        return simulation_service.run_replay(req.seed)
