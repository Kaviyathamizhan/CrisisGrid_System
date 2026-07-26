"""
backend/models/schemas/simulation.py
Pydantic schemas for API requests, responses, grid steps, metrics, and WebSocket frames.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = False
    device: str = "cpu"
    version: str = "2.0.0"


class SeedsResponse(BaseModel):
    seeds: List[int] = Field(default_factory=lambda: [123, 42, 999])


class SimulationRequest(BaseModel):
    seed: int = Field(default=123, description="RNG seed for scenario reproducibility")
    mode: str = Field(default="live", description="Execution mode: 'live' or 'replay'")


class MetricsSummary(BaseModel):
    final_survival: float
    population_saved: int
    initial_population: int
    total_reward: float
    agent_reliability: float
    active_emergencies: int
    resource_efficiency: float


class EventItem(BaseModel):
    step: int
    text: str
    type: str  # 'info' | 'warning' | 'critical' | 'success'


class SimulationResponse(BaseModel):
    agent_type: str = "trained"
    seed: int
    mode: str
    steps: List[Dict[str, Any]]
    survival_curve: List[float]
    severity_curve: List[float]
    events: List[EventItem]
    metrics: MetricsSummary


class ComparisonMeta(BaseModel):
    survival_delta: float = Field(..., description="Trained final survival minus random final survival (percentage points)")
    population_saved_delta: int = Field(..., description="Trained survivors saved minus random survivors saved")
    policies_match: bool = Field(..., description="True if trained and random trajectories are identical (should be False)")
    decision_similarity: float = Field(..., description="Fraction of timesteps where trained and random actions matched (0.0 to 1.0)")


class ComparisonResponse(BaseModel):
    """Response containing both trained and random baseline trajectories for side-by-side comparison."""
    seed: int
    mode: str
    trained: SimulationResponse
    random: SimulationResponse
    comparison: ComparisonMeta


class WSFrame(BaseModel):
    type: str  # "init" | "step" | "complete" | "error"
    step: Optional[int] = None
    data: Dict[str, Any]
