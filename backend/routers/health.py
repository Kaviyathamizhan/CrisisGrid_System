"""
backend/routers/health.py
Health check endpoint for CrisisGrid FastAPI backend.
"""

from fastapi import APIRouter
from backend.core.config import settings
from backend.services.inference_service import inference_service
from backend.models.schemas.simulation import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=inference_service.is_loaded,
        device=inference_service.device,
        version=settings.VERSION
    )
