"""
backend/routers/health.py
Health check endpoint for CrisisGrid FastAPI backend.
"""

from fastapi import APIRouter
from backend.core.config import settings
from backend.services.inference_service import inference_service
from backend.models.schemas.simulation import HealthResponse

router = APIRouter(tags=["Health"])


import os

try:
    import psutil
    has_psutil = True
except ImportError:
    has_psutil = False


@router.get("/health", response_model=HealthResponse)
async def health_check():
    cpu_pct = None
    mem_mb = None

    if has_psutil:
        try:
            proc = psutil.Process(os.getpid())
            cpu_pct = round(proc.cpu_percent(interval=None), 1)
            mem_mb = round(proc.memory_info().rss / (1024 * 1024), 1)
        except Exception:
            pass

    return HealthResponse(
        status="ok",
        model_loaded=inference_service.is_loaded,
        device=inference_service.device,
        version=settings.VERSION,
        cpu_usage_pct=cpu_pct,
        memory_mb=mem_mb
    )
