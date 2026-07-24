"""
backend/main.py
FastAPI application entry point for CrisisGrid AI Operations Platform.
Loads model ONCE during server startup via lifespan context manager.
"""

import sys
import os
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI

# Add repo root to import path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.middleware.cors import setup_cors
from backend.middleware.logging import RequestLoggingMiddleware
from backend.middleware.error_handler import setup_error_handlers
from backend.services.inference_service import inference_service
from backend.routers import health, simulation, websocket

logger = get_logger("Main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to perform background startup model loading and cleanup."""
    logger.info("Initializing CrisisGrid FastAPI Backend...")
    logger.info("Triggering background PyTorch model loading thread...")
    
    # Launch model loading in background thread so server binds port immediately
    thread = threading.Thread(target=inference_service.load_model, daemon=True)
    thread.start()
        
    logger.info("Server startup complete. Ready to receive HTTP & WebSocket connections.")
    yield
    logger.info("Shutting down CrisisGrid FastAPI Backend...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# Setup Middlewares
setup_cors(app)
app.add_middleware(RequestLoggingMiddleware)
setup_error_handlers(app)

# Include Routers
app.include_router(health.router, prefix=settings.API_PREFIX)
app.include_router(simulation.router, prefix=settings.API_PREFIX)
app.include_router(websocket.router, prefix=settings.API_PREFIX)


@app.get("/")
async def root():
    return {
        "title": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "api_health": f"{settings.API_PREFIX}/health"
    }
