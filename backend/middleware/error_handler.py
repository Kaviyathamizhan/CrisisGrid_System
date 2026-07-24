"""
backend/middleware/error_handler.py
Global error handler middleware catching unhandled exceptions and returning JSON errors.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from backend.core.logging import get_logger

logger = get_logger("ErrorHandler")


def setup_error_handlers(app: FastAPI):
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled Exception on {request.url.path}: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "An internal server error occurred.",
                "detail": str(exc)
            }
        )
