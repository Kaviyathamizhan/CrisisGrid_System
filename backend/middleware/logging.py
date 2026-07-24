"""
backend/middleware/logging.py
HTTP Request logging middleware to record response latencies.
"""

import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from backend.core.logging import get_logger

logger = get_logger("HTTPLogger")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        logger.info(f"{request.method} {request.url.path} - Status: {response.status_code} - Duration: {process_time:.2f}ms")
        return response
