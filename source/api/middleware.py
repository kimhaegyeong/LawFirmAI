# -*- coding: utf-8 -*-
"""
API Middleware
미들웨어 설정 및 관리
"""

import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


def setup_middleware(app: FastAPI, config: Config):
    """미들웨어 설정"""

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Trusted host middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure appropriately for production
    )

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request, call_next):
        """요청 로깅 미들웨어"""
        start_time = time.time()

        response = await call_next(request)

        process_time = time.time() - start_time

        logger.info(
            "Request processed",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            process_time=process_time
        )

        return response

    logger.info("Middleware configured successfully")
