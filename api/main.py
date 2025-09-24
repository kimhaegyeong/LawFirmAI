#!/usr/bin/env python3
"""
LawFirmAI - FastAPI Backend
RESTful API 서버
"""

import os
import sys
import logging
from pathlib import Path

# Add source directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "source"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from api.endpoints import setup_routes
from api.middleware import setup_middleware
from utils.config import Config
from utils.logger import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """FastAPI 애플리케이션 생성"""
    try:
        # Initialize configuration
        config = Config()
        
        # Create FastAPI app
        app = FastAPI(
            title="LawFirmAI API",
            description="법률 AI 어시스턴트 RESTful API",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json"
        )
        
        # Setup middleware
        setup_middleware(app, config)
        
        # Setup routes
        setup_routes(app, config)
        
        # Health check endpoint
        @app.get("/health")
        async def health_check():
            """헬스체크 엔드포인트"""
            return {
                "status": "healthy",
                "service": "LawFirmAI API",
                "version": "1.0.0"
            }
        
        # Root endpoint
        @app.get("/")
        async def root():
            """루트 엔드포인트"""
            return {
                "message": "LawFirmAI API",
                "version": "1.0.0",
                "docs": "/docs"
            }
        
        # Global exception handler
        @app.exception_handler(Exception)
        async def global_exception_handler(request, exc):
            """글로벌 예외 처리"""
            logger.error(f"Global exception: {exc}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": str(exc) if os.getenv("DEBUG", "false").lower() == "true" else "An error occurred"
                }
            )
        
        logger.info("FastAPI application created successfully")
        return app
        
    except Exception as e:
        logger.error(f"Failed to create FastAPI application: {e}")
        raise

def main():
    """메인 실행 함수"""
    try:
        logger.info("Starting LawFirmAI FastAPI server...")
        
        # Create app
        app = create_app()
        
        # Get configuration
        config = Config()
        host = config.get("API_HOST", "0.0.0.0")
        port = int(config.get("API_PORT", 8000))
        debug = config.get("DEBUG", "false").lower() == "true"
        
        # Run server
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=debug,
            log_level="info" if not debug else "debug"
        )
        
    except Exception as e:
        logger.error(f"Failed to start FastAPI server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
