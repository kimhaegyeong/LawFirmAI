"""
CORS 미들웨어 설정
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.config import APIConfig
import logging

logger = logging.getLogger(__name__)


def setup_cors(app: FastAPI, config: APIConfig):
    """CORS 설정"""
    import sys
    cors_origins = config.get_cors_origins()
    logger.info(f"CORS origins 설정: {cors_origins}")
    print(f"✅ CORS 미들웨어 설정 완료: {cors_origins}", flush=True)
    sys.stdout.flush()
    
    # CORS 미들웨어는 가장 먼저 추가되어야 함
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],  # 모든 메서드 허용
        allow_headers=["*"],  # 모든 헤더 허용
        expose_headers=["*"],  # 모든 헤더 노출
        max_age=600,  # preflight 캐시 시간
    )

