#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
FastAPI 서버 실행 스크립트
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..api.endpoints import setup_routes
from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)

def create_app():
    """FastAPI 앱 생성"""
    app = FastAPI(
        title="LawFirmAI API",
        description="지능형 법률 AI 어시스턴트 API",
        version="2.0.0"
    )

    # CORS 미들웨어 추가
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 설정 로드
    config = Config()

    # 라우트 설정
    setup_routes(app, config)

    return app

def main():
    """메인 함수"""
    import uvicorn

    app = create_app()

    logger.info("LawFirmAI API 서버 시작")
    logger.info("서버 주소: http://localhost:8000")
    logger.info("API 문서: http://localhost:8000/docs")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main()
