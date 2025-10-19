# -*- coding: utf-8 -*-
"""
LawFirmAI - 정적 파일 서빙이 포함된 Gradio 애플리케이션
FastAPI를 통한 정적 파일 서빙으로 manifest.json 404 오류 해결
"""

import os
import sys
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 기존 app.py의 모든 클래스와 함수 import
from gradio.app import (
    HuggingFaceSpacesApp, 
    create_gradio_interface,
    app_instance,
    logger
)

def create_app_with_static():
    """정적 파일 서빙이 포함된 FastAPI 앱 생성"""
    
    # FastAPI 앱 생성
    app = FastAPI(
        title="LawFirmAI API",
        description="법률 AI 어시스턴트 API with Static File Serving",
        version="1.0.0"
    )
    
    # CORS 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 정적 파일 마운트
    static_dir = Path("gradio/static")
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        logger.info(f"Static files mounted at /static from {static_dir}")
    else:
        logger.warning(f"Static directory not found: {static_dir}")
    
    # Gradio 인터페이스 생성
    interface = create_gradio_interface()
    
    # Gradio 앱을 FastAPI에 마운트
    app = gr.mount_gradio_app(app, interface, path="/")
    
    return app

def main():
    """메인 함수 - 정적 파일 서빙이 포함된 앱 실행"""
    logger.info("Starting LawFirmAI with static file serving...")
    
    # FastAPI 앱 생성
    app = create_app_with_static()
    
    # uvicorn으로 실행
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        log_level="info"
    )

if __name__ == "__main__":
    main()
