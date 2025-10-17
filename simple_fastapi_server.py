#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 FastAPI 서버 테스트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import sqlite3
import time

# 간단한 요청/응답 모델
class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    confidence: float
    sources: list
    processing_time: float

class SystemStatusResponse(BaseModel):
    status: str
    timestamp: float
    database_status: str
    total_articles: int
    version: str

def get_database_path():
    """데이터베이스 파일의 절대 경로를 반환"""
    import os
    # 현재 스크립트의 디렉토리를 기준으로 데이터베이스 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "data", "lawfirm.db")
    return db_path

def create_simple_app():
    """간단한 FastAPI 앱 생성"""
    app = FastAPI(
        title="LawFirmAI API (Simple)",
        description="간단한 법률 AI 어시스턴트 API",
        version="1.0.0"
    )
    
    # CORS 미들웨어 추가
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        """루트 엔드포인트"""
        return {
            "message": "LawFirmAI API 서버가 실행 중입니다",
            "version": "1.0.0",
            "status": "running"
        }
    
    @app.get("/health")
    async def health_check():
        """헬스 체크 엔드포인트"""
        return {"status": "healthy", "timestamp": time.time()}
    
    @app.get("/api/v1/system/status", response_model=SystemStatusResponse)
    async def system_status():
        """시스템 상태 확인"""
        try:
            # 데이터베이스 연결 테스트
            db_path = get_database_path()
            print(f"데이터베이스 경로: {db_path}")  # 디버깅용
            
            if not os.path.exists(db_path):
                return SystemStatusResponse(
                    status="error",
                    timestamp=time.time(),
                    database_status=f"Database file not found: {db_path}",
                    total_articles=0,
                    version="1.0.0"
                )
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM assembly_articles")
            article_count = cursor.fetchone()[0]
            conn.close()
            
            return SystemStatusResponse(
                status="healthy",
                timestamp=time.time(),
                database_status="connected",
                total_articles=article_count,
                version="1.0.0"
            )
        except Exception as e:
            print(f"데이터베이스 연결 오류: {e}")  # 디버깅용
            return SystemStatusResponse(
                status="error",
                timestamp=time.time(),
                database_status=f"error: {str(e)}",
                total_articles=0,
                version="1.0.0"
            )
    
    @app.post("/api/v1/chat", response_model=ChatResponse)
    async def simple_chat(request: ChatRequest):
        """간단한 채팅 엔드포인트"""
        start_time = time.time()
        
        try:
            # 간단한 응답 생성
            response_text = f"테스트 응답: '{request.message}'에 대한 답변입니다."
            
            # 데이터베이스에서 관련 정보 검색
            db_path = get_database_path()
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 간단한 검색
            search_query = request.message[:10]  # 처음 10자만 사용
            cursor.execute("""
                SELECT law_id, article_number, article_content 
                FROM assembly_articles 
                WHERE article_content LIKE ? 
                LIMIT 3
            """, (f"%{search_query}%",))
            
            results = cursor.fetchall()
            sources = []
            
            for law_id, article_number, content in results:
                sources.append({
                    "law_name": law_id,  # law_id를 law_name으로 표시
                    "article_number": article_number,
                    "content": content[:100] + "..." if len(content) > 100 else content
                })
            
            conn.close()
            
            processing_time = time.time() - start_time
            
            return ChatResponse(
                response=response_text,
                confidence=0.8,
                sources=sources,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ChatResponse(
                response=f"오류가 발생했습니다: {str(e)}",
                confidence=0.0,
                sources=[],
                processing_time=processing_time
            )
    
    @app.get("/api/v1/search")
    async def simple_search(q: str = "손해배상", limit: int = 5):
        """간단한 검색 엔드포인트"""
        try:
            db_path = get_database_path()
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT law_id, article_number, article_content 
                FROM assembly_articles 
                WHERE article_content LIKE ? 
                LIMIT ?
            """, (f"%{q}%", limit))
            
            results = cursor.fetchall()
            conn.close()
            
            search_results = []
            for law_id, article_number, content in results:
                search_results.append({
                    "law_name": law_id,  # law_id를 law_name으로 표시
                    "article_number": article_number,
                    "content": content[:200] + "..." if len(content) > 200 else content,
                    "relevance": 0.8  # 임시 관련도
                })
            
            return {
                "query": q,
                "results": search_results,
                "total_count": len(search_results),
                "processing_time": 0.1
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app

def main():
    """메인 함수"""
    import uvicorn
    
    app = create_simple_app()
    
    print("LawFirmAI 간단한 API 서버 시작")
    print("서버 주소: http://localhost:8000")
    print("API 문서: http://localhost:8000/docs")
    print("헬스 체크: http://localhost:8000/health")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main()
