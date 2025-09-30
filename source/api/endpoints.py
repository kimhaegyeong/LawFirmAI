# -*- coding: utf-8 -*-
"""
API Endpoints
RESTful API 엔드포인트 정의
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
from pydantic import BaseModel
from source.utils.config import Config
from source.utils.logger import get_logger
from source.services.chat_service import ChatService

logger = get_logger(__name__)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    confidence: float
    sources: list
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str

def setup_routes(app, config: Config):
    """라우트 설정"""
    
    # Initialize services
    chat_service = ChatService(config)
    
    # Create API router
    api_router = APIRouter(prefix="/api/v1")
    
    @api_router.post("/chat", response_model=ChatResponse)
    async def chat_endpoint(request: ChatRequest):
        """채팅 엔드포인트"""
        try:
            logger.info(f"Chat request received: {request.message[:100]}...")
            
            result = chat_service.process_message(
                message=request.message,
                context=request.context
            )
            
            return ChatResponse(**result)
            
        except Exception as e:
            logger.error(f"Chat endpoint error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @api_router.get("/health", response_model=HealthResponse)
    async def health_check():
        """헬스체크 엔드포인트"""
        return HealthResponse(
            status="healthy",
            service="LawFirmAI API",
            version="1.0.0"
        )
    
    # Include router in app
    app.include_router(api_router)
    
    logger.info("API routes configured successfully")
