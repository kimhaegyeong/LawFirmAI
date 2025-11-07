"""
헬스체크 엔드포인트
"""
from fastapi import APIRouter
from datetime import datetime

from api.services.chat_service import get_chat_service

router = APIRouter()


@router.get("/health")
async def health_check():
    """헬스체크"""
    chat_service = get_chat_service()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "chat_service_available": chat_service.is_available()
    }

