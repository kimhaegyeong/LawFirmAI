"""
헬스체크 엔드포인트
"""
from fastapi import APIRouter
from datetime import datetime
import asyncio

from api.services.chat_service import get_chat_service
from api.schemas.health import HealthResponse

router = APIRouter()

# 초기화 상태 추적
_warmup_started = False
_warmup_task = None

async def _warmup_chat_service():
    """ChatService 초기화 (비동기)"""
    global _warmup_started
    if _warmup_started:
        return
    
    _warmup_started = True
    try:
        chat_service = get_chat_service()
        # 초기화가 완료될 때까지 대기하지 않음 (백그라운드에서 진행)
        return chat_service
    except Exception as e:
        # 초기화 실패는 무시 (다음 요청에서 재시도)
        pass

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스체크 (ChatService 초기화 트리거)"""
    global _warmup_task
    
    # 백그라운드에서 초기화 시작 (이미 시작되지 않은 경우)
    if _warmup_task is None or _warmup_task.done():
        _warmup_task = asyncio.create_task(_warmup_chat_service())
    
    # ChatService 상태 확인 (초기화 중이어도 응답 반환)
    try:
        chat_service = get_chat_service()
        chat_service_available = chat_service.is_available()
    except Exception:
        chat_service_available = False
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        chat_service_available=chat_service_available
    )

