"""
헬스체크 관련 스키마
"""
from pydantic import BaseModel, Field
from datetime import datetime


class HealthResponse(BaseModel):
    """헬스체크 응답 스키마"""
    status: str = Field(..., description="상태 (healthy/unhealthy)")
    timestamp: str = Field(..., description="타임스탬프")
    chat_service_available: bool = Field(..., description="ChatService 사용 가능 여부")

