"""
피드백 관련 스키마
"""
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    """피드백 요청 스키마"""
    session_id: str = Field(..., description="세션 ID")
    message_id: Optional[str] = Field(None, description="메시지 ID")
    rating: int = Field(..., ge=1, le=5, description="평점 (1-5)")
    comment: Optional[str] = Field(None, description="의견")
    feedback_type: str = Field("general", description="피드백 유형")


class FeedbackResponse(BaseModel):
    """피드백 응답 스키마"""
    feedback_id: str = Field(..., description="피드백 ID")
    session_id: str = Field(..., description="세션 ID")
    message_id: Optional[str] = Field(None, description="메시지 ID")
    rating: int = Field(..., description="평점")
    comment: Optional[str] = Field(None, description="의견")
    timestamp: datetime = Field(..., description="타임스탬프")

