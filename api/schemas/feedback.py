"""
피드백 관련 스키마
"""
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import re


class FeedbackRequest(BaseModel):
    """피드백 요청 스키마"""
    session_id: str = Field(..., description="세션 ID")
    message_id: Optional[str] = Field(None, description="메시지 ID")
    rating: int = Field(..., ge=1, le=5, description="평점 (1-5)")
    comment: Optional[str] = Field(None, max_length=5000, description="의견")
    feedback_type: str = Field("general", max_length=50, description="피드백 유형")
    
    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v):
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        if not re.match(uuid_pattern, v, re.IGNORECASE):
            raise ValueError('유효하지 않은 세션 ID 형식입니다')
        return v
    
    @field_validator('message_id')
    @classmethod
    def validate_message_id(cls, v):
        if v:
            uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
            if not re.match(uuid_pattern, v, re.IGNORECASE):
                raise ValueError('유효하지 않은 메시지 ID 형식입니다')
        return v
    
    @field_validator('comment')
    @classmethod
    def validate_comment(cls, v):
        if v and len(v) > 5000:
            raise ValueError('의견은 5000자를 초과할 수 없습니다')
        return v
    
    @field_validator('feedback_type')
    @classmethod
    def validate_feedback_type(cls, v):
        allowed_types = ['general', 'accuracy', 'helpfulness', 'speed', 'other']
        if v not in allowed_types:
            raise ValueError(f'피드백 유형은 {allowed_types} 중 하나여야 합니다')
        return v


class FeedbackResponse(BaseModel):
    """피드백 응답 스키마"""
    feedback_id: str = Field(..., description="피드백 ID")
    session_id: str = Field(..., description="세션 ID")
    message_id: Optional[str] = Field(None, description="메시지 ID")
    rating: int = Field(..., description="평점")
    comment: Optional[str] = Field(None, description="의견")
    timestamp: datetime = Field(..., description="타임스탬프")

