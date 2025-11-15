"""
세션 관련 스키마
"""
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class SessionCreate(BaseModel):
    """세션 생성 스키마"""
    title: Optional[str] = Field(None, max_length=255, description="세션 제목")
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v):
        if v and len(v.strip()) == 0:
            raise ValueError('세션 제목은 비어있을 수 없습니다')
        if v and len(v) > 255:
            raise ValueError('세션 제목은 255자를 초과할 수 없습니다')
        return v.strip() if v else None


class SessionUpdate(BaseModel):
    """세션 업데이트 스키마"""
    title: Optional[str] = Field(None, max_length=255, description="세션 제목")
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v):
        if v and len(v.strip()) == 0:
            raise ValueError('세션 제목은 비어있을 수 없습니다')
        if v and len(v) > 255:
            raise ValueError('세션 제목은 255자를 초과할 수 없습니다')
        return v.strip() if v else None


class SessionResponse(BaseModel):
    """세션 응답 스키마"""
    session_id: str = Field(..., description="세션 ID")
    title: Optional[str] = Field(None, description="세션 제목")
    created_at: Optional[str] = Field(None, description="생성 시간")
    updated_at: Optional[str] = Field(None, description="수정 시간")
    message_count: int = Field(0, description="메시지 개수")
    user_id: Optional[str] = Field(None, description="사용자 ID")
    ip_address: Optional[str] = Field(None, description="IP 주소")
    
    class Config:
        from_attributes = True
        extra = "ignore"  # 추가 필드는 무시


class SessionListResponse(BaseModel):
    """세션 목록 응답 스키마"""
    sessions: List[SessionResponse] = Field(..., description="세션 목록")
    total: int = Field(..., description="전체 개수")
    page: int = Field(1, description="현재 페이지")
    page_size: int = Field(10, description="페이지 크기")

