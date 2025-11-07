"""
세션 관련 스키마
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    """세션 생성 스키마"""
    title: Optional[str] = Field(None, description="세션 제목")
    category: Optional[str] = Field(None, description="카테고리")


class SessionUpdate(BaseModel):
    """세션 업데이트 스키마"""
    title: Optional[str] = Field(None, description="세션 제목")
    category: Optional[str] = Field(None, description="카테고리")


class SessionResponse(BaseModel):
    """세션 응답 스키마"""
    session_id: str = Field(..., description="세션 ID")
    title: Optional[str] = Field(None, description="세션 제목")
    category: Optional[str] = Field(None, description="카테고리")
    created_at: Optional[str] = Field(None, description="생성 시간")
    updated_at: Optional[str] = Field(None, description="수정 시간")
    message_count: int = Field(0, description="메시지 개수")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="메타데이터")
    
    class Config:
        from_attributes = True


class SessionListResponse(BaseModel):
    """세션 목록 응답 스키마"""
    sessions: List[SessionResponse] = Field(..., description="세션 목록")
    total: int = Field(..., description="전체 개수")
    page: int = Field(1, description="현재 페이지")
    page_size: int = Field(10, description="페이지 크기")

