"""
히스토리 관련 스키마
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class HistoryQuery(BaseModel):
    """히스토리 조회 쿼리 스키마"""
    session_id: Optional[str] = Field(None, description="세션 ID 필터")
    category: Optional[str] = Field(None, description="카테고리 필터")
    search: Optional[str] = Field(None, description="검색어")
    page: int = Field(1, description="페이지 번호")
    page_size: int = Field(10, description="페이지 크기")
    sort_by: str = Field("updated_at", description="정렬 기준")
    sort_order: str = Field("desc", description="정렬 순서 (asc/desc)")


class MessageResponse(BaseModel):
    """메시지 응답 스키마"""
    message_id: str = Field(..., description="메시지 ID")
    session_id: str = Field(..., description="세션 ID")
    role: str = Field(..., description="역할 (user/assistant)")
    content: str = Field(..., description="메시지 내용")
    timestamp: datetime = Field(..., description="타임스탬프")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="메타데이터")


class HistoryResponse(BaseModel):
    """히스토리 응답 스키마"""
    messages: List[MessageResponse] = Field(..., description="메시지 목록")
    total: int = Field(..., description="전체 개수")
    page: int = Field(1, description="현재 페이지")
    page_size: int = Field(10, description="페이지 크기")


class ExportRequest(BaseModel):
    """내보내기 요청 스키마"""
    session_ids: List[str] = Field(..., description="세션 ID 목록")
    format: str = Field("json", description="내보내기 형식 (json/txt)")

