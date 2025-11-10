"""
히스토리 관련 스키마
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import re


class HistoryQuery(BaseModel):
    """히스토리 조회 쿼리 스키마"""
    session_id: Optional[str] = Field(None, description="세션 ID 필터")
    category: Optional[str] = Field(None, max_length=100, description="카테고리 필터")
    search: Optional[str] = Field(None, max_length=200, description="검색어")
    page: int = Field(1, ge=1, description="페이지 번호")
    page_size: int = Field(10, ge=1, le=100, description="페이지 크기")
    sort_by: str = Field("updated_at", description="정렬 기준")
    sort_order: str = Field("desc", description="정렬 순서 (asc/desc)")
    
    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v):
        if v:
            uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
            if not re.match(uuid_pattern, v, re.IGNORECASE):
                raise ValueError('유효하지 않은 세션 ID 형식입니다')
        return v
    
    @field_validator('category')
    @classmethod
    def validate_category(cls, v):
        if v and len(v) > 100:
            raise ValueError('카테고리는 100자를 초과할 수 없습니다')
        if v and not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('카테고리는 영숫자, 하이픈, 언더스코어만 허용됩니다')
        return v
    
    @field_validator('search')
    @classmethod
    def validate_search(cls, v):
        if v and len(v) > 200:
            raise ValueError('검색어는 200자를 초과할 수 없습니다')
        return v
    
    @field_validator('sort_by')
    @classmethod
    def validate_sort_by(cls, v):
        allowed_fields = ['created_at', 'updated_at', 'title', 'message_count']
        if v not in allowed_fields:
            raise ValueError(f'정렬 기준은 {allowed_fields} 중 하나여야 합니다')
        return v
    
    @field_validator('sort_order')
    @classmethod
    def validate_sort_order(cls, v):
        if v not in ['asc', 'desc']:
            raise ValueError('정렬 순서는 asc 또는 desc여야 합니다')
        return v


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
    session_ids: List[str] = Field(..., min_items=1, max_items=100, description="세션 ID 목록")
    format: str = Field("json", description="내보내기 형식 (json/txt)")
    
    @field_validator('session_ids')
    @classmethod
    def validate_session_ids(cls, v):
        if not v:
            raise ValueError('세션 ID 목록은 비어있을 수 없습니다')
        if len(v) > 100:
            raise ValueError('세션 ID는 최대 100개까지 내보낼 수 있습니다')
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        for session_id in v:
            if not re.match(uuid_pattern, session_id, re.IGNORECASE):
                raise ValueError(f'유효하지 않은 세션 ID 형식입니다: {session_id}')
        return v
    
    @field_validator('format')
    @classmethod
    def validate_format(cls, v):
        allowed_formats = ['json', 'txt']
        if v not in allowed_formats:
            raise ValueError(f'내보내기 형식은 {allowed_formats} 중 하나여야 합니다')
        return v

