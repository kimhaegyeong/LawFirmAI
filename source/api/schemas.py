"""
API Schemas
Pydantic 모델 정의
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    """문서 유형 열거형"""
    CONTRACT = "contract"
    CASE = "case"
    LAW = "law"
    GENERAL = "general"


class SearchType(str, Enum):
    """검색 유형 열거형"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


# 요청 모델
class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    message: str = Field(..., description="사용자 메시지", min_length=1, max_length=10000)
    context: Optional[str] = Field(None, description="추가 컨텍스트", max_length=5000)
    session_id: Optional[str] = Field(None, description="세션 ID", max_length=100)
    
    @validator('message')
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('메시지는 비어있을 수 없습니다')
        return v.strip()


class DocumentUploadRequest(BaseModel):
    """문서 업로드 요청 모델"""
    title: str = Field(..., description="문서 제목", min_length=1, max_length=500)
    content: str = Field(..., description="문서 내용", min_length=10, max_length=100000)
    document_type: DocumentType = Field(..., description="문서 유형")
    source_url: Optional[str] = Field(None, description="출처 URL", max_length=1000)
    
    @validator('title')
    def validate_title(cls, v):
        if not v or not v.strip():
            raise ValueError('제목은 비어있을 수 없습니다')
        return v.strip()
    
    @validator('content')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError('내용은 비어있을 수 없습니다')
        return v.strip()


class SearchRequest(BaseModel):
    """검색 요청 모델"""
    query: str = Field(..., description="검색 쿼리", min_length=1, max_length=1000)
    search_type: SearchType = Field(SearchType.HYBRID, description="검색 유형")
    limit: int = Field(10, description="결과 개수", ge=1, le=100)
    
    @validator('query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('검색 쿼리는 비어있을 수 없습니다')
        return v.strip()


class AnalysisRequest(BaseModel):
    """분석 요청 모델"""
    document: DocumentUploadRequest = Field(..., description="분석할 문서")
    analysis_type: str = Field("general", description="분석 유형")


# 응답 모델
class ChatResponse(BaseModel):
    """채팅 응답 모델"""
    response: str = Field(..., description="AI 응답")
    confidence: float = Field(..., description="신뢰도", ge=0.0, le=1.0)
    sources: List[Dict[str, Any]] = Field(..., description="참조 소스")
    processing_time: float = Field(..., description="처리 시간 (초)")
    model: str = Field(..., description="사용된 모델")
    retrieved_docs_count: Optional[int] = Field(None, description="검색된 문서 수")


class DocumentResponse(BaseModel):
    """문서 응답 모델"""
    id: int = Field(..., description="문서 ID")
    title: str = Field(..., description="문서 제목")
    content: str = Field(..., description="문서 내용")
    document_type: DocumentType = Field(..., description="문서 유형")
    source_url: Optional[str] = Field(None, description="출처 URL")
    created_at: datetime = Field(..., description="생성 시간")
    updated_at: datetime = Field(..., description="수정 시간")


class SearchResult(BaseModel):
    """검색 결과 모델"""
    document_id: int = Field(..., description="문서 ID")
    title: str = Field(..., description="문서 제목")
    content: str = Field(..., description="문서 내용")
    similarity: Optional[float] = Field(None, description="유사도 점수")
    search_type: str = Field(..., description="검색 유형")
    matched_keywords: Optional[List[str]] = Field(None, description="매칭된 키워드")
    chunk_index: Optional[int] = Field(None, description="청크 인덱스")


class SearchResponse(BaseModel):
    """검색 응답 모델"""
    results: List[SearchResult] = Field(..., description="검색 결과")
    total_count: int = Field(..., description="총 결과 수")
    query: str = Field(..., description="검색 쿼리")
    search_type: str = Field(..., description="검색 유형")
    processing_time: float = Field(..., description="처리 시간 (초)")


class AnalysisResult(BaseModel):
    """분석 결과 모델"""
    summary: str = Field(..., description="문서 요약")
    risk_factors: List[Dict[str, Any]] = Field(..., description="위험 요소")
    key_clauses: List[Dict[str, Any]] = Field(..., description="핵심 조항")
    missing_elements: List[str] = Field(..., description="누락된 요소")
    recommendations: List[str] = Field(..., description="권장사항")
    entities: Dict[str, List[str]] = Field(..., description="추출된 엔티티")
    word_count: int = Field(..., description="단어 수")
    confidence: float = Field(..., description="분석 신뢰도")


class AnalysisResponse(BaseModel):
    """분석 응답 모델"""
    analysis: AnalysisResult = Field(..., description="분석 결과")
    document_id: int = Field(..., description="문서 ID")
    processing_time: float = Field(..., description="처리 시간 (초)")


class HealthResponse(BaseModel):
    """헬스체크 응답 모델"""
    status: str = Field(..., description="서비스 상태")
    service: str = Field(..., description="서비스명")
    version: str = Field(..., description="버전")
    timestamp: datetime = Field(..., description="체크 시간")
    models: Dict[str, Any] = Field(..., description="모델 상태")


class ErrorResponse(BaseModel):
    """에러 응답 모델"""
    error: str = Field(..., description="에러 메시지")
    detail: Optional[str] = Field(None, description="상세 에러 정보")
    timestamp: datetime = Field(..., description="에러 발생 시간")


class StatsResponse(BaseModel):
    """통계 응답 모델"""
    total_documents: int = Field(..., description="총 문서 수")
    total_embeddings: int = Field(..., description="총 임베딩 수")
    model_status: Dict[str, Any] = Field(..., description="모델 상태")
    vector_store_stats: Dict[str, Any] = Field(..., description="벡터 저장소 통계")
    database_stats: Dict[str, Any] = Field(..., description="데이터베이스 통계")


# 유틸리티 모델
class PaginationParams(BaseModel):
    """페이지네이션 파라미터"""
    page: int = Field(1, description="페이지 번호", ge=1)
    size: int = Field(10, description="페이지 크기", ge=1, le=100)
    
    @property
    def offset(self) -> int:
        return (self.page - 1) * self.size


class SortParams(BaseModel):
    """정렬 파라미터"""
    sort_by: str = Field("created_at", description="정렬 기준")
    sort_order: str = Field("desc", description="정렬 순서")
    
    @validator('sort_order')
    def validate_sort_order(cls, v):
        if v not in ['asc', 'desc']:
            raise ValueError('정렬 순서는 asc 또는 desc여야 합니다')
        return v


class FilterParams(BaseModel):
    """필터 파라미터"""
    document_type: Optional[DocumentType] = Field(None, description="문서 유형 필터")
    date_from: Optional[datetime] = Field(None, description="시작 날짜")
    date_to: Optional[datetime] = Field(None, description="종료 날짜")
    keyword: Optional[str] = Field(None, description="키워드 필터", max_length=100)
