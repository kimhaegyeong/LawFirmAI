"""
채팅 관련 스키마
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """채팅 요청 스키마"""
    message: str = Field(..., description="사용자 메시지")
    session_id: Optional[str] = Field(None, description="세션 ID")
    context: Optional[str] = Field(None, description="추가 컨텍스트")
    enable_checkpoint: bool = Field(True, description="체크포인트 사용 여부")


class ChatResponse(BaseModel):
    """채팅 응답 스키마"""
    answer: str = Field(..., description="AI 답변")
    sources: List[str] = Field(default_factory=list, description="참고 출처")
    confidence: float = Field(..., description="신뢰도 (0.0 ~ 1.0)")
    legal_references: List[str] = Field(default_factory=list, description="법률 참조")
    processing_steps: List[str] = Field(default_factory=list, description="처리 단계")
    session_id: str = Field(..., description="세션 ID")
    processing_time: float = Field(..., description="처리 시간 (초)")
    query_type: str = Field(default="", description="질문 유형")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="메타데이터")
    errors: List[str] = Field(default_factory=list, description="에러 목록")
    warnings: List[str] = Field(default_factory=list, description="경고 메시지 목록")


class StreamingChatRequest(BaseModel):
    """스트리밍 채팅 요청 스키마"""
    message: str = Field(..., description="사용자 메시지")
    session_id: Optional[str] = Field(None, description="세션 ID")
    context: Optional[str] = Field(None, description="추가 컨텍스트")

