"""
채팅 관련 스키마
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
import re


class ChatRequest(BaseModel):
    """채팅 요청 스키마"""
    message: str = Field(
        ..., 
        description="사용자 메시지",
        min_length=1,
        max_length=10000
    )
    session_id: Optional[str] = Field(None, description="세션 ID")
    context: Optional[str] = Field(None, max_length=5000, description="추가 컨텍스트")
    enable_checkpoint: bool = Field(True, description="체크포인트 사용 여부")
    image_base64: Optional[str] = Field(None, description="Base64 인코딩된 이미지 (OCR 처리용)")
    file_base64: Optional[str] = Field(None, description="Base64 인코딩된 파일 (이미지, 텍스트, PDF, DOCX)")
    filename: Optional[str] = Field(None, max_length=255, description="파일명")
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('메시지는 비어있을 수 없습니다')
        dangerous_patterns = [
            r'(\bOR\b|\bAND\b)\s*[\'"]?\d+[\'"]?\s*=\s*[\'"]?\d+[\'"]?',
            r'(\bOR\b|\bAND\b)\s*[\'"]?[\w]+[\'"]?\s*=\s*[\'"]?[\w]+[\'"]?',
            r';\s*(DROP|DELETE|UPDATE|INSERT|CREATE|ALTER)',
            r'(\bUNION\b|\bSELECT\b).*(\bFROM\b|\bWHERE\b)',
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError('잘못된 입력 형식입니다')
        
        # XSS 패턴 검사
        xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>',
        ]
        for pattern in xss_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError('잘못된 입력 형식입니다')
        
        return v.strip()
    
    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v):
        if v:
            uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
            if not re.match(uuid_pattern, v, re.IGNORECASE):
                raise ValueError('유효하지 않은 세션 ID 형식입니다')
        return v
    
    @field_validator('context')
    @classmethod
    def validate_context(cls, v):
        if v:
            if len(v) > 5000:
                raise ValueError('컨텍스트는 5000자를 초과할 수 없습니다')
        return v
    
    @field_validator('file_base64')
    @classmethod
    def validate_file_base64(cls, v):
        if v:
            # Base64 크기 제한 (10MB로 축소)
            max_base64_size = 10 * 1024 * 1024  # 10MB
            if len(v) > max_base64_size:
                raise ValueError('파일 크기가 너무 큽니다. 최대 10MB까지 업로드 가능합니다.')
            # Base64 형식 검증
            import base64
            try:
                if v.startswith('data:'):
                    v = v.split(',', 1)[1]
                base64.b64decode(v, validate=True)
            except Exception:
                raise ValueError('유효하지 않은 Base64 형식입니다.')
        return v
    
    @field_validator('filename')
    @classmethod
    def validate_filename(cls, v):
        if v:
            # 파일명 검증
            from api.services.file_validator import validate_filename
            is_valid, error_msg = validate_filename(v)
            if not is_valid:
                raise ValueError(error_msg)
        return v


class ContinueAnswerRequest(BaseModel):
    """계속 읽기 요청 스키마"""
    session_id: str = Field(..., description="세션 ID")
    message_id: str = Field(..., description="메시지 ID")
    chunk_index: int = Field(..., description="요청할 청크 인덱스 (0부터 시작)")
    
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
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        if not re.match(uuid_pattern, v, re.IGNORECASE):
            raise ValueError('유효하지 않은 메시지 ID 형식입니다')
        return v
    
    @field_validator('chunk_index')
    @classmethod
    def validate_chunk_index(cls, v):
        if v < 0:
            raise ValueError('청크 인덱스는 0 이상이어야 합니다')
        return v


class ContinueAnswerResponse(BaseModel):
    """계속 읽기 응답 스키마"""
    content: str = Field(..., description="청크 내용")
    chunk_index: int = Field(..., description="청크 인덱스")
    total_chunks: int = Field(..., description="전체 청크 수")
    has_more: bool = Field(..., description="더 많은 청크가 있는지 여부")


class SourceInfo(BaseModel):
    """출처 정보 (상세)"""
    name: str = Field(..., description="출처명")
    type: str = Field(..., description="출처 타입 (statute_article, case_paragraph 등)")
    url: Optional[str] = Field(None, description="출처 URL")
    metadata: Optional[Dict[str, Any]] = Field(None, description="추가 메타데이터")


class AnswerChunkInfo(BaseModel):
    """답변 청크 정보"""
    chunk_index: int = Field(..., description="청크 인덱스 (0부터 시작)")
    total_chunks: int = Field(..., description="전체 청크 수")
    has_more: bool = Field(..., description="더 많은 청크가 있는지 여부")
    is_complete: bool = Field(..., description="전체 답변이 완료되었는지 여부")


class ChatResponse(BaseModel):
    """채팅 응답 스키마"""
    answer: str = Field(..., description="AI 답변")
    sources_by_type: Optional[Dict[str, List[SourceInfo]]] = Field(default_factory=lambda: {
        "statute_article": [],
        "case_paragraph": [],
        "decision_paragraph": [],
        "interpretation_paragraph": []
    }, description="참고 출처 타입별 그룹화 (유일한 필요한 필드)")
    confidence: float = Field(..., description="신뢰도 (0.0 ~ 1.0)")
    # 하위 호환성을 위해 deprecated 필드도 포함 (점진적 제거)
    sources: List[str] = Field(default_factory=list, description="참고 출처 (deprecated: sources_by_type에서 재구성 가능)")
    sources_detail: List[SourceInfo] = Field(default_factory=list, description="참고 출처 상세 정보 (deprecated: sources_by_type에서 재구성 가능)")
    legal_references: List[str] = Field(default_factory=list, description="법률 참조 (deprecated: sources_by_type에서 재구성 가능)")
    related_questions: List[str] = Field(default_factory=list, description="연관 질문")
    processing_steps: List[str] = Field(default_factory=list, description="처리 단계")
    session_id: str = Field(..., description="세션 ID")
    processing_time: float = Field(..., description="처리 시간 (초)")
    query_type: str = Field(default="", description="질문 유형")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="메타데이터")
    errors: List[str] = Field(default_factory=list, description="에러 목록")
    warnings: List[str] = Field(default_factory=list, description="경고 메시지 목록")
    chunk_info: Optional[AnswerChunkInfo] = Field(None, description="답변 청크 정보 (분할 전송 시)")
    message_id: Optional[str] = Field(None, description="메시지 ID (계속 읽기 기능용)")


class StreamingChatRequest(BaseModel):
    """스트리밍 채팅 요청 스키마"""
    message: str = Field(
        ..., 
        description="사용자 메시지",
        min_length=1,
        max_length=10000
    )
    session_id: Optional[str] = Field(None, description="세션 ID")
    context: Optional[str] = Field(None, max_length=5000, description="추가 컨텍스트")
    image_base64: Optional[str] = Field(None, description="Base64 인코딩된 이미지 (OCR 처리용)")
    file_base64: Optional[str] = Field(None, description="Base64 인코딩된 파일 (이미지, 텍스트, PDF, DOCX)")
    filename: Optional[str] = Field(None, max_length=255, description="파일명")
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('메시지는 비어있을 수 없습니다')
        dangerous_patterns = [
            r'(\bOR\b|\bAND\b)\s*[\'"]?\d+[\'"]?\s*=\s*[\'"]?\d+[\'"]?',
            r'(\bOR\b|\bAND\b)\s*[\'"]?[\w]+[\'"]?\s*=\s*[\'"]?[\w]+[\'"]?',
            r';\s*(DROP|DELETE|UPDATE|INSERT|CREATE|ALTER)',
            r'(\bUNION\b|\bSELECT\b).*(\bFROM\b|\bWHERE\b)',
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError('잘못된 입력 형식입니다')
        
        # XSS 패턴 검사
        xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>',
        ]
        for pattern in xss_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError('잘못된 입력 형식입니다')
        
        return v.strip()
    
    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v):
        if v:
            uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
            if not re.match(uuid_pattern, v, re.IGNORECASE):
                raise ValueError('유효하지 않은 세션 ID 형식입니다')
        return v
    
    @field_validator('context')
    @classmethod
    def validate_context(cls, v):
        if v:
            if len(v) > 5000:
                raise ValueError('컨텍스트는 5000자를 초과할 수 없습니다')
        return v
    
    @field_validator('file_base64')
    @classmethod
    def validate_file_base64(cls, v):
        if v:
            # Base64 크기 제한 (10MB로 축소)
            max_base64_size = 10 * 1024 * 1024  # 10MB
            if len(v) > max_base64_size:
                raise ValueError('파일 크기가 너무 큽니다. 최대 10MB까지 업로드 가능합니다.')
            # Base64 형식 검증
            import base64
            try:
                if v.startswith('data:'):
                    v = v.split(',', 1)[1]
                base64.b64decode(v, validate=True)
            except Exception:
                raise ValueError('유효하지 않은 Base64 형식입니다.')
        return v
    
    @field_validator('filename')
    @classmethod
    def validate_filename(cls, v):
        if v:
            # 파일명 검증
            from api.services.file_validator import validate_filename
            is_valid, error_msg = validate_filename(v)
            if not is_valid:
                raise ValueError(error_msg)
        return v


