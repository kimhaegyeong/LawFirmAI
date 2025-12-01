"""
구조화된 답변 모델 정의

답변 생성에 사용된 문서와 커버리지 정보를 추적하기 위한 Pydantic 모델
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class DocumentUsageInfo(BaseModel):
    """답변 생성에 사용된 문서 정보"""
    document_number: int = Field(..., description="문서 번호 (예: 1, 2, 3)")
    document_id: Optional[str] = Field(None, description="문서 고유 ID")
    source: str = Field(..., description="출처 (예: 민법 제750조, 대법원 2020다12345)")
    source_type: str = Field(..., description="출처 타입 (statute_article, precedent_content 등)")
    
    # 사용 정보
    used_in_answer: bool = Field(True, description="답변 본문에서 실제로 인용되었는지 여부")
    citation_count: int = Field(0, description="답변 본문에서 인용된 횟수")
    citation_positions: List[int] = Field(default_factory=list, description="인용된 위치 (문자 인덱스)")
    
    # 관련성 정보
    relevance_score: Optional[float] = Field(None, description="검색 관련성 점수")
    semantic_similarity: Optional[float] = Field(None, description="의미적 유사도")
    
    # 사용 근거 (LLM이 이 문서를 왜 사용했는지)
    usage_rationale: Optional[str] = Field(None, description="이 문서를 사용한 이유/근거")
    
    # 문서 내용 (선택적)
    key_content: Optional[str] = Field(None, description="문서의 핵심 내용 (요약)")


class CoverageMetrics(BaseModel):
    """전체 커버리지 메트릭"""
    # 키워드 커버리지
    keyword_coverage: float = Field(0.0, description="키워드 커버리지 (0.0-1.0)")
    keyword_total: int = Field(0, description="전체 키워드 수")
    keyword_matched: int = Field(0, description="매칭된 키워드 수")
    
    # 인용 커버리지
    citation_coverage: float = Field(0.0, description="인용 커버리지 (0.0-1.0)")
    citation_count: int = Field(0, description="답변에 포함된 인용 수")
    citation_expected: int = Field(0, description="예상 인용 수")
    
    # 문서 활용도
    document_usage_rate: float = Field(0.0, description="문서 활용도 (0.0-1.0)")
    documents_used: int = Field(0, description="사용된 문서 수")
    documents_total: int = Field(0, description="검색된 전체 문서 수")
    documents_min_required: int = Field(0, description="최소 요구 문서 수")
    
    # 법률 참조 커버리지
    legal_reference_coverage: Optional[float] = Field(None, description="법률 참조 커버리지")
    legal_references_found: int = Field(0, description="발견된 법률 참조 수")
    
    # 전체 커버리지 (종합 점수)
    overall_coverage: float = Field(0.0, description="전체 커버리지 점수 (0.0-1.0)")
    
    # 커버리지 세부 정보
    coverage_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="커버리지 세부 항목별 점수"
    )


class StructuredAnswer(BaseModel):
    """구조화된 답변 모델"""
    # 답변 본문 (표 제외)
    answer_text: str = Field(..., description="답변 본문")
    
    # 문서 사용 정보
    document_usage: List[DocumentUsageInfo] = Field(
        default_factory=list,
        description="답변 생성에 사용된 문서 목록"
    )
    
    # 커버리지 메트릭
    coverage: CoverageMetrics = Field(
        default_factory=CoverageMetrics,
        description="전체 커버리지 메트릭"
    )
    
    # 기존 필드 (하위 호환성)
    sources: List[str] = Field(default_factory=list, description="참고 출처 목록")
    structure_confidence: float = Field(0.0, description="구조 신뢰도")
    
    # 메타데이터
    created_at: datetime = Field(default_factory=datetime.now, description="생성 시간")
    query_type: Optional[str] = Field(None, description="질문 유형")
    
    def to_legacy_format(self) -> Dict[str, Any]:
        """기존 형식으로 변환 (하위 호환성)"""
        return {
            "answer": self.answer_text,
            "sources": self.sources,
            "structure_confidence": self.structure_confidence,
            # 새로운 필드들은 metadata에 포함
            "document_usage": [doc.model_dump() for doc in self.document_usage],
            "coverage": self.coverage.model_dump()
        }
    
    def get_used_documents(self) -> List[DocumentUsageInfo]:
        """실제로 사용된 문서만 반환"""
        return [doc for doc in self.document_usage if doc.used_in_answer]
    
    def get_unused_documents(self) -> List[DocumentUsageInfo]:
        """검색되었지만 사용되지 않은 문서 반환"""
        return [doc for doc in self.document_usage if not doc.used_in_answer]
    
    @classmethod
    def from_answer_state(cls, answer_state: Dict[str, Any]) -> "StructuredAnswer":
        """AnswerState (dict)에서 StructuredAnswer 생성"""
        # document_usage 복원
        document_usage = []
        if answer_state.get("document_usage"):
            for doc_dict in answer_state["document_usage"]:
                if isinstance(doc_dict, dict):
                    document_usage.append(DocumentUsageInfo(**doc_dict))
        
        # coverage 복원
        coverage = CoverageMetrics()
        if answer_state.get("coverage"):
            coverage = CoverageMetrics(**answer_state["coverage"])
        
        return cls(
            answer_text=answer_state.get("answer", ""),
            document_usage=document_usage,
            coverage=coverage,
            sources=answer_state.get("sources", []),
            structure_confidence=answer_state.get("structure_confidence", 0.0),
            query_type=answer_state.get("query_type")
        )
    
    def to_answer_state(self) -> Dict[str, Any]:
        """AnswerState 형식으로 변환"""
        return {
            "answer": self.answer_text,
            "sources": self.sources,
            "structure_confidence": self.structure_confidence,
            "document_usage": [doc.model_dump() for doc in self.document_usage] if self.document_usage else None,
            "coverage": self.coverage.model_dump() if self.coverage else None
        }

