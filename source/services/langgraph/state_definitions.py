# -*- coding: utf-8 -*-
"""
LangGraph State Definitions
LangGraph 워크플로우 상태 정의 모듈
"""

from operator import add
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class LegalWorkflowState(TypedDict):
    """법률 워크플로우 상태 정의"""

    # 입력 데이터
    query: str
    session_id: str

    # 질문 분류 결과
    query_type: str  # "simple", "complex", "contract_review", "precedent_search"
    confidence: float

    # 멀티턴 처리 결과 (새로 추가)
    is_multi_turn: bool
    original_query: str  # 원본 쿼리 (대명사 해결 전)
    resolved_query: str  # 대명사 해결된 최종 쿼리
    multi_turn_confidence: float  # 멀티턴 해결 신뢰도
    multi_turn_reasoning: str  # 멀티턴 해결 추론 과정
    conversation_history: Annotated[List[Dict[str, Any]], add]  # 대화 이력
    conversation_context: Optional[Dict[str, Any]]  # 대화 맥락 정보

    # 키워드 추출 결과
    extracted_keywords: List[str]
    search_query: str  # 강화된 검색 쿼리

    # 문서 검색 결과
    retrieved_docs: Annotated[List[Dict[str, Any]], add]
    search_metadata: Dict[str, Any]

    # 컨텍스트 분석 결과
    analysis: Optional[str]
    legal_references: List[str]

    # 답변 처리 중간 결과 (새로 추가)
    enhanced_answer: Optional[str]  # 구조화된 답변
    structure_confidence: float  # 구조화 후 신뢰도
    format_metadata: Optional[Dict[str, Any]]  # 포맷팅 메타데이터
    quality_metrics: Optional[Dict[str, Any]]  # 품질 메트릭
    legal_citations: Optional[List[Dict[str, Any]]]  # 법적 인용

    # 최종 답변
    answer: str
    sources: List[str]

    # 처리 과정 메타데이터
    processing_steps: Annotated[List[str], add]
    errors: Annotated[List[str], add]
    metadata: Dict[str, Any]

    # 성능 메트릭
    processing_time: float
    tokens_used: int

    # 재시도 제어 (조건부 흐름 및 재귀적 그래프용)
    retry_count: int
    quality_check_passed: bool
    needs_enhancement: bool
    skip_document_search: bool


class AgentWorkflowState(TypedDict):
    """에이전트 워크플로우 상태 정의 (향후 확장용)"""

    # 기본 상태
    query: str
    session_id: str

    # 에이전트 상태
    current_agent: str
    agent_history: Annotated[List[Dict[str, Any]], add]

    # 작업 결과
    task_results: Annotated[List[Dict[str, Any]], add]
    final_result: Optional[Dict[str, Any]]

    # 메타데이터
    processing_steps: Annotated[List[str], add]
    errors: Annotated[List[str], add]
    metadata: Dict[str, Any]


class StreamingWorkflowState(TypedDict):
    """스트리밍 워크플로우 상태 정의 (향후 확장용)"""

    # 기본 상태
    query: str
    session_id: str

    # 스트리밍 데이터
    stream_chunks: Annotated[List[str], add]
    is_streaming: bool

    # 최종 결과
    final_answer: str
    sources: List[str]

    # 메타데이터
    processing_steps: Annotated[List[str], add]
    errors: Annotated[List[str], add]
    metadata: Dict[str, Any]


# 상태 초기화 헬퍼 함수들
def create_initial_legal_state(query: str, session_id: str) -> LegalWorkflowState:
    """법률 워크플로우 초기 상태 생성"""
    return LegalWorkflowState(
        query=query,
        session_id=session_id,
        query_type="",
        confidence=0.0,
        # 멀티턴 관련 필드 초기화
        is_multi_turn=False,
        original_query=query,
        resolved_query=query,
        multi_turn_confidence=1.0,
        multi_turn_reasoning="",
        conversation_history=[],
        conversation_context=None,
        # 기존 필드들
        extracted_keywords=[],
        search_query=query,
        retrieved_docs=[],
        search_metadata={},
        analysis=None,
        legal_references=[],
        # 답변 처리 중간 결과
        enhanced_answer=None,
        structure_confidence=0.0,
        format_metadata=None,
        quality_metrics=None,
        legal_citations=None,
        answer="",
        sources=[],
        processing_steps=[],
        errors=[],
        metadata={},
        processing_time=0.0,
        tokens_used=0,
        retry_count=0,
        quality_check_passed=False,
        needs_enhancement=False,
        skip_document_search=False
    )


def create_initial_agent_state(query: str, session_id: str) -> AgentWorkflowState:
    """에이전트 워크플로우 초기 상태 생성"""
    return AgentWorkflowState(
        query=query,
        session_id=session_id,
        current_agent="",
        agent_history=[],
        task_results=[],
        final_result=None,
        processing_steps=[],
        errors=[],
        metadata={}
    )


def create_initial_streaming_state(query: str, session_id: str) -> StreamingWorkflowState:
    """스트리밍 워크플로우 초기 상태 생성"""
    return StreamingWorkflowState(
        query=query,
        session_id=session_id,
        stream_chunks=[],
        is_streaming=False,
        final_answer="",
        sources=[],
        processing_steps=[],
        errors=[],
        metadata={}
    )
