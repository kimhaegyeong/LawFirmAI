# -*- coding: utf-8 -*-
"""
LangGraph State Definitions
LangGraph 워크플로우 상태 정의 모듈

기존 flat 구조를 유지하여 최대 호환성 확보
"""

from operator import add
from typing import Annotated, Any, Dict, List, Optional, TypedDict

# Configuration constants
from .state_utils import (
    MAX_CONVERSATION_HISTORY,
    MAX_DOCUMENT_CONTENT_LENGTH,
    MAX_PROCESSING_STEPS,
    MAX_RETRIEVED_DOCS,
)

# Re-export configuration constants
__all__ = [
    "LegalWorkflowState",
    "AgentWorkflowState",
    "StreamingWorkflowState",
    "create_initial_legal_state",
    "create_flat_legal_state",
    "create_initial_agent_state",
    "create_initial_streaming_state",
    "MAX_RETRIEVED_DOCS",
    "MAX_DOCUMENT_CONTENT_LENGTH",
    "MAX_CONVERSATION_HISTORY",
    "MAX_PROCESSING_STEPS",
]


class LegalWorkflowState(TypedDict):
    """
    법률 워크플로우 상태 정의 - Flat 구조 (레거시 호환용)

    Note: 이 클래스는 하위 호환성을 위해 유지됩니다.
    새로운 코드는 modular_states.py의 modular 구조를 사용하는 것이 권장됩니다.

    create_initial_legal_state() 함수는 이제 modular 구조를 반환합니다.
    """

    # 입력 데이터
    query: str
    session_id: str

    # 질문 분류 결과
    query_type: str
    confidence: float

    # 긴급도 평가 결과
    urgency_level: str
    urgency_reasoning: str
    emergency_type: Optional[str]

    # 법률 분야 분류
    legal_field: str
    legal_domain: str

    # 법령 검증 결과
    legal_validity_check: bool
    legal_basis_validation: Optional[Dict[str, Any]]
    outdated_laws: List[str]

    # 문서 분석 결과
    document_type: Optional[str]
    document_analysis: Optional[Dict[str, Any]]
    key_clauses: List[Dict[str, Any]]
    potential_issues: List[Dict[str, Any]]

    # 전문가 라우팅
    complexity_level: str
    requires_expert: bool
    expert_subgraph: Optional[str]

    # 멀티턴 처리 결과
    is_multi_turn: bool
    multi_turn_confidence: float
    conversation_history: List[Dict[str, Any]]
    conversation_context: Optional[Dict[str, Any]]

    # 키워드 추출 결과
    extracted_keywords: List[str]
    search_query: str
    ai_keyword_expansion: Optional[Dict[str, Any]]

    # 문서 검색 결과
    retrieved_docs: List[Dict[str, Any]]

    # 컨텍스트 분석 결과
    analysis: Optional[str]
    legal_references: List[str]

    # 답변 처리 중간 결과
    structure_confidence: float
    legal_citations: Optional[List[Dict[str, Any]]]

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

    # 재시도 제어
    retry_count: int
    quality_check_passed: bool
    needs_enhancement: bool


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


def create_flat_legal_state(query: str, session_id: str) -> LegalWorkflowState:
    """
    법률 워크플로우 초기 상태 생성 (Flat 구조)

    레거시 코드 및 타입 정의와 일치하는 flat 구조를 반환합니다.
    LegalWorkflowState TypedDict의 모든 필드가 최상위 레벨에 있습니다.

    Args:
        query: 사용자 질문
        session_id: 세션 ID

    Returns:
        Flat 구조의 LegalWorkflowState (모든 필드가 최상위 레벨)
    """
    # TypedDict는 직접 인스턴스화할 수 없으므로 dict를 반환
    return {
        # 입력 데이터
        "query": query,
        "session_id": session_id,

        # 질문 분류 결과
        "query_type": "",
        "confidence": 0.0,

        # 긴급도 평가 결과
        "urgency_level": "medium",
        "urgency_reasoning": "",
        "emergency_type": None,

        # 법률 분야 분류
        "legal_field": "general",
        "legal_domain": "general",

        # 법령 검증 결과
        "legal_validity_check": True,
        "legal_basis_validation": None,
        "outdated_laws": [],

        # 문서 분석 결과
        "document_type": None,
        "document_analysis": None,
        "key_clauses": [],
        "potential_issues": [],

        # 전문가 라우팅
        "complexity_level": "simple",
        "requires_expert": False,
        "expert_subgraph": None,

        # 멀티턴 처리 결과
        "is_multi_turn": False,
        "multi_turn_confidence": 1.0,
        "conversation_history": [],
        "conversation_context": None,

        # 키워드 추출 결과
        "extracted_keywords": [],
        "search_query": query,
        "ai_keyword_expansion": None,

        # 문서 검색 결과
        "retrieved_docs": [],

        # 컨텍스트 분석 결과
        "analysis": None,
        "legal_references": [],

        # 답변 처리 중간 결과
        "structure_confidence": 0.0,
        "legal_citations": None,

        # 최종 답변
        "answer": "",
        "sources": [],

        # 처리 과정 메타데이터
        "processing_steps": [],
        "errors": [],
        "metadata": {},

        # 성능 메트릭
        "processing_time": 0.0,
        "tokens_used": 0,

        # 재시도 제어
        "retry_count": 0,
        "quality_check_passed": False,
        "needs_enhancement": False,
    }


def create_initial_legal_state(query: str, session_id: str) -> LegalWorkflowState:
    """
    법률 워크플로우 초기 상태 생성 (Modular 구조)

    이 함수는 modular 구조를 반환합니다.
    Flat 구조가 필요하면 create_flat_legal_state()를 사용하세요.
    """
    # modular_states의 create_initial_legal_state 사용
    from .modular_states import create_initial_legal_state as create_modular_state
    return create_modular_state(query, session_id)


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
