# -*- coding: utf-8 -*-
"""
Modular State Definitions
최소 공유 State 구조를 위한 모듈화된 State 정의

93개 필드를 11개 그룹으로 모듈화하여:
- 각 노드가 필요한 데이터만 접근
- LangSmith 로깅 시 불필요한 데이터 전송 방지
- 메모리 사용 최적화
"""

from operator import add
from typing import Annotated, Any, Dict, List, Optional, TypedDict


# ============================================
# 1. Input State - 모든 노드가 필요
# ============================================
class InputState(TypedDict):
    """입력 데이터 - 모든 노드에서 사용"""
    query: str
    session_id: str


# ============================================
# 2. Classification State - 분류, 긴급도, 라우팅
# ============================================
class ClassificationState(TypedDict):
    """분류 및 라우팅 결과"""
    query_type: str  # "simple", "complex", "contract_review", etc.
    confidence: float
    legal_field: str  # "civil", "criminal", "family", etc.
    legal_domain: str  # LegalDomain enum value

    # 긴급도 평가 결과
    urgency_level: str  # "low", "medium", "high", "critical"
    urgency_reasoning: str
    emergency_type: Optional[str]

    # 라우팅 결과
    complexity_level: str  # "simple", "medium", "complex"
    requires_expert: bool
    expert_subgraph: Optional[str]


# ============================================
# 3. Search State - 검색 관련 데이터
# ============================================
class SearchState(TypedDict, total=False):
    """검색 결과 - retrieved_docs는 pruned됨"""
    search_query: str
    search_type: str  # "exact", "semantic", "hybrid"
    retrieved_docs: Annotated[List[Dict[str, Any]], add]
    search_results: List[Dict[str, Any]]
    search_metadata: Dict[str, Any]


# ============================================
# 4. Analysis State - 분석 결과
# ============================================
class AnalysisState(TypedDict, total=False):
    """문서 분석 결과"""
    analysis_result: Dict[str, Any]
    reasoning_steps: Annotated[List[str], add]
    extracted_insights: List[str]
    legal_issues: List[str]


# ============================================
# 5. Answer State - 최종 답변
# ============================================
class AnswerState(TypedDict, total=False):
    """최종 답변 데이터"""
    answer: str
    confidence: float
    sources: Annotated[List[str], add]
    citations: List[Dict[str, Any]]


# ============================================
# 6. Document State - 문서 처리
# ============================================
class DocumentState(TypedDict, total=False):
    """문서 처리 관련"""
    document_context: str
    document_summary: str
    document_terms: List[str]


# ============================================
# 7. Multi-Turn State - 대화 컨텍스트
# ============================================
class MultiTurnState(TypedDict, total=False):
    """대화 컨텍스트 관리"""
    conversation_history: Annotated[List[Dict[str, str]], add]
    previous_answer: str
    follow_up_needed: bool
    clarification_questions: List[str]


# ============================================
# 8. Validation State - 검증 결과
# ============================================
class ValidationState(TypedDict, total=False):
    """답변 검증 결과"""
    is_valid: bool
    validation_errors: Annotated[List[str], add]
    quality_score: float
    needs_improvement: bool


# ============================================
# 9. Control State - 워크플로우 제어
# ============================================
class ControlState(TypedDict, total=False):
    """워크플로우 제어 상태"""
    current_step: str
    processing_steps: Annotated[List[str], add]
    errors: Annotated[List[str], add]
    retry_count: int
    should_retry: bool


# ============================================
# 10. Common State - 공통 메타데이터
# ============================================
class CommonState(TypedDict, total=False):
    """공통 메타데이터"""
    metadata: Dict[str, Any]
    timestamp: str
    execution_time: float
    model_used: str


# ============================================
# 11. Combined Modular State
# ============================================
class ModularLegalWorkflowState(
    InputState,
    ClassificationState,
    SearchState,
    AnalysisState,
    AnswerState,
    DocumentState,
    MultiTurnState,
    ValidationState,
    ControlState,
    CommonState,
):
    """모듈화된 Legal Workflow State"""
    pass


# ============================================
# Helper Functions
# ============================================
def create_default_classification() -> ClassificationState:
    """기본 분류 상태 생성"""
    return {
        "query_type": "simple",
        "confidence": 0.0,
        "legal_field": "",
        "legal_domain": "",
        "urgency_level": "low",
        "urgency_reasoning": "",
        "emergency_type": None,
        "complexity_level": "simple",
        "requires_expert": False,
        "expert_subgraph": None,
    }


def create_default_search() -> SearchState:
    """기본 검색 상태 생성"""
    return {
        "search_query": "",
        "search_type": "hybrid",
        "retrieved_docs": [],
        "search_results": [],
        "search_metadata": {},
    }


def create_default_analysis() -> AnalysisState:
    """기본 분석 상태 생성"""
    return {
        "analysis_result": {},
        "reasoning_steps": [],
        "extracted_insights": [],
        "legal_issues": [],
    }


def create_default_answer() -> AnswerState:
    """기본 답변 상태 생성"""
    return {
        "answer": "",
        "confidence": 0.0,
        "sources": [],
        "citations": [],
    }


def create_default_document() -> DocumentState:
    """기본 문서 상태 생성"""
    return {
        "document_context": "",
        "document_summary": "",
        "document_terms": [],
    }


def create_default_multi_turn() -> MultiTurnState:
    """기본 대화 상태 생성"""
    return {
        "conversation_history": [],
        "previous_answer": "",
        "follow_up_needed": False,
        "clarification_questions": [],
    }


def create_default_validation() -> ValidationState:
    """기본 검증 상태 생성"""
    return {
        "is_valid": False,
        "validation_errors": [],
        "quality_score": 0.0,
        "needs_improvement": True,
    }


def create_default_control() -> ControlState:
    """기본 제어 상태 생성"""
    return {
        "current_step": "start",
        "processing_steps": [],
        "errors": [],
        "retry_count": 0,
        "should_retry": False,
    }


def create_default_common() -> CommonState:
    """기본 공통 상태 생성"""
    return {
        "metadata": {},
        "timestamp": "",
        "execution_time": 0.0,
        "model_used": "",
    }


def create_initial_legal_state(query: str, session_id: str) -> ModularLegalWorkflowState:
    """
    초기 Legal Workflow State 생성 (Modular 구조)
    
    Args:
        query: 사용자 질문
        session_id: 세션 ID
    
    Returns:
        ModularLegalWorkflowState: 초기화된 모듈화된 상태
    """
    import time
    
    # Input State
    state: ModularLegalWorkflowState = {
        "query": query,
        "session_id": session_id,
    }
    
    # 각 모듈의 기본값 추가
    state.update(create_default_classification())
    state.update(create_default_search())
    state.update(create_default_analysis())
    state.update(create_default_answer())
    state.update(create_default_document())
    state.update(create_default_multi_turn())
    state.update(create_default_validation())
    state.update(create_default_control())
    state.update(create_default_common())
    
    # 타임스탬프 설정
    state["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    return state

