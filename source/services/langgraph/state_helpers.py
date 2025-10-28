# -*- coding: utf-8 -*-
"""
State Access Helpers
모듈화된 State에 대한 접근 편의 함수들

각 노드에서 필요한 State 그룹만 쉽게 접근할 수 있도록
헬퍼 함수를 제공합니다.
"""

import logging
from typing import Any, Dict, List, Optional

from .modular_states import (
    AnalysisState,
    AnswerState,
    ClassificationState,
    ControlState,
    DocumentState,
    InputState,
    LegalWorkflowState,
    MultiTurnState,
    SearchState,
    ValidationState,
)

logger = logging.getLogger(__name__)


# ============================================
# Input State Access
# ============================================
def get_input(state: LegalWorkflowState) -> InputState:
    """입력 데이터 반환"""
    return state["input"]


def get_query(state: LegalWorkflowState) -> str:
    """질문만 반환"""
    return state["input"]["query"]


def get_session_id(state: LegalWorkflowState) -> str:
    """세션 ID만 반환"""
    return state["input"]["session_id"]


# ============================================
# Classification State Access
# ============================================
def get_classification(state: LegalWorkflowState) -> ClassificationState:
    """분류 결과 반환"""
    return state.get("classification", {})  # type: ignore


def set_classification(state: LegalWorkflowState, data: Dict[str, Any]):
    """분류 결과 설정"""
    if state["classification"] is None:
        from .modular_states import create_default_classification
        state["classification"] = create_default_classification()

    state["classification"].update(data)  # type: ignore


def update_classification(state: LegalWorkflowState, **kwargs):
    """분류 결과 업데이트 (keyword arguments)"""
    if state["classification"] is None:
        from .modular_states import create_default_classification
        state["classification"] = create_default_classification()

    for key, value in kwargs.items():
        if key in state["classification"]:  # type: ignore
            state["classification"][key] = value  # type: ignore


# ============================================
# Search State Access
# ============================================
def get_search(state: LegalWorkflowState) -> SearchState:
    """검색 결과 반환"""
    return state.get("search", {})  # type: ignore


def set_search(state: LegalWorkflowState, data: Dict[str, Any]):
    """검색 결과 설정"""
    if state["search"] is None:
        from .modular_states import create_default_search
        state["search"] = create_default_search(get_query(state))

    state["search"].update(data)  # type: ignore


def get_retrieved_docs(state: LegalWorkflowState) -> List[Dict[str, Any]]:
    """검색된 문서만 반환"""
    search = get_search(state)
    return search.get("retrieved_docs", [])


def set_retrieved_docs(state: LegalWorkflowState, docs: List[Dict[str, Any]]):
    """검색된 문서 설정"""
    set_search(state, {"retrieved_docs": docs})


# ============================================
# Analysis State Access
# ============================================
def get_analysis(state: LegalWorkflowState) -> AnalysisState:
    """분석 결과 반환"""
    return state.get("analysis", {})  # type: ignore


def set_analysis(state: LegalWorkflowState, data: Dict[str, Any]):
    """분석 결과 설정"""
    if state["analysis"] is None:
        from .modular_states import create_default_analysis
        state["analysis"] = create_default_analysis()

    state["analysis"].update(data)  # type: ignore


# ============================================
# Answer State Access
# ============================================
def get_answer(state: LegalWorkflowState) -> AnswerState:
    """답변 결과 반환"""
    return state.get("answer", {})  # type: ignore


def set_answer(state: LegalWorkflowState, data: Dict[str, Any]):
    """답변 결과 설정"""
    if state["answer"] is None:
        from .modular_states import create_default_answer
        state["answer"] = create_default_answer()

    state["answer"].update(data)  # type: ignore


def get_answer_text(state: LegalWorkflowState) -> str:
    """답변 텍스트만 반환"""
    answer = get_answer(state)
    return answer.get("answer", "")


def set_answer_text(state: LegalWorkflowState, text: str):
    """답변 텍스트만 설정"""
    set_answer(state, {"answer": text})


# ============================================
# Document State Access
# ============================================
def get_document(state: LegalWorkflowState) -> DocumentState:
    """문서 분석 결과 반환"""
    return state.get("document", {})  # type: ignore


def set_document(state: LegalWorkflowState, data: Dict[str, Any]):
    """문서 분석 결과 설정"""
    if state["document"] is None:
        from .modular_states import create_default_document
        state["document"] = create_default_document()

    state["document"].update(data)  # type: ignore


# ============================================
# MultiTurn State Access
# ============================================
def get_multi_turn(state: LegalWorkflowState) -> MultiTurnState:
    """멀티턴 상태 반환"""
    return state.get("multi_turn", {})  # type: ignore


def set_multi_turn(state: LegalWorkflowState, data: Dict[str, Any]):
    """멀티턴 상태 설정"""
    if state["multi_turn"] is None:
        from .modular_states import create_default_multi_turn
        state["multi_turn"] = create_default_multi_turn()

    state["multi_turn"].update(data)  # type: ignore


# ============================================
# Validation State Access
# ============================================
def get_validation(state: LegalWorkflowState) -> ValidationState:
    """검증 결과 반환"""
    return state.get("validation", {})  # type: ignore


def set_validation(state: LegalWorkflowState, data: Dict[str, Any]):
    """검증 결과 설정"""
    if state["validation"] is None:
        from .modular_states import create_default_validation
        state["validation"] = create_default_validation()

    state["validation"].update(data)  # type: ignore


# ============================================
# Control State Access
# ============================================
def get_control(state: LegalWorkflowState) -> ControlState:
    """제어 상태 반환"""
    return state.get("control", {})  # type: ignore


def set_control(state: LegalWorkflowState, data: Dict[str, Any]):
    """제어 상태 설정"""
    if state["control"] is None:
        from .modular_states import create_default_control
        state["control"] = create_default_control()

    state["control"].update(data)  # type: ignore


# ============================================
# Common State Access
# ============================================
def get_common(state: LegalWorkflowState) -> Dict[str, Any]:
    """공통 상태 반환"""
    return state.get("common", {})


def get_processing_steps(state: LegalWorkflowState) -> List[str]:
    """처리 단계 반환"""
    common = get_common(state)
    return common.get("processing_steps", [])


def add_processing_step(state: LegalWorkflowState, step: str):
    """처리 단계 추가"""
    if "common" not in state:
        from .modular_states import create_default_common
        state["common"] = create_default_common()

    state["common"]["processing_steps"].append(step)


def get_errors(state: LegalWorkflowState) -> List[str]:
    """에러 목록 반환"""
    common = get_common(state)
    return common.get("errors", [])


def add_error(state: LegalWorkflowState, error: str):
    """에러 추가"""
    if "common" not in state:
        from .modular_states import create_default_common
        state["common"] = create_default_common()

    state["common"]["errors"].append(error)


def get_metadata(state: LegalWorkflowState) -> Dict[str, Any]:
    """메타데이터 반환"""
    common = get_common(state)
    return common.get("metadata", {})


def set_metadata(state: LegalWorkflowState, data: Dict[str, Any]):
    """메타데이터 설정"""
    if "common" not in state:
        from .modular_states import create_default_common
        state["common"] = create_default_common()

    state["common"]["metadata"].update(data)


# ============================================
# Backward Compatibility Helpers
# ============================================
def get_field(state: LegalWorkflowState, field_path: str) -> Any:
    """
    필드 경로로 접근 (레거시 코드 호환용)

    Examples:
        get_field(state, "query") -> input["query"]
        get_field(state, "urgency_level") -> classification["urgency_level"]
        get_field(state, "retrieved_docs") -> search["retrieved_docs"]
    """
    path_mapping = {
        # Input
        "query": lambda s: get_query(s),
        "session_id": lambda s: get_session_id(s),

        # Classification
        "query_type": lambda s: get_classification(s).get("query_type", ""),
        "confidence": lambda s: get_classification(s).get("confidence", 0.0),
        "legal_field": lambda s: get_classification(s).get("legal_field", ""),
        "legal_domain": lambda s: get_classification(s).get("legal_domain", ""),
        "urgency_level": lambda s: get_classification(s).get("urgency_level", ""),
        "urgency_reasoning": lambda s: get_classification(s).get("urgency_reasoning", ""),
        "emergency_type": lambda s: get_classification(s).get("emergency_type"),
        "complexity_level": lambda s: get_classification(s).get("complexity_level", ""),
        "requires_expert": lambda s: get_classification(s).get("requires_expert", False),
        "expert_subgraph": lambda s: get_classification(s).get("expert_subgraph"),

        # Search
        "search_query": lambda s: get_search(s).get("search_query", ""),
        "extracted_keywords": lambda s: get_search(s).get("extracted_keywords", []),
        "ai_keyword_expansion": lambda s: get_search(s).get("ai_keyword_expansion"),
        "retrieved_docs": lambda s: get_retrieved_docs(s),

        # Analysis
        "analysis": lambda s: get_analysis(s).get("analysis"),
        "legal_references": lambda s: get_analysis(s).get("legal_references", []),
        "legal_citations": lambda s: get_analysis(s).get("legal_citations"),

        # Answer
        "answer": lambda s: get_answer_text(s),
        "sources": lambda s: get_answer(s).get("sources", []),
        "enhanced_answer": lambda s: get_answer(s).get("enhanced_answer"),
        "structure_confidence": lambda s: get_answer(s).get("structure_confidence", 0.0),

        # Document
        "document_type": lambda s: get_document(s).get("document_type"),
        "document_analysis": lambda s: get_document(s).get("document_analysis"),
        "key_clauses": lambda s: get_document(s).get("key_clauses", []),
        "potential_issues": lambda s: get_document(s).get("potential_issues", []),

        # MultiTurn
        "is_multi_turn": lambda s: get_multi_turn(s).get("is_multi_turn", False),
        "multi_turn_confidence": lambda s: get_multi_turn(s).get("multi_turn_confidence", 1.0),
        "conversation_history": lambda s: get_multi_turn(s).get("conversation_history", []),
        "conversation_context": lambda s: get_multi_turn(s).get("conversation_context"),

        # Validation
        "legal_validity_check": lambda s: get_validation(s).get("legal_validity_check", True),
        "legal_basis_validation": lambda s: get_validation(s).get("legal_basis_validation"),
        "outdated_laws": lambda s: get_validation(s).get("outdated_laws", []),

        # Control
        "retry_count": lambda s: get_control(s).get("retry_count", 0),
        "quality_check_passed": lambda s: get_control(s).get("quality_check_passed", False),
        "needs_enhancement": lambda s: get_control(s).get("needs_enhancement", False),

        # Common
        "processing_steps": lambda s: get_processing_steps(s),
        "errors": lambda s: get_errors(s),
        "metadata": lambda s: get_metadata(s),
        "processing_time": lambda s: get_common(s).get("processing_time", 0.0),
        "tokens_used": lambda s: get_common(s).get("tokens_used", 0),
    }

    if field_path in path_mapping:
        return path_mapping[field_path](state)

    logger.warning(f"Unknown field path: {field_path}")
    return None


def set_field(state: LegalWorkflowState, field_path: str, value: Any):
    """
    필드 경로로 설정 (레거시 코드 호환용)
    """
    if field_path in ["query", "session_id"]:
        state["input"][field_path] = value  # type: ignore
    elif field_path in ["query_type", "confidence", "legal_field", "legal_domain",
                        "urgency_level", "urgency_reasoning", "emergency_type",
                        "complexity_level", "requires_expert", "expert_subgraph"]:
        update_classification(state, **{field_path: value})
    elif field_path in ["search_query", "extracted_keywords", "ai_keyword_expansion"]:
        set_search(state, {field_path: value})
    elif field_path == "retrieved_docs":
        set_retrieved_docs(state, value)
    elif field_path in ["analysis", "legal_references", "legal_citations"]:
        set_analysis(state, {field_path: value})
    elif field_path == "answer":
        set_answer_text(state, value)
    elif field_path in ["sources", "enhanced_answer", "structure_confidence"]:
        set_answer(state, {field_path: value})
    elif field_path in ["document_type", "document_analysis", "key_clauses", "potential_issues"]:
        set_document(state, {field_path: value})
    elif field_path in ["is_multi_turn", "multi_turn_confidence", "conversation_history", "conversation_context"]:
        set_multi_turn(state, {field_path: value})
    elif field_path in ["legal_validity_check", "legal_basis_validation", "outdated_laws"]:
        set_validation(state, {field_path: value})
    elif field_path in ["retry_count", "quality_check_passed", "needs_enhancement"]:
        set_control(state, {field_path: value})
    elif field_path in ["processing_time", "tokens_used"]:
        common = get_common(state)
        common[field_path] = value  # type: ignore
    elif field_path == "metadata":
        set_metadata(state, value if isinstance(value, dict) else {})
    else:
        logger.warning(f"Unknown field path for setting: {field_path}")
