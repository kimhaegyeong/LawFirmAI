# -*- coding: utf-8 -*-
"""
State Access Helpers
Flat 및 Modular 구조 모두 지원하는 State 접근 헬퍼 함수들

이 모듈은 다음 기능을 제공합니다:
- Flat 구조와 Modular 구조 자동 감지
- 일관된 필드 접근 API (get_field, set_field)
- State 그룹 자동 초기화 (ensure_state_group)
- 중첩 구조 안전 접근 (get_nested_value)

각 노드는 _get_state_value와 _set_state_value를 통해
두 구조 모두에서 작동할 수 있습니다.
"""

import logging
from typing import Any, Dict, List

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
    # classification 키가 없거나 None인 경우 기본값 생성
    if "classification" not in state or state["classification"] is None:
        from .modular_states import create_default_classification
        state["classification"] = create_default_classification()

    for key, value in kwargs.items():
        if key in state["classification"]:  # type: ignore
            state["classification"][key] = value  # type: ignore


# ============================================
# Search State Access
# ============================================
def get_search(state: LegalWorkflowState) -> Dict[str, Any]:
    """검색 결과 반환 (확장 가능한 딕셔너리)"""
    search = state.get("search")
    if search is None:
        # 기본값 생성 (필요한 모든 키 포함)
        from .modular_states import create_default_search
        query = get_query(state)
        return create_default_search(query)
    return search  # type: ignore


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
    """
    분석 결과 설정 (중첩 방지)

    Args:
        state: LegalWorkflowState
        data: 설정할 데이터 딕셔너리
    """
    if state["analysis"] is None:
        from .modular_states import create_default_analysis
        state["analysis"] = create_default_analysis()

    # "analysis" 키가 있고 값이 딕셔너리인 경우, 중첩 구조라면 평탄화
    if "analysis" in data and isinstance(data["analysis"], dict):
        nested_analysis = data["analysis"]
        # 이미 AnalysisState 형태면 (중첩된 경우)
        if "analysis" in nested_analysis or "legal_references" in nested_analysis:
            # 중첩 해제 - 재귀적으로 평탄화
            flattened_data = {}

            # 가장 안쪽의 analysis 값 추출
            current = nested_analysis
            depth = 0
            max_depth = 10  # 무한 재귀 방지

            while isinstance(current, dict) and "analysis" in current and isinstance(current["analysis"], dict) and depth < max_depth:
                current = current["analysis"]
                depth += 1

            # 평탄화된 데이터 구성
            flattened_data["analysis"] = current.get("analysis") if isinstance(current, dict) else current

            # legal_references와 legal_citations도 재귀적으로 추출
            refs_source = nested_analysis
            while isinstance(refs_source, dict) and "analysis" in refs_source:
                if "legal_references" in refs_source:
                    flattened_data["legal_references"] = refs_source["legal_references"]
                    break
                refs_source = refs_source["analysis"]

            if "legal_references" not in flattened_data:
                flattened_data["legal_references"] = nested_analysis.get("legal_references", [])

            # legal_citations도 동일하게 처리
            citations_source = nested_analysis
            while isinstance(citations_source, dict) and "analysis" in citations_source:
                if "legal_citations" in citations_source:
                    flattened_data["legal_citations"] = citations_source["legal_citations"]
                    break
                citations_source = citations_source["analysis"]

            if "legal_citations" not in flattened_data:
                flattened_data["legal_citations"] = nested_analysis.get("legal_citations")

            # 평탄화된 데이터로 업데이트
            state["analysis"].update(flattened_data)  # type: ignore
            return

    # 정상적인 경우 그대로 업데이트
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
# Utility Functions
# ============================================
def get_nested_value(state: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    중첩 구조에서 안전하게 값 가져오기

    Args:
        state: State 객체 (Flat 또는 Modular)
        keys: 키 경로 (예: "input", "query" 또는 "classification", "query_type")
        default: 기본값

    Returns:
        찾은 값 또는 기본값
    """
    current = state
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def ensure_state_group(state: LegalWorkflowState, group_name: str) -> None:
    """
    State 그룹이 없으면 기본값으로 초기화

    Args:
        state: LegalWorkflowState
        group_name: 그룹 이름 ("classification", "search", "analysis", 등)
    """
    if state.get(group_name) is None:
        from .modular_states import (
            create_default_analysis,
            create_default_answer,
            create_default_classification,
            create_default_common,
            create_default_control,
            create_default_document,
            create_default_multi_turn,
            create_default_search,
            create_default_validation,
        )

        defaults = {
            "classification": create_default_classification,
            "search": lambda: create_default_search(get_query(state)),
            "analysis": create_default_analysis,
            "answer": create_default_answer,
            "document": create_default_document,
            "multi_turn": create_default_multi_turn,
            "validation": create_default_validation,
            "control": create_default_control,
            "common": create_default_common,
        }

        if group_name in defaults:
            state[group_name] = defaults[group_name]()  # type: ignore


def is_modular_state(state: Dict[str, Any]) -> bool:
    """
    State가 Modular 구조인지 확인

    Args:
        state: State 객체

    Returns:
        True if modular, False if flat
    """
    return "input" in state and isinstance(state.get("input"), dict)


# ============================================
# Backward Compatibility Helpers
# ============================================
def get_field(state: Dict[str, Any], field_path: str) -> Any:
    """
    Flat 및 Modular 구조 모두 지원하는 필드 접근

    Examples:
        get_field(state, "query") -> input["query"] 또는 state["query"]
        get_field(state, "urgency_level") -> classification["urgency_level"] 또는 state["urgency_level"]
        get_field(state, "retrieved_docs") -> search["retrieved_docs"] 또는 state["retrieved_docs"]
    """
    # Flat 구조 확인 (input이 없거나 input이 dict가 아닌 경우)
    if not is_modular_state(state):
        # Flat 구조에서 직접 접근
        return state.get(field_path)

    # Modular 구조 처리
    path_mapping = {
        # Input
        "query": lambda s: get_query(s),
        "session_id": lambda s: get_session_id(s),

        # Classification
        # 중요: get_classification이 None을 반환할 수 있으므로 안전하게 처리
        "query_type": lambda s: (get_classification(s) or {}).get("query_type", ""),
        "confidence": lambda s: (get_classification(s) or {}).get("confidence", 0.0),
        "legal_field": lambda s: (get_classification(s) or {}).get("legal_field", ""),
        "legal_domain": lambda s: (get_classification(s) or {}).get("legal_domain", ""),
        "urgency_level": lambda s: (get_classification(s) or {}).get("urgency_level", ""),
        "urgency_reasoning": lambda s: (get_classification(s) or {}).get("urgency_reasoning", ""),
        "emergency_type": lambda s: (get_classification(s) or {}).get("emergency_type"),
        "complexity_level": lambda s: (get_classification(s) or {}).get("complexity_level", ""),
        "query_complexity": lambda s: (get_classification(s) or {}).get("query_complexity", "") or (get_classification(s) or {}).get("complexity_level", ""),
        "needs_search": lambda s: (get_classification(s) or {}).get("needs_search", True),
        "requires_expert": lambda s: (get_classification(s) or {}).get("requires_expert", False),
        "expert_subgraph": lambda s: (get_classification(s) or {}).get("expert_subgraph"),

        # Search
        # 중요: get_search이 None을 반환할 수 있으므로 안전하게 처리
        "search_query": lambda s: (get_search(s) or {}).get("search_query", ""),
        "extracted_keywords": lambda s: (get_search(s) or {}).get("extracted_keywords", []),
        "ai_keyword_expansion": lambda s: (get_search(s) or {}).get("ai_keyword_expansion"),
        "retrieved_docs": lambda s: get_retrieved_docs(s),
        "optimized_queries": lambda s: (get_search(s) or {}).get("optimized_queries") or {},
        "search_params": lambda s: (get_search(s) or {}).get("search_params") or {},
        "is_retry_search": lambda s: (get_search(s) or {}).get("is_retry_search", False),
        "search_start_time": lambda s: (get_search(s) or {}).get("search_start_time", 0.0),
        "search_cache_hit": lambda s: (get_search(s) or {}).get("search_cache_hit", False),
        "semantic_results": lambda s: (get_search(s) or {}).get("semantic_results", []),
        "keyword_results": lambda s: (get_search(s) or {}).get("keyword_results", []),
        "semantic_count": lambda s: (get_search(s) or {}).get("semantic_count", 0),
        "keyword_count": lambda s: (get_search(s) or {}).get("keyword_count", 0),
        "merged_documents": lambda s: (get_search(s) or {}).get("merged_documents", []),
        "keyword_weights": lambda s: (get_search(s) or {}).get("keyword_weights", {}),
        "prompt_optimized_context": lambda s: (get_search(s) or {}).get("prompt_optimized_context", {}),

        # Analysis
        # 중요: get_analysis 등이 None을 반환할 수 있으므로 안전하게 처리
        "analysis": lambda s: (get_analysis(s) or {}).get("analysis"),
        "legal_references": lambda s: (get_analysis(s) or {}).get("legal_references", []),
        "legal_citations": lambda s: (get_analysis(s) or {}).get("legal_citations"),

        # Answer
        "answer": lambda s: get_answer_text(s),
        "sources": lambda s: (get_answer(s) or {}).get("sources", []),
        "structure_confidence": lambda s: (get_answer(s) or {}).get("structure_confidence", 0.0),

        # Document
        "document_type": lambda s: (get_document(s) or {}).get("document_type"),
        "document_analysis": lambda s: (get_document(s) or {}).get("document_analysis"),
        "key_clauses": lambda s: (get_document(s) or {}).get("key_clauses", []),
        "potential_issues": lambda s: (get_document(s) or {}).get("potential_issues", []),

        # MultiTurn
        "is_multi_turn": lambda s: (get_multi_turn(s) or {}).get("is_multi_turn", False),
        "multi_turn_confidence": lambda s: (get_multi_turn(s) or {}).get("multi_turn_confidence", 1.0),
        "conversation_history": lambda s: (get_multi_turn(s) or {}).get("conversation_history", []),
        "conversation_context": lambda s: (get_multi_turn(s) or {}).get("conversation_context"),

        # Validation
        "legal_validity_check": lambda s: (get_validation(s) or {}).get("legal_validity_check", True),
        "legal_basis_validation": lambda s: (get_validation(s) or {}).get("legal_basis_validation"),
        "outdated_laws": lambda s: (get_validation(s) or {}).get("outdated_laws", []),

        # Control
        "retry_count": lambda s: (get_control(s) or {}).get("retry_count", 0),
        "quality_check_passed": lambda s: (get_control(s) or {}).get("quality_check_passed", False),
        "needs_enhancement": lambda s: (get_control(s) or {}).get("needs_enhancement", False),

        # Common
        "processing_steps": lambda s: get_processing_steps(s),
        "errors": lambda s: get_errors(s),
        "metadata": lambda s: get_metadata(s),
        "processing_time": lambda s: (get_common(s) or {}).get("processing_time", 0.0),
        "tokens_used": lambda s: (get_common(s) or {}).get("tokens_used", 0),
    }

    if field_path in path_mapping:
        return path_mapping[field_path](state)

    # 개선 사항 5: Optional 필드 안전 접근 로직 추가
    # category와 uploaded_document는 optional 필드이므로 경고 레벨을 낮춤
    optional_fields = ["category", "uploaded_document"]
    if field_path in optional_fields:
        logger.debug(f"Optional field path accessed: {field_path} (not in path_mapping)")
        # Optional 필드에 대한 기본값 반환
        if field_path == "category":
            return None
        elif field_path == "uploaded_document":
            return None
    else:
        # expanded_queries는 SearchState에 정의되어 있지만 LangGraph reducer가 인식하지 못할 수 있음
        # 이는 경고가 아닌 정상적인 동작일 수 있으므로 DEBUG 레벨로 변경
        if field_path == "expanded_queries":
            logger.debug(f"Unknown field path (expected for expanded_queries): {field_path}")
        else:
            logger.debug(f"Unknown field path: {field_path}")
    return None


def set_field(state: Dict[str, Any], field_path: str, value: Any):
    """
    Flat 및 Modular 구조 모두 지원하는 필드 설정

    Args:
        state: LegalWorkflowState (Flat 또는 Modular)
        field_path: 필드 경로
        value: 설정할 값
    """
    # Flat 구조 확인
    if not is_modular_state(state):
        # Flat 구조에서 직접 설정
        state[field_path] = value
        return

    # Modular 구조 처리
    if field_path in ["query", "session_id"]:
        state["input"][field_path] = value  # type: ignore
    elif field_path in ["query_type", "confidence", "legal_field", "legal_domain",
                        "urgency_level", "urgency_reasoning", "emergency_type",
                        "complexity_level", "query_complexity", "needs_search",
                        "requires_expert", "expert_subgraph"]:
        update_classification(state, **{field_path: value})  # type: ignore
    elif field_path in ["search_query", "extracted_keywords", "ai_keyword_expansion",
                        "optimized_queries", "search_params", "expanded_queries", "semantic_results", "keyword_results",
                        "semantic_count", "keyword_count", "merged_documents", "keyword_weights",
                        "prompt_optimized_context", "structured_documents", "search_metadata",
                        "search_quality_evaluation", "search_quality", "is_retry_search", "search_start_time", "search_cache_hit"]:
        set_search(state, {field_path: value})  # type: ignore
    elif field_path == "retrieved_docs":
        set_retrieved_docs(state, value)  # type: ignore
    elif field_path in ["analysis", "legal_references", "legal_citations"]:
        set_analysis(state, {field_path: value})  # type: ignore
    elif field_path == "answer":
        set_answer_text(state, value)  # type: ignore
    elif field_path in ["sources", "structure_confidence"]:
        set_answer(state, {field_path: value})  # type: ignore
    elif field_path in ["document_type", "document_analysis", "key_clauses", "potential_issues", "uploaded_document"]:
        set_document(state, {field_path: value})  # type: ignore
    elif field_path in ["is_multi_turn", "multi_turn_confidence", "conversation_history", "conversation_context"]:
        set_multi_turn(state, {field_path: value})  # type: ignore
    elif field_path in ["legal_validity_check", "legal_basis_validation", "outdated_laws"]:
        set_validation(state, {field_path: value})  # type: ignore
    elif field_path in ["retry_count", "quality_check_passed", "needs_enhancement"]:
        set_control(state, {field_path: value})  # type: ignore
    elif field_path in ["processing_time", "tokens_used", "processing_steps", "errors"]:
        # common 필드 처리
        if "common" not in state or state["common"] is None:
            from .modular_states import create_default_common
            state["common"] = create_default_common()
        state["common"][field_path] = value  # type: ignore
    elif field_path == "metadata":
        set_metadata(state, value if isinstance(value, dict) else {})  # type: ignore
    elif field_path == "quality_metrics":
        # quality_metrics는 common.metadata에 저장
        if "common" not in state or state["common"] is None:
            from .modular_states import create_default_common
            state["common"] = create_default_common()
        if "metadata" not in state["common"]:
            state["common"]["metadata"] = {}
        state["common"]["metadata"]["quality_metrics"] = value
    else:
        # expanded_queries는 SearchState에 정의되어 있지만 LangGraph reducer가 인식하지 못할 수 있음
        # 이는 경고가 아닌 정상적인 동작일 수 있으므로 DEBUG 레벨로 변경
        if field_path == "expanded_queries":
            logger.debug(f"Unknown field path for setting (expected for expanded_queries): {field_path}")
        else:
            logger.debug(f"Unknown field path for setting: {field_path}")
