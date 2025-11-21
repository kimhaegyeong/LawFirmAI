# -*- coding: utf-8 -*-
"""
State Adapter Layer
기존 flat State 구조와 새 nested State 구조 간의 변환

단계적 마이그레이션을 위해 기존 코드가 변경 없이
작동하도록 변환 레이어를 제공합니다.
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from typing import Any, Dict, Optional

from .modular_states import LegalWorkflowState
try:
    from core.workflow.node_input_output_spec import validate_node_input
except ImportError:
    try:
        from ..workflow.node_input_output_spec import validate_node_input
    except ImportError:
        # Fallback: validate_node_input이 없어도 동작하도록
        def validate_node_input(node_name: str, state: Dict[str, Any]) -> tuple[bool, Optional[str]]:
            """Fallback: 항상 검증 통과"""
            return True, None

logger = get_logger(__name__)


def _extract_analysis_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analysis State를 중첩 없이 추출

    Args:
        state: Flat 또는 Nested 구조의 state

    Returns:
        AnalysisState 딕셔너리 (중첩 없이)
    """
    analysis_data = state.get("analysis")

    # analysis_data가 이미 AnalysisState 딕셔너리인 경우 (nested 구조)
    if isinstance(analysis_data, dict):
        # 중첩 구조 확인: {"analysis": {...}, ...} 형태인지 확인
        if "analysis" in analysis_data and isinstance(analysis_data["analysis"], dict):
            # 이중 중첩 방지: analysis["analysis"]가 또 딕셔너리면 재귀적으로 평탄화
            nested = analysis_data["analysis"]
            if isinstance(nested, dict) and ("analysis" in nested or "legal_references" in nested):
                # 재귀적으로 평탄화 - 최대 10단계까지
                current = nested
                depth = 0
                max_depth = 10

                while isinstance(current, dict) and "analysis" in current and isinstance(current["analysis"], dict) and depth < max_depth:
                    current = current["analysis"]
                    depth += 1

                # 평탄화된 구조 반환
                return {
                    "analysis": current.get("analysis") if isinstance(current, dict) else current,
                    "legal_references": nested.get("legal_references", analysis_data.get("legal_references", state.get("legal_references", []))),
                    "legal_citations": nested.get("legal_citations") or analysis_data.get("legal_citations") or state.get("legal_citations")
                }
            else:
                # 정상적인 AnalysisState 구조
                return {
                    "analysis": analysis_data.get("analysis"),
                    "legal_references": analysis_data.get("legal_references", []),
                    "legal_citations": analysis_data.get("legal_citations")
                }
        else:
            # analysis 필드가 없거나 문자열인 경우 - 정상적인 AnalysisState
            return {
                "analysis": analysis_data.get("analysis") if "analysis" in analysis_data else analysis_data,
                "legal_references": analysis_data.get("legal_references", []),
                "legal_citations": analysis_data.get("legal_citations")
            }
    else:
        # analysis_data가 문자열이거나 None인 경우 (flat 구조)
        return {
            "analysis": analysis_data,
            "legal_references": state.get("legal_references", []),
            "legal_citations": state.get("legal_citations")
        }


class StateAdapter:
    """
    LegalWorkflowState의 flat 접근을 nested로 자동 변환

    기존 코드:
        state["errors"]
        state["query"]
        state["retrieved_docs"]

    새 구조:
        state["common"]["errors"]
        state["input"]["query"]
        state["search"]["retrieved_docs"]
    """

    @staticmethod
    def to_nested(state: Dict[str, Any]) -> LegalWorkflowState:
        """
        Flat 구조를 Nested 구조로 변환

        Args:
            state: 기존 flat 구조의 state

        Returns:
            새로운 nested 구조의 LegalWorkflowState
        """
        if not isinstance(state, dict):
            return state  # type: ignore

        # 이미 nested 구조인지 확인
        if "input" in state and isinstance(state["input"], dict):
            # input이 있지만 query가 없으면 확인
            if not state["input"].get("query"):
                # state의 다른 위치에서 query 찾기
                query_from_state = state.get("query")
                if query_from_state:
                    state["input"]["query"] = query_from_state
                    self.logger.debug("Restored query in existing input group")
            return state  # type: ignore

        # Flat 구조를 nested로 변환
        nested = {
            "input": {
                "query": state.get("query", ""),
                "session_id": state.get("session_id", "")
            },
            "classification": {
                "query_type": state.get("query_type", ""),
                "confidence": state.get("confidence", 0.0),
                "legal_field": state.get("legal_field", "general"),
                "legal_domain": state.get("legal_domain", "general"),
                "urgency_level": state.get("urgency_level", "medium"),
                "urgency_reasoning": state.get("urgency_reasoning", ""),
                "emergency_type": state.get("emergency_type"),
                "complexity_level": state.get("complexity_level", "simple"),
                "requires_expert": state.get("requires_expert", False),
                "expert_subgraph": state.get("expert_subgraph")
            },
            "search": {
                "search_query": state.get("search_query", state.get("query", "")),
                "extracted_keywords": state.get("extracted_keywords", []),
                "ai_keyword_expansion": state.get("ai_keyword_expansion"),
                "retrieved_docs": state.get("retrieved_docs", [])
            },
            # Analysis: 중첩 구조 방지 - state.get("analysis")가 이미 AnalysisState 딕셔너리인지 확인
            "analysis": _extract_analysis_state(state),
            "answer": {
                "answer": state.get("answer", ""),
                "sources": state.get("sources", []),
                "structure_confidence": state.get("structure_confidence", 0.0)
            },
            "document": {
                "document_type": state.get("document_type"),
                "document_analysis": state.get("document_analysis"),
                "key_clauses": state.get("key_clauses", []),
                "potential_issues": state.get("potential_issues", [])
            },
            "multi_turn": {
                "is_multi_turn": state.get("is_multi_turn", False),
                "multi_turn_confidence": state.get("multi_turn_confidence", 1.0),
                "conversation_history": state.get("conversation_history", []),
                "conversation_context": state.get("conversation_context")
            },
            "validation": {
                "legal_validity_check": state.get("legal_validity_check", True),
                "legal_basis_validation": state.get("legal_basis_validation"),
                "outdated_laws": state.get("outdated_laws", [])
            },
            "control": {
                "retry_count": state.get("retry_count", 0),
                "quality_check_passed": state.get("quality_check_passed", False),
                "needs_enhancement": state.get("needs_enhancement", False)
            },
            "common": {
                "processing_steps": state.get("processing_steps", []),
                "errors": state.get("errors", []),
                "metadata": state.get("metadata", {}),
                "processing_time": state.get("processing_time", 0.0),
                "tokens_used": state.get("tokens_used", 0)
            }
        }

        return nested  # type: ignore

    @staticmethod
    def to_flat(state: LegalWorkflowState) -> Dict[str, Any]:
        """
        Nested 구조를 Flat 구조로 변환

        Args:
            state: Nested 구조의 LegalWorkflowState

        Returns:
            Flat 구조의 dict (기존 코드 호환용)
        """
        if not isinstance(state, dict):
            return {}

        # Input
        query = state.get("input", {}).get("query", "") if isinstance(state.get("input"), dict) else state.get("query", "")
        session_id = state.get("input", {}).get("session_id", "") if isinstance(state.get("input"), dict) else state.get("session_id", "")

        # Classification
        classification = state.get("classification", {})
        if not isinstance(classification, dict):
            classification = {}

        # Search
        search = state.get("search", {})
        if not isinstance(search, dict):
            search = {}

        # Analysis
        analysis = state.get("analysis", {})
        if not isinstance(analysis, dict):
            analysis = {}

        # Answer
        answer = state.get("answer", "")
        # answer가 문자열인 경우 그대로 유지 (딕셔너리로 변환하지 않음)
        if isinstance(answer, str):
            # 문자열 answer를 그대로 사용
            pass
        elif isinstance(answer, dict):
            # 이미 딕셔너리인 경우 그대로 사용
            pass
        else:
            # 다른 타입인 경우 빈 문자열로 변환
            answer = ""

        # Document
        document = state.get("document", {})
        if not isinstance(document, dict):
            document = {}

        # MultiTurn
        multi_turn = state.get("multi_turn", {})
        if not isinstance(multi_turn, dict):
            multi_turn = {}

        # Validation
        validation = state.get("validation", {})
        if not isinstance(validation, dict):
            validation = {}

        # Control
        control = state.get("control", {})
        if not isinstance(control, dict):
            control = {}

        # Common
        common = state.get("common", {})
        if not isinstance(common, dict):
            common = {}

        # Flat 구조로 변환
        flat = {
            # Input
            "query": query,
            "session_id": session_id,

            # Classification
            "query_type": classification.get("query_type", ""),
            "confidence": classification.get("confidence", 0.0),
            "legal_field": classification.get("legal_field", "general"),
            "legal_domain": classification.get("legal_domain", "general"),
            "urgency_level": classification.get("urgency_level", "medium"),
            "urgency_reasoning": classification.get("urgency_reasoning", ""),
            "emergency_type": classification.get("emergency_type"),
            "complexity_level": classification.get("complexity_level", "simple"),
            "requires_expert": classification.get("requires_expert", False),
            "expert_subgraph": classification.get("expert_subgraph"),

            # Search
            "search_query": search.get("search_query", query),
            "extracted_keywords": search.get("extracted_keywords", []),
            "ai_keyword_expansion": search.get("ai_keyword_expansion"),
            "retrieved_docs": search.get("retrieved_docs", []),

            # Analysis
            "analysis": analysis.get("analysis"),
            "legal_references": analysis.get("legal_references", []),
            "legal_citations": analysis.get("legal_citations"),

            # Answer
            "answer": answer if isinstance(answer, str) else answer.get("answer", "") if isinstance(answer, dict) else "",
            "sources": answer.get("sources", []) if isinstance(answer, dict) else [],
            "structure_confidence": answer.get("structure_confidence", 0.0) if isinstance(answer, dict) else 0.0,

            # Document
            "document_type": document.get("document_type"),
            "document_analysis": document.get("document_analysis"),
            "key_clauses": document.get("key_clauses", []),
            "potential_issues": document.get("potential_issues", []),

            # MultiTurn
            "is_multi_turn": multi_turn.get("is_multi_turn", False),
            "multi_turn_confidence": multi_turn.get("multi_turn_confidence", 1.0),
            "conversation_history": multi_turn.get("conversation_history", []),
            "conversation_context": multi_turn.get("conversation_context"),

            # Validation
            "legal_validity_check": validation.get("legal_validity_check", True),
            "legal_basis_validation": validation.get("legal_basis_validation"),
            "outdated_laws": validation.get("outdated_laws", []),

            # Control
            "retry_count": control.get("retry_count", 0),
            "quality_check_passed": control.get("quality_check_passed", False),
            "needs_enhancement": control.get("needs_enhancement", False),

            # Common
            "processing_steps": common.get("processing_steps", []),
            "errors": common.get("errors", []),
            "metadata": common.get("metadata", {}),
            "processing_time": common.get("processing_time", 0.0),
            "tokens_used": common.get("tokens_used", 0)
        }

        return flat


def adapt_state(state: Dict[str, Any]) -> LegalWorkflowState:
    """
    State를 자동으로 적절한 구조로 변환

    사용자 코드에서 편의를 위해 제공되는 함수
    """
    return StateAdapter.to_nested(state)


def flatten_state(state: LegalWorkflowState) -> Dict[str, Any]:
    """
    Nested State를 Flat 구조로 변환

    기존 API 호환성을 위해 제공되는 함수
    """
    return StateAdapter.to_flat(state)


def validate_state_for_node(
    state: Dict[str, Any],
    node_name: str,
    auto_convert: bool = True
) -> tuple[bool, Optional[str], Dict[str, Any]]:
    """
    노드 실행 전 State 검증 및 변환

    Args:
        state: State 객체 (flat 또는 nested)
        node_name: 노드 이름
        auto_convert: 자동 변환 여부

    Returns:
        (is_valid, error_message, converted_state) 튜플
    """
    # 0. State가 딕셔너리인지 확인
    if not isinstance(state, dict):
        error_msg = (
            f"State must be a dict for node {node_name}, "
            f"got {type(state).__name__}"
        )
        logger.error(error_msg)
        # 빈 딕셔너리 반환 (에러 발생 방지)
        return False, error_msg, {}

    # 1. 자동 변환 (필요한 경우)
    converted_state = state

    # 디버깅: 원본 state의 query 확인
    original_query = state.get("query") or (state.get("input") and isinstance(state.get("input"), dict) and state["input"].get("query"))
    if node_name == "classify_query":
        print(f"[DEBUG] state_adapter.validate_state_for_node ({node_name}): original state query='{original_query[:50] if original_query else 'EMPTY'}...'")
        print(f"[DEBUG] state_adapter.validate_state_for_node ({node_name}): state has input={bool(state.get('input'))}, state keys={list(state.keys())}")

    if auto_convert:
        # Nested 구조가 아니면 변환
        if "input" not in state or not isinstance(state.get("input"), dict):
            converted_state = adapt_state(state)

            # 디버깅: 변환 후 query 확인
            converted_query = converted_state.get("query") or (converted_state.get("input") and isinstance(converted_state.get("input"), dict) and converted_state["input"].get("query"))
            if node_name == "classify_query":
                print(f"[DEBUG] state_adapter.validate_state_for_node ({node_name}): after adapt_state query='{converted_query[:50] if converted_query else 'EMPTY'}...'")
        else:
            # 이미 nested 구조인 경우
            nested_query = state.get("input", {}).get("query", "") if isinstance(state.get("input"), dict) else ""
            if node_name == "classify_query":
                print(f"[DEBUG] state_adapter.validate_state_for_node ({node_name}): already nested, query='{nested_query[:50] if nested_query else 'EMPTY'}...'")

    # 2. Input 유효성 검증
    is_valid, error = validate_node_input(node_name, converted_state)

    return is_valid, error, converted_state
