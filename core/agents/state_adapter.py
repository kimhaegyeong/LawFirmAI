# -*- coding: utf-8 -*-
"""
State Adapter Layer
기존 flat State 구조와 새 nested State 구조 간의 변환

단계적 마이그레이션을 위해 기존 코드가 변경 없이
작동하도록 변환 레이어를 제공합니다.
"""

import logging
from typing import Any, Dict

from .modular_states import LegalWorkflowState

logger = logging.getLogger(__name__)


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
            "analysis": {
                "analysis": state.get("analysis"),
                "legal_references": state.get("legal_references", []),
                "legal_citations": state.get("legal_citations")
            },
            "answer": {
                "answer": state.get("answer", ""),
                "sources": state.get("sources", []),
                "enhanced_answer": state.get("enhanced_answer"),
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
        answer = state.get("answer", {})
        if not isinstance(answer, dict):
            answer = {}

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
            "answer": answer.get("answer", ""),
            "sources": answer.get("sources", []),
            "enhanced_answer": answer.get("enhanced_answer"),
            "structure_confidence": answer.get("structure_confidence", 0.0),

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
