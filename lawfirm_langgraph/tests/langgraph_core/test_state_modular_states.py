# -*- coding: utf-8 -*-
"""
Modular States 테스트
langgraph_core/state/modular_states.py 단위 테스트
"""

import pytest
from typing import Dict, Any

from lawfirm_langgraph.langgraph_core.state.modular_states import (
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
    LegalWorkflowState,
    create_initial_legal_state
)


class TestModularStates:
    """Modular States 테스트"""
    
    def test_input_state_structure(self):
        """InputState 구조 테스트"""
        state: InputState = {
            "query": "테스트 질문",
            "session_id": "test_session"
        }
        
        assert state["query"] == "테스트 질문"
        assert state["session_id"] == "test_session"
    
    def test_classification_state_structure(self):
        """ClassificationState 구조 테스트"""
        state: ClassificationState = {
            "query_type": "legal_advice",
            "confidence": 0.9,
            "legal_field": "계약법",
            "legal_domain": "civil",
            "urgency_level": "medium",
            "urgency_reasoning": "일반적인 질문",
            "emergency_type": None,
            "complexity_level": "moderate",
            "requires_expert": False,
            "expert_subgraph": None
        }
        
        assert state["query_type"] == "legal_advice"
        assert state["confidence"] == 0.9
    
    def test_search_state_structure(self):
        """SearchState 구조 테스트"""
        state: SearchState = {
            "search_query": "계약 해지",
            "extracted_keywords": ["계약", "해지"],
            "retrieved_docs": []
        }
        
        assert state["search_query"] == "계약 해지"
        assert isinstance(state["extracted_keywords"], list)
    
    def test_analysis_state_structure(self):
        """AnalysisState 구조 테스트"""
        state: AnalysisState = {
            "analysis": "법률 분석 결과",
            "legal_references": ["민법 제543조"],
            "legal_citations": []
        }
        
        assert state["analysis"] == "법률 분석 결과"
        assert isinstance(state["legal_references"], list)
    
    def test_answer_state_structure(self):
        """AnswerState 구조 테스트"""
        state: AnswerState = {
            "answer": "테스트 답변",
            "sources": ["source1", "source2"],
            "structure_confidence": 0.85
        }
        
        assert state["answer"] == "테스트 답변"
        assert isinstance(state["sources"], list)
    
    def test_create_modular_legal_state(self):
        """모듈화된 법률 상태 생성 테스트"""
        state = create_initial_legal_state(
            query="테스트 질문",
            session_id="test_session"
        )
        
        assert isinstance(state, dict)
        assert "input" in state
        assert state["input"]["query"] == "테스트 질문"
        assert state["input"]["session_id"] == "test_session"
        assert "classification" in state
        assert "search" in state

