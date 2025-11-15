# -*- coding: utf-8 -*-
"""
State Definitions 테스트
langgraph_core/state/state_definitions.py 단위 테스트
"""

import pytest
from typing import Dict, Any

from lawfirm_langgraph.langgraph_core.state.state_definitions import (
    LegalWorkflowState,
    create_initial_legal_state,
    create_flat_legal_state,
    MAX_RETRIEVED_DOCS,
    MAX_DOCUMENT_CONTENT_LENGTH,
    MAX_CONVERSATION_HISTORY,
    MAX_PROCESSING_STEPS
)


class TestCreateInitialLegalState:
    """create_initial_legal_state 테스트"""
    
    def test_create_initial_legal_state_basic(self):
        """기본 초기 상태 생성 테스트"""
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
    
    def test_create_initial_legal_state_with_optional(self):
        """선택적 파라미터 포함 초기 상태 생성 테스트 (modular 구조)"""
        state = create_initial_legal_state(
            query="테스트 질문",
            session_id="test_session"
        )
        
        assert isinstance(state, dict)
        assert "input" in state
        assert state["input"]["query"] == "테스트 질문"
        assert "classification" in state
        assert isinstance(state["classification"], dict)
    
    def test_create_initial_legal_state_defaults(self):
        """기본값 포함 초기 상태 생성 테스트"""
        state = create_initial_legal_state(
            query="테스트 질문",
            session_id="test_session"
        )
        
        assert isinstance(state, dict)
        assert "query_type" in state or "classification" in state
        assert "retrieved_docs" in state or "search" in state


class TestCreateFlatLegalState:
    """create_flat_legal_state 테스트"""
    
    def test_create_flat_legal_state_basic(self):
        """기본 평면 상태 생성 테스트"""
        state = create_flat_legal_state(
            query="테스트 질문",
            session_id="test_session"
        )
        
        assert isinstance(state, dict)
        assert state.get("query") == "테스트 질문"
        assert state.get("session_id") == "test_session"
    
    def test_create_flat_legal_state_structure(self):
        """평면 상태 구조 테스트"""
        state = create_flat_legal_state(
            query="테스트 질문",
            session_id="test_session"
        )
        
        assert isinstance(state, dict)
        assert "query" in state
        assert "session_id" in state

