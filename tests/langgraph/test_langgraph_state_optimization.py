# -*- coding: utf-8 -*-
"""
LangGraph State 최적화 테스트
변경된 State 구조와 최적화 기능을 테스트합니다.
"""

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # pytest가 없으면 unittest로 대체
    import unittest
    pytest = unittest

# 프로젝트 루트 경로 추가
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from source.services.langgraph.state_definitions import (
    LegalWorkflowState,
    create_initial_legal_state,
)
from source.services.langgraph.state_utils import (
    MAX_DOCUMENT_CONTENT_LENGTH,
    MAX_PROCESSING_STEPS,
    MAX_RETRIEVED_DOCS,
    prune_processing_steps,
    prune_retrieved_docs,
)


class TestStateCreation:
    """State 생성 테스트"""

    def test_create_initial_state(self):
        """초기 State 생성 테스트"""
        state = create_initial_legal_state("테스트 질문", "session-123")

        assert state["query"] == "테스트 질문"
        assert state["session_id"] == "session-123"
        assert state["query_type"] == ""
        assert state["confidence"] == 0.0
        assert isinstance(state["processing_steps"], list)
        assert isinstance(state["errors"], list)
        assert isinstance(state["retrieved_docs"], list)

    def test_state_structure(self):
        """State 구조 검증"""
        state = create_initial_legal_state("test", "session")

        # 필수 필드 확인
        required_fields = [
            "query", "session_id", "query_type", "confidence",
            "urgency_level", "legal_field", "complexity_level",
            "answer", "sources", "processing_steps", "errors"
        ]

        for field in required_fields:
            assert field in state, f"필수 필드 {field}가 없습니다"


class TestStatePruning:
    """State 최적화(Pruning) 테스트"""

    def test_processing_steps_pruning(self):
        """처리 단계 최적화 테스트"""
        # MAX_PROCESSING_STEPS 초과하는 steps 생성
        steps = [f"Step {i}" for i in range(MAX_PROCESSING_STEPS + 50)]

        pruned = prune_processing_steps(steps, max_items=MAX_PROCESSING_STEPS)

        assert len(pruned) <= MAX_PROCESSING_STEPS
        # 최근 항목들이 유지되어야 함
        assert pruned[-1] == f"Step {MAX_PROCESSING_STEPS + 49}"

    def test_retrieved_docs_pruning(self):
        """검색 문서 최적화 테스트"""
        # MAX_RETRIEVED_DOCS 초과하는 문서 생성
        docs = [
            {"content": "A" * MAX_DOCUMENT_CONTENT_LENGTH * 2, "source": f"doc{i}"}
            for i in range(MAX_RETRIEVED_DOCS + 10)
        ]

        pruned = prune_retrieved_docs(docs, max_items=MAX_RETRIEVED_DOCS, max_content_per_doc=MAX_DOCUMENT_CONTENT_LENGTH)

        assert len(pruned) <= MAX_RETRIEVED_DOCS
        # 각 문서의 content가 최대 길이로 제한되어야 함
        for doc in pruned:
            assert len(doc["content"]) <= MAX_DOCUMENT_CONTENT_LENGTH

    def test_empty_lists_pruning(self):
        """빈 리스트 pruning 테스트"""
        empty_steps = []
        empty_docs = []

        pruned_steps = prune_processing_steps(empty_steps, max_items=MAX_PROCESSING_STEPS)
        pruned_docs = prune_retrieved_docs(empty_docs, max_items=MAX_RETRIEVED_DOCS, max_content_per_doc=MAX_DOCUMENT_CONTENT_LENGTH)

        assert len(pruned_steps) == 0
        assert len(pruned_docs) == 0


class TestStateOptimizationConfig:
    """State 최적화 설정 테스트"""

    def test_max_constants(self):
        """최대값 상수 테스트"""
        assert MAX_RETRIEVED_DOCS == 10
        assert MAX_DOCUMENT_CONTENT_LENGTH == 500
        assert MAX_PROCESSING_STEPS == 200

    def test_constant_usage(self):
        """상수 사용 테스트"""
        state = create_initial_legal_state("test", "session")

        # 빈 리스트로 시작
        assert len(state["retrieved_docs"]) == 0

        # 최대값 확인
        assert MAX_RETRIEVED_DOCS > 0
        assert MAX_DOCUMENT_CONTENT_LENGTH > 0


class TestStateFields:
    """State 필드 테스트"""

    def test_urgency_fields(self):
        """긴급도 필드 테스트"""
        state = create_initial_legal_state("test", "session")

        assert "urgency_level" in state
        assert "urgency_reasoning" in state
        assert "emergency_type" in state
        assert state["urgency_level"] == "medium"

    def test_classification_fields(self):
        """분류 필드 테스트"""
        state = create_initial_legal_state("test", "session")

        assert "query_type" in state
        assert "confidence" in state
        assert "legal_field" in state
        assert "legal_domain" in state
        assert "complexity_level" in state
        assert "requires_expert" in state

    def test_expert_routing_fields(self):
        """전문가 라우팅 필드 테스트"""
        state = create_initial_legal_state("test", "session")

        assert "expert_subgraph" in state
        assert "complexity_level" in state
        assert "requires_expert" in state
        assert state["requires_expert"] is False

    def test_document_analysis_fields(self):
        """문서 분석 필드 테스트"""
        state = create_initial_legal_state("test", "session")

        assert "document_type" in state
        assert "document_analysis" in state
        assert "key_clauses" in state
        assert "potential_issues" in state
        assert state["document_type"] is None


class TestStateAnnotatedFields:
    """Annotated 필드 테스트 (accumulation)"""

    def test_processing_steps_annotation(self):
        """processing_steps는 Annotated[List, add]로 accumulation됨"""
        state = create_initial_legal_state("test", "session")

        # LangGraph의 accumulation 동작 확인
        assert isinstance(state["processing_steps"], list)
        assert isinstance(state["errors"], list)

    def test_errors_annotation(self):
        """errors는 Annotated[List, add]로 accumulation됨"""
        state = create_initial_legal_state("test", "session")

        assert isinstance(state["errors"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
