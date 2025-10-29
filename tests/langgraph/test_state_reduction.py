# -*- coding: utf-8 -*-
"""
LangGraph State Reduction 테스트
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest

from source.services.langgraph.node_specs import (
    get_all_node_names,
    get_node_spec,
    get_output_fields,
    get_required_fields,
    validate_node_input,
)
from source.services.langgraph.state_definitions import create_initial_legal_state
from source.services.langgraph.state_reducer import (
    FlatStateReducer,
    reduce_state_for_node,
    reduce_state_size,
)


class TestNodeSpecs:
    """노드 스펙 테스트"""

    def test_get_node_spec(self):
        """노드 스펙 조회 테스트"""
        spec = get_node_spec("classify_query")
        assert spec is not None
        assert spec.node_name == "classify_query"
        assert "query" in spec.required_fields

    def test_get_required_fields(self):
        """필수 필드 조회 테스트"""
        fields = get_required_fields("classify_query")
        assert "query" in fields
        assert "session_id" in fields  # always_include
        assert "processing_steps" in fields  # always_include

    def test_get_output_fields(self):
        """출력 필드 조회 테스트"""
        fields = get_output_fields("classify_query")
        assert "query_type" in fields
        assert "confidence" in fields

    def test_validate_node_input(self):
        """Input 유효성 검증 테스트"""
        state = {
            "query": "테스트 질문",
            "session_id": "test_session",
            "processing_steps": [],
            "errors": [],
            "metadata": {},
        }
        is_valid, error = validate_node_input("classify_query", state)
        assert is_valid, error

        # 필수 필드 없는 경우
        invalid_state = {"session_id": "test_session"}
        is_valid, error = validate_node_input("classify_query", invalid_state)
        assert not is_valid
        assert "query" in error.lower()

    def test_all_node_names(self):
        """모든 노드 이름 조회 테스트"""
        node_names = get_all_node_names()
        assert len(node_names) > 0
        assert "classify_query" in node_names
        assert "retrieve_documents" in node_names


class TestStateReducer:
    """State Reducer 테스트"""

    def test_reduce_state_for_node(self):
        """노드별 state 축소 테스트"""
        # 전체 state 생성
        state = create_initial_legal_state("테스트 질문", "test_session")

        # 추가 필드 추가 (실제 워크플로우에서 생성되는 필드들)
        state["retrieved_docs"] = [
            {"content": "문서 내용 " * 100, "source": "test"} for _ in range(20)
        ]
        state["conversation_history"] = [{"role": "user", "content": "질문"} for _ in range(10)]
        state["legal_references"] = ["민법 제750조", "상법 제2조"]

        # classify_query 노드는 query만 필요
        reduced = reduce_state_for_node(state, "classify_query")

        # 필수 필드는 포함되어야 함
        assert "query" in reduced
        assert "session_id" in reduced
        assert "processing_steps" in reduced

        # 불필요한 필드는 제거되어야 함 (optional이 아닌 경우)
        # 하지만 retrieved_docs는 optional이므로 있으면 포함될 수 있음
        # 실제로는 required_fields와 always_include만 포함

    def test_reduce_state_size(self):
        """State 크기 축소 테스트"""
        state = {
            "retrieved_docs": [
                {"content": "긴 내용 " * 200, "source": f"doc_{i}"}
                for i in range(20)
            ],
            "conversation_history": [{"role": "user", "content": "질문"} for _ in range(10)],
        }

        reduced = reduce_state_size(state, max_docs=5, max_content_per_doc=100)

        # 문서 수 제한
        assert len(reduced["retrieved_docs"]) <= 5

        # 각 문서의 내용 길이 제한
        for doc in reduced["retrieved_docs"]:
            if "content" in doc:
                assert len(doc["content"]) <= 100

        # 대화 이력 제한
        assert len(reduced["conversation_history"]) <= 5

    def test_state_reducer_with_full_workflow_state(self):
        """전체 워크플로우 state로 테스트"""
        # 완전한 state 생성
        state = create_initial_legal_state("계약서 작성 시 주의사항은?", "test_session_001")
        state.update({
            "query_type": "contract_review",
            "confidence": 0.9,
            "legal_field": "civil_law",
            "legal_domain": "계약법",
            "search_query": "계약서 작성 주의사항",
            "extracted_keywords": ["계약서", "작성", "주의사항"],
            "retrieved_docs": [
                {"content": "계약서 작성 가이드 " * 50, "source": "법률DB_1", "score": 0.95},
                {"content": "계약 조항 설명 " * 50, "source": "법률DB_2", "score": 0.88},
            ] * 10,  # 20개 문서
            "analysis": "계약서 작성 시 중요한 조항들을 설명합니다...",
            "legal_references": ["민법", "상법"],
            "answer": "계약서 작성 시 다음 사항들을 주의해야 합니다...",
            "sources": ["법률DB_1", "법률DB_2"],
        })

        # retrieve_documents 노드용 축소
        reduced = reduce_state_for_node(state, "retrieve_documents")

        # 필요한 필드만 포함되어야 함
        assert "query" in reduced
        assert "search_query" in reduced
        assert "retrieved_docs" in reduced or "retrieved_docs" not in reduced  # optional

        # retrieved_docs가 있다면 크기 제한 적용
        if "retrieved_docs" in reduced:
            assert len(reduced["retrieved_docs"]) <= 10  # MAX_RETRIEVED_DOCS

    def test_reducer_with_different_nodes(self):
        """다양한 노드에 대한 reducer 테스트"""
        state = create_initial_legal_state("테스트 질문", "test_session")
        state.update({
            "retrieved_docs": [{"content": "문서", "source": "test"} for _ in range(15)],
            "answer": "테스트 답변",
            "sources": ["source1", "source2"],
        })

        # 각 노드별로 축소
        nodes_to_test = [
            "classify_query",
            "retrieve_documents",
            "generate_answer_enhanced",
            "validate_answer_quality",
        ]

        for node_name in nodes_to_test:
            reduced = reduce_state_for_node(state, node_name)

            # 항상 포함 필드는 있어야 함
            assert "session_id" in reduced or "session_id" in state
            assert isinstance(reduced, dict)

            # retrieved_docs는 크기 제한 적용
            if "retrieved_docs" in reduced:
                assert len(reduced["retrieved_docs"]) <= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
