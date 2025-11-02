# -*- coding: utf-8 -*-
"""
State Refactoring 통합 테스트
Flat ↔ Modular 변환 및 호환성 테스트
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest

from core.agents.modular_states import (
    create_initial_legal_state as create_modular_state,
)
from core.agents.state_adapter import StateAdapter
from core.agents.state_definitions import (
    create_initial_legal_state as create_flat_state,
)
from core.agents.state_helpers import (
    get_field,
    is_modular_state,
    set_field,
)
from core.agents.state_reduction import (
    reduce_state_for_node,
    reduce_state_size,
)


class TestStateRefactoring:
    """State 리팩토링 통합 테스트"""

    def test_create_flat_state(self):
        """Flat 구조 생성 테스트"""
        flat_state = create_flat_state("테스트 질문", "session_123")

        assert "query" in flat_state
        assert "session_id" in flat_state
        assert flat_state["query"] == "테스트 질문"
        assert not is_modular_state(flat_state)

    def test_create_modular_state(self):
        """Modular 구조 생성 테스트"""
        modular_state = create_modular_state("테스트 질문", "session_123")

        assert "input" in modular_state
        assert "classification" in modular_state
        assert modular_state["input"]["query"] == "테스트 질문"
        assert is_modular_state(modular_state)

    def test_flat_to_modular_conversion(self):
        """Flat → Modular 변환 테스트"""
        flat_state = create_flat_state("테스트 질문", "session_123")
        flat_state["query_type"] = "legal_advice"
        flat_state["confidence"] = 0.9

        modular_state = StateAdapter.to_nested(flat_state)

        assert is_modular_state(modular_state)
        assert get_field(modular_state, "query") == "테스트 질문"
        assert get_field(modular_state, "query_type") == "legal_advice"
        assert get_field(modular_state, "confidence") == 0.9

    def test_modular_to_flat_conversion(self):
        """Modular → Flat 변환 테스트"""
        modular_state = create_modular_state("테스트 질문", "session_123")
        set_field(modular_state, "query_type", "legal_advice")
        set_field(modular_state, "confidence", 0.9)

        flat_state = StateAdapter.to_flat(modular_state)

        assert not is_modular_state(flat_state)
        assert flat_state["query"] == "테스트 질문"
        assert flat_state["query_type"] == "legal_advice"
        assert flat_state["confidence"] == 0.9

    def test_field_access_compatibility(self):
        """필드 접근 호환성 테스트"""
        flat_state = create_flat_state("테스트 질문", "session_123")
        modular_state = create_modular_state("테스트 질문", "session_123")

        # 같은 필드에 같은 값 설정
        set_field(flat_state, "query_type", "legal_advice")
        set_field(modular_state, "query_type", "legal_advice")

        # 같은 값 가져오기
        assert get_field(flat_state, "query_type") == "legal_advice"
        assert get_field(modular_state, "query_type") == "legal_advice"
        assert get_field(flat_state, "query") == "테스트 질문"
        assert get_field(modular_state, "query") == "테스트 질문"

    def test_state_reduction_flat(self):
        """Flat 구조 State Reduction 테스트"""
        flat_state = create_flat_state("테스트 질문", "session_123")
        flat_state["query_type"] = "legal_advice"
        flat_state["retrieved_docs"] = [{"content": "doc1"}, {"content": "doc2"}]

        # classify_query 노드에 필요한 필드만 추출
        reduced = reduce_state_for_node(flat_state, "classify_query")

        assert "query" in reduced or "input" in reduced
        assert "query_type" in reduced or "classification" in reduced

    def test_state_reduction_modular(self):
        """Modular 구조 State Reduction 테스트"""
        modular_state = create_modular_state("테스트 질문", "session_123")
        set_field(modular_state, "query_type", "legal_advice")
        set_field(modular_state, "retrieved_docs", [{"content": "doc1"}, {"content": "doc2"}])

        # classify_query 노드에 필요한 그룹만 추출
        reduced = reduce_state_for_node(modular_state, "classify_query")

        assert "input" in reduced
        assert "classification" in reduced
        assert "common" in reduced

    def test_state_size_reduction_modular(self):
        """Modular 구조에서 State 크기 제한 테스트"""
        modular_state = create_modular_state("테스트 질문", "session_123")

        # 많은 문서 추가
        large_docs = [{"content": "doc" * 100} for _ in range(20)]
        set_field(modular_state, "retrieved_docs", large_docs)

        # 크기 제한 적용
        reduced = reduce_state_size(modular_state, max_docs=10, max_content_per_doc=500)

        # Modular 구조 유지 확인
        assert is_modular_state(reduced)

        # 문서 수 제한 확인
        final_docs = get_field(reduced, "retrieved_docs")
        assert len(final_docs) <= 10

    def test_state_size_reduction_flat(self):
        """Flat 구조에서 State 크기 제한 테스트"""
        flat_state = create_flat_state("테스트 질문", "session_123")

        # 많은 문서 추가
        large_docs = [{"content": "doc" * 100} for _ in range(20)]
        flat_state["retrieved_docs"] = large_docs

        # 크기 제한 적용
        reduced = reduce_state_size(flat_state, max_docs=10, max_content_per_doc=500)

        # Flat 구조 유지 확인
        assert not is_modular_state(reduced)

        # 문서 수 제한 확인
        assert len(reduced["retrieved_docs"]) <= 10

    def test_round_trip_conversion(self):
        """Round-trip 변환 테스트 (Flat → Modular → Flat)"""
        original_flat = create_flat_state("테스트 질문", "session_123")
        original_flat["query_type"] = "legal_advice"
        original_flat["confidence"] = 0.9
        original_flat["retrieved_docs"] = [{"content": "doc1"}]

        # Flat → Modular
        modular = StateAdapter.to_nested(original_flat)
        assert is_modular_state(modular)

        # Modular → Flat
        back_to_flat = StateAdapter.to_flat(modular)

        # 주요 필드 일치 확인
        assert back_to_flat["query"] == original_flat["query"]
        assert back_to_flat["query_type"] == original_flat["query_type"]
        assert back_to_flat["confidence"] == original_flat["confidence"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
