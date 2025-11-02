# -*- coding: utf-8 -*-
"""
State Helpers 테스트
Flat 및 Modular 구조 지원 테스트
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest  # noqa: E402

from core.agents.modular_states import (  # noqa: E402
    create_initial_legal_state as create_modular_state,
)
from core.agents.state_definitions import (  # noqa: E402
    create_flat_legal_state as create_flat_state,
)
from core.agents.state_helpers import (  # noqa: E402
    ensure_state_group,
    get_answer_text,
    get_classification,
    get_field,
    get_nested_value,
    get_query,
    is_modular_state,
    set_field,
)


class TestStateHelpers:
    """State Helper 함수 테스트"""

    def test_is_modular_state_flat(self):
        """Flat 구조 감지 테스트"""
        flat_state = create_flat_state("테스트 질문", "session_123")
        assert not is_modular_state(flat_state)

    def test_is_modular_state_modular(self):
        """Modular 구조 감지 테스트"""
        modular_state = create_modular_state("테스트 질문", "session_123")
        assert is_modular_state(modular_state)

    def test_get_field_flat(self):
        """Flat 구조에서 필드 접근 테스트"""
        flat_state = create_flat_state("테스트 질문", "session_123")

        assert get_field(flat_state, "query") == "테스트 질문"
        assert get_field(flat_state, "session_id") == "session_123"
        assert get_field(flat_state, "query_type") == ""
        assert get_field(flat_state, "confidence") == 0.0

    def test_get_field_modular(self):
        """Modular 구조에서 필드 접근 테스트"""
        modular_state = create_modular_state("테스트 질문", "session_123")

        assert get_field(modular_state, "query") == "테스트 질문"
        assert get_field(modular_state, "session_id") == "session_123"
        assert get_field(modular_state, "query_type") == ""
        assert get_field(modular_state, "confidence") == 0.0

    def test_set_field_flat(self):
        """Flat 구조에서 필드 설정 테스트"""
        flat_state = create_flat_state("테스트 질문", "session_123")

        set_field(flat_state, "query_type", "legal_advice")
        assert flat_state["query_type"] == "legal_advice"

        set_field(flat_state, "confidence", 0.9)
        assert flat_state["confidence"] == 0.9

    def test_set_field_modular(self):
        """Modular 구조에서 필드 설정 테스트"""
        modular_state = create_modular_state("테스트 질문", "session_123")

        set_field(modular_state, "query_type", "legal_advice")
        assert get_field(modular_state, "query_type") == "legal_advice"

        set_field(modular_state, "confidence", 0.9)
        assert get_field(modular_state, "confidence") == 0.9

    def test_get_nested_value(self):
        """중첩 값 접근 테스트"""
        modular_state = create_modular_state("테스트 질문", "session_123")

        # 정상 경로
        assert get_nested_value(modular_state, "input", "query") == "테스트 질문"
        assert get_nested_value(modular_state, "classification", "query_type") == ""

        # 존재하지 않는 경로
        assert get_nested_value(modular_state, "nonexistent", "field", default="default") == "default"

    def test_ensure_state_group(self):
        """State 그룹 초기화 테스트"""
        modular_state = create_modular_state("테스트 질문", "session_123")

        # None으로 설정
        modular_state["classification"] = None
        ensure_state_group(modular_state, "classification")

        assert modular_state["classification"] is not None
        assert isinstance(modular_state["classification"], dict)
        assert "query_type" in modular_state["classification"]

    def test_get_query_modular(self):
        """Modular 구조에서 query 접근 테스트"""
        modular_state = create_modular_state("테스트 질문", "session_123")
        assert get_query(modular_state) == "테스트 질문"

    def test_get_classification_modular(self):
        """Modular 구조에서 classification 접근 테스트"""
        modular_state = create_modular_state("테스트 질문", "session_123")
        classification = get_classification(modular_state)

        assert isinstance(classification, dict)
        assert "query_type" in classification
        assert "confidence" in classification

    def test_answer_field_access(self):
        """Answer 필드 접근 테스트"""
        modular_state = create_modular_state("테스트 질문", "session_123")

        # answer 필드 접근
        answer = get_answer_text(modular_state)
        assert answer == ""

        # answer 설정
        set_field(modular_state, "answer", "답변입니다")
        assert get_answer_text(modular_state) == "답변입니다"

    def test_compatibility_between_structures(self):
        """Flat과 Modular 구조 간 호환성 테스트"""
        flat_state = create_flat_state("테스트 질문", "session_123")
        modular_state = create_modular_state("테스트 질문", "session_123")

        # 같은 필드에 같은 값 설정
        set_field(flat_state, "query_type", "legal_advice")
        set_field(modular_state, "query_type", "legal_advice")

        # 같은 값 가져오기
        assert get_field(flat_state, "query_type") == "legal_advice"
        assert get_field(modular_state, "query_type") == "legal_advice"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
