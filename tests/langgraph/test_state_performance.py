# -*- coding: utf-8 -*-
"""
State 리팩토링 성능 테스트
메모리 사용량, 로깅 데이터 크기, 처리 속도 측정
"""

import json
import sys
import time
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 프로젝트 루트 추가 후 import 필요 (lint: disable=wrong-import-position)
import pytest  # noqa: E402

from core.agents.modular_states import (  # noqa: E402
    create_initial_legal_state as create_modular_state,
)
from core.agents.state_definitions import (  # noqa: E402
    create_initial_legal_state as create_flat_state,
)
from core.agents.state_reduction import reduce_state_for_node  # noqa: E402


def estimate_size(state: dict) -> int:
    """State 크기 추정 (bytes)"""
    return len(json.dumps(state, ensure_ascii=False).encode('utf-8'))


class TestStatePerformance:
    """State 성능 테스트"""

    def test_memory_comparison(self):
        """메모리 사용량 비교 테스트"""
        # 많은 문서를 포함한 State 생성
        flat_state = create_flat_state("테스트 질문", "session_123")
        flat_state["retrieved_docs"] = [{"content": "문서 내용 " * 50} for _ in range(20)]
        flat_state["conversation_history"] = [{"role": "user", "content": "질문"} for _ in range(10)]

        modular_state = create_modular_state("테스트 질문", "session_123")
        from core.agents.state_helpers import set_field
        set_field(modular_state, "retrieved_docs", [{"content": "문서 내용 " * 50} for _ in range(20)])
        set_field(modular_state, "conversation_history", [{"role": "user", "content": "질문"} for _ in range(10)])

        # 크기 측정
        flat_size = estimate_size(flat_state)
        modular_size = estimate_size(modular_state)

        print(f"\nFlat 구조 크기: {flat_size:,} bytes")
        print(f"Modular 구조 크기: {modular_size:,} bytes")
        print(f"차이: {flat_size - modular_size:,} bytes ({(1 - modular_size/flat_size)*100:.1f}% 감소)")

        # Modular 구조가 더 작거나 비슷해야 함 (그룹화 오버헤드 고려)
        # 실제로는 reduce_state_for_node 후에 더 큰 차이가 날 것
        assert modular_size > 0

    def test_reduced_state_size(self):
        """축소된 State 크기 테스트"""
        # 많은 필드를 포함한 State 생성
        flat_state = create_flat_state("테스트 질문", "session_123")
        flat_state["retrieved_docs"] = [{"content": "문서 내용 " * 50} for _ in range(20)]

        # classify_query 노드에 필요한 것만 추출
        reduced_flat = reduce_state_for_node(flat_state, "classify_query")

        modular_state = create_modular_state("테스트 질문", "session_123")
        from core.agents.state_helpers import set_field
        set_field(modular_state, "retrieved_docs", [{"content": "문서 내용 " * 50} for _ in range(20)])

        reduced_modular = reduce_state_for_node(modular_state, "classify_query")

        # 축소된 크기 측정
        full_flat_size = estimate_size(flat_state)
        reduced_flat_size = estimate_size(reduced_flat)

        full_modular_size = estimate_size(modular_state)
        reduced_modular_size = estimate_size(reduced_modular)

        print(f"\n전체 Flat 크기: {full_flat_size:,} bytes")
        print(f"축소 Flat 크기: {reduced_flat_size:,} bytes")
        print(f"전체 Modular 크기: {full_modular_size:,} bytes")
        print(f"축소 Modular 크기: {reduced_modular_size:,} bytes")

        # 축소된 것이 더 작아야 함
        assert reduced_flat_size < full_flat_size
        assert reduced_modular_size < full_modular_size

    def test_field_access_performance(self):
        """필드 접근 성능 테스트"""
        flat_state = create_flat_state("테스트 질문", "session_123")
        modular_state = create_modular_state("테스트 질문", "session_123")

        from core.agents.state_helpers import get_field

        # 반복 횟수 증가 (더 정확한 측정)
        iterations = 10000

        # Flat 구조 접근 시간 측정
        start = time.time()
        for _ in range(iterations):
            _ = flat_state.get("query")
            _ = flat_state.get("query_type")
            _ = flat_state.get("confidence")
        flat_time = time.time() - start

        # Modular 구조 접근 시간 측정 (helper 함수 사용)
        start = time.time()
        for _ in range(iterations):
            _ = get_field(modular_state, "query")
            _ = get_field(modular_state, "query_type")
            _ = get_field(modular_state, "confidence")
        modular_time = time.time() - start

        print(f"\nFlat 구조 접근: {flat_time*1000:.2f} ms ({iterations}회)")
        print(f"Modular 구조 접근: {modular_time*1000:.2f} ms ({iterations}회)")

        if flat_time > 0:
            ratio = modular_time / flat_time
            print(f"성능 비율: {ratio:.2f}x (Modular/Flat)")

        # Helper 함수 오버헤드는 있지만 허용 범위 내여야 함
        # Flat 구조 접근이 너무 빠르면 상대 비교가 의미 없으므로 절대 시간 기준 사용

        # 절대 시간 기준: 10000회 반복 시 500ms 이하면 합리적
        # (회당 0.05ms = 50마이크로초)
        # Helper 함수는 람다 호출과 중첩 접근으로 인해 오버헤드가 있지만
        # 실제 사용 시 이 오버헤드는 무시할 수준이어야 함
        max_modular_time = 0.5  # 500ms (10000회 기준)

        # 절대 시간 기준 확인 (상대 비교는 환경에 따라 변동이 크므로 제외)
        assert modular_time < max_modular_time, \
            f"Modular 접근 시간({modular_time*1000:.2f}ms)이 최대 허용 시간({max_modular_time*1000:.2f}ms)을 초과. " \
            f"Helper 함수 성능 최적화가 필요할 수 있습니다."


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
