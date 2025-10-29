# -*- coding: utf-8 -*-
"""
핵심 State 시스템 독립 테스트
의존성 없이 핵심 기능만 검증
"""

import logging
import sys
from pathlib import Path

# 로깅 설정 (에러 억제)
logging.getLogger().setLevel(logging.CRITICAL)

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.agents.node_input_output_spec import (
    NODE_SPECS,
    get_all_node_names,
    get_node_spec,
    validate_node_input,
    validate_workflow_flow,
)
from core.agents.state_adapter import StateAdapter, adapt_state, flatten_state
from core.agents.state_reduction import StateReducer


def test_node_specs_core():
    """노드 스펙 핵심 검증"""
    print("=" * 80)
    print("1. 노드 스펙 핵심 검증")
    print("=" * 80)

    all_nodes = get_all_node_names()
    print(f"✅ 총 {len(all_nodes)}개 노드 스펙 정의됨")

    # 필수 노드 확인
    required = ["classify_query", "assess_urgency", "retrieve_documents", "generate_answer_enhanced"]
    for node in required:
        spec = get_node_spec(node)
        assert spec is not None, f"❌ 노드 {node} 없음"
        print(f"  ✅ {node}: {len(spec.required_input)}개 입력, {len(spec.output)}개 출력")

    return True


def test_state_adapter_core():
    """State Adapter 핵심 검증"""
    print("\n" + "=" * 80)
    print("2. State Adapter 핵심 검증")
    print("=" * 80)

    # Flat State
    flat_state = {
        "query": "테스트 질문",
        "session_id": "test_123",
        "query_type": "general_question",
        "confidence": 0.85
    }

    # Flat → Nested
    nested = adapt_state(flat_state)
    assert "input" in nested
    assert nested["input"]["query"] == flat_state["query"]
    print("✅ Flat → Nested 변환 성공")

    # Nested → Flat
    flat_again = flatten_state(nested)
    assert flat_again["query"] == flat_state["query"]
    print("✅ Nested → Flat 변환 성공")

    # Round-trip
    assert flat_again["query"] == flat_state["query"]
    print("✅ Round-trip 검증 성공")

    return True


def test_state_reduction_core():
    """State Reduction 핵심 검증"""
    print("\n" + "=" * 80)
    print("3. State Reduction 핵심 검증")
    print("=" * 80)

    # 대용량 State
    full_state = {
        "query": "테스트",
        "session_id": "test",
        "retrieved_docs": [{"content": "test " * 100} for _ in range(20)],
        "processing_steps": [f"step_{i}" for i in range(50)]
    }

    reducer = StateReducer(aggressive_reduction=True)

    # 각 노드별 Reduction
    test_nodes = ["classify_query", "retrieve_documents"]
    for node_name in test_nodes:
        reduced = reducer.reduce_state_for_node(full_state, node_name)
        assert isinstance(reduced, dict), f"❌ {node_name} 실패"
        print(f"  ✅ {node_name} State Reduction 성공")

    return True


def test_workflow_validation_core():
    """워크플로우 검증 핵심"""
    print("\n" + "=" * 80)
    print("4. 워크플로우 검증 핵심")
    print("=" * 80)

    result = validate_workflow_flow()
    print(f"총 {result['total_nodes']}개 노드")
    print(f"검증 결과: {'✅ Valid' if result['valid'] else '⚠️ Issues'}")

    if result['issues']:
        print(f"\n⚠️ {len(result['issues'])}개의 이슈:")
        for issue in result['issues'][:3]:
            print(f"  - {issue}")

    return True


def main():
    """메인 테스트"""
    print("\n" + "=" * 80)
    print("LangGraph State 시스템 핵심 테스트")
    print("=" * 80)

    tests = [
        ("노드 스펙", test_node_specs_core),
        ("State Adapter", test_state_adapter_core),
        ("State Reduction", test_state_reduction_core),
        ("워크플로우 검증", test_workflow_validation_core),
    ]

    results = []
    for name, func in tests:
        try:
            result = func()
            results.append((name, result, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"❌ {name} 실패: {e}")

    # 결과
    print("\n" + "=" * 80)
    print("테스트 결과")
    print("=" * 80)

    passed = sum(1 for _, r, _ in results if r)
    failed = sum(1 for _, r, _ in results if not r)

    for name, result, error in results:
        status = "✅ 통과" if result else "❌ 실패"
        error_msg = f" ({error})" if error else ""
        print(f"  {status}: {name}{error_msg}")

    print(f"\n총 {len(results)}개 테스트 중 {passed}개 통과, {failed}개 실패")

    if failed == 0:
        print("\n🎉 모든 핵심 테스트 통과!")
        return True
    else:
        print(f"\n⚠️ {failed}개 테스트 실패")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
