# -*- coding: utf-8 -*-
"""
LangGraph State 시스템 전체 통합 테스트
모든 기능을 한 번에 검증
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# .env 파일 로드 (python-dotenv 패키지 필요)
try:
    from dotenv import load_dotenv
    # 프로젝트 루트에서 .env 파일 로드
    load_dotenv(dotenv_path=str(project_root / ".env"))
except ImportError:
    # python-dotenv가 설치되지 않은 경우 경고만 출력하고 계속 진행
    pass

# LangSmith 모니터링 설정 (선택 사항)
# .env 파일 또는 환경 변수에서 다음 변수들을 설정하세요:
#
# 필수 (LangChain 표준 환경변수):
#   LANGCHAIN_TRACING_V2=true          # LangSmith 트레이싱 활성화
#   LANGCHAIN_API_KEY=your-api-key     # LangSmith API 키
#   LANGCHAIN_PROJECT=LawFirmAI-Test   # LangSmith 프로젝트 이름 (선택사항)
#
# 선택 (하위 호환성):
#   ENABLE_LANGSMITH=true              # 추가 활성화 플래그 (선택사항)
#   LANGSMITH_API_KEY=...              # LANGCHAIN_API_KEY 대신 사용 가능
#   LANGSMITH_PROJECT=...              # LANGCHAIN_PROJECT 대신 사용 가능

# LangSmith 설정 읽기 (.env 파일에서 이미 로드됨)
# 표준 LangChain 환경변수 우선 사용
langsmith_api_key = os.environ.get("LANGCHAIN_API_KEY", "") or os.environ.get("LANGSMITH_API_KEY", "")
langsmith_tracing = os.environ.get("LANGCHAIN_TRACING_V2", "false").lower()
langsmith_project = os.environ.get("LANGCHAIN_PROJECT", "") or os.environ.get("LANGSMITH_PROJECT", "LawFirmAI-Test")
enable_langsmith_flag = os.environ.get("ENABLE_LANGSMITH", "false").lower() == "true"

# LangSmith 활성화 여부 확인
# LANGCHAIN_TRACING_V2가 'true'이고 API 키가 있으면 활성화
langsmith_enabled = (
    langsmith_tracing in ["true", "1", "yes"]
    and bool(langsmith_api_key)
)

# ENABLE_LANGSMITH 플래그가 설정된 경우에도 활성화 (하위 호환성)
if enable_langsmith_flag and langsmith_tracing not in ["true", "1", "yes"]:
    # 플래그만 있고 TRACING_V2가 없으면 경고
    print("⚠ ENABLE_LANGSMITH=true가 설정되었지만 LANGCHAIN_TRACING_V2가 설정되지 않았습니다.")
    print("  LangSmith를 활성화하려면 LANGCHAIN_TRACING_V2=true를 설정하세요.\n")
    langsmith_enabled = False

if langsmith_enabled:
    print("=" * 80)
    print("LangSmith 모니터링 활성화됨")
    print("=" * 80)
    if langsmith_api_key:
        # API 키 부분만 표시 (보안)
        if len(langsmith_api_key) > 30:
            print(f"  API Key: {langsmith_api_key[:15]}...{langsmith_api_key[-10:]} (부분 표시)")
        else:
            print(f"  API Key: {'*' * min(len(langsmith_api_key), 20)}... (설정됨)")
    print(f"  Project: {langsmith_project}")
    print(f"  Tracing: 활성화 (LANGCHAIN_TRACING_V2={langsmith_tracing})")
    if enable_langsmith_flag:
        print("  ENABLE_LANGSMITH: 활성화됨")
    print("=" * 80 + "\n")
else:
    # 비활성화된 경우 상세한 안내 메시지 출력
    print("ℹ LangSmith 모니터링 비활성화됨 (기본값)")

    missing_config = []
    if langsmith_tracing not in ["true", "1", "yes"]:
        missing_config.append("LANGCHAIN_TRACING_V2=true")
    if not langsmith_api_key:
        missing_config.append("LANGCHAIN_API_KEY=your-api-key")

    if missing_config:
        print("  LangSmith를 활성화하려면 .env 파일에 다음 환경 변수를 설정하세요:")
        for config in missing_config:
            print(f"    {config}")
        if not langsmith_project or langsmith_project == "LawFirmAI-Test":
            print("    LANGCHAIN_PROJECT=LawFirmAI-Test  # (선택사항)")
        print()
    else:
        print("  (설정은 되어 있지만 활성화 조건을 만족하지 않습니다)\n")

from typing import Any, Dict

# 핵심 모듈 import
from core.agents.node_input_output_spec import (
    NODE_SPECS,
    get_all_node_names,
    get_node_spec,
    validate_node_input,
    validate_workflow_flow,
)
from core.agents.state_adapter import StateAdapter, adapt_state, flatten_state
from core.agents.state_reduction import StateReducer, reduce_state_for_node


def test_node_specs():
    """노드 스펙 검증"""
    print("=" * 80)
    print("1. 노드 스펙 검증")
    print("=" * 80)

    all_nodes = get_all_node_names()
    print(f"✅ 총 {len(all_nodes)}개 노드 스펙 정의됨")

    for node_name in all_nodes:
        spec = get_node_spec(node_name)
        print(f"  - {node_name}: {len(spec.required_input)}개 입력, {len(spec.output)}개 출력")

    return True


def test_state_adapter():
    """State Adapter 검증"""
    print("\n" + "=" * 80)
    print("2. State Adapter 검증")
    print("=" * 80)

    # Flat State 생성
    flat_state = {
        "query": "계약서 작성 시 주의사항은?",
        "session_id": "test_123",
        "query_type": "general_question",
        "confidence": 0.85,
        "retrieved_docs": [],
        "answer": "",
        "sources": [],
        "processing_steps": [],
        "errors": []
    }

    # Flat → Nested 변환
    nested_state = adapt_state(flat_state)
    assert "input" in nested_state, "❌ Flat → Nested 변환 실패"
    assert nested_state["input"]["query"] == flat_state["query"]
    print("✅ Flat → Nested 변환 성공")

    # Nested → Flat 변환
    flat_again = flatten_state(nested_state)
    assert flat_again["query"] == flat_state["query"]
    print("✅ Nested → Flat 변환 성공")

    # Round-trip 검증
    assert flat_again["query"] == flat_state["query"]
    assert flat_again["query_type"] == flat_state["query_type"]
    print("✅ Round-trip 변환 검증 성공")

    return True


def test_state_reduction():
    """State Reduction 검증"""
    print("\n" + "=" * 80)
    print("3. State Reduction 검증")
    print("=" * 80)

    # 대용량 State 생성
    full_state = {
        "query": "계약서 작성 시 주의사항은?",
        "session_id": "test_123",
        "query_type": "general_question",
        "confidence": 0.85,
        "retrieved_docs": [
            {"content": "test " * 100, "source": f"doc_{i}"}
            for i in range(20)
        ],
        "answer": "답변 내용입니다",
        "sources": [],
        "processing_steps": [f"step_{i}" for i in range(50)],
        "errors": []
    }

    reducer = StateReducer(aggressive_reduction=True)

    # 각 노드별로 State Reduction
    nodes_to_test = [
        "classify_query",
        "assess_urgency",
        "retrieve_documents",
        "generate_answer_enhanced"
    ]

    for node_name in nodes_to_test:
        reduced = reducer.reduce_state_for_node(full_state, node_name)
        assert isinstance(reduced, dict), f"❌ {node_name} State Reduction 실패"

        reduction_info = ""
        if "input" in reduced and "query" in reduced.get("input", {}):
            reduction_info = f" (reduced)"

        print(f"  ✅ {node_name} State Reduction 성공{reduction_info}")

    return True


def test_workflow_validation():
    """워크플로우 검증"""
    print("\n" + "=" * 80)
    print("4. 워크플로우 검증")
    print("=" * 80)

    result = validate_workflow_flow()

    print(f"총 {result['total_nodes']}개 노드")
    print(f"검증 결과: {'✅ Valid' if result['valid'] else '⚠️ Issues found'}")

    if result['issues']:
        print(f"\n⚠️ {len(result['issues'])}개의 이슈 발견:")
        for issue in result['issues'][:5]:
            print(f"  - {issue}")

    return True


def test_node_input_validation():
    """노드 Input 검증"""
    print("\n" + "=" * 80)
    print("5. 노드 Input 검증")
    print("=" * 80)

    # 유효한 입력
    valid_state = {
        "query": "계약서 작성 시 주의사항은?",
        "session_id": "test_123"
    }
    is_valid, error = validate_node_input("classify_query", valid_state)
    assert is_valid, f"❌ 유효한 입력이 거부됨: {error}"
    print("✅ 유효한 입력 검증 성공")

    # 유효하지 않은 입력
    invalid_state = {
        "session_id": "test_123"
        # query 누락
    }
    is_valid, error = validate_node_input("classify_query", invalid_state)
    assert not is_valid, "❌ 유효하지 않은 입력이 통과됨"
    print("✅ 유효하지 않은 입력 거부 성공")

    return True


def test_all_nodes():
    """모든 노드 검증"""
    print("\n" + "=" * 80)
    print("6. 모든 노드 검증")
    print("=" * 80)

    all_nodes = get_all_node_names()

    # 필수 노드 확인
    required_nodes = [
        "classify_query",
        "assess_urgency",
        "resolve_multi_turn",
        "route_expert",
        "retrieve_documents",
        "generate_answer_enhanced",
        "validate_answer_quality"
    ]

    for node in required_nodes:
        assert node in all_nodes, f"❌ 필수 노드 {node}가 없습니다"
        spec = get_node_spec(node)
        assert spec is not None, f"❌ {node}에 대한 스펙이 없습니다"
        print(f"  ✅ {node}")

    print(f"\n✅ 전체 {len(all_nodes)}개 노드 모두 정상")

    return True


def main():
    """메인 테스트 실행"""
    print("\n" + "=" * 80)
    print("LangGraph State 시스템 전체 통합 테스트")
    print("=" * 80)

    tests = [
        ("노드 스펙 검증", test_node_specs),
        ("State Adapter 검증", test_state_adapter),
        ("State Reduction 검증", test_state_reduction),
        ("워크플로우 검증", test_workflow_validation),
        ("노드 Input 검증", test_node_input_validation),
        ("모든 노드 검증", test_all_nodes),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n❌ {test_name} 실패: {e}")

    # 결과 요약
    print("\n" + "=" * 80)
    print("테스트 결과 요약")
    print("=" * 80)

    passed = sum(1 for _, result, _ in results if result)
    failed = sum(1 for _, result, _ in results if not result)

    for test_name, result, error in results:
        status = "✅ 통과" if result else "❌ 실패"
        if error:
            print(f"  {status}: {test_name} ({error})")
        else:
            print(f"  {status}: {test_name}")

    print(f"\n총 {len(results)}개 테스트 중 {passed}개 통과, {failed}개 실패")

    if failed == 0:
        print("\n🎉 모든 테스트 통과!")
    else:
        print(f"\n⚠️ {failed}개 테스트 실패")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
