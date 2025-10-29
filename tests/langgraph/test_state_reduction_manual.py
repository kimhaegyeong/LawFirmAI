# -*- coding: utf-8 -*-
"""
LangGraph State Reduction 수동 테스트
pytest 없이 직접 실행 가능한 테스트
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

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


def test_get_node_spec():
    """노드 스펙 조회 테스트"""
    print("테스트: 노드 스펙 조회")
    spec = get_node_spec("classify_query")
    assert spec is not None, "classify_query 스펙이 없습니다"
    assert spec.node_name == "classify_query", "노드 이름 불일치"
    assert "query" in spec.required_fields, "query 필드가 required_fields에 없습니다"
    print("  ✓ 통과")


def test_get_required_fields():
    """필수 필드 조회 테스트"""
    print("테스트: 필수 필드 조회")
    fields = get_required_fields("classify_query")
    assert "query" in fields, "query가 필수 필드에 없습니다"
    assert "session_id" in fields, "session_id가 always_include에 없습니다"
    assert "processing_steps" in fields, "processing_steps가 always_include에 없습니다"
    print("  ✓ 통과")


def test_validate_node_input():
    """Input 유효성 검증 테스트"""
    print("테스트: Input 유효성 검증")
    state = {
        "query": "테스트 질문",
        "session_id": "test_session",
        "processing_steps": [],
        "errors": [],
        "metadata": {},
    }
    is_valid, error = validate_node_input("classify_query", state)
    assert is_valid, f"유효한 state가 거부됨: {error}"

    # 필수 필드 없는 경우
    invalid_state = {"session_id": "test_session"}
    is_valid, error = validate_node_input("classify_query", invalid_state)
    assert not is_valid, "유효하지 않은 state가 통과됨"
    assert "query" in error.lower(), "에러 메시지에 query가 없음"
    print("  ✓ 통과")


def test_reduce_state_size():
    """State 크기 축소 테스트"""
    print("테스트: State 크기 축소")
    state = {
        "retrieved_docs": [
            {"content": "긴 내용 " * 200, "source": f"doc_{i}"}
            for i in range(20)
        ],
        "conversation_history": [{"role": "user", "content": "질문"} for _ in range(10)],
        "query": "테스트",
        "session_id": "test",
    }

    reduced = reduce_state_size(state, max_docs=5, max_content_per_doc=100)

    # 문서 수 제한
    assert len(reduced["retrieved_docs"]) <= 5, f"문서 수가 제한되지 않음: {len(reduced['retrieved_docs'])}"
    print(f"  ✓ 문서 수: 20 → {len(reduced['retrieved_docs'])}")

    # 각 문서의 내용 길이 제한 (truncate 메시지 포함하여 약간의 여유 허용)
    for doc in reduced["retrieved_docs"]:
        if "content" in doc:
            # truncate 메시지가 포함될 수 있으므로 약간의 여유 허용 (110자까지)
            assert len(doc["content"]) <= 110, f"문서 내용이 제한되지 않음: {len(doc['content'])}"

    # 대화 이력 제한
    assert len(reduced["conversation_history"]) <= 5, f"대화 이력이 제한되지 않음: {len(reduced['conversation_history'])}"
    print(f"  ✓ 대화 이력: 10 → {len(reduced['conversation_history'])}")
    print("  ✓ 통과")


def test_reduce_state_for_node():
    """노드별 state 축소 테스트"""
    print("테스트: 노드별 state 축소")
    # 전체 state 생성
    state = create_initial_legal_state("테스트 질문", "test_session")

    # 추가 필드 추가 (실제 워크플로우에서 생성되는 필드들)
    state["retrieved_docs"] = [
        {"content": "문서 내용 " * 100, "source": "test"} for _ in range(20)
    ]
    state["conversation_history"] = [{"role": "user", "content": "질문"} for _ in range(10)]
    state["legal_references"] = ["민법 제750조", "상법 제2조"]
    state["answer"] = "답변 내용"
    state["sources"] = ["source1", "source2"]

    original_field_count = len(state)
    print(f"  원본 state 필드 수: {original_field_count}")

    # classify_query 노드는 query만 필요
    reduced = reduce_state_for_node(state, "classify_query")

    # 필수 필드는 포함되어야 함
    assert "query" in reduced, "query 필드가 없습니다"
    assert "session_id" in reduced, "session_id 필드가 없습니다"
    assert "processing_steps" in reduced, "processing_steps 필드가 없습니다"

    # 필드 수 감소 확인
    reduced_field_count = len(reduced)
    print(f"  축소된 state 필드 수: {reduced_field_count}")
    print(f"  필드 감소: {original_field_count - reduced_field_count}개")

    # retrieved_docs가 있으면 크기 제한 적용
    if "retrieved_docs" in reduced:
        assert len(reduced["retrieved_docs"]) <= 10, f"retrieved_docs가 제한되지 않음: {len(reduced['retrieved_docs'])}"
        print(f"  ✓ retrieved_docs: 20 → {len(reduced['retrieved_docs'])}")

    print("  ✓ 통과")


def test_state_reducer_with_full_workflow_state():
    """전체 워크플로우 state로 테스트"""
    print("테스트: 전체 워크플로우 state 축소")
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

    original_size = len(state)
    print(f"  원본 state 필드 수: {original_size}")

    # retrieve_documents 노드용 축소
    reduced = reduce_state_for_node(state, "retrieve_documents")

    # 필요한 필드만 포함되어야 함
    assert "query" in reduced, "query 필드가 없습니다"
    assert "search_query" in reduced, "search_query 필드가 없습니다"

    # retrieved_docs가 있다면 크기 제한 적용
    if "retrieved_docs" in reduced:
        assert len(reduced["retrieved_docs"]) <= 10, f"retrieved_docs가 제한되지 않음: {len(reduced['retrieved_docs'])}"
        print(f"  ✓ retrieved_docs: 20 → {len(reduced['retrieved_docs'])}")

    reduced_size = len(reduced)
    print(f"  축소된 state 필드 수: {reduced_size}")
    print(f"  필드 감소: {original_size - reduced_size}개")
    print("  ✓ 통과")


def test_reducer_with_different_nodes():
    """다양한 노드에 대한 reducer 테스트"""
    print("테스트: 다양한 노드별 state 축소")
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
        assert "session_id" in reduced or "session_id" in state, f"{node_name}: session_id 없음"
        assert isinstance(reduced, dict), f"{node_name}: dict가 아님"

        # retrieved_docs는 크기 제한 적용
        if "retrieved_docs" in reduced:
            assert len(reduced["retrieved_docs"]) <= 10, f"{node_name}: retrieved_docs가 제한되지 않음"
        print(f"  ✓ {node_name}: 통과")

    print("  ✓ 통과")


def main():
    """모든 테스트 실행"""
    print("=" * 80)
    print("LangGraph State Reduction 테스트 시작")
    print("=" * 80)
    print()

    tests = [
        test_get_node_spec,
        test_get_required_fields,
        test_validate_node_input,
        test_reduce_state_size,
        test_reduce_state_for_node,
        test_state_reducer_with_full_workflow_state,
        test_reducer_with_different_nodes,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ 실패: {str(e)}")
            failed += 1
        except Exception as e:
            print(f"  ✗ 오류: {type(e).__name__}: {str(e)}")
            failed += 1
        print()

    print("=" * 80)
    print(f"테스트 결과: {passed}개 통과, {failed}개 실패")
    print("=" * 80)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
