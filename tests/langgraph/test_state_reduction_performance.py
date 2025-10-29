# -*- coding: utf-8 -*-
"""
State Reduction 성능 테스트
메모리 사용량, 처리 속도, 데이터 전송량 비교
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pytest
except ImportError:
    pytest = None

from core.agents.node_input_output_spec import get_all_node_names
from core.agents.state_adapter import StateAdapter
from core.agents.state_reduction import (
    StateReducer,
    reduce_state_for_node,
    reduce_state_size,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_state() -> dict:
    """테스트용 대용량 State 생성"""
    return {
        "query": "계약서 작성 시 주의사항은?",
        "session_id": "test_session_123",
        "query_type": "general_question",
        "confidence": 0.85,
        "legal_field": "civil_law",
        "legal_domain": "일반",
        "urgency_level": "medium",
        "urgency_reasoning": "긴급도 평가 결과",
        "emergency_type": None,
        "complexity_level": "simple",
        "requires_expert": False,
        "expert_subgraph": None,
        "is_multi_turn": False,
        "multi_turn_confidence": 1.0,
        "conversation_history": [],
        "conversation_context": None,
        "search_query": "계약서 작성 시 주의사항",
        "extracted_keywords": ["계약서", "주의사항", "법률"],
        "ai_keyword_expansion": {
            "keywords": ["계약서", "주의사항", "법률"],
            "expanded": ["계약", "계약서", "주의사항", "법률", "법령"],
            "confidence": 0.9
        },
        "retrieved_docs": [
            {"content": "테스트 문서 내용 " * 500, "source": f"doc_{i}", "metadata": {"title": f"문서 {i}"}}
            for i in range(20)
        ],
        "analysis": "법률 분석 결과",
        "legal_references": ["민법", "계약법"],
        "legal_citations": [{"law": "민법", "article": "제1조"}],
        "answer": "계약서 작성 시 주요 주의사항은 다음과 같습니다...",
        "sources": ["doc_1", "doc_2"],
        "enhanced_answer": "계약서 작성 시 주요 주의사항은 다음과 같습니다...",
        "structure_confidence": 0.95,
        "document_type": None,
        "document_analysis": None,
        "key_clauses": [],
        "potential_issues": [],
        "legal_validity_check": True,
        "legal_basis_validation": {"confidence": 0.9},
        "outdated_laws": [],
        "quality_check_passed": True,
        "quality_score": 0.85,
        "retry_count": 0,
        "needs_enhancement": False,
        "processing_steps": [f"단계 {i}" for i in range(50)],
        "errors": [],
        "metadata": {"version": "1.0"},
        "processing_time": 0.0,
        "tokens_used": 1000
    }


def estimate_size(obj) -> int:
    """객체 크기 추정 (bytes)"""
    return sys.getsizeof(str(obj))


class TestStateReductionPerformance:
    """State Reduction 성능 테스트"""

    def test_memory_usage_reduction(self):
        """메모리 사용량 감소 테스트"""
        full_state = create_test_state()
        full_size = estimate_size(full_state)

        reducer = StateReducer(aggressive_reduction=True)

        # 각 노드별로 State Reduction 적용
        nodes = [
            "classify_query",
            "assess_urgency",
            "resolve_multi_turn",
            "route_expert",
            "retrieve_documents",
            "generate_answer_enhanced"
        ]

        total_reduced_size = 0
        for node_name in nodes:
            reduced = reducer.reduce_state_for_node(full_state, node_name)
            reduced_size = estimate_size(reduced)
            reduction_pct = (1 - reduced_size / full_size) * 100 if full_size > 0 else 0

            logger.info(
                f"{node_name}: {reduction_pct:.1f}% reduction "
                f"({full_size:.0f} → {reduced_size:.0f} bytes)"
            )

            total_reduced_size += reduced_size

            # 감소율 검증
            assert reduction_pct > 0, f"{node_name}에서 감소가 발생하지 않음"

        # 평균 감소율 계산
        avg_reduction = sum([
            (1 - estimate_size(reducer.reduce_state_for_node(full_state, node)) / full_size) * 100
            for node in nodes
        ]) / len(nodes)

        logger.info(f"평균 State Reduction: {avg_reduction:.1f}%")
        assert avg_reduction > 50, f"평균 감소율이 50% 미만: {avg_reduction:.1f}%"

    def test_processing_speed(self):
        """처리 속도 개선 테스트"""
        full_state = create_test_state()
        reducer = StateReducer(aggressive_reduction=True)

        nodes = get_all_node_names()

        # State Reduction 없이 처리 시간 측정
        start = time.time()
        for node_name in nodes[:5]:
            _ = full_state  # 모의 처리
        time_without_reduction = time.time() - start

        # State Reduction 적용 후 처리 시간 측정
        start = time.time()
        for node_name in nodes[:5]:
            reduced = reducer.reduce_state_for_node(full_state, node_name)
            _ = reduced  # 모의 처리
        time_with_reduction = time.time() - start

        logger.info(
            f"처리 시간: Reduction 없이 {time_without_reduction:.4f}s, "
            f"Reduction 적용 {time_with_reduction:.4f}s"
        )

    def test_state_size_reduction(self):
        """State 크기 제한 테스트"""
        large_state = {
            "retrieved_docs": [
                {"content": "test " * 1000} for _ in range(50)
            ],
            "conversation_history": [f"turn_{i}" for i in range(20)]
        }

        reduced = reduce_state_size(large_state, max_docs=10, max_content_per_doc=500)

        assert len(reduced["retrieved_docs"]) <= 10
        for doc in reduced["retrieved_docs"]:
            assert len(doc.get("content", "")) <= 503  # 500 + "..."

        assert len(reduced["conversation_history"]) <= 5

    def test_flat_vs_nested_conversion(self):
        """Flat ↔ Nested 변환 성능 테스트"""
        flat_state = create_test_state()

        start = time.time()
        nested_state = StateAdapter.to_nested(flat_state)
        to_nested_time = time.time() - start

        start = time.time()
        flat_again = StateAdapter.to_flat(nested_state)
        to_flat_time = time.time() - start

        logger.info(f"Flat → Nested: {to_nested_time*1000:.2f}ms")
        logger.info(f"Nested → Flat: {to_flat_time*1000:.2f}ms")

        # 주요 필드 동일성 확인
        assert flat_again["query"] == flat_state["query"]
        assert flat_again["query_type"] == flat_state["query_type"]


class TestWorkflowIntegration:
    """워크플로우 통합 성능 테스트"""

    def test_full_workflow_performance(self):
        """전체 워크플로우 성능 테스트"""
        # 시뮬레이션: 전체 워크플로우 실행
        initial_state = create_test_state()
        reducer = StateReducer(aggressive_reduction=True)

        workflow_nodes = [
            "classify_query",
            "assess_urgency",
            "resolve_multi_turn",
            "route_expert",
            "expand_keywords_ai",
            "retrieve_documents",
            "process_legal_terms",
            "generate_answer_enhanced",
            "validate_answer_quality"
        ]

        start_time = time.time()
        total_memory_reduction = 0

        for node_name in workflow_nodes:
            reduced = reducer.reduce_state_for_node(initial_state, node_name)
            reduced_size = estimate_size(reduced)
            original_size = estimate_size(initial_state)
            reduction = (1 - reduced_size / original_size) * 100 if original_size > 0 else 0

            total_memory_reduction += reduction

        total_time = time.time() - start_time
        avg_reduction = total_memory_reduction / len(workflow_nodes)

        logger.info(f"전체 워크플로우 실행 시간: {total_time:.4f}s")
        logger.info(f"평균 메모리 감소율: {avg_reduction:.1f}%")

        assert total_time < 1.0, "웨크플로우 실행 시간이 너무 김"
        assert avg_reduction > 40, f"평균 감소율이 낮음: {avg_reduction:.1f}%"


def benchmark_state_operations():
    """State 연산 벤치마크"""
    logger.info("🔍 State Reduction 성능 벤치마크 시작")

    state = create_test_state()
    reducer = StateReducer(aggressive_reduction=True)

    results = []

    for node_name in get_all_node_names():
        start = time.time()
        reduced = reducer.reduce_state_for_node(state, node_name)
        elapsed = time.time() - start

        original_size = estimate_size(state)
        reduced_size = estimate_size(reduced)
        reduction_pct = (1 - reduced_size / original_size) * 100 if original_size > 0 else 0

        results.append({
            "node": node_name,
            "time_ms": elapsed * 1000,
            "original_size": original_size,
            "reduced_size": reduced_size,
            "reduction_pct": reduction_pct
        })

    # 결과 출력
    logger.info("=" * 80)
    logger.info(f"{'Node':<30} {'Time(ms)':<12} {'Size(bytes)':<15} {'Reduction(%)':<15}")
    logger.info("-" * 80)

    for r in sorted(results, key=lambda x: x["reduction_pct"], reverse=True):
        logger.info(
            f"{r['node']:<30} {r['time_ms']:<12.2f} "
            f"{r['original_size']}→{r['reduced_size']:<4} {r['reduction_pct']:<15.1f}"
        )

    # 통계
    avg_time = sum(r["time_ms"] for r in results) / len(results)
    avg_reduction = sum(r["reduction_pct"] for r in results) / len(results)
    total_original = sum(r["original_size"] for r in results)
    total_reduced = sum(r["reduced_size"] for r in results)

    logger.info("-" * 80)
    logger.info(f"평균 처리 시간: {avg_time:.2f}ms")
    logger.info(f"평균 감소율: {avg_reduction:.1f}%")
    logger.info(f"전체 크기: {total_original:,} → {total_reduced:,} bytes")

    return results


if __name__ == "__main__":
    # 간단한 테스트 실행
    benchmark_state_operations()
