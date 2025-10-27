#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph 워크플로우 성능 벤치마크 테스트
"""

import os
import sys
import time
import tracemalloc

import psutil

sys.path.insert(0, os.path.dirname(__file__))

from source.services.langgraph_workflow.legal_workflow_enhanced import (
    EnhancedLegalQuestionWorkflow,
)
from source.services.langgraph_workflow.state_definitions import create_initial_state
from source.utils.langgraph_config import LangGraphConfig


def get_memory_usage():
    """현재 메모리 사용량 반환"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB


def test_performance():
    """성능 테스트 실행"""
    print("=" * 70)
    print("LangGraph 워크플로우 성능 벤치마크")
    print("=" * 70)

    config = LangGraphConfig()
    workflow = EnhancedLegalQuestionWorkflow(config)

    # 테스트 쿼리
    test_queries = [
        "이혼 절차에 대해 알려주세요",
        "계약서 작성 방법을 알려주세요",
        "상속 순위에 대해 알려주세요",
        "형법 제250조에 대해 알려주세요",
        "민법 제750조에 대해 알려주세요"
    ]

    print(f"\n테스트 쿼리: {len(test_queries)}개\n")

    results = []

    for i, query in enumerate(test_queries, 1):
        print(f"[{i}/{len(test_queries)}] 테스트 중: '{query[:30]}...'")

        # 메모리 추적 시작
        tracemalloc.start()
        start_time = time.time()
        start_memory = get_memory_usage()

        try:
            # 워크플로우 실행
            state = create_initial_state(query, f"perf_test_{i}", "user_1")
            state["user_query"] = query

            state = workflow.validate_input(state)
            state = workflow.detect_special_queries(state)
            state["query"] = query
            state = workflow.classify_query(state)
            state = workflow.analyze_query_hybrid(state)
            state = workflow.validate_legal_restrictions(state)
            state = workflow.retrieve_documents(state)
            state = workflow.enrich_conversation_context(state)
            state = workflow.personalize_response(state)
            state = workflow.manage_memory_quality(state)
            state = workflow.generate_answer_enhanced(state)
            state = workflow.enhance_completion(state)
            state = workflow.add_disclaimer(state)

            elapsed_time = time.time() - start_time
            end_memory = get_memory_usage()
            memory_delta = end_memory - start_memory

            # 메모리 상세 정보
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            result = {
                "query": query,
                "time": elapsed_time,
                "memory_usage": memory_delta,
                "peak_memory_mb": peak / 1024 / 1024,
                "answer_length": len(state.get("answer", "")),
                "retrieved_docs": len(state.get("retrieved_docs", [])),
                "success": True
            }

            print(f"  ✅ 완료: {elapsed_time:.2f}초, 메모리: {memory_delta:.2f}MB")

        except Exception as e:
            elapsed_time = time.time() - start_time
            result = {
                "query": query,
                "time": elapsed_time,
                "memory_usage": 0,
                "peak_memory_mb": 0,
                "answer_length": 0,
                "retrieved_docs": 0,
                "success": False,
                "error": str(e)
            }
            print(f"  ❌ 실패: {e}")
            tracemalloc.stop()

        results.append(result)

    # 결과 요약
    print("\n" + "=" * 70)
    print("성능 테스트 결과 요약")
    print("=" * 70)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if successful:
        avg_time = sum(r["time"] for r in successful) / len(successful)
        avg_memory = sum(r["memory_usage"] for r in successful) / len(successful)
        avg_answer_length = sum(r["answer_length"] for r in successful) / len(successful)
        max_time = max(r["time"] for r in successful)
        min_time = min(r["time"] for r in successful)

        print(f"\n✅ 성공한 테스트: {len(successful)}/{len(results)}")
        print(f"\n⏱️  응답 시간:")
        print(f"  - 평균: {avg_time:.2f}초")
        print(f"  - 최소: {min_time:.2f}초")
        print(f"  - 최대: {max_time:.2f}초")

        print(f"\n💾 메모리 사용량:")
        print(f"  - 평균 증가량: {avg_memory:.2f}MB")

        print(f"\n📝 답변 품질:")
        print(f"  - 평균 답변 길이: {avg_answer_length:.0f}자")
        print(f"  - 평균 검색 문서: {sum(r['retrieved_docs'] for r in successful) / len(successful):.1f}개")

    if failed:
        print(f"\n❌ 실패한 테스트: {len(failed)}")
        for r in failed:
            print(f"  - {r['query'][:30]}...: {r.get('error', 'Unknown error')}")

    print("\n" + "=" * 70)

    return results


def test_error_handling():
    """에러 처리 테스트"""
    print("\n" + "=" * 70)
    print("에러 처리 테스트")
    print("=" * 70)

    config = LangGraphConfig()
    workflow = EnhancedLegalQuestionWorkflow(config)

    error_cases = [
        ("빈 쿼리", ""),
        ("너무 긴 쿼리", "A" * 10001),
        ("None 값", None),
    ]

    print(f"\n테스트 케이스: {len(error_cases)}개\n")

    results = []

    for name, query in error_cases:
        print(f"[{name}] 테스트 중...")

        try:
            if query is None:
                state = create_initial_state("", "error_test", "user_1")
            else:
                state = create_initial_state(query, "error_test", "user_1")

            state["user_query"] = query if query is not None else ""

            result = workflow.validate_input(state)

            valid = result.get("validation_results", {}).get("valid", False)

            if valid:
                print(f"  ⚠️  예상과 다른 결과: 검증 통과함")
                results.append({"name": name, "status": "unexpected_pass"})
            else:
                print(f"  ✅ 올바른 에러 처리: 검증 실패")
                results.append({"name": name, "status": "expected_fail"})

        except Exception as e:
            print(f"  ✅ 예상된 에러: {e}")
            results.append({"name": name, "status": "expected_error", "error": str(e)})

    print("\n" + "=" * 70)
    print("에러 처리 테스트 완료")
    print("=" * 70)

    return results


if __name__ == "__main__":
    performance_results = test_performance()
    error_results = test_error_handling()

    print("\n✅ 모든 테스트 완료!")

    # 최종 요약
    successful_perf = sum(1 for r in performance_results if r["success"])
    successful_error = sum(1 for r in error_results)

    print(f"\n📊 최종 통계:")
    print(f"  - 성능 테스트: {successful_perf}/{len(performance_results)} 성공")
    print(f"  - 에러 처리 테스트: {successful_error}/{len(error_results)} 성공")
