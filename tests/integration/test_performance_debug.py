#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
성능 디버깅 테스트 - 병목 지점 파악
"""

import cProfile
import os
import pstats
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from source.services.langgraph_workflow.legal_workflow_enhanced import (
    EnhancedLegalQuestionWorkflow,
)
from source.services.langgraph_workflow.state_definitions import create_initial_state
from source.utils.langgraph_config import LangGraphConfig


def test_performance_debug():
    """성능 병목 지점 분석"""
    print("=" * 70)
    print("성능 병목 지점 분석")
    print("=" * 70)

    config = LangGraphConfig()

    # 워크플로우 초기화 시간 측정
    print("\n[1] 워크플로우 초기화...")
    start = time.time()
    workflow = EnhancedLegalQuestionWorkflow(config)
    init_time = time.time() - start
    print(f"초기화 시간: {init_time:.2f}초")

    # 테스트 쿼리
    query = "이혼 절차에 대해 알려주세요"
    print(f"\n[2] 쿼리 처리: '{query}'")

    state = create_initial_state(query, "perf_test", "user_1")
    state["user_query"] = query

    # 각 단계별 시간 측정
    steps = [
        ("입력 검증", lambda s: workflow.validate_input(s)),
        ("특수 쿼리 감지", lambda s: workflow.detect_special_queries(s)),
        ("질문 분류", lambda s: workflow.classify_query(s)),
        ("하이브리드 분석", lambda s: workflow.analyze_query_hybrid(s)),
        ("법률 제한 검증", lambda s: workflow.validate_legal_restrictions(s)),
        ("문서 검색", lambda s: workflow.retrieve_documents(s)),
        ("Phase 1", lambda s: workflow.enrich_conversation_context(s)),
        ("Phase 2", lambda s: workflow.personalize_response(s)),
        ("Phase 3", lambda s: workflow.manage_memory_quality(s)),
        ("답변 생성", lambda s: workflow.generate_answer_enhanced(s)),
        ("후처리 1", lambda s: workflow.enhance_completion(s)),
        ("후처리 2", lambda s: workflow.add_disclaimer(s)),
    ]

    total_time = 0
    step_times = []

    for step_name, step_func in steps:
        start = time.time()
        state = step_func(state)
        elapsed = time.time() - start
        total_time += elapsed
        step_times.append((step_name, elapsed))
        print(f"  - {step_name}: {elapsed:.3f}초")

    print("\n" + "=" * 70)
    print("성능 요약")
    print("=" * 70)
    print(f"초기화: {init_time:.2f}초")
    print(f"총 처리 시간: {total_time:.2f}초")
    print(f"평균 단계 시간: {total_time/len(steps):.3f}초")

    # 가장 느린 단계 상위 3개
    step_times.sort(key=lambda x: x[1], reverse=True)
    print(f"\n🐌 가장 느린 단계 TOP 3:")
    for i, (step_name, elapsed) in enumerate(step_times[:3], 1):
        print(f"  {i}. {step_name}: {elapsed:.3f}초 ({elapsed/total_time*100:.1f}%)")

    # 병목 지점 제안
    bottleneck_threshold = total_time * 0.3  # 30% 이상 소요
    bottlenecks = [s for s in step_times if s[1] > bottleneck_threshold]

    if bottlenecks:
        print(f"\n⚠️  병목 지점 발견 ({bottleneck_threshold:.2f}초 이상):")
        for step_name, elapsed in bottlenecks:
            print(f"  - {step_name}: {elapsed:.3f}초")

    return state


if __name__ == "__main__":
    state = test_performance_debug()
    print("\n✅ 성능 분석 완료")
