# -*- coding: utf-8 -*-
"""
노드 통합 테스트 (Phase 1-3)
"""
import asyncio
import logging
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

from core.agents.workflow_service import LangGraphWorkflowService
from infrastructure.utils.langgraph_config import LangGraphConfig


def clear_global_cache():
    """테스트 격리를 위한 global cache 초기화"""
    try:
        from core.agents.node_wrappers import _global_search_results_cache
        if _global_search_results_cache is not None:
            _global_search_results_cache.clear()
    except (ImportError, AttributeError, TypeError):
        pass


async def test_simple_query_integration():
    """간단한 질문 테스트 (통합 노드 검증)"""
    print("=" * 80)
    print("테스트: 간단한 질문 (통합 노드 검증)")
    print("=" * 80)

    # 테스트 격리를 위한 cache 초기화
    clear_global_cache()

    config = LangGraphConfig.from_env()
    workflow_service = LangGraphWorkflowService(config)

    query = "안녕하세요"
    print(f"\n질문: {query}")

    start = time.time()
    result = await workflow_service.process_query(query)
    elapsed = time.time() - start

    print(f"\n[결과]")
    print(f"  시간: {elapsed:.2f}초")
    print(f"  복잡도: {result.get('query_complexity', 'unknown')}")
    print(f"  검색 필요: {result.get('needs_search', True)}")
    print(f"  답변 길이: {len(result.get('answer', ''))}자")

    # 통합 노드 검증
    processing_steps = result.get('processing_steps', [])
    # processing_steps는 문자열 리스트이거나 딕셔너리 리스트일 수 있음
    step_texts = []
    for step in processing_steps:
        if isinstance(step, dict):
            step_texts.append(step.get('step', '') or str(step))
        elif isinstance(step, str):
            step_texts.append(step)
        else:
            step_texts.append(str(step))

    has_format_and_prepare = any('포맷팅' in step or '최종 준비' in step or '포맷팅' in step for step in step_texts)

    success = (
        result.get('query_complexity') == 'simple' and
        result.get('needs_search') == False and
        has_format_and_prepare
    )

    if success:
        print("  ✅ [PASS] 통합 노드 정상 작동")
    else:
        print(f"  ❌ [FAIL] 통합 노드 검증 실패")
        print(f"        processing_steps: {step_texts[-5:]}")

    return success


async def test_moderate_query_integration():
    """중간 복잡도 질문 테스트 (통합 노드 검증)"""
    print("\n" + "=" * 80)
    print("테스트: 중간 복잡도 질문 (통합 노드 검증)")
    print("=" * 80)

    # 테스트 격리를 위한 cache 초기화
    clear_global_cache()

    config = LangGraphConfig.from_env()
    workflow_service = LangGraphWorkflowService(config)

    query = "민법 제111조의 내용을 알려주세요"
    print(f"\n질문: {query}")

    start = time.time()
    result = await workflow_service.process_query(query)
    elapsed = time.time() - start

    print(f"\n[결과]")
    print(f"  시간: {elapsed:.2f}초")
    print(f"  복잡도: {result.get('query_complexity', 'unknown')}")
    print(f"  검색 필요: {result.get('needs_search', True)}")
    print(f"  답변 길이: {len(result.get('answer', ''))}자")

    # 통합 노드 검증
    processing_steps = result.get('processing_steps', [])
    # processing_steps는 문자열 리스트이거나 딕셔너리 리스트일 수 있음
    step_texts = []
    for step in processing_steps:
        if isinstance(step, dict):
            step_texts.append(step.get('step', '') or str(step))
        elif isinstance(step, str):
            step_texts.append(step)
        else:
            step_texts.append(str(step))

    has_documents_and_terms = any('문서 컨텍스트' in step or '용어' in step or '문서' in step for step in step_texts)
    has_format_and_prepare = any('포맷팅' in step or '최종 준비' in step or '포맷팅' in step for step in step_texts)

    success = (
        result.get('query_complexity') == 'moderate' and
        result.get('needs_search') == True and
        has_documents_and_terms and
        has_format_and_prepare
    )

    if success:
        print("  ✅ [PASS] 통합 노드 정상 작동")
    else:
        print(f"  ❌ [FAIL] 통합 노드 검증 실패")
        print(f"        has_documents_and_terms: {has_documents_and_terms}")
        print(f"        has_format_and_prepare: {has_format_and_prepare}")

    return success




async def main():
    """통합 테스트 실행"""
    print("\n" + "=" * 80)
    print("노드 통합 테스트 (Phase 1-3)")
    print("=" * 80)

    results = []

    # 테스트 1: 간단한 질문
    try:
        result1 = await test_simple_query_integration()
        results.append(("간단한 질문", result1))
    except Exception as e:
        print(f"  ❌ [ERROR] 간단한 질문 테스트 실패: {e}")
        results.append(("간단한 질문", False))

    # 테스트 2: 중간 복잡도 질문
    try:
        result2 = await test_moderate_query_integration()
        results.append(("중간 복잡도 질문", result2))
    except Exception as e:
        print(f"  ❌ [ERROR] 중간 복잡도 질문 테스트 실패: {e}")
        results.append(("중간 복잡도 질문", False))


    # 결과 요약
    print("\n" + "=" * 80)
    print("📊 테스트 결과 요약")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")

    print(f"\n전체: {passed}/{total} 테스트 통과")

    if passed == total:
        print("\n✅ 모든 테스트 통과!")
        return 0
    else:
        print(f"\n⚠️ {total - passed}개 테스트 실패")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
