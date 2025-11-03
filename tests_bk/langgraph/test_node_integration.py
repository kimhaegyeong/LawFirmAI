# -*- coding: utf-8 -*-
"""
?�드 ?�합 ?�스??(Phase 1-3)
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

from source.agents.workflow_service import LangGraphWorkflowService
from infrastructure.utils.langgraph_config import LangGraphConfig


def clear_global_cache():
    """?�스??격리�??�한 global cache 초기??""
    try:
        from source.agents.node_wrappers import _global_search_results_cache
        if _global_search_results_cache is not None:
            _global_search_results_cache.clear()
    except (ImportError, AttributeError, TypeError):
        pass


async def test_simple_query_integration():
    """간단??질문 ?�스??(?�합 ?�드 검�?"""
    print("=" * 80)
    print("?�스?? 간단??질문 (?�합 ?�드 검�?")
    print("=" * 80)

    # ?�스??격리�??�한 cache 초기??
    clear_global_cache()

    config = LangGraphConfig.from_env()
    workflow_service = LangGraphWorkflowService(config)

    query = "?�녕?�세??
    print(f"\n질문: {query}")

    start = time.time()
    result = await workflow_service.process_query(query)
    elapsed = time.time() - start

    print(f"\n[결과]")
    print(f"  ?�간: {elapsed:.2f}�?)
    print(f"  복잡?? {result.get('query_complexity', 'unknown')}")
    print(f"  검???�요: {result.get('needs_search', True)}")
    print(f"  ?��? 길이: {len(result.get('answer', ''))}??)

    # ?�합 ?�드 검�?
    processing_steps = result.get('processing_steps', [])
    # processing_steps??문자??리스?�이거나 ?�셔?�리 리스?�일 ???�음
    step_texts = []
    for step in processing_steps:
        if isinstance(step, dict):
            step_texts.append(step.get('step', '') or str(step))
        elif isinstance(step, str):
            step_texts.append(step)
        else:
            step_texts.append(str(step))

    has_format_and_prepare = any('?�맷?? in step or '최종 준�? in step or '?�맷?? in step for step in step_texts)

    success = (
        result.get('query_complexity') == 'simple' and
        result.get('needs_search') == False and
        has_format_and_prepare
    )

    if success:
        print("  ??[PASS] ?�합 ?�드 ?�상 ?�동")
    else:
        print(f"  ??[FAIL] ?�합 ?�드 검�??�패")
        print(f"        processing_steps: {step_texts[-5:]}")

    return success


async def test_moderate_query_integration():
    """중간 복잡??질문 ?�스??(?�합 ?�드 검�?"""
    print("\n" + "=" * 80)
    print("?�스?? 중간 복잡??질문 (?�합 ?�드 검�?")
    print("=" * 80)

    # ?�스??격리�??�한 cache 초기??
    clear_global_cache()

    config = LangGraphConfig.from_env()
    workflow_service = LangGraphWorkflowService(config)

    query = "민법 ??11조의 ?�용???�려주세??
    print(f"\n질문: {query}")

    start = time.time()
    result = await workflow_service.process_query(query)
    elapsed = time.time() - start

    print(f"\n[결과]")
    print(f"  ?�간: {elapsed:.2f}�?)
    print(f"  복잡?? {result.get('query_complexity', 'unknown')}")
    print(f"  검???�요: {result.get('needs_search', True)}")
    print(f"  ?��? 길이: {len(result.get('answer', ''))}??)

    # ?�합 ?�드 검�?
    processing_steps = result.get('processing_steps', [])
    # processing_steps??문자??리스?�이거나 ?�셔?�리 리스?�일 ???�음
    step_texts = []
    for step in processing_steps:
        if isinstance(step, dict):
            step_texts.append(step.get('step', '') or str(step))
        elif isinstance(step, str):
            step_texts.append(step)
        else:
            step_texts.append(str(step))

    has_documents_and_terms = any('문서 컨텍?�트' in step or '?�어' in step or '문서' in step for step in step_texts)
    has_format_and_prepare = any('?�맷?? in step or '최종 준�? in step or '?�맷?? in step for step in step_texts)

    success = (
        result.get('query_complexity') == 'moderate' and
        result.get('needs_search') == True and
        has_documents_and_terms and
        has_format_and_prepare
    )

    if success:
        print("  ??[PASS] ?�합 ?�드 ?�상 ?�동")
    else:
        print(f"  ??[FAIL] ?�합 ?�드 검�??�패")
        print(f"        has_documents_and_terms: {has_documents_and_terms}")
        print(f"        has_format_and_prepare: {has_format_and_prepare}")

    return success




async def main():
    """?�합 ?�스???�행"""
    print("\n" + "=" * 80)
    print("?�드 ?�합 ?�스??(Phase 1-3)")
    print("=" * 80)

    results = []

    # ?�스??1: 간단??질문
    try:
        result1 = await test_simple_query_integration()
        results.append(("간단??질문", result1))
    except Exception as e:
        print(f"  ??[ERROR] 간단??질문 ?�스???�패: {e}")
        results.append(("간단??질문", False))

    # ?�스??2: 중간 복잡??질문
    try:
        result2 = await test_moderate_query_integration()
        results.append(("중간 복잡??질문", result2))
    except Exception as e:
        print(f"  ??[ERROR] 중간 복잡??질문 ?�스???�패: {e}")
        results.append(("중간 복잡??질문", False))


    # 결과 ?�약
    print("\n" + "=" * 80)
    print("?�� ?�스??결과 ?�약")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "??PASS" if result else "??FAIL"
        print(f"  {test_name}: {status}")

    print(f"\n?�체: {passed}/{total} ?�스???�과")

    if passed == total:
        print("\n??모든 ?�스???�과!")
        return 0
    else:
        print(f"\n?�️ {total - passed}�??�스???�패")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
