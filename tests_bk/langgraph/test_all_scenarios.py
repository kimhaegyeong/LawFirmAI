# -*- coding: utf-8 -*-
"""
Adaptive RAG �?그래???�순??최적???�체 ?�나리오 ?�스??
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


async def test_simple_query():
    """간단??질문 ?�스??(?�사�?"""
    print("=" * 80)
    print("?�스??1: 간단??질문 (?�사�?")
    print("=" * 80)

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
    print(f"  검??문서 ?? {len(result.get('retrieved_docs', []))}�?)
    print(f"  ?��? 길이: {len(result.get('answer', ''))}??)

    success = (
        result.get('query_complexity') == 'simple' and
        result.get('needs_search') == False and
        len(result.get('retrieved_docs', [])) == 0
    )

    if success:
        print("  ??[PASS] 간단??질문 ?�상 처리 (검???�략)")
    else:
        print(f"  ??[FAIL] ?�상: simple, needs_search=False")
        print(f"        ?�제: {result.get('query_complexity')}, needs_search={result.get('needs_search')}")

    return success


async def test_moderate_query():
    """중간 복잡??질문 ?�스??""
    print("\n" + "=" * 80)
    print("?�스??2: 중간 복잡??질문 (법령 조회)")
    print("=" * 80)

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
    print(f"  검??문서 ?? {len(result.get('retrieved_docs', []))}�?)
    print(f"  ?��? 길이: {len(result.get('answer', ''))}??)

    success = (
        result.get('query_complexity') == 'moderate' and
        result.get('needs_search') == True
    )

    if success:
        print("  ??[PASS] 중간 복잡??질문 ?�상 처리 (검???�행)")
    else:
        print(f"  ??[FAIL] ?�상: moderate, needs_search=True")
        print(f"        ?�제: {result.get('query_complexity')}, needs_search={result.get('needs_search')}")

    return success


async def test_complex_query():
    """복잡??질문 ?�스??""
    print("\n" + "=" * 80)
    print("?�스??3: 복잡??질문 (비교/분석)")
    print("=" * 80)

    config = LangGraphConfig.from_env()
    workflow_service = LangGraphWorkflowService(config)

    query = "민법�??�법??차이?�을 비교?�주?�요"
    print(f"\n질문: {query}")

    start = time.time()
    result = await workflow_service.process_query(query)
    elapsed = time.time() - start

    print(f"\n[결과]")
    print(f"  ?�간: {elapsed:.2f}�?)
    print(f"  복잡?? {result.get('query_complexity', 'unknown')}")
    print(f"  검???�요: {result.get('needs_search', True)}")
    print(f"  검??문서 ?? {len(result.get('retrieved_docs', []))}�?)
    print(f"  ?��? 길이: {len(result.get('answer', ''))}??)

    success = (
        result.get('query_complexity') in ['moderate', 'complex'] and
        result.get('needs_search') == True
    )

    if success:
        print(f"  ??[PASS] 복잡??질문 ?�상 처리 (검???�행, 복잡?? {result.get('query_complexity')})")
    else:
        print(f"  ??[FAIL] ?�상: moderate ?�는 complex, needs_search=True")
        print(f"        ?�제: {result.get('query_complexity')}, needs_search={result.get('needs_search')}")

    return success


async def main():
    """?�체 ?�스???�행"""
    print("\n" + "=" * 80)
    print("Adaptive RAG �?그래???�순??최적???�스??)
    print("=" * 80)

    results = []

    # ?�스??1: 간단??질문
    try:
        result1 = await test_simple_query()
        results.append(("간단??질문", result1))
    except Exception as e:
        print(f"  ??[ERROR] 간단??질문 ?�스???�패: {e}")
        results.append(("간단??질문", False))

    # ?�스??2: 중간 복잡??질문
    try:
        result2 = await test_moderate_query()
        results.append(("중간 복잡??질문", result2))
    except Exception as e:
        print(f"  ??[ERROR] 중간 복잡??질문 ?�스???�패: {e}")
        results.append(("중간 복잡??질문", False))

    # ?�스??3: 복잡??질문
    try:
        result3 = await test_complex_query()
        results.append(("복잡??질문", result3))
    except Exception as e:
        print(f"  ??[ERROR] 복잡??질문 ?�스???�패: {e}")
        results.append(("복잡??질문", False))

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
