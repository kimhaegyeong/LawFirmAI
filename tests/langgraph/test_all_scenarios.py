# -*- coding: utf-8 -*-
"""
Adaptive RAG 및 그래프 단순화 최적화 전체 시나리오 테스트
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


async def test_simple_query():
    """간단한 질문 테스트 (인사말)"""
    print("=" * 80)
    print("테스트 1: 간단한 질문 (인사말)")
    print("=" * 80)

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
    print(f"  검색 문서 수: {len(result.get('retrieved_docs', []))}개")
    print(f"  답변 길이: {len(result.get('answer', ''))}자")

    success = (
        result.get('query_complexity') == 'simple' and
        result.get('needs_search') == False and
        len(result.get('retrieved_docs', [])) == 0
    )

    if success:
        print("  ✅ [PASS] 간단한 질문 정상 처리 (검색 생략)")
    else:
        print(f"  ❌ [FAIL] 예상: simple, needs_search=False")
        print(f"        실제: {result.get('query_complexity')}, needs_search={result.get('needs_search')}")

    return success


async def test_moderate_query():
    """중간 복잡도 질문 테스트"""
    print("\n" + "=" * 80)
    print("테스트 2: 중간 복잡도 질문 (법령 조회)")
    print("=" * 80)

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
    print(f"  검색 문서 수: {len(result.get('retrieved_docs', []))}개")
    print(f"  답변 길이: {len(result.get('answer', ''))}자")

    success = (
        result.get('query_complexity') == 'moderate' and
        result.get('needs_search') == True
    )

    if success:
        print("  ✅ [PASS] 중간 복잡도 질문 정상 처리 (검색 실행)")
    else:
        print(f"  ❌ [FAIL] 예상: moderate, needs_search=True")
        print(f"        실제: {result.get('query_complexity')}, needs_search={result.get('needs_search')}")

    return success


async def test_complex_query():
    """복잡한 질문 테스트"""
    print("\n" + "=" * 80)
    print("테스트 3: 복잡한 질문 (비교/분석)")
    print("=" * 80)

    config = LangGraphConfig.from_env()
    workflow_service = LangGraphWorkflowService(config)

    query = "민법과 상법의 차이점을 비교해주세요"
    print(f"\n질문: {query}")

    start = time.time()
    result = await workflow_service.process_query(query)
    elapsed = time.time() - start

    print(f"\n[결과]")
    print(f"  시간: {elapsed:.2f}초")
    print(f"  복잡도: {result.get('query_complexity', 'unknown')}")
    print(f"  검색 필요: {result.get('needs_search', True)}")
    print(f"  검색 문서 수: {len(result.get('retrieved_docs', []))}개")
    print(f"  답변 길이: {len(result.get('answer', ''))}자")

    success = (
        result.get('query_complexity') in ['moderate', 'complex'] and
        result.get('needs_search') == True
    )

    if success:
        print(f"  ✅ [PASS] 복잡한 질문 정상 처리 (검색 실행, 복잡도: {result.get('query_complexity')})")
    else:
        print(f"  ❌ [FAIL] 예상: moderate 또는 complex, needs_search=True")
        print(f"        실제: {result.get('query_complexity')}, needs_search={result.get('needs_search')}")

    return success


async def main():
    """전체 테스트 실행"""
    print("\n" + "=" * 80)
    print("Adaptive RAG 및 그래프 단순화 최적화 테스트")
    print("=" * 80)

    results = []

    # 테스트 1: 간단한 질문
    try:
        result1 = await test_simple_query()
        results.append(("간단한 질문", result1))
    except Exception as e:
        print(f"  ❌ [ERROR] 간단한 질문 테스트 실패: {e}")
        results.append(("간단한 질문", False))

    # 테스트 2: 중간 복잡도 질문
    try:
        result2 = await test_moderate_query()
        results.append(("중간 복잡도 질문", result2))
    except Exception as e:
        print(f"  ❌ [ERROR] 중간 복잡도 질문 테스트 실패: {e}")
        results.append(("중간 복잡도 질문", False))

    # 테스트 3: 복잡한 질문
    try:
        result3 = await test_complex_query()
        results.append(("복잡한 질문", result3))
    except Exception as e:
        print(f"  ❌ [ERROR] 복잡한 질문 테스트 실패: {e}")
        results.append(("복잡한 질문", False))

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
