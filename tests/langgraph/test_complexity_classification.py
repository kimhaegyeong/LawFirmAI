# -*- coding: utf-8 -*-
"""
LLM 기반 복잡??분류 ?�스??
"""

import sys
import time
from pathlib import Path

# ?�로?�트 루트 경로 추�?
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from source.agents.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow, QueryComplexity
from infrastructure.utils.langgraph_config import LangGraphConfig


def test_llm_complexity_classification():
    """LLM 기반 복잡??분류 ?�스??""
    print("=" * 80)
    print("LLM 기반 복잡??분류 ?�스??)
    print("=" * 80)

    # ?�정 로드 (LLM ?�성??
    config = LangGraphConfig.from_env()
    config.use_llm_for_complexity = True

    workflow = EnhancedLegalQuestionWorkflow(config)

    test_cases = [
        {
            "query": "?�녕?�세??,
            "query_type": "general_question",
            "expected_complexity": QueryComplexity.SIMPLE,
            "expected_needs_search": False
        },
        {
            "query": "민법 ??23조의 ?�용???�려주세??,
            "query_type": "law_inquiry",
            "expected_complexity": QueryComplexity.MODERATE,
            "expected_needs_search": True
        },
        {
            "query": "계약 ?��??� ?�제??차이??무엇?��???",
            "query_type": "legal_advice",
            "expected_complexity": QueryComplexity.COMPLEX,
            "expected_needs_search": True
        },
        {
            "query": "?�혼 ?�차�??�려주세??,
            "query_type": "procedure_guide",
            "expected_complexity": QueryComplexity.COMPLEX,
            "expected_needs_search": True
        }
    ]

    results = []

    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        query_type = test_case["query_type"]
        expected_complexity = test_case["expected_complexity"]
        expected_needs_search = test_case["expected_needs_search"]

        print(f"\n[?�스??{i}] {query}")
        print(f"  ?�상: {expected_complexity.value}, needs_search={expected_needs_search}")

        try:
            complexity, needs_search = workflow._classify_complexity_with_llm(query, query_type)

            print(f"  ?�제: {complexity.value}, needs_search={needs_search}")

            # 복잡?��? ?�상 범위 ?�인지 ?�인 (?�격??매칭 ?�??범위 체크)
            if complexity == expected_complexity:
                status = "??PASS"
            elif complexity == QueryComplexity.MODERATE and expected_complexity == QueryComplexity.SIMPLE:
                # LLM??보수?�으�??�단??경우 ?�용
                status = "?�️  MODERATE (보수???�단, ?�용)"
            else:
                status = "??FAIL"

            # needs_search???�확???�치?�야 ??
            if needs_search != expected_needs_search:
                status = "??FAIL (needs_search 불일�?"

            print(f"  결과: {status}")

            results.append({
                "query": query,
                "expected": expected_complexity.value,
                "actual": complexity.value,
                "expected_search": expected_needs_search,
                "actual_search": needs_search,
                "status": "PASS" if status.startswith("??) else "FAIL"
            })

        except Exception as e:
            print(f"  ??ERROR: {e}")
            results.append({
                "query": query,
                "status": "ERROR",
                "error": str(e)
            })

    # ?�약
    print("\n" + "=" * 80)
    print("?�스???�약")
    print("=" * 80)

    passed = sum(1 for r in results if r.get("status") == "PASS")
    failed = sum(1 for r in results if r.get("status") == "FAIL")
    errors = sum(1 for r in results if r.get("status") == "ERROR")

    print(f"?�과: {passed}�?)
    print(f"?�패: {failed}�?)
    print(f"?�류: {errors}�?)

    return passed == len(test_cases)


def test_fallback_classification():
    """?�백 복잡??분류 ?�스??""
    print("\n" + "=" * 80)
    print("?�백 복잡??분류 ?�스??)
    print("=" * 80)

    config = LangGraphConfig.from_env()
    workflow = EnhancedLegalQuestionWorkflow(config)

    test_cases = [
        ("?�녕?�세??, QueryComplexity.SIMPLE, False),
        ("민법 ??23�?, QueryComplexity.MODERATE, True),
        ("계약 ?��??� ?�제??차이", QueryComplexity.COMPLEX, True),
        ("", QueryComplexity.MODERATE, True),
    ]

    all_passed = True

    for query, expected_complexity, expected_needs_search in test_cases:
        print(f"\n질문: '{query}'")
        try:
            complexity, needs_search = workflow._fallback_complexity_classification(query)

            passed = (complexity == expected_complexity and needs_search == expected_needs_search)
            status = "??PASS" if passed else "??FAIL"

            print(f"  ?�상: {expected_complexity.value}, needs_search={expected_needs_search}")
            print(f"  ?�제: {complexity.value}, needs_search={needs_search}")
            print(f"  결과: {status}")

            if not passed:
                all_passed = False
        except Exception as e:
            print(f"  ??ERROR: {e}")
            all_passed = False

    return all_passed


def test_caching():
    """캐싱 ?�작 ?�스??""
    print("\n" + "=" * 80)
    print("캐싱 ?�작 ?�스??)
    print("=" * 80)

    config = LangGraphConfig.from_env()
    config.use_llm_for_complexity = True

    workflow = EnhancedLegalQuestionWorkflow(config)

    query = "민법 ??23조의 ?�용???�려주세??
    query_type = "law_inquiry"

    # 캐시 초기??
    workflow._complexity_cache.clear()

    print(f"\n질문: {query}")

    # �?번째 ?�출 (캐시 미스)
    print("\n[�?번째 ?�출]")
    start1 = time.time()
    complexity1, needs_search1 = workflow._classify_complexity_with_llm(query, query_type)
    elapsed1 = time.time() - start1
    print(f"  결과: {complexity1.value}, needs_search={needs_search1}")
    print(f"  ?�간: {elapsed1:.3f}�?)

    # ??번째 ?�출 (캐시 ?�트)
    print("\n[??번째 ?�출]")
    start2 = time.time()
    complexity2, needs_search2 = workflow._classify_complexity_with_llm(query, query_type)
    elapsed2 = time.time() - start2
    print(f"  결과: {complexity2.value}, needs_search={needs_search2}")
    print(f"  ?�간: {elapsed2:.3f}�?)

    # 결과 비교
    if complexity1 == complexity2 and needs_search1 == needs_search2:
        print("  ??결과 ?�치")
        if elapsed2 < elapsed1 * 0.1:  # 캐시???�씬 빠름
            print("  ??캐시 ?�트 ?�인 (??번째 ?�출???�씬 빠름)")
            return True
        else:
            print("  ?�️  캐시가 ?�상보다 ?�림")
            return False
    else:
        print("  ??결과 불일�?)
        return False


def test_parse_complexity_response():
    """JSON ?�싱 ?�스??""
    print("\n" + "=" * 80)
    print("JSON ?�싱 ?�스??)
    print("=" * 80)

    config = LangGraphConfig.from_env()
    workflow = EnhancedLegalQuestionWorkflow(config)

    test_cases = [
        (
            '{"complexity": "simple", "needs_search": false, "reasoning": "?�사�?}',
            {"complexity": "simple", "needs_search": False}
        ),
        (
            '{"complexity": "moderate", "needs_search": true, "reasoning": "법령 조회"}',
            {"complexity": "moderate", "needs_search": True}
        ),
        (
            'Some text before {"complexity": "complex", "needs_search": true} some text after',
            {"complexity": "complex", "needs_search": True}
        ),
        (
            'Invalid response text with "complexity": "simple"',
            {"complexity": "simple", "needs_search": False}
        ),
        (
            'No valid JSON here',
            {"complexity": "moderate", "needs_search": True}
        )
    ]

    all_passed = True

    for i, (response, expected) in enumerate(test_cases, 1):
        print(f"\n[?�스??{i}]")
        print(f"  ?�력: {response[:50]}...")

        result = workflow._parse_complexity_response(response)

        if result:
            complexity_match = result.get("complexity") == expected.get("complexity")
            search_match = result.get("needs_search") == expected.get("needs_search")

            if complexity_match and search_match:
                print(f"  ??PASS: {result.get('complexity')}, needs_search={result.get('needs_search')}")
            else:
                print(f"  ??FAIL: ?�상={expected}, ?�제={result}")
                all_passed = False
        else:
            print(f"  ??FAIL: ?�싱 ?�패")
            all_passed = False

    return all_passed


def main():
    """메인 ?�스???�행"""
    import time

    print("=" * 80)
    print("LLM 기반 복잡??분류 종합 ?�스??)
    print("=" * 80)

    results = {}

    # ?�스??1: LLM 기반 분류
    try:
        results["llm_classification"] = test_llm_complexity_classification()
    except Exception as e:
        print(f"\n??LLM 분류 ?�스???�패: {e}")
        results["llm_classification"] = False

    # ?�스??2: ?�백 분류
    try:
        results["fallback"] = test_fallback_classification()
    except Exception as e:
        print(f"\n???�백 ?�스???�패: {e}")
        results["fallback"] = False

    # ?�스??3: JSON ?�싱
    try:
        results["parsing"] = test_parse_complexity_response()
    except Exception as e:
        print(f"\n???�싱 ?�스???�패: {e}")
        results["parsing"] = False

    # ?�스??4: 캐싱
    try:
        results["caching"] = test_caching()
    except Exception as e:
        print(f"\n??캐싱 ?�스???�패: {e}")
        results["caching"] = False

    # 최종 ?�약
    print("\n" + "=" * 80)
    print("최종 결과")
    print("=" * 80)

    for test_name, result in results.items():
        status = "??PASS" if result else "??FAIL"
        print(f"  {test_name}: {status}")

    total_passed = sum(1 for r in results.values() if r)
    total_tests = len(results)

    print(f"\n?�체: {total_passed}/{total_tests} ?�스???�과")

    if total_passed == total_tests:
        print("\n??모든 ?�스???�과!")
        return 0
    else:
        print(f"\n?�️ {total_tests - total_passed}�??�스???�패")
        return 1


if __name__ == "__main__":
    import time
    exit_code = main()
    sys.exit(exit_code)
