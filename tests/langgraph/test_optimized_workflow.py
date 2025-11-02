# -*- coding: utf-8 -*-
"""
최적화된 워크플로우 테스트 (Adaptive RAG + 그래프 단순화 + 병렬 실행)
"""
import asyncio
import logging
import sys
import time
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from core.agents.workflow_service import LangGraphWorkflowService
from infrastructure.utils.langgraph_config import LangGraphConfig


class WorkflowPerformanceTester:
    """워크플로우 성능 테스터"""

    def __init__(self):
        """테스터 초기화"""
        config = LangGraphConfig.from_env()
        self.workflow_service = LangGraphWorkflowService(config)
        self.results = []

    async def test_simple_query(self):
        """간단한 질문 테스트 (Adaptive RAG - 검색 스킵)"""
        print("\n" + "=" * 80)
        print("테스트 1: 간단한 질문 (Adaptive RAG - 검색 스킵)")
        print("=" * 80)

        test_queries = [
            "안녕하세요",
            "고마워요",
            "계약이란 무엇인가요?",
            "법률 용어 '소송'의 의미를 알려주세요"
        ]

        for query in test_queries:
            print(f"\n📝 질문: {query}")
            start_time = time.time()

            try:
                result = await self.workflow_service.process_query(query)
                elapsed_time = time.time() - start_time

                # 결과 분석
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                processing_steps = result.get("processing_steps", [])
                query_complexity = result.get("query_complexity", "unknown")
                needs_search = result.get("needs_search", True)

                print(f"  ⏱️  응답 시간: {elapsed_time:.2f}초")
                print(f"  📊 복잡도: {query_complexity}")
                print(f"  🔍 검색 필요: {needs_search}")
                print(f"  📄 답변 길이: {len(answer)}자")
                print(f"  📚 소스 수: {len(sources)}개")
                print(f"  🔄 처리 단계 수: {len(processing_steps)}개")

                # 검색 스킵 확인
                if query_complexity == "simple":
                    if needs_search == False:
                        print("  ✅ 검색 스킵 확인됨 (Adaptive RAG 작동)")
                    else:
                        print("  ⚠️  검색 스킵 예상되었으나 실행됨")
                else:
                    print(f"  ⚠️  간단한 질문으로 분류되지 않음 (복잡도: {query_complexity})")

                # 답변 내용 미리보기
                if answer:
                    preview = answer[:100] + "..." if len(answer) > 100 else answer
                    print(f"  💬 답변 미리보기: {preview}")

                self.results.append({
                    "query": query,
                    "type": "simple",
                    "elapsed_time": elapsed_time,
                    "complexity": query_complexity,
                    "needs_search": needs_search,
                    "sources_count": len(sources),
                    "steps_count": len(processing_steps),
                    "answer_length": len(answer)
                })

            except Exception as e:
                print(f"  ❌ 오류 발생: {e}")
                logger.exception(f"테스트 실패: {query}")

    async def test_complexity_classification(self):
        """복잡도 분류 테스트"""
        print("\n" + "=" * 80)
        print("테스트: 복잡도 분류 확인")
        print("=" * 80)

        test_cases = [
            ("안녕하세요", "simple"),
            ("계약이란 무엇인가요?", "simple"),
            ("민법 제111조의 내용을 알려주세요", "moderate"),
        ]

        passed = 0
        failed = 0

        for query, expected_complexity in test_cases:
            print(f"\n📝 질문: {query}")
            print(f"  예상 복잡도: {expected_complexity}")

            try:
                result = await self.workflow_service.process_query(query)
                actual_complexity = result.get("query_complexity", "unknown")
                needs_search = result.get("needs_search", True)

                print(f"  실제 복잡도: {actual_complexity}")
                print(f"  검색 필요: {needs_search}")

                if actual_complexity == expected_complexity:
                    print("  ✅ 올바르게 분류됨")
                    passed += 1
                else:
                    print(f"  ⚠️  복잡도 불일치 (예상: {expected_complexity}, 실제: {actual_complexity})")
                    failed += 1

                self.results.append({
                    "query": query,
                    "type": "classification",
                    "expected": expected_complexity,
                    "actual": actual_complexity,
                    "passed": actual_complexity == expected_complexity
                })

            except Exception as e:
                print(f"  ❌ 오류 발생: {e}")
                logger.exception(f"테스트 실패: {query}")
                failed += 1

        print(f"\n📊 결과: {passed}개 통과, {failed}개 실패")
        return failed == 0

    async def test_moderate_query(self):
        """중간 복잡도 질문 테스트 (검색 수행)"""
        print("\n" + "=" * 80)
        print("테스트 2: 중간 복잡도 질문 (검색 수행)")
        print("=" * 80)

        test_queries = [
            "민법 제111조의 내용을 알려주세요",
            "계약 해지 조건은 무엇인가요?",
            "이혼 소송 절차를 알려주세요"
        ]

        for query in test_queries:
            print(f"\n📝 질문: {query}")
            start_time = time.time()

            try:
                result = await self.workflow_service.process_query(query)
                elapsed_time = time.time() - start_time

                # 결과 분석
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                processing_steps = result.get("processing_steps", [])
                query_complexity = result.get("query_complexity", "unknown")
                needs_search = result.get("needs_search", True)
                retrieved_docs = result.get("retrieved_docs", [])

                print(f"  ⏱️  응답 시간: {elapsed_time:.2f}초")
                print(f"  📊 복잡도: {query_complexity}")
                print(f"  🔍 검색 필요: {needs_search}")
                print(f"  📄 답변 길이: {len(answer)}자")
                print(f"  📚 소스 수: {len(sources)}개")
                print(f"  📖 검색 문서 수: {len(retrieved_docs)}개")
                print(f"  🔄 처리 단계 수: {len(processing_steps)}개")

                # 검색 수행 확인
                if needs_search == True and len(retrieved_docs) > 0:
                    print("  ✅ 검색 수행 확인됨")
                elif needs_search == True and len(retrieved_docs) == 0:
                    print("  ⚠️  검색 필요했으나 문서를 찾지 못함")

                self.results.append({
                    "query": query,
                    "type": "moderate",
                    "elapsed_time": elapsed_time,
                    "complexity": query_complexity,
                    "needs_search": needs_search,
                    "sources_count": len(sources),
                    "retrieved_docs_count": len(retrieved_docs),
                    "steps_count": len(processing_steps),
                    "answer_length": len(answer)
                })

            except Exception as e:
                print(f"  ❌ 오류 발생: {e}")
                logger.exception(f"테스트 실패: {query}")

    async def test_complex_query(self):
        """복잡한 질문 테스트 (전체 플로우)"""
        print("\n" + "=" * 80)
        print("테스트 3: 복잡한 질문 (전체 플로우)")
        print("=" * 80)

        test_queries = [
            "이혼과 재혼의 차이점과 각각의 법적 절차를 비교해주세요",
            "계약 해지와 해제의 차이는 무엇인가요?",
            "최근 판례를 바탕으로 손해배상 청구 방법을 설명해주세요"
        ]

        for query in test_queries:
            print(f"\n📝 질문: {query}")
            start_time = time.time()

            try:
                result = await self.workflow_service.process_query(query)
                elapsed_time = time.time() - start_time

                # 결과 분석
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                processing_steps = result.get("processing_steps", [])
                query_complexity = result.get("query_complexity", "unknown")
                needs_search = result.get("needs_search", True)
                retrieved_docs = result.get("retrieved_docs", [])

                print(f"  ⏱️  응답 시간: {elapsed_time:.2f}초")
                print(f"  📊 복잡도: {query_complexity}")
                print(f"  🔍 검색 필요: {needs_search}")
                print(f"  📄 답변 길이: {len(answer)}자")
                print(f"  📚 소스 수: {len(sources)}개")
                print(f"  📖 검색 문서 수: {len(retrieved_docs)}개")
                print(f"  🔄 처리 단계 수: {len(processing_steps)}개")

                # 처리 단계 상세
                if processing_steps:
                    print(f"  📋 처리 단계:")
                    for idx, step in enumerate(processing_steps[:10], 1):  # 최대 10개만
                        print(f"     {idx}. {step}")

                self.results.append({
                    "query": query,
                    "type": "complex",
                    "elapsed_time": elapsed_time,
                    "complexity": query_complexity,
                    "needs_search": needs_search,
                    "sources_count": len(sources),
                    "retrieved_docs_count": len(retrieved_docs),
                    "steps_count": len(processing_steps),
                    "answer_length": len(answer)
                })

            except Exception as e:
                print(f"  ❌ 오류 발생: {e}")
                logger.exception(f"테스트 실패: {query}")

    def print_summary(self):
        """테스트 결과 요약 출력"""
        print("\n" + "=" * 80)
        print("📊 테스트 결과 요약")
        print("=" * 80)

        if not self.results:
            print("❌ 테스트 결과가 없습니다.")
            return

        # 유형별 통계
        simple_results = [r for r in self.results if r["type"] == "simple"]
        moderate_results = [r for r in self.results if r["type"] == "moderate"]
        complex_results = [r for r in self.results if r["type"] == "complex"]

        print(f"\n📈 전체 통계:")
        print(f"  - 총 테스트: {len(self.results)}개")
        print(f"  - 간단한 질문: {len(simple_results)}개")
        print(f"  - 중간 질문: {len(moderate_results)}개")
        print(f"  - 복잡한 질문: {len(complex_results)}개")

        # 성능 통계
        if simple_results:
            avg_time = sum(r["elapsed_time"] for r in simple_results) / len(simple_results)
            min_time = min(r["elapsed_time"] for r in simple_results)
            max_time = max(r["elapsed_time"] for r in simple_results)
            print(f"\n⚡ 간단한 질문 성능:")
            print(f"  - 평균 응답 시간: {avg_time:.2f}초")
            print(f"  - 최소 시간: {min_time:.2f}초")
            print(f"  - 최대 시간: {max_time:.2f}초")
            search_skipped = sum(1 for r in simple_results if r.get("needs_search") == False)
            print(f"  - 검색 스킵률: {search_skipped}/{len(simple_results)} ({search_skipped/len(simple_results)*100:.1f}%)")

        if moderate_results:
            avg_time = sum(r["elapsed_time"] for r in moderate_results) / len(moderate_results)
            min_time = min(r["elapsed_time"] for r in moderate_results)
            max_time = max(r["elapsed_time"] for r in moderate_results)
            print(f"\n⚡ 중간 질문 성능:")
            print(f"  - 평균 응답 시간: {avg_time:.2f}초")
            print(f"  - 최소 시간: {min_time:.2f}초")
            print(f"  - 최대 시간: {max_time:.2f}초")
            avg_docs = sum(r.get("retrieved_docs_count", 0) for r in moderate_results) / len(moderate_results)
            print(f"  - 평균 검색 문서 수: {avg_docs:.1f}개")

        if complex_results:
            avg_time = sum(r["elapsed_time"] for r in complex_results) / len(complex_results)
            min_time = min(r["elapsed_time"] for r in complex_results)
            max_time = max(r["elapsed_time"] for r in complex_results)
            print(f"\n⚡ 복잡한 질문 성능:")
            print(f"  - 평균 응답 시간: {avg_time:.2f}초")
            print(f"  - 최소 시간: {min_time:.2f}초")
            print(f"  - 최대 시간: {max_time:.2f}초")
            avg_docs = sum(r.get("retrieved_docs_count", 0) for r in complex_results) / len(complex_results)
            print(f"  - 평균 검색 문서 수: {avg_docs:.1f}개")
            avg_steps = sum(r.get("steps_count", 0) for r in complex_results) / len(complex_results)
            print(f"  - 평균 처리 단계 수: {avg_steps:.1f}개")

        # 전체 평균
        if self.results:
            avg_time_all = sum(r["elapsed_time"] for r in self.results if "elapsed_time" in r) / len([r for r in self.results if "elapsed_time" in r])
            print(f"\n📊 전체 평균 응답 시간: {avg_time_all:.2f}초")

        print("\n" + "=" * 80)


async def main():
    """메인 테스트 실행"""
    print("=" * 80)
    print("🚀 최적화된 워크플로우 테스트 시작")
    print("=" * 80)
    print("\n테스트 항목:")
    print("  1. 간단한 질문 (Adaptive RAG - 검색 스킵)")
    print("  2. 중간 복잡도 질문 (검색 수행)")
    print("  3. 복잡한 질문 (전체 플로우)")
    print("  4. 복잡도 분류 테스트")
    print("  5. 성능 측정 및 요약")

    tester = WorkflowPerformanceTester()

    try:
        # 테스트 실행
        await tester.test_simple_query()
        await tester.test_moderate_query()
        await tester.test_complex_query()
        await tester.test_complexity_classification()

        # 결과 요약
        tester.print_summary()

        print("\n✅ 모든 테스트 완료!")

    except KeyboardInterrupt:
        print("\n⚠️  테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        logger.exception("테스트 실행 중 오류")


if __name__ == "__main__":
    asyncio.run(main())
