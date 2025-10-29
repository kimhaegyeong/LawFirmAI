# -*- coding: utf-8 -*-
"""
Langfuse 통합 테스트
개선된 LangGraph 워크플로우를 Langfuse로 모니터링하는 테스트
"""

import asyncio
import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 테스트 환경 설정
os.environ["USE_LANGGRAPH"] = "true"
os.environ["LANGGRAPH_ENABLED"] = "true"
os.environ["LANGGRAPH_CHECKPOINT_STORAGE"] = "memory"  # 빠른 테스트를 위해 메모리 사용

from source.services.langgraph.workflow_service import LangGraphWorkflowService
from source.utils.langgraph_config import LangGraphConfig


class LangfuseIntegrationTest:
    """Langfuse 통합 테스트 클래스"""

    def __init__(self):
        self.config = LangGraphConfig()
        self.service = LangGraphWorkflowService(self.config)

    async def run_single_query(self, query: str, session_id: str, query_number: int):
        """단일 질의 실행"""
        print(f"\n{'='*80}")
        print(f"질의 #{query_number}: {query}")
        print(f"{'='*80}")

        try:
            # LangGraph 워크플로우 실행
            result = await self.service.process_query(
                query=query,
                session_id=session_id
            )

            # 결과 출력
            print(f"\n✓ 처리 완료")
            print(f"  답변 길이: {len(result.get('answer', ''))}자")
            print(f"  신뢰도: {result.get('confidence', 0):.2%}")
            print(f"  검색된 문서 수: {len(result.get('retrieved_docs', []))}개")
            print(f"  처리 단계 수: {len(result.get('processing_steps', []))}개")

            # 키워드 확장 정보
            if result.get('ai_keyword_expansion'):
                expansion = result['ai_keyword_expansion']
                print(f"  AI 키워드 확장: {expansion.get('method')}")
                print(f"    - 원본 키워드: {len(expansion.get('original_keywords', []))}개")
                print(f"    - 확장 키워드: {len(expansion.get('expanded_keywords', []))}개")
                print(f"    - 신뢰도: {expansion.get('confidence', 0):.2%}")

            # 처리 단계 출력
            steps = result.get('processing_steps', [])
            if steps:
                print(f"\n  처리 단계:")
                for i, step in enumerate(steps[-5:], 1):  # 마지막 5개만
                    print(f"    {i}. {step}")

            return result

        except Exception as e:
            print(f"\n✗ 처리 실패: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def run_all_tests(self):
        """전체 테스트 실행"""
        print("=" * 80)
        print("Langfuse 통합 테스트 시작")
        print("개선된 LangGraph 워크플로우 테스트")
        print("=" * 80)

        # 테스트 케이스 정의
        test_cases = [
            {
                "query": "손해배상 청구 방법을 알려주세요",
                "session_id": "test_session_001",
                "description": "기본 법률 조언 질문"
            },
            {
                "query": "계약 위반 시 법적 조치 방법",
                "session_id": "test_session_002",
                "description": "계약 관련 질문"
            },
            {
                "query": "민사소송에서 승소하기 위한 증거 수집 방법",
                "session_id": "test_session_003",
                "description": "민사소송 절차 질문"
            },
            {
                "query": "계약서에 따르면 배송 지연 시 어떻게 대응해야 하나요?",
                "session_id": "test_session_004",
                "description": "구체적 사안 질문"
            },
            {
                "query": "이전에 소개해주신 손해배상 청구에서 과실비율은 어떻게 결정되나요?",
                "session_id": "test_session_005",
                "description": "멀티턴 질문 (이전 질문 참조)"
            }
        ]

        results = []

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n\n{'#'*80}")
            print(f"테스트 케이스 #{i}/{len(test_cases)}")
            print(f"설명: {test_case['description']}")
            print(f"{'#'*80}")

            result = await self.run_single_query(
                query=test_case['query'],
                session_id=test_case['session_id'],
                query_number=i
            )

            if result:
                results.append({
                    'case': i,
                    'query': test_case['query'],
                    'result': result
                })

            # 각 테스트 사이에 짧은 대기
            await asyncio.sleep(1)

        # 통계 출력
        print("\n" + "=" * 80)
        print("테스트 결과 통계")
        print("=" * 80)

        total_queries = len(test_cases)
        successful_queries = len(results)

        print(f"\n총 질의 수: {total_queries}")
        print(f"성공한 질의: {successful_queries}")
        print(f"실패한 질의: {total_queries - successful_queries}")

        if results:
            # 평균 값 계산
            total_confidence = sum(r['result'].get('confidence', 0) for r in results)
            total_docs = sum(len(r['result'].get('retrieved_docs', [])) for r in results)
            total_steps = sum(len(r['result'].get('processing_steps', [])) for r in results)

            print(f"\n평균 신뢰도: {total_confidence/successful_queries:.2%}")
            print(f"평균 검색 문서 수: {total_docs/successful_queries:.1f}개")
            print(f"평균 처리 단계 수: {total_steps/successful_queries:.1f}개")

            # AI 키워드 확장 통계
            ai_expansions = [r for r in results if r['result'].get('ai_keyword_expansion')]
            if ai_expansions:
                print(f"\nAI 키워드 확장 실행: {len(ai_expansions)}회")
                gemini_count = len([r for r in ai_expansions
                                     if r['result']['ai_keyword_expansion'].get('method') == 'gemini_ai'])
                fallback_count = len([r for r in ai_expansions
                                       if r['result']['ai_keyword_expansion'].get('method') == 'fallback'])
                print(f"  - Gemini AI: {gemini_count}회")
                print(f"  - Fallback: {fallback_count}회")

        print("\n" + "=" * 80)
        print("Langfuse 모니터링 확인")
        print("=" * 80)
        print("\nLangfuse 대시보드에서 다음 정보를 확인할 수 있습니다:")
        print("  - 각 노드의 실행 시간")
        print("  - 노드 간 데이터 흐름")
        print("  - AI 키워드 확장 과정")
        print("  - 에러 및 경고 메시지")
        print("\nLangfuse URL: http://localhost:3000 (로컬 설정인 경우)")

        return results


async def main():
    """메인 함수"""
    test_runner = LangfuseIntegrationTest()
    results = await test_runner.run_all_tests()

    print("\n" + "=" * 80)
    print("테스트 완료!")
    print("=" * 80)

    return results


if __name__ == "__main__":
    # Langfuse 설정 확인
    try:
        from langfuse import Langfuse
        langfuse_client = Langfuse()
        print("✓ Langfuse 초기화 성공")
    except Exception as e:
        print(f"⚠ Langfuse 초기화 실패: {e}")
        print("  Langfuse 없이도 테스트는 진행됩니다.")

    # 테스트 실행
    asyncio.run(main())
