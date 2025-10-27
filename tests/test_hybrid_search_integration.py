# -*- coding: utf-8 -*-
"""
하이브리드 검색 통합 테스트
SemanticSearchEngine + LangGraph 워크플로우 통합 검증
"""

import asyncio
import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 테스트 환경 설정
os.environ["USE_LANGGRAPH"] = "true"
os.environ["LANGGRAPH_ENABLED"] = "true"

from source.services.langgraph.workflow_service import LangGraphWorkflowService
from source.utils.langgraph_config import LangGraphConfig


async def test_hybrid_search_integration():
    """하이브리드 검색 통합 테스트"""
    print("\n" + "="*80)
    print("하이브리드 검색 LangGraph 워크플로우 테스트")
    print("="*80 + "\n")

    try:
        # 설정 로드
        config = LangGraphConfig.from_env()
        print("✅ LangGraph 설정 로드 완료")

        # 워크플로우 서비스 초기화
        workflow_service = LangGraphWorkflowService(config)
        print("✅ LangGraphWorkflowService 초기화 완료")

        # 테스트 쿼리들 - 다양한 검색 시나리오
        test_queries = [
            {
                "query": "이혼 절차는 어떻게 진행하나요?",
                "description": "가족법 - 하이브리드 검색 테스트",
                "expected_sources": ["family_law", "civil_law"]
            },
            {
                "query": "계약서에서 손해배상 조항은 어떻게 작성해야 하나요?",
                "description": "민사법 - 의미적 검색 강점 테스트",
                "expected_sources": ["contract_review", "civil_law"]
            },
            {
                "query": "부당해고의 구제 절차는?",
                "description": "노동법 - 키워드 검색 테스트",
                "expected_sources": ["labor_law"]
            },
            {
                "query": "특허권 침해 시 어떻게 대응하나요?",
                "description": "지적재산권법 - 혼합 검색 테스트",
                "expected_sources": ["intellectual_property"]
            }
        ]

        results = []

        for i, test_case in enumerate(test_queries, 1):
            print(f"\n{'='*80}")
            print(f"테스트 {i}/{len(test_queries)}: {test_case['description']}")
            print(f"질문: {test_case['query']}")
            print(f"{'='*80}\n")

            try:
                # 워크플로우 실행
                result = await workflow_service.process_query(
                    query=test_case['query'],
                    enable_checkpoint=False
                )

                # 결과 분석
                print(f"✅ 처리 완료: {result.get('answer', '')[:100]}...")

                # 검색 메타데이터 확인
                metadata = result.get('metadata', {})
                search_metadata = result.get('search_metadata', {})

                print(f"\n📊 검색 메타데이터:")
                print(f"  - 의미적 검색 결과: {search_metadata.get('semantic_results_count', 0)}개")
                print(f"  - 키워드 검색 결과: {search_metadata.get('keyword_results_count', 0)}개")
                print(f"  - 최종 문서 수: {search_metadata.get('final_count', 0)}개")
                print(f"  - 검색 모드: {search_metadata.get('search_mode', 'N/A')}")
                print(f"  - 검색 시간: {search_metadata.get('search_time', 0):.3f}초")

                # 처리 단계 확인
                steps = result.get('processing_steps', [])
                print(f"\n🔍 처리 단계:")
                for step in steps:
                    if '검색' in step or '하이브리드' in step:
                        print(f"  • {step}")

                # 소스 확인
                sources = result.get('sources', [])
                print(f"\n📚 검색된 소스 ({len(sources)}개):")
                for j, source in enumerate(sources[:5], 1):
                    print(f"  {j}. {source}")

                results.append({
                    'test_case': test_case,
                    'success': True,
                    'result': result,
                    'search_metadata': search_metadata
                })

            except Exception as e:
                print(f"❌ 테스트 실패: {str(e)}")
                import traceback
                traceback.print_exc()
                results.append({
                    'test_case': test_case,
                    'success': False,
                    'error': str(e)
                })

        # 종합 결과 출력
        print(f"\n{'='*80}")
        print("종합 테스트 결과")
        print(f"{'='*80}\n")

        success_count = sum(1 for r in results if r.get('success'))
        total_count = len(results)

        print(f"✅ 성공: {success_count}/{total_count}")
        print(f"❌ 실패: {total_count - success_count}/{total_count}\n")

        # 하이브리드 검색 분석
        hybrid_search_used = sum(1 for r in results
                                 if r.get('success') and
                                 r.get('search_metadata', {}).get('search_mode') == 'hybrid')

        print(f"🔍 하이브리드 검색 모드 사용: {hybrid_search_used}/{success_count}회")

        # 검색 성능 분석
        if success_count > 0:
            avg_semantic = sum(r.get('search_metadata', {}).get('semantic_results_count', 0)
                              for r in results if r.get('success')) / success_count
            avg_keyword = sum(r.get('search_metadata', {}).get('keyword_results_count', 0)
                             for r in results if r.get('success')) / success_count
            avg_final = sum(r.get('search_metadata', {}).get('final_count', 0)
                           for r in results if r.get('success')) / success_count

            print(f"\n📈 평균 검색 성능:")
            print(f"  - 의미적 검색 결과: {avg_semantic:.1f}개")
            print(f"  - 키워드 검색 결과: {avg_keyword:.1f}개")
            print(f"  - 최종 선택 문서: {avg_final:.1f}개")

        # 상세 결과 출력
        print(f"\n{'='*80}")
        print("상세 결과")
        print(f"{'='*80}\n")

        for i, result in enumerate(results, 1):
            if result.get('success'):
                test_case = result['test_case']
                metadata = result['search_metadata']
                print(f"\n{i}. {test_case['description']}")
                print(f"   검색: 의미적 {metadata.get('semantic_results_count', 0)}개 + "
                      f"키워드 {metadata.get('keyword_results_count', 0)}개 → "
                      f"최종 {metadata.get('final_count', 0)}개")
            else:
                print(f"\n{i}. ❌ 실패: {result['test_case']['description']}")
                print(f"   오류: {result.get('error', 'Unknown error')}")

        print(f"\n{'='*80}")
        print("✅ 하이브리드 검색 통합 테스트 완료")
        print(f"{'='*80}\n")

        return results

    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


async def test_semantic_search_only():
    """의미적 검색 엔진 단독 테스트"""
    print("\n" + "="*80)
    print("SemanticSearchEngine 단독 테스트")
    print("="*80 + "\n")

    try:
        from source.services.semantic_search_engine import SemanticSearchEngine

        # SemanticSearchEngine 초기화
        print("SemanticSearchEngine 초기화 중...")
        search_engine = SemanticSearchEngine()
        print("✅ SemanticSearchEngine 초기화 완료")

        # 테스트 쿼리
        test_queries = [
            "이혼 절차",
            "계약서 손해배상",
            "부당해고 구제",
            "특허권 침해"
        ]

        for query in test_queries:
            print(f"\n검색 쿼리: '{query}'")
            print("-" * 80)

            results = search_engine.search(query, k=5)

            if results:
                print(f"검색 결과: {len(results)}개\n")
                for i, result in enumerate(results[:3], 1):
                    print(f"{i}. [Score: {result.get('score', 0):.3f}]")
                    print(f"   텍스트: {result.get('text', '')[:100]}...")
                    print(f"   소스: {result.get('source', 'Unknown')}")
                    print()
            else:
                print("검색 결과 없음\n")

        print("✅ SemanticSearchEngine 단독 테스트 완료")

    except Exception as e:
        print(f"❌ SemanticSearchEngine 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("하이브리드 검색 통합 테스트 실행\n")

    # 1. SemanticSearchEngine 단독 테스트
    asyncio.run(test_semantic_search_only())

    # 2. 전체 워크플로우 통합 테스트
    asyncio.run(test_hybrid_search_integration())
