# -*- coding: utf-8 -*-
"""
용어 통합 기능 포함 LangGraph 워크플로우 테스트
TermIntegrationSystem 통합 후 워크플로우 검증
"""

import asyncio
import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# lawfirm_langgraph 경로 추가
lawfirm_langgraph_path = project_root / "lawfirm_langgraph"
sys.path.insert(0, str(lawfirm_langgraph_path))

# 테스트 환경 설정
os.environ["USE_LANGGRAPH"] = "true"
os.environ["LANGGRAPH_ENABLED"] = "true"

from langgraph_core.services.workflow_service import LangGraphWorkflowService
from langgraph_core.config.langgraph_config import LangGraphConfig


async def test_term_integration_workflow():
    """용어 통합 워크플로우 테스트"""
    print("\n" + "="*80)
    print("용어 통합 LangGraph 워크플로우 테스트")
    print("="*80 + "\n")

    try:
        # 설정 로드
        config = LangGraphConfig.from_env()
        print("✅ LangGraph 설정 로드 완료")

        # 워크플로우 서비스 초기화
        workflow_service = LangGraphWorkflowService(config)
        print("✅ LangGraphWorkflowService 초기화 완료")

        # 테스트 쿼리들 - 법률 용어가 많은 질문들
        test_queries = [
            {
                "query": "이혼 절차와 양육권 분쟁에 대해 알려주세요",
                "description": "가족법 - 용어 통합 테스트"
            },
            {
                "query": "계약서 작성 시 손해배상 조항과 위약금 조항의 차이점은?",
                "description": "민사법 - 중복 용어 정리 테스트"
            },
            {
                "query": "해고 제한 조건과 부당해고 방지에 대해 설명해주세요",
                "description": "노동법 - 유사 용어 그룹화 테스트"
            },
        ]

        results = []

        for i, test_case in enumerate(test_queries, 1):
            query = test_case["query"]
            description = test_case["description"]

            print(f"\n{'='*80}")
            print(f"테스트 {i}: {description}")
            print(f"{'='*80}")
            print(f"질문: {query}\n")

            try:
                # 워크플로우 실행
                result = await workflow_service.process_query(query)

                # 결과 검증
                assert "answer" in result, "답변이 없습니다"
                assert len(result["answer"]) > 0, "답변이 비어있습니다"
                assert "confidence" in result, "신뢰도 정보가 없습니다"

                # 결과 출력
                answer_length = len(result["answer"])
                confidence = result.get("confidence", 0.0)
                processing_time = result.get("processing_time", 0.0)
                sources_count = len(result.get("sources", []))

                print(f"📊 처리 결과:")
                print(f"   - 답변 길이: {answer_length}자")
                print(f"   - 신뢰도: {confidence:.2f}")
                print(f"   - 처리 시간: {processing_time:.2f}초")
                print(f"   - 출처: {sources_count}개")

                # 용어 통합 결과 확인
                metadata = result.get("metadata", {})
                extracted_terms = metadata.get("extracted_terms", [])
                unique_terms = metadata.get("unique_terms", 0)
                total_terms = metadata.get("total_terms_extracted", 0)

                print(f"\n📝 용어 통합 결과:")
                print(f"   - 총 추출된 용어: {total_terms}개")
                print(f"   - 고유 용어 (통합 후): {unique_terms}개")
                print(f"   - 중복 제거율: {(1 - unique_terms/total_terms)*100:.1f}%" if total_terms > 0 else "   - 중복 제거율: N/A")

                if extracted_terms:
                    print(f"\n   🔑 주요 법률 용어 (최대 10개):")
                    for j, term in enumerate(extracted_terms[:10], 1):
                        print(f"   {j}. {term}")
                    if len(extracted_terms) > 10:
                        print(f"   ... 외 {len(extracted_terms) - 10}개")

                # 처리 단계 확인
                processing_steps = result.get("processing_steps", [])
                if processing_steps:
                    print(f"\n   🔄 처리 단계:")
                    for step in processing_steps:
                        if "용어" in step:
                            print(f"      ✅ {step}")
                        else:
                            print(f"      • {step}")

                # 오류 확인
                errors = result.get("errors", [])
                if errors:
                    print(f"\n   ⚠️ 오류 발생:")
                    for error in errors:
                        print(f"      - {error}")

                results.append({
                    "success": True,
                    "description": description,
                    "query": query,
                    "processing_time": processing_time,
                    "confidence": confidence,
                    "extracted_terms_count": unique_terms,
                    "total_terms": total_terms
                })

                print(f"\n   ✅ 테스트 {i} 성공")

            except Exception as e:
                print(f"\n   ❌ 테스트 {i} 실패: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "success": False,
                    "description": description,
                    "query": query,
                    "error": str(e)
                })

        # 결과 요약
        print("\n" + "="*80)
        print("테스트 결과 요약")
        print("="*80)

        successful_tests = [r for r in results if r["success"]]
        failed_tests = [r for r in results if not r["success"]]

        print(f"\n✅ 성공: {len(successful_tests)}/{len(results)}")
        print(f"❌ 실패: {len(failed_tests)}/{len(results)}")

        if successful_tests:
            print(f"\n📊 성공한 테스트 통계:")
            avg_time = sum(r["processing_time"] for r in successful_tests) / len(successful_tests)
            avg_confidence = sum(r["confidence"] for r in successful_tests) / len(successful_tests)
            avg_unique_terms = sum(r["extracted_terms_count"] for r in successful_tests) / len(successful_tests)
            avg_total_terms = sum(r["total_terms"] for r in successful_tests) / len(successful_tests)

            print(f"   - 평균 처리 시간: {avg_time:.2f}초")
            print(f"   - 평균 신뢰도: {avg_confidence:.2f}")
            print(f"   - 평균 고유 용어: {avg_unique_terms:.0f}개")
            print(f"   - 평균 총 용어: {avg_total_terms:.0f}개")

        if failed_tests:
            print(f"\n❌ 실패한 테스트:")
            for test in failed_tests:
                print(f"   - {test['description']}: {test.get('error', 'Unknown error')}")

        if all(r["success"] for r in results):
            print("\n" + "="*80)
            print("✅ 모든 테스트가 성공했습니다!")
            print("="*80 + "\n")
            return True
        else:
            print("\n" + "="*80)
            print("⚠️ 일부 테스트가 실패했습니다.")
            print("="*80 + "\n")
            return False

    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_single_query_detailed():
    """단일 쿼리에 대한 상세 테스트"""
    print("\n" + "="*80)
    print("상세 단일 쿼리 테스트")
    print("="*80 + "\n")

    try:
        config = LangGraphConfig.from_env()
        workflow_service = LangGraphWorkflowService(config)

        query = "이혼 절차와 양육권 분쟁, 상속 문제에 대해 상세히 알려주세요"

        print(f"테스트 질문: {query}\n")

        result = await workflow_service.process_query(query)

        print("\n" + "="*80)
        print("워크플로우 처리 결과")
        print("="*80)
        print(f"\n답변:\n{result['answer']}\n")

        # 메타데이터 상세 출력
        metadata = result.get("metadata", {})
        print(f"메타데이터:")
        for key, value in metadata.items():
            if isinstance(value, list) and len(value) > 5:
                print(f"  - {key}: {len(value)}개 항목")
            else:
                print(f"  - {key}: {value}")

        print(f"\n처리 단계:")
        for step in result.get("processing_steps", []):
            print(f"  • {step}")

        if result.get("errors"):
            print(f"\n오류:")
            for error in result["errors"]:
                print(f"  ⚠️ {error}")

        return True

    except Exception as e:
        print(f"\n❌ 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_tests():
    """전체 테스트 실행"""
    print("\n" + "="*80)
    print("용어 통합 워크플로우 테스트 시작")
    print("="*80)

    # 비동기 테스트 실행
    result1 = asyncio.run(test_term_integration_workflow())
    result2 = asyncio.run(test_single_query_detailed())

    success = result1 and result2

    print("\n" + "="*80)
    if success:
        print("✅ 모든 테스트 완료!")
    else:
        print("⚠️ 일부 테스트 실패")
    print("="*80 + "\n")

    return success


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
