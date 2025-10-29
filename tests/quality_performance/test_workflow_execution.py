# -*- coding: utf-8 -*-
"""
LangGraph 워크플로우 실행 테스트
UnifiedPromptManager 통합 후 실제 워크플로우 실행 검증
"""

import asyncio
import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 테스트 환경 설정
os.environ["USE_LANGGRAPH"] = "true"
os.environ["LANGGRAPH_ENABLED"] = "true"

from source.services.langgraph.workflow_service import LangGraphWorkflowService
from source.utils.langgraph_config import LangGraphConfig


async def test_workflow_execution():
    """워크플로우 실행 테스트"""
    print("\n" + "="*80)
    print("LangGraph 워크플로우 실행 테스트")
    print("="*80 + "\n")

    try:
        # 설정 로드
        config = LangGraphConfig.from_env()
        print("✅ LangGraph 설정 로드 완료")

        # 워크플로우 서비스 초기화
        workflow_service = LangGraphWorkflowService(config)
        print("✅ LangGraphWorkflowService 초기화 완료")

        # 테스트 쿼리들
        test_queries = [
            ("이혼 절차에 대해 알려주세요", "가족법"),
            ("계약서 작성 시 주의사항은?", "민사법"),
            ("해고 제한 조건은 무엇인가요?", "노동법"),
            ("절도죄의 처벌은?", "형사법"),
            ("민법 제750조에 대해 알려주세요", "민사법 조문"),
        ]

        results = []

        for query, description in test_queries:
            print(f"\n📋 테스트: {description}")
            print(f"   질문: {query}")

            try:
                # 워크플로우 실행
                result = await workflow_service.process_query(query)

                # 결과 검증
                assert "answer" in result
                assert len(result["answer"]) > 0
                assert "confidence" in result
                assert "sources" in result

                # 결과 출력
                answer_length = len(result["answer"])
                confidence = result.get("confidence", 0.0)
                processing_time = result.get("processing_time", 0.0)

                print(f"   ✅ 처리 완료")
                print(f"   - 답변 길이: {answer_length}자")
                print(f"   - 신뢰도: {confidence:.2f}")
                print(f"   - 처리 시간: {processing_time:.2f}초")
                print(f"   - 출처: {len(result['sources'])}개")

                # 처리 단계 확인
                if result.get("processing_steps"):
                    steps = result["processing_steps"]
                    if any("UnifiedPromptManager" in step for step in steps):
                        print(f"   - UnifiedPromptManager 사용 확인됨")

                results.append(True)

            except Exception as e:
                print(f"   ❌ 처리 실패: {e}")
                results.append(False)

        # 결과 요약
        print("\n" + "="*80)
        print("결과 요약")
        print("="*80)

        passed = sum(results)
        total = len(results)
        print(f"\n✅ 성공: {passed}/{total}")
        print(f"❌ 실패: {total - passed}/{total}")

        if all(results):
            print("\n✅ 모든 워크플로우 테스트가 성공했습니다!")
            return True
        else:
            print("\n⚠️ 일부 테스트가 실패했습니다.")
            return False

    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_quick_test():
    """빠른 테스트 실행"""
    print("\n" + "="*80)
    print("빠른 워크플로우 테스트")
    print("="*80 + "\n")

    # 비동기 테스트 실행
    result = asyncio.run(test_workflow_execution())

    return result


if __name__ == "__main__":
    success = run_quick_test()

    if success:
        print("\n" + "="*80)
        print("✅ 모든 테스트 완료!")
        print("="*80 + "\n")
    else:
        print("\n" + "="*80)
        print("⚠️ 일부 테스트 실패")
        print("="*80 + "\n")
