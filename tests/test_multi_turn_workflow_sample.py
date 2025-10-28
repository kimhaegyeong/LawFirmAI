# -*- coding: utf-8 -*-
"""
LangGraph 멀티턴 워크플로우 샘플 테스트
실제 대화 시나리오를 통한 멀티턴 처리 검증
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from source.services.conversation_manager import ConversationManager, ConversationTurn
from source.services.langgraph.legal_workflow_enhanced import (
    EnhancedLegalQuestionWorkflow,
)
from source.services.langgraph.state_definitions import create_initial_legal_state
from source.utils.langgraph_config import LangGraphConfig


class MultiTurnWorkflowTester:
    """멀티턴 워크플로우 테스터"""

    def __init__(self):
        """테스터 초기화"""
        self.config = LangGraphConfig.from_env()
        self.workflow = EnhancedLegalQuestionWorkflow(self.config)
        self.conversation_manager = ConversationManager()
        self.session_id = "test_multi_turn_session_001"

    def setup_conversation_history(self):
        """대화 이력 설정"""
        print("\n=== 대화 이력 설정 ===")

        # 1차 질문: 손해배상 청구 방법
        print("\n[1차 질문]")
        query1 = "손해배상 청구 방법을 알려주세요"
        response1 = """민법 제750조에 따른 손해배상 청구 방법을 설명드리겠습니다.

손해배상 청구는 다음과 같은 절차로 진행됩니다:
1. 손해배상 청구의 법적 근거: 민법 제750조
2. 불법행위로 인한 손해의 발생
3. 과실 또는 고의 요건
4. 인과관계의 존재
5. 손해배상 청구 절차

상세 내용은 민법 제750조를 참고하시기 바랍니다."""

        context1 = self.conversation_manager.add_turn(
            self.session_id, query1, response1, "legal_advice"
        )
        print(f"Q: {query1}")
        print(f"A: {response1[:100]}...")
        print(f"✓ 엔티티 추출: {context1.entities}")

        # 2차 질문: 계약 해지
        print("\n[2차 질문]")
        query2 = "계약 해지 절차는 어떻게 되나요?"
        response2 = """계약 해지 절차는 다음과 같습니다:

1. 계약 해지 사유 확인
   - 위약 행위 발생
   - 계약 조건 미이행
   - 법적 해지 사유 발생

2. 상대방에게 해지 통지
   - 서면으로 해지 의사 표시
   - 해지 사유 명시

3. 해지 효과
   - 계약 효력 소멸
   - 원상회복 청구 가능

관련 법령: 민법 제552조, 제553조"""

        context2 = self.conversation_manager.add_turn(
            self.session_id, query2, response2, "procedure_guide"
        )
        print(f"Q: {query2}")
        print(f"A: {response2[:100]}...")
        print(f"✓ 엔티티 추출: {context2.entities}")

        print(f"\n총 턴 수: {len(context2.turns)}")
        print(f"총 엔티티 수: {sum(len(entities) for entities in context2.entities.values())}")

    async def test_multi_turn_workflow(self):
        """멀티턴 워크플로우 테스트"""
        print("\n\n=== 멀티턴 워크플로우 테스트 ===\n")

        # 대화 이력 설정
        self.setup_conversation_history()

        # 멀티턴 질문들 테스트
        test_queries = [
            {
                "name": "멀티턴 질문 1: 대명사 참조",
                "query": "그것에 대해 더 자세히 알려주세요",
                "expected_pattern": "손해배상|계약"
            },
            {
                "name": "멀티턴 질문 2: 위의 사건 참조",
                "query": "위의 손해배상 사건에서 과실비율은 어떻게 정해지나요?",
                "expected_pattern": "손해배상"
            },
            {
                "name": "멀티턴 질문 3: 해당 조문 참조",
                "query": "그 계약 해지 절차에서 원상회복은 어떻게 진행되나요?",
                "expected_pattern": "계약|해지"
            },
            {
                "name": "단일 턴 질문: 새로운 질문",
                "query": "임대차 계약금 반환 조건은?",
                "expected_pattern": "임대차"
            }
        ]

        results = []

        for i, test_case in enumerate(test_queries, 1):
            print(f"\n{'='*60}")
            print(f"[테스트 {i}] {test_case['name']}")
            print(f"{'='*60}")
            print(f"질문: {test_case['query']}")

            try:
                # 초기 상태 생성
                initial_state = create_initial_legal_state(test_case['query'], self.session_id)

                # 대화 맥락을 conversation_manager에 연결
                self.workflow.conversation_manager = self.conversation_manager

                # resolve_multi_turn 노드 실행
                resolved_state = self.workflow.resolve_multi_turn(initial_state)

                # 결과 출력
                print(f"\n✓ 멀티턴 처리 완료")
                print(f"  - 멀티턴 여부: {resolved_state.get('is_multi_turn')}")
                print(f"  - 원본 질문: {resolved_state.get('original_query')}")
                print(f"  - 해결된 질문: {resolved_state.get('resolved_query')}")
                print(f"  - 신뢰도: {resolved_state.get('multi_turn_confidence', 1.0):.2f}")
                print(f"  - 추론: {resolved_state.get('multi_turn_reasoning', 'N/A')}")

                # 검증
                if resolved_state.get('is_multi_turn'):
                    resolved_query = resolved_state.get('resolved_query', '')
                    import re
                    if re.search(test_case['expected_pattern'], resolved_query):
                        print(f"\n✅ 성공: 해결된 질문에 예상 패턴 '{test_case['expected_pattern']}' 포함")
                    else:
                        print(f"\n⚠ 경고: 해결된 질문에 예상 패턴 '{test_case['expected_pattern']}' 포함되지 않음")
                elif "단일 턴" in test_case['name']:
                    print(f"\n✅ 성공: 단일 턴 질문으로 올바르게 처리")
                else:
                    print(f"\n❌ 실패: 멀티턴 질문으로 감지되지 않음")

                results.append({
                    "name": test_case['name'],
                    "is_multi_turn": resolved_state.get('is_multi_turn'),
                    "original": resolved_state.get('original_query'),
                    "resolved": resolved_state.get('resolved_query'),
                    "confidence": resolved_state.get('multi_turn_confidence', 1.0)
                })

            except Exception as e:
                print(f"\n❌ 에러 발생: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "name": test_case['name'],
                    "error": str(e)
                })

        # 최종 요약
        print(f"\n\n{'='*60}")
        print("=== 테스트 결과 요약 ===")
        print(f"{'='*60}\n")

        for i, result in enumerate(results, 1):
            if "error" in result:
                print(f"{i}. {result['name']}: ❌ 에러 - {result['error']}")
            elif result.get('is_multi_turn'):
                print(f"{i}. {result['name']}: ✅ 멀티턴 감지")
                print(f"   원본: {result['original']}")
                print(f"   해결: {result['resolved']}")
                print(f"   신뢰도: {result['confidence']:.2f}")
            else:
                print(f"{i}. {result['name']}: ℹ 단일턴 처리")
                print(f"   질문: {result['original']}")

        return results

    async def test_full_workflow_integration(self):
        """전체 워크플로우 통합 테스트"""
        print("\n\n=== 전체 워크플로우 통합 테스트 ===")

        # 대화 맥락 설정
        self.workflow.conversation_manager = self.conversation_manager

        # 간단한 멀티턴 질문으로 전체 워크플로우 테스트
        query = "그 손해배상 관련 내용을 더 자세히 알려주세요"

        print(f"질문: {query}")
        print("전체 워크플로우 실행 중... (시간이 걸릴 수 있습니다)")

        try:
            # 전체 워크플로우 실행
            initial_state = create_initial_legal_state(query, self.session_id)

            # 워크플로우 그래프 컴파일 및 실행
            compiled_graph = self.workflow.graph.compile()

            # 워크플로우 실행 (async)
            result = await compiled_graph.ainvoke(initial_state, {})

            # 결과 확인
            print("\n✓ 워크플로우 실행 완료")
            print(f"\n{'='*60}")
            print("결과 분석")
            print(f"{'='*60}")
            print(f"  원본 질문: {result.get('original_query', query)}")
            print(f"  멀티턴 여부: {result.get('is_multi_turn', False)}")
            print(f"  해결된 질문: {result.get('resolved_query', query)}")
            print(f"  질문 유형: {result.get('query_type', 'N/A')}")
            print(f"  신뢰도: {result.get('confidence', 0.0):.2f}")
            print(f"  답변 길이: {len(result.get('answer', ''))}자")
            print(f"  처리 시간: {result.get('processing_time', 0.0):.2f}초")
            print(f"  검색된 문서 수: {len(result.get('retrieved_docs', []))}")
            print(f"  출처 수: {len(result.get('sources', []))}")

            print(f"\n처리 단계:")
            for i, step in enumerate(result.get('processing_steps', [])[:8], 1):
                print(f"  {i}. {step}")

            # 답변 미리보기
            answer = result.get('answer', '')
            if answer:
                print(f"\n답변 미리보기:")
                print(f"  {answer[:300]}...")

            # 에러 확인
            if result.get('errors'):
                print(f"\n⚠ 에러 발생:")
                for error in result.get('errors', [])[:3]:
                    print(f"  - {error}")

            return result

        except Exception as e:
            print(f"\n❌ 워크플로우 실행 실패: {e}")
            import traceback
            traceback.print_exc()
            return None


async def main():
    """메인 함수"""
    print("=" * 60)
    print("LangGraph 멀티턴 워크플로우 샘플 테스트")
    print("=" * 60)

    tester = MultiTurnWorkflowTester()

    # 1. 멀티턴 워크플로우 테스트
    results = await tester.test_multi_turn_workflow()

    # 2. 전체 워크플로우 통합 테스트
    print("\n" + "=" * 60)
    print("전체 워크플로우 통합 테스트 시작")
    print("=" * 60)

    full_result = await tester.test_full_workflow_integration()

    # 추가적인 전체 플로우 테스트 케이스
    if full_result:
        print("\n=== 추가 테스트 케이스: 실제 멀티턴 시나리오 ===")

        # 실제 멀티턴 시나리오로 전체 워크플로우 실행
        test_scenarios = [
            {
                "name": "손해배상 관련 후속 질문",
                "query": "그것의 법적 근거는 무엇인가요?",
                "context_required": True
            },
            {
                "name": "계약 해지 관련 후속 질문",
                "query": "위의 절차에서 서면 통지는 필수인가요?",
                "context_required": True
            }
        ]

        for scenario in test_scenarios:
            print(f"\n[시나리오] {scenario['name']}")
            print(f"질문: {scenario['query']}")

            try:
                state = create_initial_legal_state(scenario['query'], tester.session_id)
                compiled_graph = tester.workflow.graph.compile()
                result = await compiled_graph.ainvoke(state, {})

                print(f"✓ 처리 완료")
                print(f"  - 멀티턴: {result.get('is_multi_turn', False)}")
                print(f"  - 원본: {result.get('original_query', scenario['query'])}")
                print(f"  - 해결: {result.get('resolved_query', scenario['query'])}")
                print(f"  - 답변 길이: {len(result.get('answer', ''))}자")
                print(f"  - 신뢰도: {result.get('confidence', 0.0):.2f}")

                if result.get('answer'):
                    print(f"  - 답변 미리보기: {result.get('answer', '')[:150]}...")

            except Exception as e:
                print(f"✗ 에러: {e}")

    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)

    return results, full_result


if __name__ == "__main__":
    asyncio.run(main())
