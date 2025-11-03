#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
검색 결과 포함 테스트
검색 결과가 프롬프트에 포함되어 적절한 답변을 생성하는지 확인
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# lawfirm_langgraph 경로 추가
lawfirm_langgraph_path = project_root / "lawfirm_langgraph"
sys.path.insert(0, str(lawfirm_langgraph_path))

from langgraph_core.services.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
from infrastructure.utils.langgraph_config import LangGraphConfig


def create_initial_legal_state(query: str, session_id: str) -> dict:
    """초기 법률 상태 생성"""
    return {
        "query": query,
        "session_id": session_id,
        "query_type": "",
        "retrieved_docs": [],
        "processing_steps": [],
        "metadata": {}
    }


def test_with_search_results():
    """검색 결과가 있는 경우 테스트"""
    print("\n" + "="*80)
    print("검색 결과 포함 테스트 - 검색 결과가 있을 때")
    print("="*80 + "\n")

    try:
        # 설정 로드
        config = LangGraphConfig.from_env()
        workflow = EnhancedLegalQuestionWorkflow(config)

        print("✅ 워크플로우 초기화 완료\n")

        # 테스트 케이스: 검색 결과가 있는 경우
        test_cases = [
            {
                "query": "손해배상 청구 방법을 알려주세요",
                "query_type": "legal_advice",
                "description": "민사법 - 손해배상",
                "retrieved_docs": [
                    {
                        "content": "민법 제750조 (불법행위의 내용) 타인의 고의 또는 과실로 인한 불법행위로 인하여 손해를 받은 자는 그 손해를 배상받을 수 있다. 손해배상을 청구하려면 가해자의 고의 또는 과실, 손해의 발생, 인과관계를 입증해야 한다.",
                        "source": "민법 제750조",
                        "relevance_score": 0.95,
                        "metadata": {"law_name": "민법", "article_no": "750"}
                    },
                    {
                        "content": "민법 제751조 (재산상 손해배상) 불법행위로 인하여 재산상 손해가 발생한 때에는 그 손해를 배상하여야 한다. 손해배상은 원칙적으로 금전으로 이루어지며, 손해의 범위는 통상의 손해와 특별 손해를 포함한다.",
                        "source": "민법 제751조",
                        "relevance_score": 0.92,
                        "metadata": {"law_name": "민법", "article_no": "751"}
                    },
                    {
                        "content": "대법원 2020다12345 판결에 따르면, 손해배상 청구권은 손해 발생 사실과 인과관계가 입증되어야 성립한다. 가해자의 과실과 손해 사이의 인과관계는 일반적인 사회통념에 따라 판단한다.",
                        "source": "대법원 2020다12345",
                        "relevance_score": 0.88,
                        "metadata": {"case_number": "2020다12345", "court": "대법원"}
                    }
                ]
            },
            {
                "query": "계약 해지 요건은 무엇인가요?",
                "query_type": "law_inquiry",
                "description": "민사법 - 계약 해지",
                "retrieved_docs": [
                    {
                        "content": "민법 제543조 (해지권의 행사) 계약 당사자의 일방은 계약 또는 법률의 규정에 의한 해지권을 행사할 수 있다. 해지권의 행사는 상대방에 대한 의사표시로 한다.",
                        "source": "민법 제543조",
                        "relevance_score": 0.96,
                        "metadata": {"law_name": "민법", "article_no": "543"}
                    },
                    {
                        "content": "민법 제544조 (채무불이행을 이유로 한 해지) 계약의 해지는 채무불이행이 있을 때 상당한 기간을 정하여 이행 최고를 하고 그 기간 내에 이행하지 아니한 경우에 할 수 있다. 채무자가 이행을 거부한 때에는 최고를 기다리지 아니하고 해지할 수 있다.",
                        "source": "민법 제544조",
                        "relevance_score": 0.93,
                        "metadata": {"law_name": "민법", "article_no": "544"}
                    }
                ]
            },
            {
                "query": "이혼 절차에 대해 알려주세요",
                "query_type": "legal_advice",
                "description": "가족법 - 이혼 절차",
                "retrieved_docs": [
                    {
                        "content": "민법 제834조 (협의상 이혼) 부부는 협의하여 이혼할 수 있다. 협의상 이혼은 가족관계등록법에 정한 바에 따라 신고함으로써 그 효력이 생긴다.",
                        "source": "민법 제834조",
                        "relevance_score": 0.94,
                        "metadata": {"law_name": "민법", "article_no": "834"}
                    },
                    {
                        "content": "민법 제840조 (재판상 이혼) 부부의 일방은 다음 각 호의 어느 하나에 해당하는 사유가 있는 경우에는 가정법원에 이혼을 청구할 수 있다. 1. 배우자에 부정한 행위가 있었을 때 2. 배우자가 악의로 다른 일방을 유기한 때 3. 배우자 또는 그 직계존속으로부터 심히 부적절한 대우를 받았을 때 4. 자기 또는 배우자의 직계존속으로부터 심히 부적절한 대우를 한 배우자에 대하여 이혼을 청구할 수 있다.",
                        "source": "민법 제840조",
                        "relevance_score": 0.91,
                        "metadata": {"law_name": "민법", "article_no": "840"}
                    }
                ]
            }
        ]

        results = []

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'='*80}")
            print(f"테스트 {i}/{len(test_cases)}: {test_case['description']}")
            print(f"질문: {test_case['query']}")
            print(f"검색 결과: {len(test_case['retrieved_docs'])}개")
            print(f"{'='*80}\n")

            # 초기 상태 생성
            state = create_initial_legal_state(test_case['query'], f"test-session-{i}")
            state["query_type"] = test_case['query_type']
            state["retrieved_docs"] = test_case['retrieved_docs']

            # generate_answer_enhanced 실행
            result = workflow.generate_answer_enhanced(state)

            # 결과 검증
            answer = result.get("answer", "")
            if isinstance(answer, dict):
                answer = answer.get("answer", "") or str(answer)
            if not answer:
                answer = ""

            assert answer, "답변이 생성되지 않았습니다"

            # 검색 결과가 답변에 포함되었는지 확인
            answer_lower = str(answer).lower()
            has_citation = False
            cited_sources = []

            for doc in test_case['retrieved_docs']:
                source = doc.get("source", "")
                content_preview = doc.get("content", "")[:50]

                # 출처가 답변에 포함되었는지 확인
                if source and (source in answer or any(keyword in answer for keyword in source.split())):
                    has_citation = True
                    cited_sources.append(source)

                # 조문 번호가 답변에 포함되었는지 확인
                article_no = doc.get("metadata", {}).get("article_no", "")
                if article_no and article_no in answer:
                    has_citation = True

            # 답변 길이 확인 (너무 짧으면 프롬프트가 출력된 것일 수 있음)
            answer_length = len(answer)
            is_too_short = answer_length < 100
            is_too_long = answer_length > 5000

            print(f"📝 답변 길이: {answer_length}자")
            print(f"📚 인용된 소스: {len(cited_sources)}개 / {len(test_case['retrieved_docs'])}개")
            if cited_sources:
                print(f"   인용된 소스 목록: {', '.join(cited_sources[:5])}")
            print(f"📋 답변 미리보기: {answer[:200]}...")

            # 검증 결과
            test_passed = True
            issues = []

            if not has_citation:
                test_passed = False
                issues.append("검색 결과가 답변에 인용되지 않았습니다")

            if is_too_short:
                test_passed = False
                issues.append(f"답변이 너무 짧습니다 ({answer_length}자)")

            if is_too_long:
                issues.append(f"⚠️ 답변이 매우 깁니다 ({answer_length}자)")

            if test_passed:
                print(f"✅ 테스트 통과: 검색 결과가 적절히 반영되었습니다")
                results.append(True)
            else:
                print(f"⚠️ 테스트 실패:")
                for issue in issues:
                    print(f"   - {issue}")
                results.append(False)

        # 종합 결과
        print(f"\n{'='*80}")
        print("종합 테스트 결과")
        print(f"{'='*80}")
        passed = sum(results)
        total = len(results)
        print(f"✅ 통과: {passed}/{total}")
        print(f"❌ 실패: {total - passed}/{total}")
        print(f"{'='*80}\n")

        return all(results)

    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_with_search_results()
    if success:
        print("✅ 모든 검색 결과 포함 테스트가 성공했습니다!")
    else:
        print("⚠️ 일부 테스트가 실패했습니다.")
