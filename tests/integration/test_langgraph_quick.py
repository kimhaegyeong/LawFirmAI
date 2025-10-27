#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph 빠른 테스트 - 개선된 Mock LLM 검증
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

print("=" * 70)
print("LangGraph 빠른 테스트 (개선된 Mock LLM 검증)")
print("=" * 70)

def main():
    try:
        print("\n[1/5] 워크플로우 초기화...")
        from source.services.langgraph_workflow.legal_workflow_enhanced import (
            EnhancedLegalQuestionWorkflow,
        )
        from source.services.langgraph_workflow.state_definitions import (
            create_initial_state,
        )
        from source.utils.langgraph_config import LangGraphConfig

        config = LangGraphConfig()
        workflow = EnhancedLegalQuestionWorkflow(config)
        print("✅ 초기화 완료")

        print(f"\n[2/5] LLM 확인...")
        llm_type = type(workflow.llm).__name__
        print(f"   LLM 타입: {llm_type}")

        print(f"\n[3/5] 간단한 질문 처리 테스트...")
        test_query = "이혼 절차에 대해 알려주세요"
        print(f"   질문: '{test_query}'")

        state = create_initial_state(
            query=test_query,
            session_id="test_quick_1",
            user_id="user_1"
        )
        state["user_query"] = test_query

        # 핵심 단계만 실행
        print("\n   → 입력 검증")
        state = workflow.validate_input(state)

        print("   → 특수 쿼리 감지")
        state = workflow.detect_special_queries(state)

        print("   → 질문 분류")
        state["query"] = test_query
        state = workflow.classify_query(state)

        print("   → 하이브리드 분석")
        state = workflow.analyze_query_hybrid(state)

        print("   → 법률 제한 검증")
        state = workflow.validate_legal_restrictions(state)

        print("   → 문서 검색")
        state = workflow.retrieve_documents(state)
        doc_count = len(state.get('retrieved_docs', []))
        print(f"      검색된 문서: {doc_count}개")

        print("   → Phase 병렬 처리")
        state = workflow.enrich_conversation_context(state)
        state = workflow.personalize_response(state)
        state = workflow.manage_memory_quality(state)

        print("   → 답변 생성 (Mock LLM)")
        state = workflow.generate_answer_enhanced(state)

        print("   → 후처리")
        state = workflow.enhance_completion(state)
        state = workflow.add_disclaimer(state)

        print(f"\n[4/5] 결과 검증...")

        # 답변 확인
        answer = state.get('answer', '')
        if not answer:
            print("❌ 답변이 생성되지 않았습니다")
            return False

        answer_len = len(answer)
        print(f"   답변 길이: {answer_len}자")

        if answer_len < 50:
            print(f"❌ 답변이 너무 짧습니다")
            return False

        generation_success = state.get('generation_success', False)
        print(f"   생성 성공: {generation_success}")

        generation_method = state.get('generation_method', 'unknown')
        print(f"   생성 방법: {generation_method}")

        # "관련 법률 정보를 찾을 수 없었습니다" 메시지가 없어야 함
        if "관련 법률 정보를 찾을 수 없었습니다" in answer:
            print("❌ Mock LLM이 여전히 컨텍스트를 인식하지 못함")
            return False

        # 이혼 관련 내용이 포함되어야 함
        if "이혼" in test_query and "이혼" not in answer:
            print("⚠️  답변에 질문 핵심 키워드가 포함되지 않음")

        print(f"\n[5/5] 답변 미리보기:")
        print(f"   {answer[:200]}...")

        print(f"\n{'='*70}")
        print("✅ 테스트 성공!")
        print(f"{'='*70}")

        print(f"\n처리 통계:")
        print(f"  - 처리 단계: {len(state.get('processing_steps', []))}개")
        print(f"  - 처리 시간: {state.get('processing_time', 0):.2f}초")
        print(f"  - 검색 문서: {doc_count}개")

        return True

    except Exception as e:
        print(f"\n❌ 테스트 실패: {str(e)}")
        import traceback
        print("\n상세 에러:")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
