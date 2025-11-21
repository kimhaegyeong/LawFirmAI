# -*- coding: utf-8 -*-
"""
대화 맥락 기능 테스트 스크립트
"""

import sys
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 경로 설정
script_dir = Path(__file__).parent
workflow_dir = script_dir.parent
unit_dir = workflow_dir.parent
tests_dir = unit_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(lawfirm_langgraph_dir))

try:
    from lawfirm_langgraph.core.conversation.conversation_manager import ConversationManager, ConversationContext, ConversationTurn
    from lawfirm_langgraph.core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
    from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
except ImportError:
    from core.conversation.conversation_manager import ConversationManager, ConversationContext, ConversationTurn
    from core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
    from config.langgraph_config import LangGraphConfig


def test_conversation_context_conversion():
    """ConversationContext를 LangChain Message로 변환 테스트"""
    print("\n=== 테스트 1: ConversationContext를 LangChain Message로 변환 ===")
    
    config = LangGraphConfig.from_env()
    workflow = EnhancedLegalQuestionWorkflow(config)
    session_id = "test_session_001"
    
    conversation_manager = ConversationManager()
    workflow.conversation_manager = conversation_manager
    
    # 테스트 턴 추가
    conversation_manager.add_turn(
        session_id=session_id,
        user_query="손해배상 청구 방법을 알려주세요",
        bot_response="민법 제750조에 따른 손해배상 청구 방법을 설명드리겠습니다...",
        question_type="legal_advice"
    )
    conversation_manager.add_turn(
        session_id=session_id,
        user_query="계약 해지 절차는 어떻게 되나요?",
        bot_response="계약 해지 절차는 다음과 같습니다...",
        question_type="procedure_guide"
    )
    
    context = conversation_manager.sessions.get(session_id)
    assert context is not None, "Context should not be None"
    assert len(context.turns) == 2, f"Expected 2 turns, got {len(context.turns)}"
    
    messages = workflow._convert_conversation_context_to_messages(
        context,
        max_turns=5
    )
    
    assert len(messages) == 4, f"Expected 4 messages, got {len(messages)}"
    assert messages[0].content == "손해배상 청구 방법을 알려주세요"
    assert messages[1].content.startswith("민법 제750조")
    assert messages[2].content == "계약 해지 절차는 어떻게 되나요?"
    
    print("✅ 테스트 1 통과: ConversationContext를 LangChain Message로 변환 성공")


def test_relevance_based_selection():
    """관련성 기반 선택 테스트"""
    print("\n=== 테스트 2: 관련성 기반 맥락 선택 ===")
    
    config = LangGraphConfig.from_env()
    workflow = EnhancedLegalQuestionWorkflow(config)
    session_id = "test_session_002"
    
    conversation_manager = ConversationManager()
    workflow.conversation_manager = conversation_manager
    
    # 다양한 주제의 대화 추가
    conversation_manager.add_turn(
        session_id=session_id,
        user_query="손해배상 청구 방법을 알려주세요",
        bot_response="민법 제750조에 따른 손해배상 청구 방법...",
        question_type="legal_advice"
    )
    conversation_manager.add_turn(
        session_id=session_id,
        user_query="계약 해지 절차는 어떻게 되나요?",
        bot_response="계약 해지 절차는 다음과 같습니다...",
        question_type="procedure_guide"
    )
    conversation_manager.add_turn(
        session_id=session_id,
        user_query="위의 손해배상 사건에서 과실비율은 어떻게 정해지나요?",
        bot_response="과실비율은 교통사고의 경우...",
        question_type="legal_advice"
    )
    
    context = conversation_manager.sessions.get(session_id)
    current_query = "손해배상 관련 판례를 더 찾아주세요"
    
    messages = workflow._convert_conversation_context_to_messages(
        context,
        max_turns=5,
        current_query=current_query,
        use_relevance=True
    )
    
    assert len(messages) > 0, "Messages should not be empty"
    message_contents = [msg.content for msg in messages]
    assert any("손해배상" in content for content in message_contents), "Should contain 손해배상 related content"
    
    print("✅ 테스트 2 통과: 관련성 기반 맥락 선택 성공")


def test_token_based_pruning():
    """토큰 기반 크기 관리 테스트"""
    print("\n=== 테스트 3: 토큰 기반 크기 관리 ===")
    
    config = LangGraphConfig.from_env()
    workflow = EnhancedLegalQuestionWorkflow(config)
    session_id = "test_session_003"
    
    conversation_manager = ConversationManager()
    workflow.conversation_manager = conversation_manager
    
    # 여러 턴 추가
    for i in range(10):
        conversation_manager.add_turn(
            session_id=session_id,
            user_query=f"질문 {i+1}",
            bot_response=f"답변 {i+1}입니다. " * 20,  # 각 답변 약 200자
            question_type="legal_advice"
        )
    
    context = conversation_manager.sessions.get(session_id)
    assert len(context.turns) == 10, f"Expected 10 turns, got {len(context.turns)}"
    
    # 토큰 제한으로 정리
    selected_turns = workflow._prune_conversation_history_by_tokens(
        context,
        max_tokens=1000  # 약 3-4개 턴 정도
    )
    
    assert len(selected_turns) <= 10, f"Selected turns should be <= 10, got {len(selected_turns)}"
    assert len(selected_turns) > 0, "Selected turns should not be empty"
    assert selected_turns[-1].user_query == "질문 10", "Latest turn should be included"
    
    print(f"✅ 테스트 3 통과: 토큰 기반 크기 관리 성공 ({len(context.turns)} → {len(selected_turns)} turns)")


def test_get_relevant_context():
    """관련 맥락 조회 테스트"""
    print("\n=== 테스트 4: 관련 맥락 조회 ===")
    
    conversation_manager = ConversationManager()
    session_id = "test_session_004"
    
    conversation_manager.add_turn(
        session_id=session_id,
        user_query="손해배상 청구 방법을 알려주세요",
        bot_response="민법 제750조에 따른 손해배상 청구 방법...",
        question_type="legal_advice"
    )
    conversation_manager.add_turn(
        session_id=session_id,
        user_query="계약 해지 절차는 어떻게 되나요?",
        bot_response="계약 해지 절차는 다음과 같습니다...",
        question_type="procedure_guide"
    )
    
    current_query = "손해배상 관련 판례를 더 찾아주세요"
    relevant_context = conversation_manager.get_relevant_context(
        session_id,
        current_query,
        max_turns=2
    )
    
    assert relevant_context is not None, "Relevant context should not be None"
    assert "relevant_turns" in relevant_context, "Should have relevant_turns"
    assert len(relevant_context["relevant_turns"]) > 0, "Should have at least one relevant turn"
    
    relevant_turns = relevant_context["relevant_turns"]
    assert any("손해배상" in turn.get("user_query", "") for turn in relevant_turns), "Should contain 손해배상 related turn"
    
    print("✅ 테스트 4 통과: 관련 맥락 조회 성공")


def test_token_based_conversion():
    """토큰 기반 변환 테스트"""
    print("\n=== 테스트 5: 토큰 기반 변환 ===")
    
    config = LangGraphConfig.from_env()
    workflow = EnhancedLegalQuestionWorkflow(config)
    session_id = "test_session_005"
    
    conversation_manager = ConversationManager()
    workflow.conversation_manager = conversation_manager
    
    # 긴 답변 추가
    long_response = "긴 답변입니다. " * 100  # 약 1500자
    conversation_manager.add_turn(
        session_id=session_id,
        user_query="긴 질문입니다",
        bot_response=long_response,
        question_type="legal_advice"
    )
    
    context = conversation_manager.sessions.get(session_id)
    
    messages = workflow._convert_conversation_context_to_messages(
        context,
        max_turns=10,
        max_tokens=500  # 작은 토큰 제한
    )
    
    assert len(messages) > 0, "Messages should not be empty"
    assert len(messages) <= 10, "Should not exceed max_turns"
    
    print(f"✅ 테스트 5 통과: 토큰 기반 변환 성공 ({len(messages)} messages)")


def test_pronoun_resolution_case1():
    """케이스 1: 법률 용어 명확한 참조 (그거)"""
    print("\n=== 테스트 6: 케이스 1 - 법률 용어 명확한 참조 ===")
    
    config = LangGraphConfig.from_env()
    workflow = EnhancedLegalQuestionWorkflow(config)
    session_id = "test_case1"
    
    conversation_manager = ConversationManager()
    workflow.conversation_manager = conversation_manager
    
    # 첫 번째 대화
    conversation_manager.add_turn(
        session_id=session_id,
        user_query="근로기준법 제60조 내용 알려줘",
        bot_response="근로기준법 제60조는 연차 유급휴가에 관한 규정입니다. 사용자는 1년간 80% 이상 출근한 경우 연차 유급휴가를 사용할 수 있습니다.",
        question_type="legal_advice"
    )
    
    # 두 번째 대화 (대명사 사용)
    current_query = "그거 위반하면 어떻게 돼?"
    
    # 관련 맥락 조회
    relevant_context = conversation_manager.get_relevant_context(
        session_id,
        current_query,
        max_turns=3
    )
    
    assert relevant_context is not None, "Relevant context should not be None"
    assert len(relevant_context["relevant_turns"]) > 0, "Should have relevant turns"
    
    # 메시지 변환
    context = conversation_manager.sessions.get(session_id)
    messages = workflow._convert_conversation_context_to_messages(
        context,
        max_turns=5,
        current_query=current_query,
        use_relevance=True
    )
    
    assert len(messages) > 0, "Messages should not be empty"
    # 첫 번째 대화가 포함되어야 함
    message_contents = [msg.content for msg in messages]
    assert any("근로기준법 제60조" in content for content in message_contents), "Should contain 근로기준법 제60조"
    
    print("✅ 테스트 6 통과: 법률 용어 명확한 참조 성공")


def test_pronoun_resolution_case2():
    """케이스 2: 복수 법조항 중 선택 (그거)"""
    print("\n=== 테스트 7: 케이스 2 - 복수 법조항 중 선택 ===")
    
    conversation_manager = ConversationManager()
    session_id = "test_case2"
    
    # 첫 번째 대화
    conversation_manager.add_turn(
        session_id=session_id,
        user_query="명예훼손과 모욕죄 차이 알려줘",
        bot_response="명예훼손(형법 제307조)은 사실이나 허위사실을 적시하여 명예를 훼손하는 것이고, 모욕죄(형법 제311조)는 공연히 사람을 모욕하는 죄입니다.",
        question_type="legal_advice"
    )
    
    # 두 번째 대화 (대명사 사용)
    current_query = "그거 처벌 수위는?"
    
    relevant_context = conversation_manager.get_relevant_context(
        session_id,
        current_query,
        max_turns=3
    )
    
    assert relevant_context is not None, "Relevant context should not be None"
    # 명예훼손 또는 모욕죄 관련 내용이 포함되어야 함
    relevant_turns = relevant_context.get("relevant_turns", [])
    assert len(relevant_turns) > 0, "Should have relevant turns"
    
    # 관련 턴에 명예훼손 또는 모욕죄가 포함되어야 함
    turn_contents = [turn.get("user_query", "") + " " + turn.get("bot_response", "") for turn in relevant_turns]
    assert any("명예훼손" in content or "모욕죄" in content for content in turn_contents), "Should contain 명예훼손 or 모욕죄"
    
    print("✅ 테스트 7 통과: 복수 법조항 중 선택 성공")


def test_pronoun_resolution_case3():
    """케이스 3: 판례 참조 (그때)"""
    print("\n=== 테스트 8: 케이스 3 - 판례 참조 ===")
    
    conversation_manager = ConversationManager()
    session_id = "test_case3"
    
    # 첫 번째 대화
    conversation_manager.add_turn(
        session_id=session_id,
        user_query="대법원 2020다12345 판결 요약해줘",
        bot_response="해당 판례는 임대차보호법 관련 사건으로, 임차인의 권리 보호에 관한 중요한 판례입니다.",
        question_type="precedent_search"
    )
    
    # 두 번째 대화 (대명사 사용)
    current_query = "그때 법원 판단 근거는?"
    
    relevant_context = conversation_manager.get_relevant_context(
        session_id,
        current_query,
        max_turns=3
    )
    
    assert relevant_context is not None, "Relevant context should not be None"
    relevant_turns = relevant_context.get("relevant_turns", [])
    assert len(relevant_turns) > 0, "Should have relevant turns"
    
    # 판례 관련 내용이 포함되어야 함
    turn_contents = [turn.get("user_query", "") + " " + turn.get("bot_response", "") for turn in relevant_turns]
    assert any("2020다12345" in content or "판례" in content for content in turn_contents), "Should contain 판례 reference"
    
    print("✅ 테스트 8 통과: 판례 참조 성공")


def test_pronoun_resolution_case4():
    """케이스 4: 당사자 구분 (그 사람)"""
    print("\n=== 테스트 9: 케이스 4 - 당사자 구분 ===")
    
    conversation_manager = ConversationManager()
    session_id = "test_case4"
    
    # 첫 번째 대화
    conversation_manager.add_turn(
        session_id=session_id,
        user_query="원고와 피고의 주장이 뭐야?",
        bot_response="원고는 계약 위반을 주장하고, 피고는 정당한 해지라고 주장합니다.",
        question_type="legal_advice"
    )
    
    # 두 번째 대화 (대명사 사용 - 모호함)
    current_query = "그 사람 증거는?"
    
    relevant_context = conversation_manager.get_relevant_context(
        session_id,
        current_query,
        max_turns=3
    )
    
    assert relevant_context is not None, "Relevant context should not be None"
    relevant_turns = relevant_context.get("relevant_turns", [])
    assert len(relevant_turns) > 0, "Should have relevant turns"
    
    # 원고/피고 관련 내용이 포함되어야 함
    turn_contents = [turn.get("user_query", "") + " " + turn.get("bot_response", "") for turn in relevant_turns]
    assert any("원고" in content or "피고" in content for content in turn_contents), "Should contain 원고 or 피고"
    
    print("✅ 테스트 9 통과: 당사자 구분 성공")


def test_pronoun_resolution_case5():
    """케이스 5: 절차법 단계별 참조 (두 번째 거)"""
    print("\n=== 테스트 10: 케이스 5 - 절차법 단계별 참조 ===")
    
    conversation_manager = ConversationManager()
    session_id = "test_case5"
    
    # 첫 번째 대화
    conversation_manager.add_turn(
        session_id=session_id,
        user_query="민사소송 절차 알려줘",
        bot_response="1단계 소장 제출, 2단계 답변서 제출, 3단계 변론기일이 있습니다.",
        question_type="procedure_guide"
    )
    
    # 두 번째 대화 (대명사 사용)
    current_query = "두 번째 거 기한은?"
    
    relevant_context = conversation_manager.get_relevant_context(
        session_id,
        current_query,
        max_turns=3
    )
    
    assert relevant_context is not None, "Relevant context should not be None"
    relevant_turns = relevant_context.get("relevant_turns", [])
    assert len(relevant_turns) > 0, "Should have relevant turns"
    
    # 절차 관련 내용이 포함되어야 함
    turn_contents = [turn.get("user_query", "") + " " + turn.get("bot_response", "") for turn in relevant_turns]
    assert any("절차" in content or "답변서" in content or "소장" in content for content in turn_contents), "Should contain procedure reference"
    
    print("✅ 테스트 10 통과: 절차법 단계별 참조 성공")


def test_pronoun_resolution_case6():
    """케이스 6: 계약 당사자 참조 (그쪽)"""
    print("\n=== 테스트 11: 케이스 6 - 계약 당사자 참조 ===")
    
    conversation_manager = ConversationManager()
    session_id = "test_case6"
    
    # 첫 번째 대화
    conversation_manager.add_turn(
        session_id=session_id,
        user_query="임대인과 임차인의 의무 알려줘",
        bot_response="임대인은 임대목적물을 사용·수익하게 할 의무가 있고, 임차인은 차임을 지급할 의무가 있습니다.",
        question_type="legal_advice"
    )
    
    # 두 번째 대화 (대명사 사용)
    current_query = "그쪽이 의무 위반하면?"
    
    relevant_context = conversation_manager.get_relevant_context(
        session_id,
        current_query,
        max_turns=3
    )
    
    assert relevant_context is not None, "Relevant context should not be None"
    relevant_turns = relevant_context.get("relevant_turns", [])
    assert len(relevant_turns) > 0, "Should have relevant turns"
    
    # 임대인/임차인 관련 내용이 포함되어야 함
    turn_contents = [turn.get("user_query", "") + " " + turn.get("bot_response", "") for turn in relevant_turns]
    assert any("임대인" in content or "임차인" in content for content in turn_contents), "Should contain 임대인 or 임차인"
    
    print("✅ 테스트 11 통과: 계약 당사자 참조 성공")


def test_multi_turn_conversation_flow():
    """다중 턴 대화 흐름 테스트"""
    print("\n=== 테스트 12: 다중 턴 대화 흐름 ===")
    
    config = LangGraphConfig.from_env()
    workflow = EnhancedLegalQuestionWorkflow(config)
    session_id = "test_multiturn"
    
    conversation_manager = ConversationManager()
    workflow.conversation_manager = conversation_manager
    
    # 여러 턴의 대화 추가
    turns = [
        ("근로기준법 제60조 내용 알려줘", "근로기준법 제60조는 연차 유급휴가에 관한 규정입니다."),
        ("그거 위반하면 어떻게 돼?", "근로기준법 제60조를 위반할 경우 처벌받을 수 있습니다."),
        ("명예훼손과 모욕죄 차이 알려줘", "명예훼손(형법 제307조)과 모욕죄(형법 제311조)의 차이는..."),
        ("그거 처벌 수위는?", "명예훼손과 모욕죄의 처벌 수위는..."),
    ]
    
    for user_query, bot_response in turns:
        conversation_manager.add_turn(
            session_id=session_id,
            user_query=user_query,
            bot_response=bot_response,
            question_type="legal_advice"
        )
    
    context = conversation_manager.sessions.get(session_id)
    assert len(context.turns) == 4, f"Expected 4 turns, got {len(context.turns)}"
    
    # 마지막 질문에 대한 관련 맥락 조회
    current_query = "그거 처벌 수위는?"
    messages = workflow._convert_conversation_context_to_messages(
        context,
        max_turns=5,
        current_query=current_query,
        use_relevance=True
    )
    
    assert len(messages) > 0, "Messages should not be empty"
    # 명예훼손/모욕죄 관련 메시지가 포함되어야 함
    message_contents = [msg.content for msg in messages]
    assert any("명예훼손" in content or "모욕죄" in content for content in message_contents), "Should contain 명예훼손 or 모욕죄"
    
    print("✅ 테스트 12 통과: 다중 턴 대화 흐름 성공")


def main():
    """모든 테스트 실행"""
    print("=" * 60)
    print("대화 맥락 기능 테스트 시작 (대명사 처리 포함)")
    print("=" * 60)
    
    tests = [
        test_conversation_context_conversion,
        test_relevance_based_selection,
        test_token_based_pruning,
        test_get_relevant_context,
        test_token_based_conversion,
        test_pronoun_resolution_case1,
        test_pronoun_resolution_case2,
        test_pronoun_resolution_case3,
        test_pronoun_resolution_case4,
        test_pronoun_resolution_case5,
        test_pronoun_resolution_case6,
        test_multi_turn_conversation_flow,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} 실패: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"테스트 결과: {passed}개 통과, {failed}개 실패")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

