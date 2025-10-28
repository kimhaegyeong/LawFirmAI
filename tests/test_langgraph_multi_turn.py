# -*- coding: utf-8 -*-
"""
LangGraph 멀티턴 로직 통합 테스트
"""

import sys
from pathlib import Path

import pytest

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime

from source.services.conversation_manager import ConversationManager, ConversationTurn
from source.services.langgraph.legal_workflow_enhanced import (
    EnhancedLegalQuestionWorkflow,
)
from source.services.langgraph.state_definitions import create_initial_legal_state
from source.utils.langgraph_config import LangGraphConfig


class TestLangGraphMultiTurnIntegration:
    """LangGraph 멀티턴 통합 테스트"""

    @pytest.fixture
    def config(self):
        """LangGraph 설정"""
        return LangGraphConfig.from_env()

    @pytest.fixture
    def workflow(self, config):
        """워크플로우 초기화"""
        return EnhancedLegalQuestionWorkflow(config)

    @pytest.fixture
    def conversation_manager(self):
        """대화 관리자 초기화"""
        manager = ConversationManager()

        # 테스트 대화 추가
        session_id = "test_session_001"

        turn1 = ConversationTurn(
            user_query="손해배상 청구 방법을 알려주세요",
            bot_response="민법 제750조에 따른 손해배상 청구 방법을 설명드리겠습니다...",
            timestamp=datetime.now(),
            question_type="legal_advice",
            entities={"laws": ["민법"], "articles": ["제750조"], "legal_terms": ["손해배상"]}
        )

        turn2 = ConversationTurn(
            user_query="계약 해지 절차는 어떻게 되나요?",
            bot_response="계약 해지 절차는 다음과 같습니다...",
            timestamp=datetime.now(),
            question_type="procedure_guide",
            entities={"legal_terms": ["계약", "해지"]}
        )

        context = manager.add_turn(session_id, turn1.user_query, turn1.bot_response, turn1.question_type)
        context = manager.add_turn(session_id, turn2.user_query, turn2.bot_response, turn2.question_type)

        return manager

    def test_multi_turn_handler_initialization(self, workflow):
        """멀티턴 핸들러 초기화 테스트"""
        assert workflow.multi_turn_handler is not None, "MultiTurnQuestionHandler가 초기화되지 않았습니다"
        assert workflow.conversation_manager is not None, "ConversationManager가 초기화되지 않았습니다"
        print("✓ 멀티턴 핸들러 초기화 성공")

    def test_state_definitions_multi_turn_fields(self):
        """상태 정의에 멀티턴 필드가 있는지 테스트"""
        state = create_initial_legal_state("테스트 질문", "test_session")

        # 멀티턴 관련 필드 확인
        assert "is_multi_turn" in state, "is_multi_turn 필드가 없습니다"
        assert "original_query" in state, "original_query 필드가 없습니다"
        assert "resolved_query" in state, "resolved_query 필드가 없습니다"
        assert "multi_turn_confidence" in state, "multi_turn_confidence 필드가 없습니다"
        assert "multi_turn_reasoning" in state, "multi_turn_reasoning 필드가 없습니다"
        assert "conversation_history" in state, "conversation_history 필드가 없습니다"
        assert "conversation_context" in state, "conversation_context 필드가 없습니다"

        print("✓ 멀티턴 필드 모두 존재")

    def test_resolve_multi_turn_node(self, workflow):
        """멀티턴 해결 노드 테스트"""
        # 테스트 상태 생성
        state = create_initial_legal_state("그것에 대해 더 자세히 알려주세요", "test_session_001")

        # 대화 맥락 시뮬레이션을 위해 conversation_manager에 직접 엑세스
        if workflow.conversation_manager:
            workflow.conversation_manager.sessions = self._create_test_context()

        # 멀티턴 해결 노드 실행
        result_state = workflow.resolve_multi_turn(state)

        # 결과 확인
        assert "is_multi_turn" in result_state
        assert "resolved_query" in result_state
        assert "original_query" in result_state

        print(f"✓ 멀티턴 노드 실행: is_multi_turn={result_state.get('is_multi_turn')}")
        print(f"  Original: {result_state.get('original_query')}")
        print(f"  Resolved: {result_state.get('resolved_query')}")

    def test_single_turn_question(self, workflow):
        """단일 턴 질문 처리 테스트"""
        state = create_initial_legal_state("손해배상 청구 방법을 알려주세요", "test_session_001")

        result_state = workflow.resolve_multi_turn(state)

        # 단일 턴 질문이므로 is_multi_turn은 False여야 함
        assert result_state.get("is_multi_turn") == False
        assert result_state.get("resolved_query") == state["query"]

        print("✓ 단일 턴 질문 처리 성공")

    def test_workflow_graph_includes_multi_turn_node(self, workflow):
        """워크플로우 그래프에 멀티턴 노드가 포함되어 있는지 테스트"""
        nodes = workflow.graph.nodes.keys()

        assert "resolve_multi_turn" in nodes, "resolve_multi_turn 노드가 그래프에 없습니다"

        print(f"✓ 워크플로우 노드: {list(nodes)}")

    def _create_test_context(self):
        """테스트용 대화 맥락 생성"""
        from datetime import datetime

        from source.services.conversation_manager import (
            ConversationContext,
            ConversationManager,
        )

        manager = ConversationManager()
        session_id = "test_session_001"

        manager.add_turn(
            session_id,
            "손해배상 청구 방법을 알려주세요",
            "민법 제750조에 따른 손해배상 청구 방법을 설명드리겠습니다...",
            "legal_advice"
        )

        manager.add_turn(
            session_id,
            "계약 해지 절차는 어떻게 되나요?",
            "계약 해지 절차는 다음과 같습니다...",
            "procedure_guide"
        )

        return manager.sessions


def test_multi_turn_integration():
    """멀티턴 통합 테스트 실행"""
    print("\n=== LangGraph 멀티턴 통합 테스트 시작 ===\n")

    # 설정 로드
    try:
        config = LangGraphConfig.from_env()
        print("✓ 설정 로드 성공")
    except Exception as e:
        print(f"✗ 설정 로드 실패: {e}")
        return

    # 워크플로우 초기화
    try:
        workflow = EnhancedLegalQuestionWorkflow(config)
        print("✓ 워크플로우 초기화 성공")

        # 멀티턴 핸들러 확인
        if workflow.multi_turn_handler:
            print("✓ 멀티턴 핸들러 초기화 완료")
        else:
            print("⚠ 멀티턴 핸들러가 초기화되지 않음")

        if workflow.conversation_manager:
            print("✓ 대화 관리자 초기화 완료")
        else:
            print("⚠ 대화 관리자가 초기화되지 않음")

    except Exception as e:
        print(f"✗ 워크플로우 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    # 워크플로우 노드 확인
    try:
        nodes = list(workflow.graph.nodes.keys())
        print(f"\n워크플로우 노드 목록: {nodes}")

        if "resolve_multi_turn" in nodes:
            print("✓ 멀티턴 노드가 워크플로우에 통합됨")
        else:
            print("✗ 멀티턴 노드가 워크플로우에 없음")

    except Exception as e:
        print(f"✗ 노드 확인 실패: {e}")

    print("\n=== LangGraph 멀티턴 통합 테스트 완료 ===\n")


if __name__ == "__main__":
    test_multi_turn_integration()
