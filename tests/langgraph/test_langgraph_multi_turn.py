# -*- coding: utf-8 -*-
"""
LangGraph 멀티턴 로직 통합 테스트
"""

import sys
from pathlib import Path

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # pytest가 없으면 unittest로 대체
    import unittest
    pytest = unittest

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime

from source.services.conversation_manager import ConversationManager, ConversationTurn
from source.services.langgraph.legal_workflow_enhanced import (
    EnhancedLegalQuestionWorkflow,
)
from source.services.langgraph.state_definitions import create_initial_legal_state
from source.utils.langgraph_config import LangGraphConfig


def test_multi_turn_integration_direct():
    """LangGraph 멀티턴 통합 테스트 (직접 실행)"""
    print("\n=== LangGraph 멀티턴 통합 테스트 시작 (직접 실행) ===\n")

    try:
        # 설정 로드
        config = LangGraphConfig.from_env()
        workflow = EnhancedLegalQuestionWorkflow(config)
        print("✓ 워크플로우 초기화 성공")

        # 멀티턴 핸들러 초기화 확인
        if hasattr(workflow, 'multi_turn_handler') and workflow.multi_turn_handler:
            print("✓ 멀티턴 핸들러 초기화 완료")
        else:
            print("⚠ 멀티턴 핸들러가 초기화되지 않음")

        if hasattr(workflow, 'conversation_manager') and workflow.conversation_manager:
            print("✓ 대화 관리자 초기화 완료")
        else:
            print("⚠ 대화 관리자가 초기화되지 않음")

        # 상태 정의 멀티턴 필드 확인
        state = create_initial_legal_state("테스트 질문", "test_session")
        multi_turn_fields = ["is_multi_turn", "original_query", "resolved_query",
                            "multi_turn_confidence", "multi_turn_reasoning",
                            "conversation_history", "conversation_context"]
        missing_fields = [f for f in multi_turn_fields if f not in state]
        if missing_fields:
            print(f"⚠ 누락된 멀티턴 필드: {missing_fields}")
        else:
            print("✓ 멀티턴 필드 모두 존재")

        # 워크플로우 노드 확인
        if hasattr(workflow, 'graph'):
            nodes = list(workflow.graph.nodes.keys())
            if "resolve_multi_turn" in nodes:
                print("✓ 멀티턴 노드가 워크플로우에 통합됨")
            else:
                print("⚠ 멀티턴 노드가 워크플로우에 없음")

        print("\n=== LangGraph 멀티턴 통합 테스트 완료 ===\n")
        return True

    except Exception as e:
        print(f"✗ 테스트 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


class TestLangGraphMultiTurnIntegration:
    """LangGraph 멀티턴 통합 테스트 (pytest용)"""

    def __init__(self):
        """pytest fixture 없이 실행 가능하도록 초기화"""
        # pytest 없이 직접 실행 가능하도록 항상 초기화
        self.config = LangGraphConfig.from_env()
        self.workflow = EnhancedLegalQuestionWorkflow(self.config)
        self.conversation_manager = self._create_conversation_manager()

    def _create_conversation_manager(self):
        """대화 관리자 생성"""
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

    # pytest가 있을 때만 fixture 정의
    if PYTEST_AVAILABLE:
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

    def test_multi_turn_handler_initialization(self, workflow=None):
        """멀티턴 핸들러 초기화 테스트"""
        # pytest fixture가 주입되지 않으면 인스턴스 변수 사용
        if workflow is None:
            if not hasattr(self, 'workflow'):
                self.__init__()
            workflow = self.workflow
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

    def test_resolve_multi_turn_node(self, workflow=None):
        """멀티턴 해결 노드 테스트"""
        # pytest fixture가 주입되지 않으면 인스턴스 변수 사용
        if workflow is None:
            if not hasattr(self, 'workflow'):
                self.__init__()
            workflow = self.workflow
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

    def test_single_turn_question(self, workflow=None):
        """단일 턴 질문 처리 테스트"""
        # pytest fixture가 주입되지 않으면 인스턴스 변수 사용
        if workflow is None:
            if not hasattr(self, 'workflow'):
                self.__init__()
            workflow = self.workflow
        state = create_initial_legal_state("손해배상 청구 방법을 알려주세요", "test_session_001")

        result_state = workflow.resolve_multi_turn(state)

        # 단일 턴 질문이므로 is_multi_turn은 False여야 함
        assert result_state.get("is_multi_turn") == False
        assert result_state.get("resolved_query") == state["query"]

        print("✓ 단일 턴 질문 처리 성공")

    def test_workflow_graph_includes_multi_turn_node(self, workflow=None):
        """워크플로우 그래프에 멀티턴 노드가 포함되어 있는지 테스트"""
        # pytest fixture가 주입되지 않으면 인스턴스 변수 사용
        if workflow is None:
            if not hasattr(self, 'workflow'):
                self.__init__()
            workflow = self.workflow
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
    # pytest가 없으면 직접 실행 가능한 함수 호출
    if PYTEST_AVAILABLE:
        test_multi_turn_integration()
    else:
        test_multi_turn_integration_direct()
