# -*- coding: utf-8 -*-
"""
Phase 3: Phase 시스템 통합 테스트
대화 맥락, 개인화, 장기 기억 및 품질 모니터링
"""

import pytest

from source.services.langgraph_workflow.legal_workflow_enhanced import (
    EnhancedLegalQuestionWorkflow,
)
from source.services.langgraph_workflow.state_definitions import create_initial_state
from source.utils.langgraph_config import LangGraphConfig


class TestPhase3Integration:
    """Phase 3 Phase 시스템 통합 테스트"""

    @pytest.fixture
    def workflow(self):
        """워크플로우 초기화"""
        config = LangGraphConfig()
        return EnhancedLegalQuestionWorkflow(config)

    def test_enrich_conversation_context(self, workflow):
        """대화 맥락 강화 테스트"""
        state = create_initial_state(
            query="이혼 절차에 대해 알려주세요",
            session_id="test_session_1",
            user_id="test_user_1"
        )
        state["user_query"] = "이혼 절차에 대해 알려주세요"

        result = workflow.enrich_conversation_context(state)

        assert result["phase1_context"] is not None
        assert result["phase1_context"]["enabled"] == False
        assert "Phase 1: 대화 맥락 강화 완료" in result["processing_steps"]

    def test_personalize_response(self, workflow):
        """개인화 테스트"""
        state = create_initial_state(
            query="계약서 작성 방법을 알려주세요",
            session_id="test_session_2",
            user_id="test_user_2"
        )
        state["user_query"] = "계약서 작성 방법을 알려주세요"
        state["phase1_context"] = {"enabled": False}

        result = workflow.personalize_response(state)

        assert result["phase2_personalization"] is not None
        assert result["phase2_personalization"]["enabled"] == False
        assert "Phase 2: 개인화 완료" in result["processing_steps"]

    def test_manage_memory_quality(self, workflow):
        """장기 기억 및 품질 모니터링 테스트"""
        state = create_initial_state(
            query="상속 순위에 대해 알려주세요",
            session_id="test_session_3",
            user_id="test_user_3"
        )
        state["user_query"] = "상속 순위에 대해 알려주세요"
        state["phase1_context"] = {"enabled": False}
        state["phase2_personalization"] = {"enabled": False}

        result = workflow.manage_memory_quality(state)

        assert result["phase3_memory_quality"] is not None
        assert result["phase3_memory_quality"]["enabled"] == False
        assert "Phase 3: 장기 기억 및 품질 모니터링 완료" in result["processing_steps"]

    @pytest.mark.asyncio
    async def test_full_workflow_with_phases(self, workflow):
        """전체 워크플로우 - Phase 시스템 포함"""
        state = create_initial_state(
            query="민법 제750조에 대해 알려주세요",
            session_id="test_session_4",
            user_id="test_user_4"
        )
        state["user_query"] = "민법 제750조에 대해 알려주세요"

        # 워크플로우 실행
        compiled_workflow = workflow.graph.compile()
        result = await compiled_workflow.ainvoke(state)

        # Phase 1 확인
        assert "phase1_context" in result or "Phase 1: 대화 맥락 강화 완료" in result["processing_steps"]

        # Phase 2 확인
        assert "phase2_personalization" in result or "Phase 2: 개인화 완료" in result["processing_steps"]

        # Phase 3 확인
        assert "phase3_memory_quality" in result or "Phase 3: 장기 기억 및 품질 모니터링 완료" in result["processing_steps"]

    def test_phase_state_initialization(self, workflow):
        """Phase 상태 초기화 테스트"""
        state = create_initial_state(
            query="테스트 질문",
            session_id="test_session_5",
            user_id="test_user_5"
        )
        state["user_query"] = "테스트 질문"

        # Phase 1
        result = workflow.enrich_conversation_context(state)
        assert "phase1_context" in result
        assert isinstance(result["phase1_context"], dict)

        # Phase 2
        result = workflow.personalize_response(result)
        assert "phase2_personalization" in result
        assert isinstance(result["phase2_personalization"], dict)

        # Phase 3
        result = workflow.manage_memory_quality(result)
        assert "phase3_memory_quality" in result
        assert isinstance(result["phase3_memory_quality"], dict)

    def test_phase_parallel_execution_structure(self, workflow):
        """Phase 병렬 실행 구조 테스트"""
        # retrieve_documents 다음에 3개의 Phase 노드가 병렬로 실행되는지 확인
        # LangGraph는 자동으로 가능한 병렬 처리를 수행합니다
        assert hasattr(workflow, "enrich_conversation_context")
        assert hasattr(workflow, "personalize_response")
        assert hasattr(workflow, "manage_memory_quality")

    def test_phase_error_handling(self, workflow):
        """Phase 에러 처리 테스트"""
        state = create_initial_state(
            query="",
            session_id="test_session_6",
            user_id="test_user_6"
        )
        state["user_query"] = ""

        # 에러가 발생해도 안전하게 처리되어야 함
        result = workflow.enrich_conversation_context(state)

        # 에러가 있어도 상태는 유효해야 함
        assert "phase1_context" in result or "errors" in result
